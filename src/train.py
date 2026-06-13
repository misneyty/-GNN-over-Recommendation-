"""LightGCN 与 HeteroGNN 的单模型训练流程。"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import Dataset
from .evaluate import classification_metrics, search_best_threshold
from .graph import build_bipartite_adjacency, build_relation_adjacencies
from .models import HeteroGNN, HeteroGNNCalibrator, LightGCN
from .negative_sampling import sample_negative_edges
from .predict import score_edges
from .split import train_valid_split
from .structural_features import StructuralFeatureStore

Adjacency = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]


def build_author_features(
    num_authors: int,
    paper_features: np.ndarray,
    train_edges: np.ndarray,
) -> np.ndarray:
    """将作者历史论文的特征取平均，得到作者初始语义画像。

    输出形状为 ``[作者数, 论文特征维度]``。没有历史论文的作者保留
    全零向量，后续可训练作者嵌入仍能为其学习表示。
    """
    # 先累加每位作者关联论文的特征，再除以该作者的历史论文数量。
    author_features = np.zeros(
        (num_authors, paper_features.shape[1]),
        dtype=np.float32,
    )
    author_counts = np.bincount(
        train_edges[:, 0],
        minlength=num_authors,
    ).astype(np.float32)
    np.add.at(
        author_features,
        train_edges[:, 0],
        paper_features[train_edges[:, 1]],
    )

    authors_with_history = author_counts > 0
    author_features[authors_with_history] /= author_counts[
        authors_with_history,
        None,
    ]
    return author_features


@dataclass
class TrainResult:
    """单模型训练入口返回的模型、图结构和验证结果。"""

    model: torch.nn.Module
    adj: Adjacency
    best_threshold: float
    valid_metrics: dict[str, float]
    calibrator: Optional[HeteroGNNCalibrator] = None
    structural_features: Optional[StructuralFeatureStore] = None


def make_edge_loader(
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """把等量正负边合并、打乱并构造成小批量数据加载器。"""
    edges = np.vstack([pos_edges, neg_edges])
    labels = np.concatenate(
        [
            np.ones(len(pos_edges), dtype=np.float32),
            np.zeros(len(neg_edges), dtype=np.float32),
        ]
    )
    # 边与标签必须使用同一个排列，确保每个 pair 的监督标签正确。
    perm = np.random.permutation(len(edges))
    edges = edges[perm]
    labels = labels[perm]
    dataset = TensorDataset(
        torch.as_tensor(edges[:, 0], dtype=torch.long),
        torch.as_tensor(edges[:, 1], dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def split_calibration_indices(
    num_positive: int,
    num_negative: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把验证 pair 划分为校准器训练、阈值调优和独立评估三部分。

    正样本和负样本分别打乱后按 40%/30%/30% 划分，确保三部分中的
    类别比例基本一致。最后 30% 在阈值和校准器确定前不会参与拟合。
    """
    rng = np.random.default_rng(seed + 2026)
    positive = rng.permutation(num_positive)
    negative = rng.permutation(num_negative) + num_positive
    positive_fit = int(num_positive * 0.4)
    positive_tune = int(num_positive * 0.7)
    negative_fit = int(num_negative * 0.4)
    negative_tune = int(num_negative * 0.7)
    # fit_indices 用于训练初始校准器。
    fit_indices = np.concatenate([positive[:positive_fit], negative[:negative_fit]])
    # tune_indices 只用于选择使 F1 最大的分类阈值。
    tune_indices = np.concatenate(
        [
            positive[positive_fit:positive_tune],
            negative[negative_fit:negative_tune],
        ]
    )
    # evaluation_indices 是最终报告指标时使用的封存子集。
    evaluation_indices = np.concatenate(
        [positive[positive_tune:], negative[negative_tune:]]
    )
    return fit_indices, tune_indices, evaluation_indices


def train(
    dataset: Dataset,
    model_name: str = "lightgcn",
    epochs: int = 20,
    dim: int = 64,
    layers: int = 2,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    valid_ratio: float = 0.1,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> TrainResult:
    """训练指定模型，并在留出的验证边上评估。

    该函数适合基础单模型实验。最终提交使用的 60/80/100 快照流程位于
    ``full_pipeline.py``，两条路线共享相同的建图和模型实现。
    """
    device = device or torch.device("cuda")
    # 验证正边从已知作者-论文边中留出，不参与训练图构建。
    train_positive, valid_positive = train_valid_split(
        dataset.train_edges,
        valid_ratio,
        seed,
    )
    valid_negative = sample_negative_edges(
        dataset.num_authors,
        dataset.num_papers,
        dataset.train_edges,
        len(valid_positive),
        seed=seed + 1,
    )

    loader = None
    if model_name == "lightgcn":
        # LightGCN 的负样本只生成一次；HeteroGNN 会在每轮重新采样。
        train_negative = sample_negative_edges(
            dataset.num_authors,
            dataset.num_papers,
            dataset.train_edges,
            len(train_positive),
            seed=seed,
        )
        loader = make_edge_loader(
            train_positive,
            train_negative,
            batch_size,
        )

    if model_name == "lightgcn":
        # 基线只使用作者-论文二部图，不使用内容特征和同类型关系。
        model = LightGCN(
            dataset.num_authors,
            dataset.num_papers,
            dim,
            layers,
        ).to(device)
        adjacency = build_bipartite_adjacency(
            dataset.num_authors,
            dataset.num_papers,
            train_positive,
        ).to(device)
    elif model_name == "heterognn":
        # 论文直接使用 feature.pkl，作者使用历史论文平均特征。
        paper_features = torch.as_tensor(
            dataset.paper_features,
            dtype=torch.float32,
        )
        author_features = torch.as_tensor(
            build_author_features(
                dataset.num_authors,
                dataset.paper_features,
                train_positive,
            ),
            dtype=torch.float32,
        )
        model = HeteroGNN(
            dataset.num_authors,
            paper_features,
            author_features,
            hidden_dim=dim,
            out_dim=dim,
            num_layers=layers,
        ).to(device)
        # 异构图同时包含作者-论文、作者-作者和论文-论文关系。
        adjacency = build_relation_adjacencies(
            dataset.num_authors,
            dataset.num_papers,
            train_positive,
            dataset.coauthor_edges,
            dataset.citation_edges,
        )
        adjacency = {name: value.to(device) for name, value in adjacency.items()}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # BCEWithLogitsLoss 内部组合 sigmoid 与二元交叉熵，数值更稳定。
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs), desc=f"train-{model_name}"):
        model.train()
        if model_name == "heterognn":
            # 每轮重新采样负边，让模型接触更多未观察到的作者-论文组合。
            train_negative = sample_negative_edges(
                dataset.num_authors,
                dataset.num_papers,
                dataset.train_edges,
                len(train_positive),
                seed=seed + epoch,
            )
            epoch_loader = make_edge_loader(
                train_positive,
                train_negative,
                batch_size,
            )
        else:
            epoch_loader = loader

        if epoch_loader is None:
            raise ValueError(f"Unsupported model: {model_name}")

        for author_ids, paper_ids, labels in epoch_loader:
            # 把当前 batch 的节点编号和标签移动到训练设备。
            author_ids = author_ids.to(device)
            paper_ids = paper_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_name == "lightgcn":
                # 每次参数更新后都需要重新计算全图节点表示。
                node_embeddings = model.encode(adjacency)
                logits = model.score(
                    node_embeddings,
                    author_ids,
                    paper_ids,
                )
            elif model_name == "heterognn":
                # 先对异构图执行消息传播，再对当前 batch 的 pair 打分。
                author_embeddings, paper_embeddings = model.encode(adjacency)
                logits = model.score(
                    author_embeddings,
                    paper_embeddings,
                    author_ids,
                    paper_ids,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            loss = loss_function(logits, labels)
            # 反向传播计算梯度，随后由 Adam 更新全部可训练参数。
            loss.backward()
            optimizer.step()

    # 验证集由等量正边和负边组成，便于比较 F1 与 Accuracy。
    valid_edges = np.vstack([valid_positive, valid_negative])
    valid_labels = np.concatenate(
        [
            np.ones(len(valid_positive), dtype=np.int64),
            np.zeros(len(valid_negative), dtype=np.int64),
        ]
    )
    model_scores = score_edges(
        model_name,
        model,
        valid_edges,
        dataset.num_authors,
        adjacency,
        device,
    )
    calibrator = None
    structural_features = None
    if model_name == "heterognn":
        # 结构特征只能基于训练图构造，不能看到验证正边，避免数据泄漏。
        structural_features = StructuralFeatureStore(
            dataset.num_authors,
            dataset.num_papers,
            train_positive,
            dataset.coauthor_edges,
            dataset.citation_edges,
            dataset.paper_features,
        )
        pair_features = structural_features.transform(valid_edges, device=device)

        calibration_indices, threshold_indices, evaluation_indices = (
            split_calibration_indices(
                len(valid_positive),
                len(valid_negative),
                seed,
            )
        )

        selection_calibrator = HeteroGNNCalibrator(seed=seed)
        # 第一阶段只在 40% 验证样本上拟合校准器。
        selection_calibrator.fit(
            model_scores[calibration_indices],
            pair_features[calibration_indices],
            valid_labels[calibration_indices],
        )
        threshold_scores = selection_calibrator.predict_proba(
            model_scores[threshold_indices],
            pair_features[threshold_indices],
        )
        best_threshold, _ = search_best_threshold(
            valid_labels[threshold_indices],
            threshold_scores,
        )
        # 阈值确定后，用前 70% 数据重新训练用于最终预测的校准器。
        final_fit_indices = np.concatenate([calibration_indices, threshold_indices])
        calibrator = HeteroGNNCalibrator(seed=seed)
        calibrator.fit(
            model_scores[final_fit_indices],
            pair_features[final_fit_indices],
            valid_labels[final_fit_indices],
        )
        evaluation_scores = calibrator.predict_proba(
            model_scores[evaluation_indices],
            pair_features[evaluation_indices],
        )
        # 最后 30% 样本仅用于报告泛化指标。
        valid_metrics = classification_metrics(
            valid_labels[evaluation_indices],
            evaluation_scores,
            best_threshold,
        )
    else:
        best_threshold, valid_metrics = search_best_threshold(
            valid_labels,
            model_scores,
        )

    return TrainResult(
        model=model,
        adj=adjacency,
        best_threshold=best_threshold,
        valid_metrics=valid_metrics,
        calibrator=calibrator,
        structural_features=structural_features,
    )
