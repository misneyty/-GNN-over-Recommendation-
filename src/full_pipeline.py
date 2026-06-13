"""可复现的 HeteroGNN 快照训练、校准、评估与预测完整流程。"""

from __future__ import annotations

import copy
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import Dataset
from .evaluate import classification_metrics, search_best_threshold
from .graph import build_relation_adjacencies
from .models import HeteroGNN, HeteroGNNCalibrator
from .negative_sampling import sample_negative_edges
from .predict import score_edges, write_submission
from .split import train_valid_split
from .structural_features import StructuralFeatureStore
from .train import build_author_features, split_calibration_indices
from .utils import configure_reproducibility, save_checkpoint


@dataclass
class FullPipelineResult:
    """完整端到端流程产生的关键结果与输出路径。"""

    best_threshold: float
    tune_metrics: dict[str, float]
    valid_metrics: dict[str, float]
    checkpoint_path: Path
    submission_path: Path
    snapshot_epochs: tuple[int, ...]


def _make_reproducible_loader(
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """为一个训练轮次构造可复现、正负平衡的数据加载器。

    NumPy 负责第一次固定排列，PyTorch ``Generator`` 控制 DataLoader
    的 shuffle。两者都使用由 epoch 推导出的固定种子。
    """
    edges = np.vstack([positive_edges, negative_edges])
    labels = np.concatenate(
        [
            np.ones(len(positive_edges), dtype=np.float32),
            np.zeros(len(negative_edges), dtype=np.float32),
        ]
    )
    # 先固定一次全局排列，使边和标签在进入 TensorDataset 前保持对应。
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(edges))
    dataset = TensorDataset(
        torch.as_tensor(edges[permutation, 0], dtype=torch.long),
        torch.as_tensor(edges[permutation, 1], dtype=torch.long),
        torch.as_tensor(labels[permutation], dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """复制一份位于 CPU 的模型参数，避免后续训练修改快照。

    ``clone`` 非常重要：如果只保存对原张量的引用，继续训练会导致
    之前的快照也被同步覆盖。
    """
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _build_model(
    dataset: Dataset,
    train_edges: np.ndarray,
    dim: int,
    layers: int,
    device: torch.device,
) -> HeteroGNN:
    """构造 HeteroGNN 的节点输入，并把模型移动到指定设备。"""
    # 论文节点直接使用原始 512 维内容特征。
    paper_features = torch.as_tensor(
        dataset.paper_features,
        dtype=torch.float32,
    )
    # 作者节点使用训练图中历史论文特征的平均值。
    author_features = torch.as_tensor(
        build_author_features(
            dataset.num_authors,
            dataset.paper_features,
            train_edges,
        ),
        dtype=torch.float32,
    )
    return HeteroGNN(
        dataset.num_authors,
        paper_features,
        author_features,
        hidden_dim=dim,
        out_dim=dim,
        num_layers=layers,
    ).to(device)


def _train_snapshots(
    dataset: Dataset,
    train_edges: np.ndarray,
    adjacency: dict[str, torch.Tensor],
    snapshot_epochs: Sequence[int],
    dim: int,
    layers: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[HeteroGNN, Dict[str, Dict[str, torch.Tensor]]]:
    """连续训练一次，并在指定 epoch 保存内存快照。

    这种方式避免从头独立训练三个模型，同时让 60、80、100 轮模型
    提供略有差异的判断，供后续校准器进行快照集成。
    """
    model = _build_model(dataset, train_edges, dim, layers, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    requested_epochs = set(snapshot_epochs)
    last_epoch = max(snapshot_epochs)
    snapshots: Dict[str, Dict[str, torch.Tensor]] = {}

    for epoch in tqdm(
        range(1, last_epoch + 1),
        desc="train-full-heterognn",
    ):
        # 每个 epoch 使用不同但可复现的随机种子重新生成负样本。
        negative_edges = sample_negative_edges(
            dataset.num_authors,
            dataset.num_papers,
            dataset.train_edges,
            len(train_edges),
            seed=seed + epoch - 1,
        )
        loader = _make_reproducible_loader(
            train_edges,
            negative_edges,
            batch_size,
            seed=seed * 1000 + epoch,
        )
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for author_ids, paper_ids, labels in loader:
            author_ids = author_ids.to(device)
            paper_ids = paper_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # 全图消息传播得到节点嵌入，再抽取当前 batch 的 pair 计算损失。
            author_embeddings, paper_embeddings = model.encode(adjacency)
            logits = model.score(
                author_embeddings,
                paper_embeddings,
                author_ids,
                paper_ids,
            )
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(labels)
            sample_count += len(labels)

        if epoch in requested_epochs:
            # 快照只保存在内存，完整训练结束后统一写入 checkpoint。
            name = f"epoch_{epoch}"
            snapshots[name] = _cpu_state_dict(model)
            print(
                f"Saved in-memory snapshot {name}; "
                f"loss={epoch_loss / sample_count:.6f}"
            )

    return model, snapshots


def _score_snapshots(
    model: HeteroGNN,
    snapshots: Dict[str, Dict[str, torch.Tensor]],
    edges: np.ndarray,
    num_authors: int,
    adjacency: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    """依次加载每个快照，对同一批边分别计算模型概率。

    返回矩阵形状为 ``[pair 数, 快照数]``，每一列对应一个 epoch。
    """
    columns = []
    for state in snapshots.values():
        # 模型结构不变，只替换当前快照的参数。
        model.load_state_dict(state)
        columns.append(
            score_edges(
                "heterognn",
                model,
                edges,
                num_authors,
                adjacency,
                device,
            )
        )
    return np.column_stack(columns)


def _environment_metadata() -> dict[str, str]:
    """记录复现实验所需的软件版本和 GPU 信息。"""
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }


def run_full_pipeline(
    dataset: Dataset,
    data_dir: str,
    checkpoint_path: str,
    submission_path: str,
    snapshot_epochs: Sequence[int] = (60, 80, 100),
    dim: int = 64,
    layers: int = 2,
    batch_size: int = 65536,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    valid_ratio: float = 0.1,
    seed: int = 0,
    device: torch.device | None = None,
) -> FullPipelineResult:
    """执行训练、校准、独立评估、测试预测和结果保存。

    这是老师从零运行项目时使用的最终入口。所有验证结构特征都只基于
    训练子图构建，测试集仅在模型和阈值确定后参与预测。
    """
    if not snapshot_epochs:
        raise ValueError("snapshot_epochs must not be empty")
    snapshot_epochs = tuple(sorted(set(int(x) for x in snapshot_epochs)))
    if snapshot_epochs[0] <= 0:
        raise ValueError("snapshot epochs must be positive")

    configure_reproducibility(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 阶段 1：固定划分训练正边和验证正边，并生成等量验证负边。
    train_edges, valid_positive = train_valid_split(
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
    valid_edges = np.vstack([valid_positive, valid_negative])
    valid_labels = np.concatenate(
        [
            np.ones(len(valid_positive), dtype=np.int64),
            np.zeros(len(valid_negative), dtype=np.int64),
        ]
    )

    # 阶段 2：只使用训练正边建立异构图，避免验证边信息泄漏。
    adjacency = build_relation_adjacencies(
        dataset.num_authors,
        dataset.num_papers,
        train_edges,
        dataset.coauthor_edges,
        dataset.citation_edges,
    )
    adjacency = {name: value.to(device) for name, value in adjacency.items()}

    # 阶段 3：从随机初始化训练一个模型，并保存 60/80/100 轮快照。
    model, snapshots = _train_snapshots(
        dataset,
        train_edges,
        adjacency,
        snapshot_epochs,
        dim,
        layers,
        batch_size,
        learning_rate,
        weight_decay,
        seed,
        device,
    )

    # 阶段 4：三个快照分别为验证 pair 打分，同时构造 89 个结构特征。
    valid_model_scores = _score_snapshots(
        model,
        snapshots,
        valid_edges,
        dataset.num_authors,
        adjacency,
        device,
    )
    feature_store = StructuralFeatureStore(
        dataset.num_authors,
        dataset.num_papers,
        train_edges,
        dataset.coauthor_edges,
        dataset.citation_edges,
        dataset.paper_features,
    )
    valid_pair_features = feature_store.transform(
        valid_edges,
        device=device,
    )
    fit_indices, tune_indices, evaluation_indices = split_calibration_indices(
        len(valid_positive),
        len(valid_negative),
        seed,
    )

    # 阶段 5：40% 样本拟合初始校准器，30% 样本搜索最佳 F1 阈值。
    selection_calibrator = HeteroGNNCalibrator(seed=seed)
    selection_calibrator.fit(
        valid_model_scores[fit_indices],
        valid_pair_features[fit_indices],
        valid_labels[fit_indices],
    )
    tune_scores = selection_calibrator.predict_proba(
        valid_model_scores[tune_indices],
        valid_pair_features[tune_indices],
    )
    best_threshold, tune_metrics = search_best_threshold(
        valid_labels[tune_indices],
        tune_scores,
    )

    # 阈值确定后，合并前 70% 样本重新训练最终校准器。
    final_fit_indices = np.concatenate([fit_indices, tune_indices])
    final_calibrator = HeteroGNNCalibrator(seed=seed)
    final_calibrator.fit(
        valid_model_scores[final_fit_indices],
        valid_pair_features[final_fit_indices],
        valid_labels[final_fit_indices],
    )
    evaluation_scores = final_calibrator.predict_proba(
        valid_model_scores[evaluation_indices],
        valid_pair_features[evaluation_indices],
    )
    valid_metrics = classification_metrics(
        valid_labels[evaluation_indices],
        evaluation_scores,
        best_threshold,
    )
    # 置换重要性用于解释快照分数和结构特征各自的贡献。
    snapshot_names = list(snapshots)
    importance = selection_calibrator.explain(
        valid_model_scores[tune_indices],
        valid_pair_features[tune_indices],
        valid_labels[tune_indices],
        feature_store.feature_names,
        model_names=snapshot_names,
    )

    # 阶段 6：模型、校准器和阈值确定后，再为全部测试候选 pair 打分。
    test_model_scores = _score_snapshots(
        model,
        snapshots,
        dataset.test_edges,
        dataset.num_authors,
        adjacency,
        device,
    )
    test_pair_features = feature_store.transform(
        dataset.test_edges,
        device=device,
    )
    test_scores = final_calibrator.predict_proba(
        test_model_scores,
        test_pair_features,
    )
    submission = write_submission(
        test_scores,
        Path(submission_path),
        best_threshold,
    )

    # 阶段 7：保存模型快照、校准器、指标、特征名和环境信息。
    training_config = {
        "data_dir": data_dir,
        "snapshot_epochs": list(snapshot_epochs),
        "dim": dim,
        "layers": layers,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "valid_ratio": valid_ratio,
        "seed": seed,
        "deterministic_algorithms": True,
        "batch_randomization": "epoch_seeded_numpy_and_pytorch_generator",
    }
    final_state = snapshots[snapshot_names[-1]]
    # model_state 保存最后一轮参数；ensemble_model_states 保存全部快照。
    save_checkpoint(
        checkpoint_path,
        model_name="heterognn_snapshot_ensemble",
        model_state=copy.deepcopy(final_state),
        ensemble_model_names=snapshot_names,
        ensemble_model_states=snapshots,
        best_threshold=best_threshold,
        tune_metrics=tune_metrics,
        valid_metrics=valid_metrics,
        num_authors=dataset.num_authors,
        num_papers=dataset.num_papers,
        calibrator=final_calibrator,
        calibrator_params=final_calibrator.model.get_params(),
        feature_names=feature_store.feature_names,
        feature_importance=importance,
        training_config=training_config,
        environment=_environment_metadata(),
    )

    checkpoint = Path(checkpoint_path)
    print("Tune metrics:", tune_metrics)
    print("Independent validation metrics:", valid_metrics)
    print("Best threshold:", best_threshold)
    print("Saved checkpoint to:", checkpoint)
    print("Saved submission to:", submission)
    return FullPipelineResult(
        best_threshold=best_threshold,
        tune_metrics=tune_metrics,
        valid_metrics=valid_metrics,
        checkpoint_path=checkpoint,
        submission_path=submission,
        snapshot_epochs=snapshot_epochs,
    )
