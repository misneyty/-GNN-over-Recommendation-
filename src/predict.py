"""批量计算作者-论文分数，并生成 Kaggle 提交文件。"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from .graph import build_bipartite_adjacency
from .utils import ensure_dir

Adjacency = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]


def score_edges(
    model_name: str,
    model: torch.nn.Module,
    edges: np.ndarray,
    num_authors: int,
    adj: Adjacency,
    device: Optional[torch.device],
    batch_size: int = 131072,
    calibrator: Optional[Any] = None,
    structural_features: Optional[Any] = None,
) -> np.ndarray:
    """分批计算作者-论文 pair 的概率，并可选执行后置校准。

    为避免重复进行昂贵的全图消息传播，节点嵌入只在批处理开始前
    计算一次。之后每个 batch 只读取对应作者和论文的向量进行打分。
    """
    device = device or torch.device("cpu")
    model.eval()
    scores: list[np.ndarray] = []

    with torch.no_grad():
        # 不同模型使用不同形式的图表示，但最终都输出 pair 的 logit。
        node_embeddings = None
        if model_name == "lightgcn":
            if adj is None:
                raise ValueError("LightGCN prediction requires adjacency")
            node_embeddings = model.encode(adj.to(device))
        elif model_name == "heterognn":
            if adj is None:
                raise ValueError("HeteroGNN prediction requires relation adjacencies")
            relation_adjacencies = {
                name: value.to(device) for name, value in adj.items()
            }
            author_embeddings, paper_embeddings = model.encode(relation_adjacencies)

        # 大测试集分批处理，避免一次性创建过大的 GPU 张量。
        for start in range(0, len(edges), batch_size):
            batch = edges[start : start + batch_size]
            author_ids = torch.as_tensor(
                batch[:, 0],
                dtype=torch.long,
                device=device,
            )
            paper_ids = torch.as_tensor(
                batch[:, 1],
                dtype=torch.long,
                device=device,
            )
            if model_name == "lightgcn":
                logits = model.score(
                    node_embeddings,
                    author_ids,
                    paper_ids,
                )
            elif model_name == "heterognn":
                logits = model.score(
                    author_embeddings,
                    paper_embeddings,
                    author_ids,
                    paper_ids,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    # 神经网络输出先经过 sigmoid，得到属于正样本的基础概率。
    model_scores = np.concatenate(scores)
    if calibrator is None:
        return model_scores
    if structural_features is None:
        raise ValueError("Calibrated prediction requires structural features")
    # 校准器同时使用神经模型分数和 89 个结构/语义特征。
    pair_features = structural_features.transform(edges, device=device)
    return calibrator.predict_proba(model_scores, pair_features)


def write_submission(
    scores: np.ndarray,
    output_path: Union[str, Path],
    threshold: float,
) -> Path:
    """按照竞赛要求的列名和行顺序写出二分类预测。"""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    predicted = (scores >= threshold).astype(np.int64)
    # Index 必须严格对应测试文件原始行号，因此不能对测试 pair 去重。
    submission = pd.DataFrame(
        {
            "Index": np.arange(len(predicted)),
            "Predicted": predicted,
        }
    )
    submission.to_csv(output_path, index=False)
    return output_path


def build_prediction_adjacency(dataset, train_edges: np.ndarray) -> torch.Tensor:
    """构建 LightGCN 在测试阶段使用的作者-论文图。"""
    return build_bipartite_adjacency(
        dataset.num_authors,
        dataset.num_papers,
        train_edges,
    )
