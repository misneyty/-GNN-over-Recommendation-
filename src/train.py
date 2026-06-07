import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .data import Dataset
from .evaluate import search_best_threshold
from .graph import build_bipartite_adjacency
from .models import LightGCN, MatrixFactorization
from .negative_sampling import sample_negative_edges
from .predict import score_edges
from .split import train_valid_split
from typing import Optional

class TrainResult:
    def __init__(
        self,
        model: torch.nn.Module,
        adj: Optional[torch.Tensor],
        best_threshold: float,
        valid_metrics: dict[str, float],
    ):
        self.model = model
        self.adj = adj
        self.best_threshold = best_threshold
        self.valid_metrics = valid_metrics#验证集指标

#将正负样本边合并
def make_edge_loader(
    pos_edges: np.ndarray,#正样本边（label=1）
    neg_edges: np.ndarray,#负样本边（label=0）
    batch_size: int,#批量大小
) -> DataLoader:
    edges = np.vstack([pos_edges, neg_edges])
    labels = np.concatenate([
        np.ones(len(pos_edges), dtype=np.float32),
        np.zeros(len(neg_edges), dtype=np.float32),
    ])
    perm = np.random.permutation(len(edges))
    edges = edges[perm]
    labels = labels[perm]
    #将Numpy数组转换为PyTorch张量，创建数据加载器
    dataset = TensorDataset(
        torch.as_tensor(edges[:, 0], dtype=torch.long),
        torch.as_tensor(edges[:, 1], dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    dataset: Dataset,
    model_name: str = "lightgcn",
    epochs: int = 20,
    dim: int = 64,
    layers: int = 2,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,#权重衰退，用于减少过拟合
    valid_ratio: float = 0.1,
    seed: int = 0,
    device: Optional[torch.device] = None
) -> TrainResult:
    device = device or torch.device("cuda")
    train_pos, valid_pos = train_valid_split(dataset.train_edges, valid_ratio, seed)
    train_neg = sample_negative_edges(
        dataset.num_authors,
        dataset.num_papers,
        dataset.train_edges,
        len(train_pos),
        seed=seed,
    )
    valid_neg = sample_negative_edges(
        dataset.num_authors,
        dataset.num_papers,
        dataset.train_edges,
        len(valid_pos),
        seed=seed + 1,
    )
    #数据加载器
    loader = make_edge_loader(train_pos, train_neg, batch_size)

    if model_name == "mf":
        model = MatrixFactorization(dataset.num_authors, dataset.num_papers, dim).to(device)
        adj = None
    elif model_name == "lightgcn":
        model = LightGCN(dataset.num_authors, dataset.num_papers, dim, layers).to(device)
        adj = build_bipartite_adjacency(dataset.num_authors, dataset.num_papers, train_pos).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in tqdm(range(epochs), desc=f"train-{model_name}"):
        model.train()
        for author_ids, paper_ids, labels in loader:
            author_ids = author_ids.to(device)
            paper_ids = paper_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_name == "mf":
                logits = model(author_ids, paper_ids)
            else:
                z = model.encode(adj)
                logits = model.score(z, author_ids, paper_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    valid_edges = np.vstack([valid_pos, valid_neg])
    y_true = np.concatenate([
        np.ones(len(valid_pos), dtype=np.int64),
        np.zeros(len(valid_neg), dtype=np.int64),
    ])
    y_score = score_edges(model_name, model, valid_edges, dataset.num_authors, adj, device)
    best_threshold, valid_metrics = search_best_threshold(y_true, y_score)
    return TrainResult(model=model, adj=adj, best_threshold=best_threshold, valid_metrics=valid_metrics)
