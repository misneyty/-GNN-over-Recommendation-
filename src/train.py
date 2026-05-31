from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Dataset
from .evaluate import search_best_threshold
from .graph import build_bipartite_adjacency, build_relation_adjacencies
from .models import LightGCN, MatrixFactorization, HeteroLightGCN
from .negative_sampling import sample_negative_edges, BPRDataset
from .predict import score_edges
from .split import train_valid_split


@dataclass
class TrainResult:
    model: torch.nn.Module
    adj: Union[torch.Tensor, dict[str, torch.Tensor], None]
    best_threshold: float
    valid_metrics: dict[str, float]


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
    device: torch.device | None = None,
) -> TrainResult:
    device = device or torch.device("cpu")
    train_pos, valid_pos = train_valid_split(dataset.train_edges, valid_ratio, seed)
    
    # Validation negatives are fixed for fair comparison
    valid_neg = sample_negative_edges(
        dataset.num_authors,
        dataset.num_papers,
        dataset.train_edges,
        len(valid_pos),
        seed=seed + 1,
    )
    
    # Use BPRDataset for dynamic negative sampling
    train_dataset = BPRDataset(train_pos, dataset.num_papers)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    adj = None
    if model_name == "mf":
        model = MatrixFactorization(dataset.num_authors, dataset.num_papers, dim).to(device)
    elif model_name == "lightgcn":
        model = LightGCN(dataset.num_authors, dataset.num_papers, dim, layers).to(device)
        adj = build_bipartite_adjacency(dataset.num_authors, dataset.num_papers, train_pos).to(device)
    elif model_name == "hetero_lightgcn":
        model = HeteroLightGCN(
            dataset.num_authors, 
            dataset.num_papers, 
            dataset.paper_features, 
            dim, 
            layers
        ).to(device)
        adj = build_relation_adjacencies(
            dataset.num_authors, 
            dataset.num_papers, 
            train_pos, 
            dataset.coauthor_edges, 
            dataset.citation_edges
        )
        for k, v in adj.items():
            adj[k] = v.to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in tqdm(range(epochs), desc=f"train-{model_name}"):
        model.train()
        for author_ids, pos_paper_ids, neg_paper_ids in loader:
            author_ids = author_ids.to(device)
            pos_paper_ids = pos_paper_ids.to(device)
            neg_paper_ids = neg_paper_ids.to(device)
            
            optimizer.zero_grad()
            if model_name == "mf":
                pos_scores = model(author_ids, pos_paper_ids)
                neg_scores = model(author_ids, neg_paper_ids)
            elif model_name == "lightgcn":
                z = model.encode(adj)
                pos_scores = model.score(z, author_ids, pos_paper_ids)
                neg_scores = model.score(z, author_ids, neg_paper_ids)
            elif model_name == "hetero_lightgcn":
                z = model.encode(adj["author_paper"], adj["author_author"], adj["paper_paper"])
                pos_scores = model.score(z, author_ids, pos_paper_ids)
                neg_scores = model.score(z, author_ids, neg_paper_ids)
                
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
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
