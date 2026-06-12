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


def build_author_features(
    num_authors: int,
    paper_features: np.ndarray,
    train_edges: np.ndarray,
) -> np.ndarray:
    author_features = np.zeros((num_authors, paper_features.shape[1]), dtype=np.float32)
    author_counts = np.bincount(train_edges[:, 0], minlength=num_authors).astype(np.float32)
    np.add.at(author_features, train_edges[:, 0], paper_features[train_edges[:, 1]])
    nonzero = author_counts > 0
    author_features[nonzero] /= author_counts[nonzero, None]
    return author_features


class TrainResult:
    def __init__(
        self,
        model: torch.nn.Module,
        adj: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
        best_threshold: float,
        valid_metrics: dict[str, float],
        calibrator: Optional[HeteroGNNCalibrator] = None,
        structural_features: Optional[StructuralFeatureStore] = None,
    ):
        self.model = model
        self.adj = adj
        self.best_threshold = best_threshold
        self.valid_metrics = valid_metrics
        self.calibrator = calibrator
        self.structural_features = structural_features


def make_edge_loader(
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    batch_size: int,
) -> DataLoader:
    edges = np.vstack([pos_edges, neg_edges])
    labels = np.concatenate(
        [
            np.ones(len(pos_edges), dtype=np.float32),
            np.zeros(len(neg_edges), dtype=np.float32),
        ]
    )
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
    rng = np.random.default_rng(seed + 2026)
    positive = rng.permutation(num_positive)
    negative = rng.permutation(num_negative) + num_positive
    positive_fit = int(num_positive * 0.4)
    positive_tune = int(num_positive * 0.7)
    negative_fit = int(num_negative * 0.4)
    negative_tune = int(num_negative * 0.7)
    fit_indices = np.concatenate(
        [positive[:positive_fit], negative[:negative_fit]]
    )
    tune_indices = np.concatenate(
        [
            positive[positive_fit:positive_tune],
            negative[negative_fit:negative_tune],
        ]
    )
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
    device = device or torch.device("cuda")
    train_pos, valid_pos = train_valid_split(dataset.train_edges, valid_ratio, seed)
    valid_neg = sample_negative_edges(
        dataset.num_authors,
        dataset.num_papers,
        dataset.train_edges,
        len(valid_pos),
        seed=seed + 1,
    )

    loader = None
    if model_name == "lightgcn":
        train_neg = sample_negative_edges(
            dataset.num_authors,
            dataset.num_papers,
            dataset.train_edges,
            len(train_pos),
            seed=seed,
        )
        loader = make_edge_loader(train_pos, train_neg, batch_size)

    if model_name == "lightgcn":
        model = LightGCN(dataset.num_authors, dataset.num_papers, dim, layers).to(device)
        adj = build_bipartite_adjacency(dataset.num_authors, dataset.num_papers, train_pos).to(device)
    elif model_name == "heterognn":
        paper_features = torch.as_tensor(dataset.paper_features, dtype=torch.float32)
        author_features = torch.as_tensor(
            build_author_features(dataset.num_authors, dataset.paper_features, train_pos),
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
        adj = build_relation_adjacencies(
            dataset.num_authors,
            dataset.num_papers,
            train_pos,
            dataset.coauthor_edges,
            dataset.citation_edges,
        )
        adj = {name: value.to(device) for name, value in adj.items()}
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs), desc=f"train-{model_name}"):
        model.train()
        if model_name == "heterognn":
            train_neg = sample_negative_edges(
                dataset.num_authors,
                dataset.num_papers,
                dataset.train_edges,
                len(train_pos),
                seed=seed + epoch,
            )
            epoch_loader = make_edge_loader(train_pos, train_neg, batch_size)
        else:
            epoch_loader = loader

        if epoch_loader is None:
            raise ValueError(f"Unsupported model: {model_name}")

        for author_ids, paper_ids, labels in epoch_loader:
            author_ids = author_ids.to(device)
            paper_ids = paper_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if model_name == "lightgcn":
                z = model.encode(adj)
                logits = model.score(z, author_ids, paper_ids)
            elif model_name == "heterognn":
                author_z, paper_z = model.encode(adj)
                logits = model.score(author_z, paper_z, author_ids, paper_ids)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    valid_edges = np.vstack([valid_pos, valid_neg])
    y_true = np.concatenate(
        [
            np.ones(len(valid_pos), dtype=np.int64),
            np.zeros(len(valid_neg), dtype=np.int64),
        ]
    )
    y_score = score_edges(model_name, model, valid_edges, dataset.num_authors, adj, device)
    calibrator = None
    structural_features = None
    if model_name == "heterognn":
        structural_features = StructuralFeatureStore(
            dataset.num_authors,
            dataset.num_papers,
            train_pos,
            dataset.coauthor_edges,
            dataset.citation_edges,
            dataset.paper_features,
        )
        pair_features = structural_features.transform(valid_edges, device=device)

        calibration_indices, threshold_indices, evaluation_indices = (
            split_calibration_indices(
                len(valid_pos),
                len(valid_neg),
                seed,
            )
        )

        selection_calibrator = HeteroGNNCalibrator(seed=seed)
        selection_calibrator.fit(
            y_score[calibration_indices],
            pair_features[calibration_indices],
            y_true[calibration_indices],
        )
        threshold_scores = selection_calibrator.predict_proba(
            y_score[threshold_indices],
            pair_features[threshold_indices],
        )
        best_threshold, _ = search_best_threshold(
            y_true[threshold_indices],
            threshold_scores,
        )
        final_fit_indices = np.concatenate(
            [calibration_indices, threshold_indices]
        )
        calibrator = HeteroGNNCalibrator(seed=seed)
        calibrator.fit(
            y_score[final_fit_indices],
            pair_features[final_fit_indices],
            y_true[final_fit_indices],
        )
        evaluation_scores = calibrator.predict_proba(
            y_score[evaluation_indices],
            pair_features[evaluation_indices],
        )
        valid_metrics = classification_metrics(
            y_true[evaluation_indices],
            evaluation_scores,
            best_threshold,
        )
    else:
        best_threshold, valid_metrics = search_best_threshold(y_true, y_score)

    return TrainResult(
        model=model,
        adj=adj,
        best_threshold=best_threshold,
        valid_metrics=valid_metrics,
        calibrator=calibrator,
        structural_features=structural_features,
    )
