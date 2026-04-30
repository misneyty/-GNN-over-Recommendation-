from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.baseline import MFRecommender
from models.lightgcn import LightGCN
from utils.data import load_dataset
from utils.graph import build_bipartite_adjacency, sample_negative_edges
from utils.metrics import auc_score


def build_dataloader(pos_edges: np.ndarray, neg_edges: np.ndarray, batch_size: int) -> DataLoader:
    edges = np.vstack([pos_edges, neg_edges])
    labels = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
    perm = np.random.permutation(len(edges))
    edges = edges[perm]
    labels = labels[perm]

    ds = TensorDataset(
        torch.from_numpy(edges[:, 0]).long(),
        torch.from_numpy(edges[:, 1]).long(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


def train_mf(model: MFRecommender, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for author_ids, paper_ids, labels in loader:
        logits = model(author_ids, paper_ids)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
    return total_loss / len(loader.dataset)


def train_lightgcn(
    model: LightGCN,
    adj: torch.Tensor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    z = model(adj)
    for author_ids, paper_ids, labels in loader:
        logits = model.score(z, author_ids, paper_ids)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model_name: str, model, adj: torch.Tensor | None, edges: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        if model_name == "mf":
            logits = model(
                torch.from_numpy(edges[:, 0]).long(),
                torch.from_numpy(edges[:, 1]).long(),
            )
        else:
            z = model(adj)
            logits = model.score(
                z,
                torch.from_numpy(edges[:, 0]).long(),
                torch.from_numpy(edges[:, 1]).long(),
            )
    y_score = torch.sigmoid(logits).cpu().numpy()
    y_true = np.ones(len(edges), dtype=np.float32)
    return auc_score(y_true, y_score)


def predict(model_name: str, model, adj: torch.Tensor | None, edges: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if model_name == "mf":
            logits = model(
                torch.from_numpy(edges[:, 0]).long(),
                torch.from_numpy(edges[:, 1]).long(),
            )
        else:
            z = model(adj)
            logits = model.score(
                z,
                torch.from_numpy(edges[:, 0]).long(),
                torch.from_numpy(edges[:, 1]).long(),
            )
    return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model", type=str, choices=["mf", "lightgcn"], default="mf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="predictions.csv")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    neg_edges = sample_negative_edges(
        dataset.num_authors, dataset.num_papers, dataset.ap_train, len(dataset.ap_train)
    )
    loader = build_dataloader(dataset.ap_train, neg_edges, args.batch_size)

    if args.model == "mf":
        model = MFRecommender(dataset.num_authors, dataset.num_papers, args.dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in tqdm(range(args.epochs), desc="train-mf"):
            train_mf(model, loader, optimizer)
        adj = None
    else:
        adj = build_bipartite_adjacency(
            dataset.num_authors, dataset.num_papers, dataset.ap_train
        )
        model = LightGCN(
            dataset.num_authors + dataset.num_papers,
            dataset.num_authors,
            dataset.num_papers,
            dim=args.dim,
            num_layers=args.layers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in tqdm(range(args.epochs), desc="train-lightgcn"):
            train_lightgcn(model, adj, loader, optimizer)

    scores = predict(args.model, model, adj, dataset.ap_test)
    out_path = Path(args.out)
    np.savetxt(out_path, scores, fmt="%.6f")
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
