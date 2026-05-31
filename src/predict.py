from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import build_bipartite_adjacency
from .models import LightGCN, MatrixFactorization
from .utils import ensure_dir


def score_edges(
    model_name: str,
    model: torch.nn.Module,
    edges: np.ndarray,
    num_authors: int,
    adj: torch.Tensor | dict[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
    batch_size: int = 131072,
) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        z = None
        if model_name == "lightgcn":
            if adj is None:
                raise ValueError("LightGCN prediction requires adj")
            z = model.encode(adj.to(device))
        elif model_name == "hetero_lightgcn":
            if not isinstance(adj, dict):
                raise ValueError("HeteroLightGCN prediction requires adjs dict")
            z = model.encode(
                adj["author_paper"].to(device),
                adj["author_author"].to(device),
                adj["paper_paper"].to(device),
            )

        for start in range(0, len(edges), batch_size):
            batch = edges[start : start + batch_size]
            author_ids = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            paper_ids = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
            if model_name == "mf":
                logits = model(author_ids, paper_ids)
            elif model_name in ("lightgcn", "hetero_lightgcn"):
                logits = model.score(z, author_ids, paper_ids)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores)


def write_submission(
    scores: np.ndarray,
    output_path: str | Path,
    threshold: float,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    predicted = (scores >= threshold).astype(np.int64)
    df = pd.DataFrame({"Index": np.arange(len(predicted)), "Predicted": predicted})
    df.to_csv(output_path, index=False)
    return output_path


def build_prediction_adjacency(dataset, train_edges: np.ndarray) -> torch.Tensor:
    return build_bipartite_adjacency(dataset.num_authors, dataset.num_papers, train_edges)
