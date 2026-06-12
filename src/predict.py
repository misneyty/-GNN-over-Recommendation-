from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import build_bipartite_adjacency
from .utils import ensure_dir
from typing import Any, Dict, Optional, Union

def score_edges(
    model_name: str,
    model: torch.nn.Module,
    edges: np.ndarray,
    num_authors: int,
    adj: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    device: Optional[torch.device],
    batch_size: int = 131072,
    calibrator: Optional[Any] = None,
    structural_features: Optional[Any] = None,
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
        elif model_name == "heterognn":
            if adj is None:
                raise ValueError("Heterogeneous graph model prediction requires relation adjacencies")
            adj = {name: value.to(device) for name, value in adj.items()}
            author_z, paper_z = model.encode(adj)

        for start in range(0, len(edges), batch_size):
            batch = edges[start : start + batch_size]
            author_ids = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            paper_ids = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
            if model_name == "lightgcn":
                logits = model.score(z, author_ids, paper_ids)
            elif model_name == "heterognn":
                logits = model.score(author_z, paper_z, author_ids, paper_ids)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    model_scores = np.concatenate(scores)
    if calibrator is None:
        return model_scores
    if structural_features is None:
        raise ValueError("Calibrated prediction requires structural features")
    pair_features = structural_features.transform(edges, device=device)
    return calibrator.predict_proba(model_scores, pair_features)


def write_submission(
    scores: np.ndarray,
    output_path: Union[str, Path],
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
