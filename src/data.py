from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class Dataset:
    num_authors: int
    num_papers: int
    train_edges: np.ndarray
    test_edges: np.ndarray
    coauthor_edges: np.ndarray
    citation_edges: np.ndarray
    paper_features: np.ndarray


def read_edge_list(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}, got shape {data.shape}")
    return data


def load_paper_features(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    with path.open("rb") as f:
        features = pickle.load(f)
    return np.asarray(features, dtype=np.float32)


def load_dataset(data_dir: str | Path = "data_file") -> Dataset:
    data_dir = Path(data_dir)
    train_edges = read_edge_list(data_dir / "bipartite_train_ann.txt")
    test_edges = read_edge_list(data_dir / "bipartite_test_ann.txt")
    coauthor_edges = read_edge_list(data_dir / "author_file_ann.txt")
    citation_edges = read_edge_list(data_dir / "paper_file_ann.txt")
    paper_features = load_paper_features(data_dir / "feature.pkl")

    max_author = max(
        int(train_edges[:, 0].max()),
        int(test_edges[:, 0].max()),
        int(coauthor_edges.max()),
    )
    max_paper = max(
        int(train_edges[:, 1].max()),
        int(test_edges[:, 1].max()),
        int(citation_edges.max()),
        paper_features.shape[0] - 1,
    )

    return Dataset(
        num_authors=max_author + 1,
        num_papers=max_paper + 1,
        train_edges=train_edges,
        test_edges=test_edges,
        coauthor_edges=coauthor_edges,
        citation_edges=citation_edges,
        paper_features=paper_features,
    )
