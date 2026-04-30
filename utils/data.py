from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class Dataset:
    num_authors: int
    num_papers: int
    ap_train: np.ndarray
    ap_test: np.ndarray
    aa_edges: np.ndarray
    pp_edges: np.ndarray


def _read_edge_list(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_dataset(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    ap_train = _read_edge_list(data_dir / "bipartite_train_ann.txt")
    ap_test = _read_edge_list(data_dir / "bipartite_test_ann.txt")
    aa_edges = _read_edge_list(data_dir / "author_file_ann.txt")
    pp_edges = _read_edge_list(data_dir / "paper_file_ann.txt")

    num_authors = int(max(ap_train[:, 0].max(), ap_test[:, 0].max(), aa_edges.max())) + 1
    num_papers = int(max(ap_train[:, 1].max(), ap_test[:, 1].max(), pp_edges.max())) + 1

    return Dataset(
        num_authors=num_authors,
        num_papers=num_papers,
        ap_train=ap_train,
        ap_test=ap_test,
        aa_edges=aa_edges,
        pp_edges=pp_edges,
    )


def build_id_maps(dataset: Dataset) -> Dict[str, Tuple[int, int]]:
    return {
        "author": (0, dataset.num_authors),
        "paper": (dataset.num_authors, dataset.num_authors + dataset.num_papers),
    }
