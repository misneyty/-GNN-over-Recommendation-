from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def build_bipartite_adjacency(
    num_authors: int, num_papers: int, ap_edges: np.ndarray
) -> torch.Tensor:
    num_nodes = num_authors + num_papers
    row = ap_edges[:, 0]
    col = ap_edges[:, 1] + num_authors
    indices = np.vstack([np.concatenate([row, col]), np.concatenate([col, row])])
    values = np.ones(indices.shape[1], dtype=np.float32)
    indices_t = torch.from_numpy(indices).long()
    values_t = torch.from_numpy(values)

    adj = torch.sparse_coo_tensor(indices_t, values_t, (num_nodes, num_nodes))
    return normalize_adj(adj)


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    num_nodes = adj.size(0)

    deg = torch.zeros(num_nodes, dtype=values.dtype, device=values.device)
    deg.index_add_(0, indices[0], values)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    row, col = indices[0], indices[1]
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, norm_values, adj.size())


def sample_negative_edges(
    num_authors: int, num_papers: int, pos_edges: np.ndarray, num_samples: int
) -> np.ndarray:
    rng = np.random.default_rng()
    pos_set = set((int(a), int(p)) for a, p in pos_edges)
    samples = []
    while len(samples) < num_samples:
        a = int(rng.integers(0, num_authors))
        p = int(rng.integers(0, num_papers))
        if (a, p) not in pos_set:
            samples.append((a, p))
    return np.array(samples, dtype=np.int64)
