from __future__ import annotations

import numpy as np
import torch


def build_positive_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(p)) for a, p in edges}


def sample_negative_edges(
    num_authors: int,
    num_papers: int,
    positive_edges: np.ndarray,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive_set = build_positive_set(positive_edges)
    samples: list[tuple[int, int]] = []

    while len(samples) < num_samples:
        batch_size = max(1024, (num_samples - len(samples)) * 2)
        authors = rng.integers(0, num_authors, size=batch_size)
        papers = rng.integers(0, num_papers, size=batch_size)
        for author, paper in zip(authors, papers):
            pair = (int(author), int(paper))
            if pair not in positive_set:
                samples.append(pair)
                if len(samples) == num_samples:
                    break

    return np.asarray(samples, dtype=np.int64)


class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, pos_edges: np.ndarray, num_papers: int):
        super().__init__()
        self.pos_edges = pos_edges
        self.num_papers = num_papers
        self.num_edges = len(pos_edges)
        
    def __len__(self) -> int:
        return self.num_edges

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        author = self.pos_edges[idx, 0]
        pos_paper = self.pos_edges[idx, 1]
        
        # Approximate dynamic negative sampling (uniform sampling).
        # We assume sparsity is high enough that random items are mostly negative.
        neg_paper = np.random.randint(0, self.num_papers)
        
        return (
            torch.as_tensor(author, dtype=torch.long),
            torch.as_tensor(pos_paper, dtype=torch.long),
            torch.as_tensor(neg_paper, dtype=torch.long)
        )
