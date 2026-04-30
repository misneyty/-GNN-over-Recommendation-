from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    def __init__(self, num_nodes: int, num_authors: int, num_papers: int, dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.emb = nn.Embedding(num_nodes, dim)
        self.num_layers = num_layers
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        embs = [self.emb.weight]
        h = self.emb.weight
        for _ in range(self.num_layers):
            h = torch.sparse.mm(adj, h)
            embs.append(h)
        return torch.stack(embs, dim=0).mean(dim=0)

    def score(self, z: torch.Tensor, author_ids: torch.Tensor, paper_ids: torch.Tensor) -> torch.Tensor:
        a = z[author_ids]
        p = z[paper_ids + self.num_authors]
        return (a * p).sum(dim=-1)
