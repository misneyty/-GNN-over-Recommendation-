from __future__ import annotations

import torch
from torch import nn


class MFRecommender(nn.Module):
    def __init__(self, num_authors: int, num_papers: int, dim: int = 64) -> None:
        super().__init__()
        self.author_emb = nn.Embedding(num_authors, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        nn.init.xavier_uniform_(self.author_emb.weight)
        nn.init.xavier_uniform_(self.paper_emb.weight)

    def forward(self, author_ids: torch.Tensor, paper_ids: torch.Tensor) -> torch.Tensor:
        a = self.author_emb(author_ids)
        p = self.paper_emb(paper_ids)
        return (a * p).sum(dim=-1)
