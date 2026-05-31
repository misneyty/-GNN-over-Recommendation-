from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_authors: int, num_papers: int, dim: int = 64) -> None:
        super().__init__()
        self.author_emb = nn.Embedding(num_authors, dim)
        self.paper_emb = nn.Embedding(num_papers, dim)
        nn.init.xavier_uniform_(self.author_emb.weight)
        nn.init.xavier_uniform_(self.paper_emb.weight)

    def forward(self, author_ids: torch.Tensor, paper_ids: torch.Tensor) -> torch.Tensor:
        author_z = self.author_emb(author_ids)
        paper_z = self.paper_emb(paper_ids)
        return (author_z * paper_z).sum(dim=-1)


class LightGCN(nn.Module):
    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_authors + num_papers, dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def encode(self, adj: torch.Tensor) -> torch.Tensor:
        embeddings = [self.emb.weight]
        h = self.emb.weight
        for _ in range(self.num_layers):
            h = torch.sparse.mm(adj, h)
            embeddings.append(h)
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def score(
        self,
        z: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        author_z = z[author_ids]
        paper_z = z[paper_ids + self.num_authors]
        return (author_z * paper_z).sum(dim=-1)


class PaperFeatureMLP(nn.Module):
    """A small decoder that can combine author embeddings with paper features."""

    def __init__(self, author_dim: int, paper_feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(author_dim + paper_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, author_z: torch.Tensor, paper_x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([author_z, paper_x], dim=-1)).squeeze(-1)


class HeteroLightGCN(nn.Module):
    def __init__(
        self,
        num_authors: int,
        num_papers: int,
        paper_features: np.ndarray,
        dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_authors = num_authors
        self.num_papers = num_papers
        self.num_layers = num_layers
        
        self.author_emb = nn.Embedding(num_authors, dim)
        nn.init.xavier_uniform_(self.author_emb.weight)
        
        # Initialize paper embeddings using features
        feature_tensor = torch.as_tensor(paper_features, dtype=torch.float32)
        if feature_tensor.size(1) == dim:
            self.paper_emb = nn.Embedding.from_pretrained(feature_tensor, freeze=False)
        else:
            self.paper_emb = nn.Embedding(num_papers, dim)
            nn.init.xavier_uniform_(self.paper_emb.weight)
            # A simple linear projection could be added here if needed, 
            # but standard LightGCN uses direct embeddings. We fallback to random if dim mismatch.

    def encode(
        self, 
        ap_adj: torch.Tensor, 
        aa_adj: torch.Tensor, 
        pp_adj: torch.Tensor
    ) -> torch.Tensor:
        # Initial embeddings
        h_a = self.author_emb.weight
        h_p = self.paper_emb.weight
        
        embeddings_a = [h_a]
        embeddings_p = [h_p]
        
        for _ in range(self.num_layers):
            # Message passing along each relation type
            h_combined = torch.cat([h_a, h_p], dim=0)
            h_next_ap = torch.sparse.mm(ap_adj, h_combined)
            h_next_ap_a = h_next_ap[:self.num_authors]
            h_next_ap_p = h_next_ap[self.num_authors:]
            
            h_a_next = (torch.sparse.mm(aa_adj, h_a) + h_next_ap_a) / 2.0
            h_p_next = (torch.sparse.mm(pp_adj, h_p) + h_next_ap_p) / 2.0
            h_a = h_a_next
            h_p = h_p_next
            embeddings_a.append(h_a)
            embeddings_p.append(h_p)
            
        z_a = torch.stack(embeddings_a, dim=0).mean(dim=0)
        z_p = torch.stack(embeddings_p, dim=0).mean(dim=0)
        return torch.cat([z_a, z_p], dim=0)

    def score(
        self,
        z: torch.Tensor,
        author_ids: torch.Tensor,
        paper_ids: torch.Tensor,
    ) -> torch.Tensor:
        author_z = z[author_ids]
        paper_z = z[paper_ids + self.num_authors]
        return (author_z * paper_z).sum(dim=-1)

