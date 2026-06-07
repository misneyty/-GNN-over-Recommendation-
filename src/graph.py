import numpy as np
import torch
from typing import Dict

#归一化稀疏邻接矩阵
def normalize_sparse_adjacency(adj: torch.Tensor) -> torch.Tensor:
    adj = adj.coalesce()
    indices =  adj.indices()#取出非零位置
    values = adj.values()#取出非零元素的值
    deg = torch.zeros(adj.size(0), dtype=values.dtype, device=values.device)#计算度数
    deg.index_add_(0, indices[0], values)
    deg_inv_sqrt = torch.pow(deg,-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    row,col = indices
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, norm_values, adj.size()).coalesce()

# 构建作者-论文二部图的邻接矩阵
def build_bipartite_adjacency(
    num_authors: int,
    num_papers: int,
    author_paper_edges: np.ndarray,
) -> torch.Tensor:
    num_nodes = num_authors + num_papers

    author = author_paper_edges[:, 0]
    paper = author_paper_edges[:, 1] + num_authors

    row = np.concatenate([author, paper])
    col = np.concatenate([paper, author])

    indices = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)
    values = torch.ones(indices.size(1), dtype=torch.float32)

    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    return normalize_sparse_adjacency(adj)#返回稀疏矩阵

def build_relation_adjacencies(
    num_authors: int,
    num_papers: int,
    train_edges: np.ndarray,
    coauthor_edges: np.ndarray,
    citation_edges: np.ndarray,
) -> Dict[str, torch.Tensor]:
    author_to_paper, paper_to_author = build_bipartite_relation_adjacencies(
        num_authors,
        num_papers,
        train_edges,
    )
    aa_adj = _build_homogeneous_adjacency(num_authors, coauthor_edges)
    pp_adj = _build_homogeneous_adjacency(num_papers, citation_edges)

    return {
        "author_to_paper": author_to_paper,
        "paper_to_author": paper_to_author,
        "author_author": aa_adj,
        "paper_paper": pp_adj,
    }


def build_bipartite_relation_adjacencies(
    num_authors: int,
    num_papers: int,
    author_paper_edges: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build typed author->paper and paper->author propagation matrices."""
    author = author_paper_edges[:, 0]
    paper = author_paper_edges[:, 1]

    author_deg = np.bincount(author, minlength=num_authors).astype(np.float32)
    paper_deg = np.bincount(paper, minlength=num_papers).astype(np.float32)
    norm = 1.0 / np.sqrt(author_deg[author] * paper_deg[paper])
    norm[np.isinf(norm)] = 0.0

    author_to_paper_indices = torch.as_tensor(
        np.vstack([paper, author]),
        dtype=torch.long,
    )
    paper_to_author_indices = torch.as_tensor(
        np.vstack([author, paper]),
        dtype=torch.long,
    )
    values = torch.as_tensor(norm, dtype=torch.float32)

    author_to_paper = torch.sparse_coo_tensor(
        author_to_paper_indices,
        values,
        (num_papers, num_authors),
    ).coalesce()
    paper_to_author = torch.sparse_coo_tensor(
        paper_to_author_indices,
        values,
        (num_authors, num_papers),
    ).coalesce()
    return author_to_paper, paper_to_author


def _build_homogeneous_adjacency(num_nodes: int, edges: np.ndarray) -> torch.Tensor:
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])

    indices = torch.as_tensor(np.vstack([row, col]), dtype=torch.long)
    values = torch.ones(indices.size(1), dtype=torch.float32)

    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    return normalize_sparse_adjacency(adj)
