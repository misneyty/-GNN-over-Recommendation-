"""为不同图关系构建归一化稀疏邻接矩阵。"""

from typing import Dict

import numpy as np
import torch

RelationAdjacencies = Dict[str, torch.Tensor]


def normalize_sparse_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """对邻接矩阵执行对称度归一化 ``D^(-1/2) A D^(-1/2)``。

    归一化可以抑制度数很高的节点对消息传播的过度影响，并使不同
    节点接收到的聚合结果处于更稳定的数值范围内。
    """
    adjacency = adjacency.coalesce()
    # ``indices`` 的两行分别是稀疏矩阵的行坐标和列坐标。
    indices = adjacency.indices()
    values = adjacency.values()

    # 对每个目标节点累加边权重，得到度数 D。
    degree = torch.zeros(
        adjacency.size(0),
        dtype=values.dtype,
        device=values.device,
    )
    degree.index_add_(0, indices[0], values)

    # 孤立节点的度数为 0，其逆平方根会是无穷大，需要重置为 0。
    inverse_sqrt_degree = torch.pow(degree, -0.5)
    inverse_sqrt_degree[torch.isinf(inverse_sqrt_degree)] = 0.0
    rows, columns = indices
    # 每条边 (u, v) 的新权重为 1 / sqrt(deg(u) * deg(v))。
    normalized_values = (
        inverse_sqrt_degree[rows] * values * inverse_sqrt_degree[columns]
    )
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        adjacency.size(),
    ).coalesce()


def build_bipartite_adjacency(
    num_authors: int,
    num_papers: int,
    author_paper_edges: np.ndarray,
) -> torch.Tensor:
    """为 LightGCN 基线构建统一的无向作者-论文邻接矩阵。

    LightGCN 把作者和论文放在同一个节点编号空间中，因此矩阵形状为
    ``[作者数 + 论文数, 作者数 + 论文数]``。
    """
    num_nodes = num_authors + num_papers

    author_ids = author_paper_edges[:, 0]
    # 论文编号整体加上作者数量，避免与作者编号发生冲突。
    paper_ids = author_paper_edges[:, 1] + num_authors

    # 同时加入作者->论文和论文->作者，使二部图能够双向传播消息。
    rows = np.concatenate([author_ids, paper_ids])
    columns = np.concatenate([paper_ids, author_ids])
    indices = torch.as_tensor(
        np.vstack([rows, columns]),
        dtype=torch.long,
    )
    values = torch.ones(indices.size(1), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
    )
    return normalize_sparse_adjacency(adjacency)


def build_relation_adjacencies(
    num_authors: int,
    num_papers: int,
    train_edges: np.ndarray,
    coauthor_edges: np.ndarray,
    citation_edges: np.ndarray,
) -> RelationAdjacencies:
    """构建 HeteroGNN 使用的四个有类型消息传播矩阵。

    返回的四种传播方向分别是：作者到论文、论文到作者、作者到作者、
    论文到论文。模型会为不同关系学习独立的线性变换和融合权重。
    """
    author_to_paper, paper_to_author = build_bipartite_relation_adjacencies(
        num_authors,
        num_papers,
        train_edges,
    )
    author_author = _build_homogeneous_adjacency(
        num_authors,
        coauthor_edges,
    )
    paper_paper = _build_homogeneous_adjacency(
        num_papers,
        citation_edges,
    )

    return {
        "author_to_paper": author_to_paper,
        "paper_to_author": paper_to_author,
        "author_author": author_author,
        "paper_paper": paper_paper,
    }


def build_bipartite_relation_adjacencies(
    num_authors: int,
    num_papers: int,
    author_paper_edges: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构建归一化的作者到论文、论文到作者传播矩阵。

    两类节点保留各自的编号空间，因此两个矩阵是矩形矩阵：
    ``author_to_paper`` 形状为 ``[论文数, 作者数]``；
    ``paper_to_author`` 形状为 ``[作者数, 论文数]``。
    """
    author_ids = author_paper_edges[:, 0]
    paper_ids = author_paper_edges[:, 1]

    author_degree = np.bincount(
        author_ids,
        minlength=num_authors,
    ).astype(np.float32)
    paper_degree = np.bincount(
        paper_ids,
        minlength=num_papers,
    ).astype(np.float32)
    normalization = 1.0 / np.sqrt(author_degree[author_ids] * paper_degree[paper_ids])
    normalization[np.isinf(normalization)] = 0.0

    # 稀疏矩阵乘法 ``A @ H`` 中，行表示接收消息的目标节点，
    # 列表示提供消息的源节点，因此两个方向的坐标顺序正好相反。
    author_to_paper_indices = torch.as_tensor(
        np.vstack([paper_ids, author_ids]),
        dtype=torch.long,
    )
    paper_to_author_indices = torch.as_tensor(
        np.vstack([author_ids, paper_ids]),
        dtype=torch.long,
    )
    values = torch.as_tensor(normalization, dtype=torch.float32)

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


def _build_homogeneous_adjacency(
    num_nodes: int,
    edges: np.ndarray,
) -> torch.Tensor:
    """为同一种节点构建无向、归一化的邻接矩阵。

    作者合作边和论文引用边都在这里补上反向边。对引用关系进行对称化
    是当前 GNN 的设计选择，结构特征模块仍会保留引用的正反方向。
    """
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    indices = torch.as_tensor(
        np.vstack([rows, columns]),
        dtype=torch.long,
    )
    values = torch.ones(indices.size(1), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
    )
    return normalize_sparse_adjacency(adjacency)
