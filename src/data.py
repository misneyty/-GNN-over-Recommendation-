"""从项目数据目录中读取图关系和节点特征。"""

import os
import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """推荐系统完整流程所需的数据集合。

    所有边数组均为形状 ``[边数, 2]`` 的整数矩阵。作者编号和论文编号
    分别位于各自独立的、从 0 开始的编号空间中。
    """

    num_authors: int
    num_papers: int
    train_edges: np.ndarray
    test_edges: np.ndarray
    coauthor_edges: np.ndarray
    citation_edges: np.ndarray
    paper_features: np.ndarray


def read_edge_list(path: str) -> np.ndarray:
    """读取两列边文件，并保留原始文件中的重复记录。

    当前实验将重复作者-论文 pair 视为可能的频次信息，因此读取阶段
    不主动去重。
    """
    return np.loadtxt(path, dtype=np.int64)


def load_paper_features(path: str) -> np.ndarray:
    """读取论文特征矩阵，并统一转换为 ``float32``。

    特征矩阵的第 ``i`` 行对应编号为 ``i`` 的论文节点。
    """
    with open(path, "rb") as f:
        features = pickle.load(f)
    return np.asarray(features, dtype=np.float32)


def load_dataset(data_dir: str = "data_file") -> Dataset:
    """读取全部关系文件，并推断作者、论文节点的完整编号范围。"""
    # 训练边是已经观察到的作者-论文正样本，既用于监督训练，也用于建图。
    train_edges = read_edge_list(os.path.join(data_dir, "bipartite_train_ann.txt"))

    # 测试边只用于最终预测，不能参与训练图或结构特征的构建。
    test_edges = read_edge_list(os.path.join(data_dir, "bipartite_test_ann.txt"))

    # 作者-作者边表示合作关系；论文-论文边表示引用关系。
    coauthor_edges = read_edge_list(os.path.join(data_dir, "author_file_ann.txt"))

    citation_edges = read_edge_list(os.path.join(data_dir, "paper_file_ann.txt"))

    paper_features = load_paper_features(os.path.join(data_dir, "feature.pkl"))

    # 节点数量必须覆盖训练集、测试集和关系图中出现的全部编号。
    # 这样即使某个节点只出现在测试集中，也能分配合法的嵌入位置。
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

    # 最大编号加 1，即为从 0 开始编号时的节点总数。
    return Dataset(
        num_authors=max_author + 1,
        num_papers=max_paper + 1,
        train_edges=train_edges,
        test_edges=test_edges,
        coauthor_edges=coauthor_edges,
        citation_edges=citation_edges,
        paper_features=paper_features,
    )
