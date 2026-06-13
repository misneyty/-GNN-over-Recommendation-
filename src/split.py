"""可复现的训练集与验证集划分工具。"""

import numpy as np


def train_valid_split(
    edges: np.ndarray,
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """按行随机划分训练边和验证边，并保持边内容不变。

    固定 ``seed`` 后，每次运行都会得到相同的划分结果。当前实现是
    行级划分，因此原始文件中的重复 pair 也会作为独立记录参与划分。
    """
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    # 只打乱行索引，不直接修改输入数组，便于后续复用原始数据。
    permutation = rng.permutation(len(edges))
    valid_size = max(1, int(len(edges) * valid_ratio))
    valid_indices = permutation[:valid_size]
    train_indices = permutation[valid_size:]
    return edges[train_indices], edges[valid_indices]
