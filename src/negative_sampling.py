"""为未观察到的作者-论文组合执行均匀负采样。"""

import numpy as np


def build_positive_set(edges: np.ndarray) -> set[tuple[int, int]]:
    """把正边转换为集合，以便快速判断候选 pair 是否已经存在。"""
    return {(int(author), int(paper)) for author, paper in edges}


def sample_negative_edges(
    num_authors: int,
    num_papers: int,
    positive_edges: np.ndarray,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """随机生成不属于正边集合的作者-论文 pair。

    负样本并不代表作者一定不喜欢论文，只表示该组合没有出现在已知
    正边中。训练时正负样本数量保持一致，以减轻类别不平衡。
    """
    rng = np.random.default_rng(seed)
    positive_set = build_positive_set(positive_edges)
    samples: list[tuple[int, int]] = []

    while len(samples) < num_samples:
        # 每轮多生成一些候选 pair，可以减少随机生成循环的次数。
        # 其中命中正边集合的候选会被丢弃。
        remaining = num_samples - len(samples)
        batch_size = max(1024, remaining * 2)
        authors = rng.integers(0, num_authors, size=batch_size)
        papers = rng.integers(0, num_papers, size=batch_size)

        for author, paper in zip(authors, papers):
            pair = (int(author), int(paper))
            if pair not in positive_set:
                samples.append(pair)
                if len(samples) == num_samples:
                    break

    return np.asarray(samples, dtype=np.int64)
