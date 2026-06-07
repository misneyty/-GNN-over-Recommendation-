import numpy as np
#转换成集合形式
def build_positive_set(edges: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(p)) for a, p in edges}
#负采样negative_sample
def sample_negative_edges(
    num_authors: int,
    num_papers: int,
    positive_edges: np.ndarray,
    num_samples: int,#定义负样本的数量
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