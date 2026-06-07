import numpy as np

def train_valid_split(
    edges: np.ndarray,
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    valid_size = max(1, int(len(edges) * valid_ratio))
    valid_idx = perm[:valid_size]
    train_idx = perm[valid_size:]
    return edges[train_idx], edges[valid_idx]
