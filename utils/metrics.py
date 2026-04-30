from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))
