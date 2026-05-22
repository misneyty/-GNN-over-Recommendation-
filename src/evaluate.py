from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def search_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.01,
) -> tuple[float, dict[str, float]]:
    best_threshold = start
    best_metrics = classification_metrics(y_true, y_score, start)
    for threshold in np.arange(start + step, end + 1e-12, step):
        metrics = classification_metrics(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
