"""二分类评价指标与验证集阈值搜索。"""

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    auc: Optional[float] = None,
) -> dict[str, float]:
    """根据阈值把概率转为 0/1 标签，并计算分类指标。

    ``y_score`` 是模型预测为正样本的概率；大于等于 ``threshold`` 的
    作者-论文 pair 会被判定为推荐。
    """
    y_pred = (y_score >= threshold).astype(np.int64)
    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if auc is not None:
        metrics["auc"] = float(auc)
    elif len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def search_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    start: float = 0.01,
    end: float = 0.99,
    step: float = 0.001,
) -> tuple[float, dict[str, float]]:
    """在调参数据上搜索 F1 最高的分类阈值。

    搜索范围默认为 0.01 到 0.99，步长为 0.001。返回最佳阈值以及
    该阈值对应的 F1、Accuracy 和 AUC。
    """
    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_score))

    # AUC 衡量排序能力，与具体分类阈值无关，因此只需要计算一次。
    best_threshold = start
    best_metrics = classification_metrics(y_true, y_score, start, auc=auc)
    for threshold in np.arange(start + step, end, step):
        metrics = classification_metrics(
            y_true,
            y_score,
            float(threshold),
            auc=auc,
        )
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
