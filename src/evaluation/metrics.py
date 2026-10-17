"""Imbalance-aware metrics for one batch of scores."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

METRIC_KEYS = ["pr_auc", "roc_auc", "recall_at_p", "fpr", "threshold"]


def compute_metrics(y_true, y_score, precision_target=0.90):
    """Return {pr_auc, roc_auc, recall_at_p, fpr, threshold}.

    recall_at_p is the highest recall reachable while precision >= precision_target;
    threshold is the score cut-off that achieves it and fpr is the false-positive rate there.
    If the batch has only one class, every value is NaN.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in METRIC_KEYS}

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # the curve's final point (precision=1, recall=0) has no threshold, so drop it
    meets_target = precision[:-1] >= precision_target
    if meets_target.any():
        best = np.argmax(np.where(meets_target, recall[:-1], -1.0))
        threshold, recall_at_p = thresholds[best], recall[best]
    else:
        threshold, recall_at_p = np.inf, 0.0

    negatives = y_true == 0
    fpr = np.mean(y_score[negatives] >= threshold)

    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall_at_p": float(recall_at_p),
        "fpr": float(fpr),
        "threshold": float(threshold),
    }
