import numpy as np
import pytest

from src.evaluation.metrics import METRIC_KEYS, compute_metrics


def test_perfect_separation():
    y = np.array([0] * 95 + [1] * 5)
    scores = np.where(y == 1, 0.9, 0.1)
    m = compute_metrics(y, scores)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_at_p"] == pytest.approx(1.0)
    assert m["fpr"] == 0.0


def test_all_legitimate_batch_returns_nans():
    m = compute_metrics(np.zeros(50), np.random.default_rng(0).random(50))
    assert set(m) == set(METRIC_KEYS)
    assert all(np.isnan(v) for v in m.values())


def test_unreachable_precision_gives_zero_recall():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.1, 0.8, 0.2])  # fraud always scored lowest
    m = compute_metrics(y, scores, precision_target=0.9)
    assert m["recall_at_p"] == 0.0
    assert m["fpr"] == 0.0


def test_recall_at_precision_picks_best_threshold():
    # top 4 scores: 3 fraud + 1 legit -> precision 0.75 at recall 0.75
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    m = compute_metrics(y, scores, precision_target=0.75)
    assert m["recall_at_p"] == pytest.approx(0.75)
    assert m["threshold"] == pytest.approx(0.6)
    assert m["fpr"] == pytest.approx(0.25)
