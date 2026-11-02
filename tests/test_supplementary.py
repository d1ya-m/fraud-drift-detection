"""Tests for the supplementary strategy: periodic retraining with an ADWIN window reset."""
import pandas as pd

from src.models.retrain import PeriodicRetrain


class CountingModel:
    name = "count"

    def __init__(self):
        self.fit_sizes = []

    def fit(self, X, y):
        self.fit_sizes.append(len(X))

    def predict_proba(self, X):
        return [0.5] * len(X)


def batch():
    return pd.DataFrame({"a": range(10)}), pd.Series([0, 1] * 5)


def test_default_periodic_is_unchanged_by_the_new_option():
    inner = CountingModel()
    m = PeriodicRetrain(inner, every=3, window=4)
    m.fit(*batch())
    for _ in range(6):
        m.partial_fit(*batch())
    assert inner.fit_sizes == [10, 40, 40]  # warm-up, then every 3 batches on the last 4
    assert m.name == "count_retrain3"


def test_reset_refits_every_batch_until_window_is_full_then_returns_to_schedule():
    inner = CountingModel()
    m = PeriodicRetrain(inner, every=3, window=4, eager_after_reset=True)
    m.fit(*batch())
    m.refit(*batch())                       # alarm: window = [alarm batch]
    for _ in range(3):
        m.partial_fit(*batch())             # eager: refit on 2, 3, 4 batches
    for _ in range(3):
        m.partial_fit(*batch())             # schedule again: one refit after 3 batches
    assert inner.fit_sizes == [10, 10, 20, 30, 40, 40]
    assert m.name == "count_retrain3_reset"


def test_without_alarm_reset_variant_behaves_like_periodic():
    a, b = CountingModel(), CountingModel()
    p, r = PeriodicRetrain(a, 3, 4), PeriodicRetrain(b, 3, 4, eager_after_reset=True)
    for m in (p, r):
        m.fit(*batch())
        for _ in range(7):
            m.partial_fit(*batch())
    assert a.fit_sizes == b.fit_sizes
