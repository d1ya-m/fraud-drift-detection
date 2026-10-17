"""Drift-triggered retraining that keeps learning after the alarm.

Plain "+ detector" refits a static model on the single alarm batch and freezes it again, so the
new model rests on very little fraud. This wrapper instead keeps refitting on every batch since
the alarm (a growing window of post-drift data) until it holds `accumulate` batches, then freezes.
Between drifts it costs nothing, like a static model.
"""
import pandas as pd

from src.models.base import BaseDetector
from src.models.baselines import has_both_classes


class AccumulateAfterDrift(BaseDetector):
    def __init__(self, inner, accumulate=6):
        """inner: a StaticModel; accumulate: post-drift batches to gather before freezing again."""
        self.inner = inner
        self.accumulate = accumulate
        self.name = f"{inner.name}_acc{accumulate}"
        self.buffer = []  # batches since the last alarm; empty when frozen

    def fit(self, X, y):
        self.buffer = []
        self.inner.fit(X, y)
        return self

    def refit(self, X, y):
        self.buffer = [(X, y)]
        self._refit_on_buffer()
        return self

    def partial_fit(self, X, y):
        if 0 < len(self.buffer) < self.accumulate:
            self.buffer.append((X, y))
            self._refit_on_buffer()
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def _refit_on_buffer(self):
        y = pd.concat([b[1] for b in self.buffer])
        if has_both_classes(y):
            self.inner.fit(pd.concat([b[0] for b in self.buffer]), y)
