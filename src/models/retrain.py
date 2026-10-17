"""Periodic retraining: keep a sliding window of recent batches and refit on it every N batches."""
from collections import deque

import pandas as pd

from src.models.base import BaseDetector
from src.models.baselines import has_both_classes


class PeriodicRetrain(BaseDetector):
    def __init__(self, inner, every=5, window=10):
        """inner: a StaticModel; every: refit interval in batches; window: batches kept for refitting."""
        self.inner = inner
        self.every = every
        self.window = window
        self.name = f"{inner.name}_retrain{every}"
        self.buffer = deque(maxlen=window)
        self.since_refit = 0

    def fit(self, X, y):
        self.buffer.clear()
        self.buffer.append((X, y))
        self.inner.fit(X, y)
        self.since_refit = 0
        return self

    def partial_fit(self, X, y):
        self.buffer.append((X, y))
        self.since_refit += 1
        if self.since_refit >= self.every:
            self._refit_on_buffer()
        return self

    def refit(self, X, y):
        # after a drift, older batches describe the old concept: start the window again
        self.buffer.clear()
        self.buffer.append((X, y))
        self._refit_on_buffer()
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def _refit_on_buffer(self):
        X = pd.concat([b[0] for b in self.buffer])
        y = pd.concat([b[1] for b in self.buffer])
        if has_both_classes(y):
            self.inner.fit(X, y)
            self.since_refit = 0
