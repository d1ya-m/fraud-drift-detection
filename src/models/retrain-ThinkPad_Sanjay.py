"""Periodic retraining: keep a sliding window of recent batches and refit on it every N batches."""
from collections import deque

import pandas as pd

from src.models.base import BaseDetector
from src.models.baselines import has_both_classes


class PeriodicRetrain(BaseDetector):
    def __init__(self, inner, every=5, window=10, eager_after_reset=False):
        """inner: a StaticModel; every: refit interval in batches; window: batches kept for refitting.

        eager_after_reset: after a drift alarm (refit), refit on EVERY new batch until the window is full
        of post-drift batches again, then go back to the fixed schedule. This combines periodic retraining
        (no blind period while a detector is still deciding) with accumulate-after-drift's fast rebuild.
        """
        self.inner = inner
        self.every = every
        self.window = window
        self.eager_after_reset = eager_after_reset
        self.name = f"{inner.name}_retrain{every}" + ("_reset" if eager_after_reset else "")
        self.buffer = deque(maxlen=window)
        self.since_refit = 0
        self.eager = False

    def fit(self, X, y):
        self.buffer.clear()
        self.buffer.append((X, y))
        self.inner.fit(X, y)
        self.since_refit = 0
        return self

    def partial_fit(self, X, y):
        self.buffer.append((X, y))
        self.since_refit += 1
        if self.eager:
            self._refit_on_buffer()
            self.eager = len(self.buffer) < self.window
        elif self.since_refit >= self.every:
            self._refit_on_buffer()
        return self

    def refit(self, X, y):
        # after a drift, older batches describe the old concept: start the window again
        self.buffer.clear()
        self.buffer.append((X, y))
        self._refit_on_buffer()
        self.eager = self.eager_after_reset
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def _refit_on_buffer(self):
        X = pd.concat([b[0] for b in self.buffer])
        y = pd.concat([b[1] for b in self.buffer])
        if has_both_classes(y):
            self.inner.fit(X, y)
            self.since_refit = 0
