"""Drift-triggered retraining schemes for static models.

AccumulateAfterDrift (this project's variant): plain "+ detector" refits a static model on the
single alarm batch and freezes it again, so the new model rests on very little fraud. This wrapper
instead keeps refitting on every batch since the alarm (a growing window of post-drift data) until
it holds `accumulate` batches, then freezes. Between drifts it costs nothing, like a static model.
It is a variant of DDM's own scheme (below): the window starts at the alarm instead of the warning.

WarningWindowRetrain (Gama et al., 2004, "Learning with Drift Detection"): DDM's warning level
starts a buffer of examples; when the drift level is reached, a new model is learnt from the
examples since the warning only. Used here as the established baseline for AccumulateAfterDrift.
"""
import numpy as np
import pandas as pd
from river import drift

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


class WarningWindowRetrain(BaseDetector):
    """DDM with its warning window, run inside the model (use it with drift_detector=None).

    On each labelled batch the model scores it, feeds DDM one 0/1 error per fraud (missed = score
    below error_threshold), and then: drift -> refit on the batches buffered since the warning
    (plus this one) and restart DDM; warning -> buffer the batch; otherwise -> clear the buffer.
    `last_alarm` tells the evaluation loop when a drift was handled, so it is recorded as an alarm.
    """

    def __init__(self, inner, error_threshold=0.5, max_buffer=10):
        self.inner = inner
        self.error_threshold = error_threshold
        self.max_buffer = max_buffer
        self.name = f"{inner.name}_ddmwarn"
        self.buffer = []
        self.last_alarm = False

    def fit(self, X, y):
        self.inner.fit(X, y)
        self.detector = drift.binary.DDM()
        self.buffer = []
        self.last_alarm = False
        return self

    def refit(self, X, y):
        self.buffer = [(X, y)]
        self._refit_on_buffer()
        return self

    def partial_fit(self, X, y):
        self.last_alarm = False
        is_fraud = np.asarray(y) == 1
        if not is_fraud.any():
            return self
        missed = self.inner.predict_proba(X)[is_fraud] < self.error_threshold
        warning = drifted = False
        for e in missed.astype(int):
            self.detector.update(e)
            warning = warning or self.detector.warning_detected
            if self.detector.drift_detected:
                drifted = True
                break
        if drifted:
            self.buffer.append((X, y))
            self._refit_on_buffer()
            self.detector = drift.binary.DDM()
            self.buffer = []
            self.last_alarm = True
        elif warning:
            self.buffer = (self.buffer + [(X, y)])[-self.max_buffer:]
        else:
            self.buffer = []
        return self

    def predict_proba(self, X):
        return self.inner.predict_proba(X)

    def _refit_on_buffer(self):
        y = pd.concat([b[1] for b in self.buffer])
        if has_both_classes(y):
            self.inner.fit(pd.concat([b[0] for b in self.buffer]), y)
