"""Drift detectors behind one tiny interface: update(errors) -> bool.

`errors` is the per-sample 0/1 error signal for one batch (the harness feeds one value per
fraudulent transaction: 1 if the model missed it). The wrapped River detector sees every value
in order; update returns True if it signalled drift anywhere in the batch.

After a detection the River detector is replaced with a fresh one: the model is about to be
refit, so its error rate will change again, and the old statistics would trigger a spurious
second alarm.

ADWIN flags a change in either direction, but only a rise in errors means the model is going
stale; a fall (e.g. a model recovering after a refit) is ignored. DDM only detects rises anyway.
"""
from river import drift


class DriftDetector:
    def __init__(self, name, prototype):
        self.name = name
        self.prototype = prototype
        self.detector = prototype.clone()

    def update(self, errors):
        for e in errors:
            before = getattr(self.detector, "estimation", None)
            self.detector.update(e)
            if not self.detector.drift_detected:
                continue
            # ADWIN drops the older part of its window on a change: if the mean went down,
            # the dropped (older) errors were higher, i.e. the model got better
            if before is not None and self.detector.estimation < before:
                continue
            self.detector = self.prototype.clone()
            return True
        return False


class NoDriftDetector:
    """Fallback that never fires, so 'no detector' runs through the same code path."""
    name = "none"

    def update(self, errors):
        return False


def make_adwin(delta=0.002):
    # clock=1 checks for a change after every value instead of every 32
    return DriftDetector("adwin", drift.ADWIN(delta=delta, clock=1))


def make_ddm(warm_start=30, drift_threshold=3.0):
    return DriftDetector("ddm", drift.binary.DDM(warm_start=warm_start, drift_threshold=drift_threshold))
