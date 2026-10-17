import numpy as np
import pytest

from src.data.loader import load_dataset
from src.data.stream import make_stream
from src.drift.detectors import NoDriftDetector, make_adwin, make_ddm
from src.evaluation.prequential import prequential_eval
from src.models.baselines import make_logreg


def error_batches(rates, n_batches=20, batch=50, seed=0):
    """Batches of 0/1 errors; error rate switches from rates[0] to rates[1] halfway."""
    rng = np.random.default_rng(seed)
    return [(rng.random(batch) < rates[i >= n_batches // 2]).astype(int) for i in range(n_batches)]


@pytest.mark.parametrize("factory", [make_adwin, make_ddm])
def test_detects_jump_in_error_rate(factory):
    det = factory()
    fired = [det.update(b) for b in error_batches((0.1, 0.6))]
    first = fired.index(True)
    assert 10 <= first <= 12


def test_adwin_ignores_falling_error_rate():
    # a model recovering after a refit must not look like a new drift
    det = make_adwin()
    assert not any(det.update(b) for b in error_batches((0.6, 0.1)))


def test_adwin_quiet_on_stationary_errors():
    det = make_adwin()
    assert not any(det.update(b) for b in error_batches((0.1, 0.1)))


def test_detector_restarts_after_firing():
    det = make_adwin()
    old = det.detector
    for b in error_batches((0.1, 0.6)):
        if det.update(b):
            break
    assert det.detector is not old


def test_no_drift_detector_never_fires():
    det = NoDriftDetector()
    assert not any(det.update(b) for b in error_batches((0.0, 1.0)))


def test_adwin_flags_synthetic_drift_and_model_recovers():
    df = load_dataset(n_samples=20_000, fraud_rate=0.02, seed=3)  # drift at the start of batch 5
    res = prequential_eval(make_logreg(), make_stream(df, batch_size=2000),
                           drift_detector=make_adwin(), warmup_batches=2)
    flagged = res.loc[res["drift_flag"] == 1, "batch_idx"].tolist()
    assert flagged and all(5 <= b <= 6 for b in flagged)
    assert res.loc[res["batch_idx"] >= 7, "pr_auc"].mean() > 0.5
    assert (res["model_name"] == "logreg+adwin").all()
