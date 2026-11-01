import numpy as np
import pandas as pd
import pytest

from src.evaluation.drift_metrics import drift_response, drift_response_table, trend_per_10_batches


def run_from(scores, start=1, name="m"):
    return pd.DataFrame({"model_name": name, "batch_idx": range(start, start + len(scores)), "pr_auc": scores})


def test_collapse_without_recovery():
    r = drift_response(run_from([0.8] * 5 + [0.1] * 5), drift_batch=6, smooth=1)
    assert r["before"] == pytest.approx(0.8)
    assert r["after"] == pytest.approx(0.1)
    assert r["drop"] == pytest.approx(0.7)
    assert r["drop_pct"] == pytest.approx(87.5)
    assert np.isnan(r["recovery_batches"])
    assert r["batches_below"] == 5


def test_recovery_counted_from_drift():
    r = drift_response(run_from([0.8] * 5 + [0.2, 0.3, 0.8, 0.8]), drift_batch=6, smooth=1)
    assert r["recovery_batches"] == 2  # back at >= 0.72 on the third post-drift batch
    assert r["batches_below"] == 2


def test_late_collapse_is_not_an_instant_recovery():
    # fine on the first post-drift batch, then collapses for good
    r = drift_response(run_from([0.8] * 5 + [0.8, 0.1, 0.1, 0.1]), drift_batch=6, smooth=1)
    assert np.isnan(r["recovery_batches"])


def test_no_drop_means_zero_recovery_time():
    r = drift_response(run_from([0.8] * 10), drift_batch=6, smooth=1)
    assert r["recovery_batches"] == 0
    assert r["drop"] == pytest.approx(0.0)


def test_straddling_batch_counts_as_neither():
    # drift at 5.5: batch 5 is mixed and must not enter either mean
    r = drift_response(run_from([0.8] * 4 + [0.0] + [0.2] * 4), drift_batch=5.5, smooth=1)
    assert r["before"] == pytest.approx(0.8)
    assert r["after"] == pytest.approx(0.2)


def test_smoothing_ignores_a_single_lucky_batch():
    scores = [0.8] * 5 + [0.1, 0.9, 0.1, 0.1, 0.8, 0.8, 0.8]
    assert drift_response(run_from(scores), drift_batch=6, smooth=1)["recovery_batches"] == 1
    assert drift_response(run_from(scores), drift_batch=6, smooth=3)["recovery_batches"] > 1


def test_trend_slope():
    assert trend_per_10_batches(run_from(list(np.linspace(0.9, 0.5, 21)))) == pytest.approx(-0.2)
    assert np.isnan(trend_per_10_batches(run_from([0.5, np.nan])))


def test_table_has_one_row_per_model():
    results = pd.concat([run_from([0.8] * 5 + [0.1] * 5, name="a"), run_from([0.8] * 10, name="b")])
    table = drift_response_table(results, drift_batch=6, smooth=1)
    assert table["model_name"].tolist() == ["a", "b"]
    assert {"before", "after", "drop", "recovery_batches"} <= set(table.columns)
