"""Tests for the additions made after the project audit (operational metrics, label delay, new baselines,
held-out drift scenarios, block bootstrap, SHAP)."""
import numpy as np
import pandas as pd
import pytest

from src.data.drift_injection import choose_random_pairs, inject_feature_swap
from src.data.loader import get_feature_columns, load_dataset
from src.data.stream import make_stream
from src.drift.detectors import make_adwin
from src.evaluation.metrics import operational_metrics
from src.evaluation.prequential import RESULT_COLUMNS, prequential_eval
from src.evaluation.stats import block_bootstrap_mean, paired_difference
from src.models.base import BaseDetector
from src.models.baselines import SingleFeatureRule, make_gradient_boosting, make_logreg, make_random_forest
from src.models.drift_retrain import WarningWindowRetrain


@pytest.fixture(scope="module")
def df():
    return load_dataset(n_samples=20_000, fraud_rate=0.02, seed=3)  # drift at the start of batch 5


# ---------------------------------------------------------------- operational metrics
def test_operational_metrics_at_fixed_threshold():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = operational_metrics(y, s, threshold=0.5)
    assert m == pytest.approx({"op_recall": 0.5, "op_precision": 0.5, "op_fpr": 1 / 3})
    assert all(np.isnan(v) for v in operational_metrics(y, s, np.inf).values())


def test_prequential_reports_operational_columns(df):
    res = prequential_eval(make_logreg(), make_stream(df, 2000), warmup_batches=2)
    assert list(res.columns) == RESULT_COLUMNS
    assert np.isnan(res["op_recall"].iloc[0])  # no labelled threshold before the first scored batch
    assert res["op_recall"].iloc[1:].notna().all()
    # a threshold fixed in advance can't beat the batch's own best threshold at the same precision
    assert res["op_recall"].mean() <= res["recall_at_p"].mean() + 0.1


# ---------------------------------------------------------------- label delay
class Recorder(BaseDetector):
    """Constant model that records which batches it learns from."""
    name = "recorder"

    def __init__(self):
        self.seen = []

    def fit(self, X, y):
        return self

    def partial_fit(self, X, y):
        self.seen.append(int(X.index[0]))
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


def test_label_delay_postpones_learning(df):
    for delay in (0, 2):
        model = Recorder()
        prequential_eval(model, make_stream(df, 2000), warmup_batches=2, label_delay=delay)
        evaluated = 10 - 2
        assert len(model.seen) == evaluated - delay
        assert model.seen[0] == 2 * 2000  # first learnt batch is always the first scored one


def test_label_delay_zero_matches_previous_behaviour(df):
    a = prequential_eval(make_logreg(), make_stream(df, 2000), drift_detector=make_adwin(), warmup_batches=2)
    b = prequential_eval(make_logreg(), make_stream(df, 2000), drift_detector=make_adwin(), warmup_batches=2,
                         label_delay=0)
    pd.testing.assert_frame_equal(a, b)


def test_label_delay_delays_alarms(df):
    flags = {}
    for delay in (0, 2):
        res = prequential_eval(make_logreg(), make_stream(df, 2000), drift_detector=make_adwin(),
                               warmup_batches=2, label_delay=delay)
        flags[delay] = res.loc[res["drift_flag"] == 1, "batch_idx"].min()
    assert flags[2] >= flags[0] + 2


def test_score_sink_collects_every_scored_row(df):
    sink = []
    prequential_eval(make_logreg(), make_stream(df, 2000), warmup_batches=2, score_sink=sink)
    scores = pd.concat(sink)
    assert len(scores) == len(df) - 2 * 2000
    assert set(scores.columns) == {"batch_idx", "row", "y", "score"}


# ---------------------------------------------------------------- new baselines and comparators
def test_rule_picks_strongest_feature_and_ranks_fraud(df):
    X, y = df[get_feature_columns(df)].iloc[:8000], df["Class"].iloc[:8000]
    rule = SingleFeatureRule().fit(X, y)
    assert rule.feature in {"V1", "V2", "V3", "Amount"}  # synthetic early fraud signal
    s = rule.predict_proba(X)
    assert np.all((s >= 0) & (s <= 1))
    assert s[y.to_numpy() == 1].mean() > s[y.to_numpy() == 0].mean()


def test_warning_window_retrain_flags_and_recovers_after_drift(df):
    model = WarningWindowRetrain(make_logreg())
    res = prequential_eval(model, make_stream(df, 2000), warmup_batches=2)
    flagged = res.loc[res["drift_flag"] == 1, "batch_idx"].tolist()
    assert flagged and min(flagged) >= 4  # no alarm well before the drift (batch 5)
    assert res.loc[res["batch_idx"] >= 8, "pr_auc"].mean() > 0.4
    assert res["model_name"].iloc[0] == "logreg_ddmwarn"


def test_gradient_boosting_uses_all_training_data():
    gb = make_gradient_boosting().estimator
    assert gb.early_stopping is False and gb.max_iter == 100


# ---------------------------------------------------------------- held-out drift scenarios
def test_gradual_drift_swaps_more_rows_later(df):
    drifted, pairs = inject_feature_swap(df, at=0.4, width=0.2, pairs=[("V1", "V7")])
    swapped = drifted["V1"].to_numpy() != df["V1"].to_numpy()
    n = len(df)
    assert not swapped[: int(0.4 * n)].any()
    assert swapped[int(0.6 * n):].all()
    window = swapped[int(0.4 * n): int(0.6 * n)]
    assert 0.3 < window.mean() < 0.7  # roughly half on a linear ramp
    assert window[: len(window) // 4].mean() < window[-len(window) // 4:].mean()


def test_random_pairs_are_disjoint_and_seeded(df):
    a = choose_random_pairs(df, k=3, seed=1)
    assert a == choose_random_pairs(df, k=3, seed=1)
    flat = [f for pair in a for f in pair]
    assert len(set(flat)) == 6 and "Amount" not in flat
    assert inject_feature_swap(df, at=0.5, k=3, pair_seed=1)[1] == a


# ---------------------------------------------------------------- uncertainty
def test_block_bootstrap_covers_true_mean():
    rng = np.random.default_rng(0)
    mean, lo, hi = block_bootstrap_mean(rng.normal(0.3, 0.2, 40), block=3)
    assert lo < 0.3 < hi and lo < mean < hi


def test_paired_difference_detects_clear_gap_and_not_noise():
    idx = pd.Index(range(20), name="batch_idx")
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.5, 0.2, 20), index=idx)
    assert paired_difference(base + 0.2, base)["significant"]
    noise = pd.Series(rng.normal(0, 0.2, 20), index=idx)
    assert not paired_difference(base + noise * 0.01, base + noise * 0.01)["significant"]


def test_paired_difference_averages_seeds_per_batch():
    idx = pd.Index(range(10), name="batch_idx")
    a = pd.DataFrame({0: np.ones(10), 1: np.ones(10) * 3}, index=idx)
    b = pd.DataFrame({0: np.zeros(10), 1: np.zeros(10)}, index=idx)
    assert paired_difference(a, b)["mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------- SHAP
def test_shap_finds_the_planted_signal_before_and_after_drift():
    from src.explain.shap_analysis import shap_importance

    syn = load_dataset(n_samples=20_000, fraud_rate=0.02, seed=0)
    X, y = syn[get_feature_columns(syn)], syn["Class"]
    n = len(syn)
    v_cols = [c for c in X.columns if c.startswith("V")]
    early = make_random_forest().fit(X.iloc[: int(0.4 * n)], y.iloc[: int(0.4 * n)])
    late = make_random_forest().fit(X.iloc[int(0.6 * n):], y.iloc[int(0.6 * n):])
    imp_early = shap_importance(early, X.iloc[: int(0.4 * n)], y.iloc[: int(0.4 * n)], max_rows=800)
    imp_late = shap_importance(late, X.iloc[int(0.6 * n):], y.iloc[int(0.6 * n):], max_rows=800)
    # Amount also changes at the drift, so only the V columns are checked
    assert set(imp_early[v_cols].nlargest(3).index) == {"V1", "V2", "V3"}
    assert set(imp_late[v_cols].nlargest(3).index) == {"V4", "V5", "V6"}
    assert (imp_early >= 0).all() and len(imp_early) == X.shape[1]
