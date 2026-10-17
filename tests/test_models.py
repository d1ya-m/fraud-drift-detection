import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.stream import make_stream
from src.evaluation.prequential import prequential_eval
from src.drift.detectors import make_adwin
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest
from src.models.drift_retrain import AccumulateAfterDrift
from src.models.online import make_arf, make_online_logreg
from src.models.retrain import PeriodicRetrain

# 20k rows, batch 2000 -> drift between batch 4 and 5 of the evaluated stream (index 5 onwards is post-drift)
DRIFT_BATCH = 5


@pytest.fixture(scope="module")
def df():
    return load_dataset(n_samples=20_000, fraud_rate=0.02, seed=3)


def run(model, df):
    return prequential_eval(model, make_stream(df, batch_size=2000), warmup_batches=2)


def post_drift_pr_auc(res):
    return res.loc[res["batch_idx"] >= DRIFT_BATCH + 2, "pr_auc"].mean()


@pytest.mark.parametrize("factory", [
    make_logreg, make_random_forest, make_gradient_boosting, make_online_logreg, make_arf,
])
def test_model_follows_contract(factory, df):
    model = factory()
    X, y = next(make_stream(df, batch_size=2000))
    scores = model.fit(X, y).predict_proba(X)
    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))
    assert model.partial_fit(X, y) is model


@pytest.mark.parametrize("factory", [make_logreg, make_random_forest, make_gradient_boosting])
def test_static_models_collapse_after_drift(factory, df):
    res = run(factory(), df)
    assert res.loc[res["batch_idx"] < DRIFT_BATCH, "pr_auc"].mean() > 0.4
    assert post_drift_pr_auc(res) < 0.2


@pytest.mark.parametrize("make_model", [
    lambda: PeriodicRetrain(make_logreg(), every=2, window=4),
    make_online_logreg,
    make_arf,
])
def test_adaptive_models_recover_after_drift(make_model, df):
    assert post_drift_pr_auc(run(make_model(), df)) > 0.4


def test_accumulate_keeps_refitting_after_alarm_then_freezes(df):
    model = AccumulateAfterDrift(make_logreg(), accumulate=3)
    batches = list(make_stream(df, batch_size=2000))
    model.fit(*batches[0])
    frozen = model.inner.model
    model.partial_fit(*batches[1])
    assert model.inner.model is frozen  # no alarm yet: behaves like a static model

    model.refit(*batches[2])  # alarm
    seen = [model.inner.model]
    for b in batches[3:6]:
        model.partial_fit(*b)
        seen.append(model.inner.model)
    # refit on alarm + 2 growing refits (buffer 2, 3 batches), then frozen
    assert len({id(m) for m in seen}) == 3
    assert seen[-1] is seen[-2]
    assert len(model.buffer) == 3


def test_accumulate_beats_single_batch_refit_after_drift(df):
    def with_adwin(model):
        return prequential_eval(model, make_stream(df, batch_size=2000), drift_detector=make_adwin(), warmup_batches=2)

    single = with_adwin(make_logreg())
    acc = with_adwin(AccumulateAfterDrift(make_logreg(), accumulate=4))
    assert post_drift_pr_auc(acc) >= post_drift_pr_auc(single)


def test_static_fit_needs_both_classes():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError):
        make_logreg().fit(X, pd.Series([0, 0, 0]))


def test_static_refit_without_fraud_keeps_model(df):
    X, y = next(make_stream(df, batch_size=2000))
    model = make_logreg().fit(X, y)
    before = model.predict_proba(X)
    model.refit(X, pd.Series(0, index=y.index))
    np.testing.assert_allclose(model.predict_proba(X), before)


def test_periodic_retrain_refits_every_n_batches(df):
    model = PeriodicRetrain(make_logreg(), every=3, window=5)
    batches = list(make_stream(df, batch_size=2000))
    model.fit(*batches[0])
    first = model.inner.model
    model.partial_fit(*batches[1])
    model.partial_fit(*batches[2])
    assert model.inner.model is first
    model.partial_fit(*batches[3])
    assert model.inner.model is not first
    assert len(model.buffer) == 4
    assert model.name == "logreg_retrain3"
