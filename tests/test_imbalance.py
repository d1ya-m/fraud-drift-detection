import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.loader import load_dataset
from src.data.stream import make_stream
from src.imbalance.samplers import METHODS, smote, undersample, with_imbalance_handling
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest


@pytest.fixture(scope="module")
def batch():
    df = load_dataset(n_samples=6000, fraud_rate=0.02, seed=0)
    return next(make_stream(df, batch_size=6000))


def fraud_ratio(y):
    y = np.asarray(y)
    return (y == 1).sum() / (y == 0).sum()


def test_smote_adds_synthetic_frauds_only(batch):
    X, y = batch
    Xr, yr = smote(X, y, ratio=0.5)
    assert fraud_ratio(yr) == pytest.approx(0.5, abs=0.01)
    assert (np.asarray(yr) == 0).sum() == (np.asarray(y) == 0).sum()  # legit rows untouched
    assert len(Xr) == len(yr) > len(X)


def test_smote_copes_with_very_few_frauds(batch):
    X, y = batch
    y = pd.Series(0, index=y.index)
    y.iloc[:3] = 1  # 3 frauds: default k_neighbors=5 would crash
    _, yr = smote(X, y, ratio=0.5)
    assert fraud_ratio(yr) == pytest.approx(0.5, abs=0.01)
    y.iloc[1:3] = 0  # 1 fraud: nothing to interpolate, data returned unchanged
    Xr, yr = smote(X, y, ratio=0.5)
    assert len(Xr) == len(X)


def test_undersample_drops_legit_only(batch):
    X, y = batch
    _, yr = undersample(X, y, ratio=0.5)
    assert fraud_ratio(yr) == pytest.approx(0.5, abs=0.01)
    assert (np.asarray(yr) == 1).sum() == (np.asarray(y) == 1).sum()


def test_class_weight_set_per_method():
    rf = RandomForestClassifier()
    assert with_imbalance_handling(rf, "weights").class_weight == "balanced_subsample"
    assert with_imbalance_handling(rf, "none").class_weight is None
    smote_pipe = with_imbalance_handling(rf, "smote")
    assert smote_pipe.steps[-1][1].class_weight is None  # don't double-correct
    assert [name for name, _ in smote_pipe.steps] == ["sampler", "clf"]


def test_sampler_goes_after_scaler():
    pipe = with_imbalance_handling(make_logreg().estimator, "smote")
    assert [name for name, _ in pipe.steps] == ["standardscaler", "sampler", "logisticregression"]


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        with_imbalance_handling(RandomForestClassifier(), "oversample-everything")


@pytest.mark.parametrize("factory", [make_logreg, make_random_forest, make_gradient_boosting])
@pytest.mark.parametrize("method", list(METHODS))
def test_every_model_trains_with_every_method(factory, method, batch):
    X, y = batch
    model = factory(imbalance=method).fit(X, y)
    scores = model.predict_proba(X)
    assert scores.shape == (len(X),)  # resampling never touches the data being scored
    assert np.all((scores >= 0) & (scores <= 1))
    assert model.name.endswith("" if method == "weights" else f"_{method}")
