import numpy as np
import pytest

from src.data.loader import load_dataset
from src.data.stream import make_stream
from src.evaluation.prequential import RESULT_COLUMNS, prequential_eval
from src.models.base import BaseDetector
from src.models.baselines import StaticModel, make_logreg


class ConstantModel(BaseDetector):
    name = "constant"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


class FrozenLogistic(StaticModel):
    """Static logistic regression that counts its refits."""

    def __init__(self):
        super().__init__("frozen_logreg", make_logreg().estimator)
        self.refits = 0

    def refit(self, X, y):
        self.refits += 1
        return super().refit(X, y)


class FireAt:
    """Fake drift detector that fires on the n-th update."""
    name = "fire_at"

    def __init__(self, n):
        self.n, self.calls = n, 0

    def update(self, error):
        self.calls += 1
        return self.calls == self.n


@pytest.fixture(scope="module")
def df():
    return load_dataset(n_samples=20_000, fraud_rate=0.02, seed=1)


def test_constant_model_one_row_per_batch(df):
    res = prequential_eval(ConstantModel(), make_stream(df, batch_size=2000))
    assert list(res.columns) == RESULT_COLUMNS
    assert len(res) == 9  # 10 batches minus the warm-up batch
    assert (res["model_name"] == "constant").all()
    assert res["drift_flag"].sum() == 0


def test_warmup_spans_several_batches(df):
    res = prequential_eval(ConstantModel(), make_stream(df, batch_size=2000), warmup_batches=3)
    assert len(res) == 7
    assert res["batch_idx"].tolist() == list(range(3, 10))


def test_frozen_model_degrades_after_drift(df):
    res = prequential_eval(FrozenLogistic(), make_stream(df, batch_size=2000))
    before = res.loc[res["batch_idx"] < 5, "pr_auc"].mean()
    after = res.loc[res["batch_idx"] >= 5, "pr_auc"].mean()
    assert before > 0.5
    assert after < before - 0.2


def test_drift_detector_triggers_refit(df):
    model = FrozenLogistic()
    res = prequential_eval(model, make_stream(df, batch_size=2000), drift_detector=FireAt(6))
    assert res["drift_flag"].sum() == 1
    assert res.loc[res["drift_flag"] == 1, "batch_idx"].item() == 6
    assert model.refits == 1
    # refitting on a post-drift batch recovers performance
    assert res.loc[res["batch_idx"] > 6, "pr_auc"].mean() > res.loc[res["batch_idx"] == 6, "pr_auc"].item()
