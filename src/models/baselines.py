"""Static models: fit once at warm-up, then frozen (partial_fit is a no-op)."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseDetector


def has_both_classes(y):
    return len(np.unique(y)) == 2


class StaticModel(BaseDetector):
    """Wraps any scikit-learn classifier with predict_proba."""

    def __init__(self, name, estimator):
        self.name = name
        self.estimator = estimator
        self.model = None

    def fit(self, X, y):
        if not has_both_classes(y):
            raise ValueError(f"{self.name}: training data needs both classes; use more warm-up batches")
        self.model = clone(self.estimator).fit(X, y)
        return self

    def refit(self, X, y):
        # a drift batch with no fraud can't train a classifier, so keep the current one
        return self.fit(X, y) if has_both_classes(y) else self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def make_logreg(seed=42):
    return StaticModel("logreg", make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    ))


def make_random_forest(seed=42):
    return StaticModel("random_forest", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    ))


def make_gradient_boosting(seed=42):
    return StaticModel("gradient_boosting", HistGradientBoostingClassifier(
        class_weight="balanced", random_state=seed,
    ))
