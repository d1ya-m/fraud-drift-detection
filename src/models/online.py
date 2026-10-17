"""Online models (River): learn from every transaction, one at a time, as the stream goes by."""
import numpy as np
from river import forest, imblearn, linear_model, optim, preprocessing

from src.models.base import BaseDetector


class OnlineModel(BaseDetector):
    """Wraps a River classifier. `prototype` is cloned fresh on fit/refit."""

    def __init__(self, name, prototype):
        self.name = name
        self.prototype = prototype
        self.model = None

    def fit(self, X, y):
        self.model = self.prototype.clone()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        for x, label in zip(X.to_dict(orient="records"), y):
            self.model.learn_one(x, int(label))
        return self

    def predict_proba(self, X):
        return np.array([
            self.model.predict_proba_one(x).get(1, 0.0)
            for x in X.to_dict(orient="records")
        ])


def _balanced(classifier, seed):
    # River's answer to class_weight: drop legitimate transactions until fraud is ~20% of what is learned
    return imblearn.RandomUnderSampler(classifier, desired_dist={0: 0.8, 1: 0.2}, seed=seed)


def make_online_logreg(seed=42):
    return OnlineModel("online_logreg", _balanced(
        preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.05)),
        seed,
    ))


def make_arf(seed=42):
    return OnlineModel("online_arf", _balanced(
        forest.ARFClassifier(n_models=10, seed=seed),
        seed,
    ))
