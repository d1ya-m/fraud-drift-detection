"""Imbalance handling for the static (scikit-learn) models: class weights or resampling.

Resampling happens only while a model is trained; scoring always runs on the real, untouched
transactions. (The online River models handle imbalance with their own undersampler.)
"""
import numpy as np
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

METHODS = {
    "none": "no handling",
    "weights": "class weights",
    "smote": "SMOTE",
    "undersample": "undersampling",
}


def _counts(y):
    n_fraud = int(np.sum(np.asarray(y) == 1))
    return n_fraud, len(y) - n_fraud


def smote(X, y, ratio=0.5, seed=42):
    """Add synthetic frauds until there are `ratio` frauds per legitimate transaction.

    SMOTE needs k_neighbors < number of frauds, and single ULB batches can hold as few as 3,
    so k shrinks to fit; with fewer than 2 frauds there is nothing to interpolate between.
    """
    n_fraud, n_legit = _counts(y)
    if n_fraud < 2 or n_fraud >= ratio * n_legit:
        return X, y
    k = min(5, n_fraud - 1)
    return SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=seed).fit_resample(X, y)


def undersample(X, y, ratio=0.5, seed=42):
    """Drop legitimate transactions at random until there are `ratio` frauds per legitimate one."""
    n_fraud, n_legit = _counts(y)
    if n_fraud == 0 or n_fraud >= ratio * n_legit:
        return X, y
    return RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(X, y)


def with_imbalance_handling(estimator, method="weights", ratio=0.5, seed=42):
    """Return a copy of `estimator` (a classifier, or a Pipeline ending in one) set up for `method`.

    weights:              class_weight on the classifier ("balanced_subsample" for random forests)
    none:                 class_weight removed, data used as-is
    smote / undersample:  a sampler inserted just before the classifier (after any scaler, since
                          SMOTE interpolates between neighbours in feature space)
    """
    if method not in METHODS:
        raise ValueError(f"unknown imbalance method {method!r}; choose from {list(METHODS)}")

    steps = list(clone(estimator).steps) if isinstance(estimator, Pipeline) else [("clf", clone(estimator))]
    clf = steps[-1][1]
    if "class_weight" in clf.get_params():
        weight = "balanced_subsample" if isinstance(clf, RandomForestClassifier) else "balanced"
        clf.set_params(class_weight=weight if method == "weights" else None)

    if method in ("smote", "undersample"):
        func = smote if method == "smote" else undersample
        sampler = FunctionSampler(func=func, kw_args={"ratio": ratio, "seed": seed}, validate=False)
        return ImbPipeline(steps[:-1] + [("sampler", sampler), steps[-1]])
    return Pipeline(steps) if len(steps) > 1 else clf
