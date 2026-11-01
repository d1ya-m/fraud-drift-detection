"""Which features a tree model relies on: mean |SHAP value| per feature for the fraud class."""
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _tree_model(model):
    """The fitted tree estimator inside a StaticModel (a resampling pipeline's samplers only act at fit time)."""
    est = model.model
    return est.steps[-1][1] if isinstance(est, Pipeline) else est


def sample_rows(X, y, max_rows=2000, seed=42):
    """All frauds plus a random sample of legitimate rows, at most max_rows in total."""
    y = pd.Series(np.asarray(y), index=X.index)
    fraud = y.index[y == 1]
    legit = y.index[y == 0]
    n_legit = max(0, min(len(legit), max_rows - len(fraud)))
    keep = fraud.union(pd.Index(np.random.default_rng(seed).choice(legit, n_legit, replace=False)))
    return X.loc[keep]


def shap_importance(model, X, y, max_rows=2000, seed=42):
    """Mean |SHAP value| per feature for the fraud class, sorted descending.

    model: a fitted StaticModel wrapping a random forest or gradient boosting model.
    Shapes differ by model (shap 0.53): random forest returns (rows, features, 2) -> fraud slice [:, :, 1];
    gradient boosting returns (rows, features) already for the fraud class.
    """
    rows = sample_rows(X, y, max_rows, seed)
    values = np.asarray(shap.TreeExplainer(_tree_model(model)).shap_values(rows))
    if values.ndim == 3:
        values = values[:, :, 1]
    return pd.Series(np.abs(values).mean(axis=0), index=X.columns).sort_values(ascending=False)
