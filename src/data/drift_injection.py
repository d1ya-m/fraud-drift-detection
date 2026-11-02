"""Inject a known concept drift into a real dataset (the controlled experiment on ULB).

Feature swap: from the drift point onwards, pairs of features exchange values for every transaction.
The data stay real, but the columns a model learned to watch stop carrying the fraud signal and other
columns start to, i.e. P(fraud | X) changes at a known time.

Pairs are either the k features most correlated with fraud swapped with the k least correlated
(default: a severe, targeted drift; note the choice uses the labels of the whole file, which only
designs the experiment and is never seen by a model), or k random pairs (pair_seed: a drift of
random, usually milder, severity). width > 0 makes the drift gradual: within the window that starts
at `at`, each row is swapped with a probability rising linearly from 0 to 1.
"""
import numpy as np

from src.data.loader import TARGET, get_feature_columns


def _candidate_features(df):
    return [c for c in get_feature_columns(df) if c != "Amount"]


def choose_swap_pairs(df, k=4):
    """Pair the k most fraud-correlated features with the k least correlated ones."""
    features = _candidate_features(df)
    corr = df[features].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    strongest, weakest = list(corr.index[:k]), list(corr.index[-k:][::-1])
    return list(zip(strongest, weakest))


def choose_random_pairs(df, k=4, seed=0):
    """k disjoint random pairs of features, ignoring how predictive they are."""
    picked = np.random.default_rng(seed).choice(_candidate_features(df), size=2 * k, replace=False)
    return [(str(a), str(b)) for a, b in zip(picked[:k], picked[k:])]


def inject_feature_swap(df, at=0.5, k=4, pairs=None, pair_seed=None, width=0.0, seed=0):
    """Return (drifted copy of df, pairs swapped). df must be time-sorted; at and width are fractions of rows."""
    if pairs is None:
        pairs = choose_random_pairs(df, k, pair_seed) if pair_seed is not None else choose_swap_pairs(df, k)
    n = len(df)
    start, end = int(at * n), min(n, int((at + width) * n))
    swap = np.zeros(n, dtype=bool)
    swap[end:] = True
    if end > start:  # gradual: probability of a row being swapped rises linearly across the window
        ramp = (np.arange(start, end) - start + 1) / (end - start)
        swap[start:end] = np.random.default_rng(seed).random(end - start) < ramp

    out = df.copy()
    for a, b in pairs:
        # explicit copies: to_numpy() can return a view, so writing column a first would
        # overwrite the values column b is about to receive
        a_vals = out[a].to_numpy(copy=True)
        b_vals = out[b].to_numpy(copy=True)
        a_new, b_new = a_vals.copy(), b_vals.copy()
        a_new[swap], b_new[swap] = b_vals[swap], a_vals[swap]
        out[a], out[b] = a_new, b_new
    return out, pairs
