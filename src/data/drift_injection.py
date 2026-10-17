"""Inject a known concept drift into a real dataset (the controlled experiment on ULB).

Feature swap: from the drift point onwards, the k features most correlated with fraud
exchange values with the k least correlated ones, for every transaction. The data stay real,
but the columns a model learned to watch stop carrying the fraud signal and other columns
start to, i.e. P(fraud | X) changes abruptly at a known time.
"""
from src.data.loader import TARGET, get_feature_columns


def choose_swap_pairs(df, k=4):
    """Pair the k most fraud-correlated features with the k least correlated ones."""
    features = [c for c in get_feature_columns(df) if c != "Amount"]
    corr = df[features].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    strongest, weakest = list(corr.index[:k]), list(corr.index[-k:][::-1])
    return list(zip(strongest, weakest))


def inject_feature_swap(df, at=0.5, k=4, pairs=None):
    """Return (drifted copy of df, pairs swapped). df must be time-sorted; `at` is the fraction of rows."""
    pairs = pairs or choose_swap_pairs(df, k)
    out = df.copy()
    start = int(at * len(out))
    for a, b in pairs:
        # explicit copies: to_numpy() can return a view, so writing column a first would
        # overwrite the values column b is about to receive
        a_vals = out[a].to_numpy(copy=True)
        b_vals = out[b].to_numpy(copy=True)
        a_vals[start:], b_vals[start:] = b_vals[start:].copy(), a_vals[start:].copy()
        out[a], out[b] = a_vals, b_vals
    return out, pairs
