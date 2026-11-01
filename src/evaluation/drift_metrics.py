"""How a model reacts to drift (objectives 2 and 3): size of the drop, recovery time, slow decay.

All functions read the per-batch results format written by prequential_eval, so they work on
any saved results CSV without rerunning a model.
"""
import math

import numpy as np
import pandas as pd


def drift_response(run, drift_batch, recover_frac=0.9, smooth=3, metric="pr_auc"):
    """Summarise one model run around a known drift that starts at (possibly fractional) drift_batch.

    before / after:   mean metric over batches entirely before / entirely after the drift
                      (the batch that straddles the drift point counts as neither)
    drop, drop_pct:   before - after, absolute and as % of before
    worst_after:      lowest `smooth`-batch rolling mean after the drift
    batches_below:    post-drift batches whose rolling mean is under recover_frac x before
    recovery_batches: batches from the drift until the rolling mean is back at or above
                      recover_frac x before, after having fallen below it.
                      0 = never fell below; NaN = fell and never came back within the stream.
    """
    run = run.sort_values("batch_idx")
    before = run.loc[run["batch_idx"] < math.floor(drift_batch), metric].mean()
    after = run.loc[run["batch_idx"] >= math.ceil(drift_batch), metric]
    # smooth within the post-drift period only, so pre-drift scores can't mask the drop
    rolling = after.rolling(smooth, min_periods=1).mean().to_numpy()
    target = recover_frac * before
    below = rolling < target

    if not below.any():
        recovery = 0.0
    else:
        first_drop = int(np.argmax(below))
        back = np.flatnonzero(~below[first_drop:] & ~np.isnan(rolling[first_drop:]))
        recovery = float(first_drop + back[0]) if back.size else np.nan

    return {
        "before": before,
        "after": after.mean(),
        "drop": before - after.mean(),
        "drop_pct": 100 * (before - after.mean()) / before if before else np.nan,
        "worst_after": np.nanmin(rolling) if len(rolling) else np.nan,
        "batches_below": int(below.sum()),
        "recovery_batches": recovery,
    }


def trend_per_10_batches(run, metric="pr_auc"):
    """Least-squares slope of the metric over time, scaled to 'change per 10 batches'.

    For data without a known drift point, a clearly negative value means slow decay.
    """
    points = run[["batch_idx", metric]].dropna()
    if len(points) < 3:
        return np.nan
    slope = np.polyfit(points["batch_idx"], points[metric], 1)[0]
    return 10 * slope


def drift_response_table(results, drift_batch, **kwargs):
    """drift_response for every model_name in a results DataFrame, one row each."""
    rows = {name: drift_response(run, drift_batch, **kwargs) for name, run in results.groupby("model_name", sort=False)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("model_name").reset_index()
