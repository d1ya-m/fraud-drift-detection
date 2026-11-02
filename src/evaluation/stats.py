"""Uncertainty for comparisons between strategies.

Per-batch scores are noisy (PR-AUC sd ~0.2 per batch on ULB) and neighbouring batches are correlated,
so a difference between two strategies is judged with a paired moving-block bootstrap over batches:
the per-batch difference (averaged over seeds when there are several runs) is resampled in blocks of
consecutive batches, which keeps short-range correlation inside each block.
"""
import numpy as np
import pandas as pd


def block_bootstrap_mean(values, block=3, n_boot=5000, seed=0, level=0.95):
    """Mean of a time-ordered series and its moving-block bootstrap confidence interval."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return x.mean(), np.quantile(means, alpha), np.quantile(means, 1 - alpha)


def paired_difference(a, b, block=3, n_boot=5000, seed=0):
    """Mean of a - b with a block-bootstrap CI.

    a, b: Series indexed by batch_idx, or DataFrames indexed by batch_idx with one column per seed
    (seeds are averaged per batch first). Returns dict(mean, lo, hi, n_batches, significant).
    """
    diff = (pd.DataFrame(a) - pd.DataFrame(b)).mean(axis=1).sort_index().dropna()
    mean, lo, hi = block_bootstrap_mean(diff, block=block, n_boot=n_boot, seed=seed)
    return {"mean": mean, "lo": lo, "hi": hi, "n_batches": len(diff), "significant": bool(lo > 0 or hi < 0)}
