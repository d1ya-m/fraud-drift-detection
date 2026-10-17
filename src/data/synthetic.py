"""Synthetic, time-ordered, imbalanced transaction stream with one abrupt concept drift.

Layout mirrors the ULB dataset (Time, V1..Vn, Amount, Class) so every downstream
module works unchanged on synthetic and real data.

Before the drift point, fraud is shifted *up* on V1-V3 and has larger amounts.
After it, fraud is shifted *down* on V4-V6 (V1-V3 look legitimate again) and
amounts are small ("card testing"). A model frozen on the early pattern should
lose most of its recall after the drift.
"""
import numpy as np
import pandas as pd


def generate_synthetic_stream(
    n_samples=50_000,
    n_features=10,
    fraud_rate=0.01,
    drift_at=0.5,
    shift=2.0,
    duration=172_800,
    seed=42,
):
    """Return a DataFrame sorted by Time with columns Time, V1..Vn, Amount, Class.

    drift_at is the fraction of the stream (by time order) where the fraud pattern changes.
    """
    if n_features < 6:
        raise ValueError("n_features must be >= 6 (V1-V3 and V4-V6 carry the two fraud patterns)")

    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, duration, n_samples))
    y = (rng.random(n_samples) < fraud_rate).astype(int)
    X = rng.standard_normal((n_samples, n_features))
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n_samples)

    after = np.arange(n_samples) >= int(drift_at * n_samples)
    early_fraud = (y == 1) & ~after
    late_fraud = (y == 1) & after

    X[np.ix_(early_fraud, [0, 1, 2])] += shift
    X[np.ix_(late_fraud, [3, 4, 5])] -= shift
    amount[early_fraud] = rng.lognormal(mean=4.0, sigma=1.2, size=early_fraud.sum())
    amount[late_fraud] = rng.lognormal(mean=1.5, sigma=0.8, size=late_fraud.sum())

    df = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(n_features)])
    df.insert(0, "Time", time)
    df["Amount"] = amount.round(2)
    df["Class"] = y
    return df
