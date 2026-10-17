import numpy as np

from src.data.drift_injection import choose_swap_pairs, inject_feature_swap
from src.data.loader import load_dataset


def test_swap_exchanges_values_after_drift_point_only():
    df = load_dataset(n_samples=4000, seed=0)
    drifted, pairs = inject_feature_swap(df, at=0.5, pairs=[("V1", "V7"), ("V2", "V8")])
    h = len(df) // 2
    assert drifted.iloc[:h].equals(df.iloc[:h])
    for a, b in pairs:
        np.testing.assert_array_equal(drifted[a].to_numpy()[h:], df[b].to_numpy()[h:])
        np.testing.assert_array_equal(drifted[b].to_numpy()[h:], df[a].to_numpy()[h:])
    assert drifted["Class"].equals(df["Class"])


def test_swap_does_not_modify_input():
    df = load_dataset(n_samples=2000, seed=0)
    before = df.copy()
    inject_feature_swap(df, at=0.5, k=2)
    assert df.equals(before)


def test_pairs_strongest_with_weakest():
    df = load_dataset(n_samples=20_000, fraud_rate=0.02, seed=0)
    pairs = choose_swap_pairs(df, k=2)
    strong = {a for a, _ in pairs}
    weak = {b for _, b in pairs}
    # the generator plants the early fraud signal in V1-V6; V7+ are pure noise
    assert strong <= {"V1", "V2", "V3", "V4", "V5", "V6"}
    assert weak.isdisjoint(strong) and "Amount" not in strong | weak
