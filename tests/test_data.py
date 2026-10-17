import numpy as np
import pandas as pd
import pytest

from src.data.loader import get_feature_columns, load_dataset
from src.data.stream import make_stream
from src.data.synthetic import generate_synthetic_stream


@pytest.fixture(scope="module")
def df():
    return load_dataset(n_samples=20_000, fraud_rate=0.02, seed=0)


def test_synthetic_has_required_columns(df):
    assert {"Time", "Class", "Amount", "V1"} <= set(df.columns)
    assert get_feature_columns(df)[0] == "V1"
    assert "Time" not in get_feature_columns(df) and "Class" not in get_feature_columns(df)


def test_synthetic_is_time_sorted(df):
    assert df["Time"].is_monotonic_increasing


def test_synthetic_fraud_rate_close_to_target(df):
    assert df["Class"].mean() == pytest.approx(0.02, abs=0.005)


def test_synthetic_fraud_pattern_drifts(df):
    half = len(df) // 2
    early = df.iloc[:half].query("Class == 1")
    late = df.iloc[half:].query("Class == 1")
    assert early["V1"].mean() > 1.0 and abs(late["V1"].mean()) < 0.5
    assert late["V4"].mean() < -1.0 and abs(early["V4"].mean()) < 0.5


def test_stream_yields_ordered_batches(df):
    batches = list(make_stream(df, batch_size=3000))
    assert len(batches) == 7  # 6 full + 1 partial
    assert sum(len(y) for _, y in batches) == len(df)
    last_times = [df.loc[X.index, "Time"].max() for X, _ in batches]
    assert last_times == sorted(last_times)
    X, y = batches[0]
    assert list(X.columns) == get_feature_columns(df) and y.name == "Class"


def test_stream_rejects_unsorted(df):
    with pytest.raises(ValueError):
        next(make_stream(df.iloc[::-1]))


def test_load_csv_with_ieee_layout(tmp_path):
    raw = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [40, 10, 30, 20],
        "isFraud": [0, 1, 0, 0],
        "card4": ["visa", None, "mastercard", "visa"],
        "C1": [1.0, np.nan, 3.0, 5.0],
    })
    path = tmp_path / "train_transaction.csv"
    raw.to_csv(path, index=False)

    df = load_dataset(path)
    assert list(df["Time"]) == [10, 20, 30, 40]
    assert "TransactionID" not in df.columns
    assert df["C1"].notna().all() and df.loc[0, "C1"] == 3.0  # median of 1, 3, 5
    assert pd.api.types.is_integer_dtype(df["card4"])


def test_generator_rejects_too_few_features():
    with pytest.raises(ValueError):
        generate_synthetic_stream(n_features=4)
