"""Contract 1: the data stream. Batches are yielded strictly in time order, never shuffled."""
from src.data.loader import TARGET, TIME, get_feature_columns, load_dataset  # noqa: F401


def make_stream(df, batch_size=2000, features=None):
    """Yield (X_batch, y_batch) in time order. The last batch may be smaller."""
    if not df[TIME].is_monotonic_increasing:
        raise ValueError("df must be sorted by Time; use load_dataset()")
    features = features or get_feature_columns(df)
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        yield batch[features], batch[TARGET]
