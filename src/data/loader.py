"""Load a dataset into the common format: a Time column, a binary Class column, numeric features."""
import pandas as pd

from src.data.synthetic import generate_synthetic_stream

TIME = "Time"
TARGET = "Class"  # 1 = fraud, 0 = legitimate

# IEEE-CIS uses different names for the same roles
_RENAMES = {"TransactionDT": TIME, "isFraud": TARGET}
_ID_COLUMNS = ["TransactionID"]


def get_feature_columns(df):
    """Every column except Time and Class, so ULB, IEEE-CIS and synthetic all work."""
    return [c for c in df.columns if c not in (TIME, TARGET)]


def load_dataset(path=None, **synthetic_kwargs):
    """Load a CSV (or generate the synthetic stream when path is None), validate it, sort by Time."""
    if path is None:
        df = generate_synthetic_stream(**synthetic_kwargs)
    else:
        df = pd.read_csv(path)
        df = df.rename(columns=_RENAMES)
        df = df.drop(columns=[c for c in _ID_COLUMNS if c in df.columns])
        df = _clean(df)

    _validate(df)
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


def _clean(df):
    """Median-impute numeric columns; fill and integer-encode categoricals (needed for IEEE-CIS).

    Note: medians are taken over the whole file, a mild look-ahead that is acceptable for
    imputation but should be mentioned in the report.
    """
    df = df.copy()
    features = get_feature_columns(df)
    numeric = df[features].select_dtypes(include="number").columns
    categorical = [c for c in features if c not in numeric]

    df[numeric] = df[numeric].fillna(df[numeric].median())
    for col in categorical:
        df[col] = df[col].fillna("missing").astype("category").cat.codes
    return df


def _validate(df):
    missing = {TIME, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{TARGET} must be binary 0/1")
    if df[get_feature_columns(df)].isna().any().any():
        raise ValueError("features contain missing values after cleaning")
