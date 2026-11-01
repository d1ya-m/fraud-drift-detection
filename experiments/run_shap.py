"""SHAP: which features a random forest relies on before vs after the injected drift.

Three tables, each from held-out rows (never the rows the model was trained on):
  before drift   model trained on the last --train-rows before the drift, explained on other pre-drift rows
  after drift    model trained on the first --train-rows after the drift, explained on later post-drift rows
  frozen, after  the BEFORE model explained on post-drift rows: what a frozen model still looks at

    python experiments/run_shap.py --data data/raw/creditcard.csv --inject-drift 0.5
    python experiments/run_shap.py                                   # synthetic
Output: results/<tag>_shap_importance.csv, results/figures/<tag>_shap.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.run_experiment import add_common_args, load_data  # noqa: E402
from src.data.loader import TARGET, get_feature_columns  # noqa: E402
from src.explain.shap_analysis import shap_importance  # noqa: E402
from src.models.baselines import make_random_forest  # noqa: E402
from src.viz.plots import plot_feature_importance  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--train-rows", type=int, default=40_000)
    p.add_argument("--max-rows", type=int, default=2000, help="rows explained per table (all frauds kept)")
    args = p.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df, tag, drift_batch = load_data(args)
    if drift_batch is None:
        raise SystemExit("needs a known drift: use synthetic data or --inject-drift")
    cut = int(drift_batch * args.batch_size)
    X, y = df[get_feature_columns(df)], df[TARGET]
    n = min(args.train_rows, cut // 2, (len(df) - cut) // 2)  # leave held-out rows on both sides
    pre_train, pre_test = slice(cut - n, cut), slice(0, cut - n)
    post_train, post_test = slice(cut, cut + n), slice(cut + n, len(df))

    before = make_random_forest(seed=args.seed).fit(X.iloc[pre_train], y.iloc[pre_train])
    after = make_random_forest(seed=args.seed).fit(X.iloc[post_train], y.iloc[post_train])
    tables = {
        "before drift": shap_importance(before, X.iloc[pre_test], y.iloc[pre_test], args.max_rows, args.seed),
        "after drift (retrained)": shap_importance(after, X.iloc[post_test], y.iloc[post_test], args.max_rows, args.seed),
        "frozen model, after drift": shap_importance(before, X.iloc[post_test], y.iloc[post_test], args.max_rows, args.seed),
    }
    table = pd.DataFrame(tables).rename_axis("feature").reset_index()
    table.to_csv(out / f"{tag}_shap_importance.csv", index=False)
    plot_feature_importance(tables, out / "figures" / f"{tag}_shap.png",
                            title=f"Random forest: top features by mean |SHAP| ({tag})")
    for label, s in tables.items():
        print(f"{label:26s} top 5: " + ", ".join(f"{f} {v:.3f}" for f, v in s.head(5).items()))


if __name__ == "__main__":
    main()
