"""Stage 5: compare imbalance handling (none / class weights / SMOTE / undersampling) for the static models.

Each static model runs frozen ("base") and with the best adaptive strategy ("+ ADWIN + accumulate").

    python experiments/run_imbalance.py                                            # synthetic
    python experiments/run_imbalance.py --data data/raw/creditcard.csv --batch-size 10000
    python experiments/run_imbalance.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.run_experiment import evaluate, load_data  # noqa: E402
from src.drift.detectors import make_adwin  # noqa: E402
from src.imbalance.samplers import METHODS  # noqa: E402
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest  # noqa: E402
from src.models.drift_retrain import AccumulateAfterDrift  # noqa: E402
from src.viz.plots import plot_grouped_bars  # noqa: E402

FAMILIES = {
    "logreg": ("Logistic regression", make_logreg),
    "random_forest": ("Random forest", make_random_forest),
    "gradient_boosting": ("Gradient boosting", make_gradient_boosting),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="CSV path; omit to use the synthetic drifting stream")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--inject-drift", type=float, metavar="FRACTION")
    p.add_argument("--models", default=",".join(FAMILIES), help=f"comma-separated subset of {list(FAMILIES)}")
    p.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated subset of {list(METHODS)}")
    p.add_argument("--ratio", type=float, default=0.5, help="frauds per legitimate transaction after resampling")
    p.add_argument("--accumulate", type=int, default=6)
    p.add_argument("--precision-target", type=float, default=0.90)
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df, tag, drift_batch = load_data(args)

    all_results = []
    for key in args.models.split(","):
        title, factory = FAMILIES[key]
        for method in args.methods.split(","):
            def build():
                return factory(seed=args.seed, imbalance=method)
            labels = {"family": title, "imbalance": METHODS[method]}
            all_results.append(evaluate(build(), None, df, args, strategy="base", **labels))
            all_results.append(evaluate(AccumulateAfterDrift(build(), accumulate=args.accumulate), make_adwin(),
                                        df, args, strategy="+ ADWIN + accumulate", **labels))

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(out / f"{tag}_imbalance.csv", index=False)

    keys = ["model_name", "family", "strategy", "imbalance"]
    summary = results.groupby(keys, sort=False).agg(
        mean_pr_auc=("pr_auc", "mean"),
        mean_recall_at_p=("recall_at_p", "mean"),
        mean_fpr=("fpr", "mean"),
        n_drift_flags=("drift_flag", "sum"),
        runtime_s=("runtime_s", "first"),
    ).reset_index()
    summary.to_csv(out / f"{tag}_imbalance_summary.csv", index=False)

    figure = plot_grouped_bars(summary, out / "figures" / f"{tag}_imbalance.png",
                               metrics=["mean_pr_auc", "mean_recall_at_p"],
                               title=f"Imbalance handling ({tag})")

    table = summary.pivot_table(index=["strategy", "family"], columns="imbalance", values="mean_pr_auc", sort=False)
    print("\nMean PR-AUC\n" + table.round(3).to_string())
    runtime = summary.pivot_table(index=["strategy", "family"], columns="imbalance", values="runtime_s", sort=False)
    print("\nRuntime (s)\n" + runtime.round(1).to_string())
    print(f"\nCSVs in {out}, figure {figure}")


if __name__ == "__main__":
    main()
