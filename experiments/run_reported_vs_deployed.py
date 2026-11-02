"""Reported vs deployed performance: the synopsis's core argument, measured.

A developer has the earliest --train-frac of the stream and reports a test score, then deploys the
model unchanged on everything that follows:
  reported, random split        shuffle the training period, test on a random 30% (the usual
                                evaluation in fraud papers); mean of --repeats shuffles
  reported, time-ordered split  train on the first 70% of the training period, test on the last 30%
  deployed                      model fitted on the whole training period, scored on each later
                                slice of the stream (--periods equal slices)

    python experiments/run_reported_vs_deployed.py                                          # synthetic
    python experiments/run_reported_vs_deployed.py --data data/raw/creditcard.csv --batch-size 10000
    python experiments/run_reported_vs_deployed.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from experiments.run_experiment import load_data  # noqa: E402
from src.data.loader import TARGET, get_feature_columns  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest  # noqa: E402
from src.viz.plots import plot_reported_vs_deployed  # noqa: E402

FAMILIES = {
    "logreg": ("Logistic regression", make_logreg),
    "random_forest": ("Random forest", make_random_forest),
    "gradient_boosting": ("Gradient boosting", make_gradient_boosting),
}
RANDOM, TIMED = "reported, random split", "reported, time-ordered split"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="CSV path; omit to use the synthetic drifting stream")
    p.add_argument("--batch-size", type=int, default=2000, help="only used by load_data's log line")
    p.add_argument("--inject-drift", type=float, metavar="FRACTION")
    p.add_argument("--models", default=",".join(FAMILIES), help=f"comma-separated subset of {list(FAMILIES)}")
    p.add_argument("--train-frac", type=float, default=0.4, help="earliest share of the stream the developer has")
    p.add_argument("--periods", type=int, default=6, help="equal slices the rest of the stream is scored in")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--repeats", type=int, default=3, help="shuffles averaged for the random split")
    p.add_argument("--precision-target", type=float, default=0.90)
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results"))
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df, tag, _ = load_data(args)
    # where the known drift sits, as a share of the stream (synthetic: generator default 0.5)
    drift_pos = args.inject_drift if args.data else 0.5
    if args.data and args.inject_drift is None:
        drift_pos = None
    X, y = df[get_feature_columns(df)], df[TARGET]
    n_train = int(args.train_frac * len(df))
    X_dev, y_dev = X.iloc[:n_train], y.iloc[:n_train]
    bounds = np.linspace(n_train, len(df), args.periods + 1).astype(int)

    def metrics(model, X_test, y_test):
        return compute_metrics(y_test, model.predict_proba(X_test), args.precision_target)

    rows = []
    for key in args.models.split(","):
        title, factory = FAMILIES[key]
        start = time.time()
        for r in range(args.repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(X_dev, y_dev, test_size=args.test_size, stratify=y_dev,
                                                      random_state=args.seed + r)
            rows.append({"family": title, "evaluation": RANDOM, "repeat": r, "n_fraud": int(y_te.sum()),
                         **metrics(factory(seed=args.seed).fit(X_tr, y_tr), X_te, y_te)})
        cut = int((1 - args.test_size) * n_train)
        timed = factory(seed=args.seed).fit(X_dev.iloc[:cut], y_dev.iloc[:cut])
        rows.append({"family": title, "evaluation": TIMED, "repeat": 0, "n_fraud": int(y_dev.iloc[cut:].sum()),
                     **metrics(timed, X_dev.iloc[cut:], y_dev.iloc[cut:])})

        deployed = factory(seed=args.seed).fit(X_dev, y_dev)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            rows.append({"family": title, "evaluation": "deployed", "repeat": 0,
                         "start_pct": 100 * lo / len(df), "end_pct": 100 * hi / len(df),
                         "n_fraud": int(y.iloc[lo:hi].sum()), **metrics(deployed, X.iloc[lo:hi], y.iloc[lo:hi])})
        print(f"  {title:22s} done in {time.time() - start:5.1f}s", flush=True)

    raw = pd.DataFrame(rows)
    raw["position_pct"] = (raw["start_pct"] + raw["end_pct"]) / 2
    raw.to_csv(out / f"{tag}_reported_vs_deployed.csv", index=False)
    figure = plot_reported_vs_deployed(
        raw, out / "figures" / f"{tag}_reported_vs_deployed.png", train_pct=100 * args.train_frac,
        drift_pct=None if drift_pos is None else 100 * drift_pos,
        title=f"Reported vs deployed PR-AUC ({tag}): trained once on the first {args.train_frac:.0%}")

    reported = raw[raw["evaluation"] != "deployed"].groupby(["family", "evaluation"], sort=False)["pr_auc"].mean()
    table = reported.unstack()[[RANDOM, TIMED]]
    deployed = raw[raw["evaluation"] == "deployed"]
    if drift_pos is not None:
        before = deployed["end_pct"] <= 100 * drift_pos
        table["deployed, before drift"] = deployed[before].groupby("family")["pr_auc"].mean()
        table["deployed, after drift"] = deployed[deployed["start_pct"] >= 100 * drift_pos].groupby("family")["pr_auc"].mean()
    else:
        table["deployed, all later data"] = deployed.groupby("family")["pr_auc"].mean()
    print("\nPR-AUC\n" + table.round(3).to_string())
    print(f"\nCSV in {out}, figure {figure}")


if __name__ == "__main__":
    main()
