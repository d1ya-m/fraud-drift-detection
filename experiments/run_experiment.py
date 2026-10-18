"""Run every model x adaptation strategy down the stream, save per-run CSVs, a summary and figures.

    python experiments/run_experiment.py                                   # synthetic data
    python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000
    python experiments/run_experiment.py --models logreg,online_logreg     # a subset
    python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.drift_injection import inject_feature_swap  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.stream import make_stream  # noqa: E402
from src.drift.detectors import make_adwin, make_ddm  # noqa: E402
from src.evaluation.prequential import prequential_eval  # noqa: E402
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest  # noqa: E402
from src.models.drift_retrain import AccumulateAfterDrift  # noqa: E402
from src.models.online import make_arf, make_online_logreg  # noqa: E402
from src.models.retrain import PeriodicRetrain  # noqa: E402
from src.viz.plots import plot_metric_over_time, plot_summary  # noqa: E402

# key -> (panel title, factory, kind). Static models get periodic retraining as an extra
# strategy; online models already update every batch, so they don't.
FAMILIES = {
    "logreg": ("Logistic regression (static)", make_logreg, "static"),
    "random_forest": ("Random forest (static)", make_random_forest, "static"),
    "gradient_boosting": ("Gradient boosting (static)", make_gradient_boosting, "static"),
    "online_logreg": ("Logistic regression (online)", make_online_logreg, "online"),
    "online_arf": ("Adaptive random forest (online)", make_arf, "online"),
}
DETECTORS = {"adwin": ("+ ADWIN", make_adwin), "ddm": ("+ DDM", make_ddm)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="CSV path; omit to use the synthetic drifting stream")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--models", default=",".join(FAMILIES), help=f"comma-separated subset of {list(FAMILIES)}")
    p.add_argument("--detectors", default=",".join(DETECTORS), help="comma-separated subset of adwin,ddm, or none")
    p.add_argument("--retrain-every", type=int, default=3, help="batches between periodic refits")
    p.add_argument("--retrain-window", type=int, default=6, help="batches kept for each periodic refit")
    p.add_argument("--accumulate", type=int, default=6,
                   help="after an ADWIN alarm, keep refitting static models until this many post-drift batches")
    p.add_argument("--precision-target", type=float, default=0.90)
    p.add_argument("--smooth", type=int, default=3, help="rolling-mean window for the over-time plots")
    p.add_argument("--n-samples", type=int, default=50_000, help="synthetic stream length")
    p.add_argument("--inject-drift", type=float, metavar="FRACTION",
                   help="real data only: swap the most fraud-predictive features from this fraction of the stream on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results"))
    return p.parse_args()


def build_runs(args):
    """Yield (family_title, strategy, model, detector_or_None) for every requested combination."""
    detectors = [] if args.detectors == "none" else args.detectors.split(",")
    for key in args.models.split(","):
        title, factory, kind = FAMILIES[key]
        yield title, "base", factory(seed=args.seed), None
        if kind == "static":
            yield (title, "periodic retrain",
                   PeriodicRetrain(factory(seed=args.seed), every=args.retrain_every, window=args.retrain_window),
                   None)
        for det in detectors:
            label, make_detector = DETECTORS[det]
            yield title, label, factory(seed=args.seed), make_detector()
        # improved method: ADWIN-triggered, then keep learning from post-drift batches
        if kind == "static" and "adwin" in detectors:
            yield (title, "+ ADWIN + accumulate",
                   AccumulateAfterDrift(factory(seed=args.seed), accumulate=args.accumulate), make_adwin())


def summarise(results, drift_batch):
    groups = results.groupby("model_name", sort=False)
    summary = pd.DataFrame({
        "family": groups["family"].first(),
        "strategy": groups["strategy"].first(),
        "mean_pr_auc": groups["pr_auc"].mean(),
        "mean_recall_at_p": groups["recall_at_p"].mean(),
        "mean_fpr": groups["fpr"].mean(),
        "n_drift_flags": groups["drift_flag"].sum(),
        "runtime_s": groups["runtime_s"].first(),
    })
    if drift_batch is not None:
        after = results[results["batch_idx"] >= drift_batch]
        summary["mean_pr_auc_after_drift"] = after.groupby("model_name")["pr_auc"].mean()
    return summary.reset_index()


def load_data(args):
    """Return (df, tag, drift_batch). drift_batch is where the known drift falls, or None."""
    if args.data:
        df = load_dataset(args.data)
        tag = Path(args.data).stem
        drift_batch = None  # real data: drift is unknown, that's what the detectors are for
        if args.inject_drift is not None:
            df, pairs = inject_feature_swap(df, at=args.inject_drift)
            tag += f"_drift{round(args.inject_drift * 100)}"
            drift_batch = args.inject_drift * len(df) / args.batch_size
            print("injected drift: swapping " + ", ".join(f"{a}<->{b}" for a, b in pairs)
                  + f" from row {int(args.inject_drift * len(df)):,}")
    else:
        df = load_dataset(n_samples=args.n_samples, seed=args.seed)
        tag = "synthetic"
        drift_batch = 0.5 * args.n_samples / args.batch_size  # generator's default drift_at=0.5

    n_batches = -(-len(df) // args.batch_size)
    print(f"{tag}: {len(df):,} rows, fraud rate {df['Class'].mean():.4f}, "
          f"{n_batches} batches of {args.batch_size}, ~{df['Class'].mean() * args.batch_size:.1f} frauds per batch")
    return df, tag, drift_batch


def evaluate(model, detector, df, args, **labels):
    """Run one model down the stream; `labels` (family, strategy, ...) become extra columns."""
    start = time.time()
    res = prequential_eval(model, make_stream(df, args.batch_size), drift_detector=detector,
                           precision_target=args.precision_target, warmup_batches=args.warmup_batches)
    for column, value in labels.items():
        res[column] = value
    res["runtime_s"] = round(time.time() - start, 1)
    print(f"  {res['model_name'][0]:36s} mean PR-AUC {res['pr_auc'].mean():.3f}   "
          f"drift flags {res['drift_flag'].sum():2d}   {res['runtime_s'][0]:6.1f}s", flush=True)
    return res


def main():
    args = parse_args()
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    df, tag, drift_batch = load_data(args)

    all_results = []
    for family, strategy, model, detector in build_runs(args):
        res = evaluate(model, detector, df, args, family=family, strategy=strategy)
        res.to_csv(out / f"{tag}_{res['model_name'][0]}.csv", index=False)
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    summary = summarise(results, drift_batch)
    summary.to_csv(out / f"{tag}_summary.csv", index=False)

    plot_metric_over_time(results, figures / f"{tag}_pr_auc_over_time.png", metric="pr_auc",
                          drift_batch=drift_batch, smooth=args.smooth)
    plot_metric_over_time(results, figures / f"{tag}_recall_over_time.png", metric="recall_at_p",
                          drift_batch=drift_batch, smooth=args.smooth)
    headline = "mean_pr_auc_after_drift" if drift_batch is not None else "mean_pr_auc"
    plot_summary(summary, figures / f"{tag}_summary.png", metric=headline)

    cols = ["model_name", "mean_pr_auc"] + (["mean_pr_auc_after_drift"] if drift_batch is not None else []) + \
           ["mean_recall_at_p", "n_drift_flags", "runtime_s"]
    print("\n" + summary[cols].sort_values(headline, ascending=False).round(3).to_string(index=False))
    print(f"\nCSVs in {out}, figures in {figures}")


if __name__ == "__main__":
    main()
