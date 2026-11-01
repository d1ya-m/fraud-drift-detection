"""Run every model x adaptation strategy down the stream, save per-run CSVs, a summary and figures.

    python experiments/run_experiment.py                                   # synthetic data
    python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000
    python experiments/run_experiment.py --models logreg,online_logreg     # a subset
    python experiments/run_experiment.py --data data/raw/creditcard.csv --batch-size 10000 --inject-drift 0.5

Drift scenarios on real data (all need --inject-drift): --swap-k (features swapped, severity),
--pair-seed (random pairs instead of the most fraud-predictive ones), --drift-width (gradual drift).
--label-delay N makes labels reach models and detectors N batches after scoring.
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
from src.evaluation.drift_metrics import drift_response_table, trend_per_10_batches  # noqa: E402
from src.evaluation.prequential import prequential_eval  # noqa: E402
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest, make_rule  # noqa: E402
from src.models.drift_retrain import AccumulateAfterDrift, WarningWindowRetrain  # noqa: E402
from src.models.online import make_arf, make_online_logreg  # noqa: E402
from src.models.retrain import PeriodicRetrain  # noqa: E402
from src.viz.plots import plot_metric_over_time, plot_summary  # noqa: E402

# key -> (panel title, factory, kind). Static models get every strategy; online models already update
# every batch, so they get only detectors; the rule is a trivial baseline, run as-is.
FAMILIES = {
    "rule": ("Single-feature rule (static)", make_rule, "rule"),
    "logreg": ("Logistic regression (static)", make_logreg, "static"),
    "random_forest": ("Random forest (static)", make_random_forest, "static"),
    "gradient_boosting": ("Gradient boosting (static)", make_gradient_boosting, "static"),
    "online_logreg": ("Logistic regression (online)", make_online_logreg, "online"),
    "online_arf": ("Adaptive random forest (online)", make_arf, "online"),
}
STRATEGIES = {
    "base": "base",
    "periodic": "periodic retrain",
    "adwin": "+ ADWIN",
    "ddm": "+ DDM",
    "accumulate": "+ ADWIN + accumulate",
    "ddmwarn": "DDM warning window",
    # supplementary (added after the held-out protocol; see run_supplementary.py). Not in the default run.
    "periodic_reset": "periodic + ADWIN reset",
}
DEFAULT_STRATEGIES = ["base", "periodic", "adwin", "ddm", "accumulate", "ddmwarn"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--models", default=",".join(FAMILIES), help=f"comma-separated subset of {list(FAMILIES)}")
    p.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES), help=f"comma-separated subset of {list(STRATEGIES)}")
    p.add_argument("--smooth", type=int, default=3, help="rolling-mean window for the over-time plots")
    return p.parse_args(argv)


def add_common_args(p):
    """Options shared by every experiment script (data, scenario, protocol settings)."""
    p.add_argument("--data", help="CSV path; omit to use the synthetic drifting stream")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--warmup-batches", type=int, default=6,
                   help="initial training batches; 6 = one periodic-retrain window, so every strategy starts "
                        "from the same amount of data")
    p.add_argument("--retrain-every", type=int, default=3, help="batches between periodic refits")
    p.add_argument("--retrain-window", type=int, default=6, help="batches kept for each periodic refit")
    p.add_argument("--accumulate", type=int, default=6,
                   help="after an ADWIN alarm, keep refitting static models until this many post-drift batches")
    p.add_argument("--adwin-delta", type=float, default=0.002, help="ADWIN confidence parameter")
    p.add_argument("--label-delay", type=int, default=0, help="batches before a batch's labels are available")
    p.add_argument("--precision-target", type=float, default=0.90)
    p.add_argument("--n-samples", type=int, default=50_000, help="synthetic stream length")
    p.add_argument("--synthetic-shift", type=float, default=2.0,
                   help="synthetic only: how far fraud is shifted on its signal features (drift severity)")
    p.add_argument("--inject-drift", type=float, metavar="FRACTION",
                   help="real data only: swap features from this fraction of the stream on")
    p.add_argument("--swap-k", type=int, default=4, help="number of feature pairs swapped (drift severity)")
    p.add_argument("--pair-seed", type=int, help="swap k random pairs (this seed) instead of the strongest features")
    p.add_argument("--drift-width", type=float, default=0.0, help="gradual drift: fraction of the stream it phases in over")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results"))


def build_runs(args):
    """Yield (family_title, strategy, model, detector_or_None) for every requested combination."""
    wanted = args.strategies.split(",")
    for key in args.models.split(","):
        title, factory, kind = FAMILIES[key]

        def make():
            return factory(seed=args.seed)

        if "base" in wanted:
            yield title, STRATEGIES["base"], make(), None
        if kind == "rule":
            continue
        if kind == "static" and "periodic" in wanted:
            yield (title, STRATEGIES["periodic"],
                   PeriodicRetrain(make(), every=args.retrain_every, window=args.retrain_window), None)
        if "adwin" in wanted:
            yield title, STRATEGIES["adwin"], make(), make_adwin(delta=args.adwin_delta)
        if "ddm" in wanted:
            yield title, STRATEGIES["ddm"], make(), make_ddm()
        if kind == "static" and "accumulate" in wanted:
            # this project's variant: ADWIN-triggered, then keep learning from post-drift batches
            yield (title, STRATEGIES["accumulate"],
                   AccumulateAfterDrift(make(), accumulate=args.accumulate), make_adwin(delta=args.adwin_delta))
        if kind == "static" and "periodic_reset" in wanted:
            # supplementary: periodic retraining whose window is reset by ADWIN, then rebuilt batch by batch
            yield (title, STRATEGIES["periodic_reset"],
                   PeriodicRetrain(make(), every=args.retrain_every, window=args.retrain_window,
                                   eager_after_reset=True), make_adwin(delta=args.adwin_delta))
        if kind == "static" and "ddmwarn" in wanted:
            # established scheme (Gama et al. 2004): retrain on the examples since DDM's warning
            yield title, STRATEGIES["ddmwarn"], WarningWindowRetrain(make()), None


def summarise(results, drift_batch):
    groups = results.groupby("model_name", sort=False)
    summary = pd.DataFrame({
        "family": groups["family"].first(),
        "strategy": groups["strategy"].first(),
        "mean_pr_auc": groups["pr_auc"].mean(),
        "mean_recall_at_p": groups["recall_at_p"].mean(),
        "mean_fpr": groups["fpr"].mean(),
        "mean_op_recall": groups["op_recall"].mean(),
        "mean_op_precision": groups["op_precision"].mean(),
        "mean_op_fpr": groups["op_fpr"].mean(),
        "n_drift_flags": groups["drift_flag"].sum(),
        "runtime_s": groups["runtime_s"].first(),
    })
    summary["pr_auc_trend_per_10_batches"] = groups.apply(trend_per_10_batches, include_groups=False)
    if drift_batch is not None:
        after = results[results["batch_idx"] >= drift_batch].groupby("model_name")
        summary["mean_pr_auc_after_drift"] = after["pr_auc"].mean()
        summary["mean_recall_at_p_after_drift"] = after["recall_at_p"].mean()
        summary["mean_op_recall_after_drift"] = after["op_recall"].mean()
        response = drift_response_table(results, drift_batch).set_index("model_name")
        for col in ["before", "drop", "drop_pct", "worst_after", "batches_below", "recovery_batches"]:
            summary[f"pr_auc_{col}" if col in ("before", "worst_after") else col] = response[col]
    return summary.reset_index()


def load_data(args):
    """Return (df, tag, drift_batch). drift_batch is where the known drift starts, or None."""
    delay = getattr(args, "label_delay", 0)
    if args.data:
        df = load_dataset(args.data)
        tag = Path(args.data).stem
        drift_batch = None  # real data: drift is unknown, that's what the detectors are for
        if args.inject_drift is not None:
            k, pair_seed, width = getattr(args, "swap_k", 4), getattr(args, "pair_seed", None), getattr(args, "drift_width", 0.0)
            df, pairs = inject_feature_swap(df, at=args.inject_drift, k=k, pair_seed=pair_seed, width=width)
            tag += f"_drift{round(args.inject_drift * 100)}"
            tag += f"_k{k}" if k != 4 else ""
            tag += f"_rand{pair_seed}" if pair_seed is not None else ""
            tag += f"_grad{round(width * 100)}" if width > 0 else ""
            drift_batch = args.inject_drift * len(df) / args.batch_size
            print("injected drift: swapping " + ", ".join(f"{a}<->{b}" for a, b in pairs)
                  + f" from row {int(args.inject_drift * len(df)):,}" + (f" over {width:.0%} of the stream" if width else ""))
    else:
        shift = getattr(args, "synthetic_shift", 2.0)
        df = load_dataset(n_samples=args.n_samples, seed=args.seed, shift=shift)
        tag = "synthetic" + (f"_shift{shift:g}" if shift != 2.0 else "")
        drift_batch = 0.5 * args.n_samples / args.batch_size  # generator's default drift_at=0.5
    tag += f"_delay{delay}" if delay else ""

    n_batches = -(-len(df) // args.batch_size)
    print(f"{tag}: {len(df):,} rows, fraud rate {df['Class'].mean():.4f}, "
          f"{n_batches} batches of {args.batch_size}, ~{df['Class'].mean() * args.batch_size:.1f} frauds per batch")
    return df, tag, drift_batch


def evaluate(model, detector, df, args, score_sink=None, **labels):
    """Run one model down the stream; `labels` (family, strategy, ...) become extra columns."""
    start = time.time()
    res = prequential_eval(model, make_stream(df, args.batch_size), drift_detector=detector,
                           precision_target=args.precision_target, warmup_batches=args.warmup_batches,
                           label_delay=getattr(args, "label_delay", 0), score_sink=score_sink)
    for column, value in labels.items():
        res[column] = value
    res["runtime_s"] = round(time.time() - start, 1)
    print(f"  {res['model_name'][0]:36s} mean PR-AUC {res['pr_auc'].mean():.3f}   "
          f"drift flags {res['drift_flag'].sum():2d}   {res['runtime_s'][0]:6.1f}s", flush=True)
    return res


def run(args, plots=True):
    """Run all requested combinations; write per-run CSVs and the summary. Returns (results, summary, tag)."""
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

    if plots:
        plot_metric_over_time(results, figures / f"{tag}_pr_auc_over_time.png", metric="pr_auc",
                              drift_batch=drift_batch, smooth=args.smooth)
        plot_metric_over_time(results, figures / f"{tag}_recall_over_time.png", metric="op_recall",
                              drift_batch=drift_batch, smooth=args.smooth)
        headline = "mean_pr_auc_after_drift" if drift_batch is not None else "mean_pr_auc"
        plot_summary(summary, figures / f"{tag}_summary.png", metric=headline)
    return results, summary, tag


def main():
    args = parse_args()
    _, summary, tag = run(args)
    after = "mean_pr_auc_after_drift" in summary.columns
    headline = "mean_pr_auc_after_drift" if after else "mean_pr_auc"
    cols = ["model_name", "mean_pr_auc"] + (["mean_pr_auc_after_drift"] if after else []) + \
           ["mean_op_recall", "mean_op_precision", "n_drift_flags", "runtime_s"]
    print("\n" + summary[cols].sort_values(headline, ascending=False).round(3).to_string(index=False))
    print(f"\nCSVs and figures in {args.out}")


if __name__ == "__main__":
    main()
