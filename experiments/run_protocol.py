"""Frozen evaluation protocol: every strategy, several seeds, development AND held-out drift scenarios.

The accumulate-after-drift method and the ADWIN "ignore falling error" rule were designed while looking at
the development scenario (ULB, drift at 50%, the 4 most fraud-predictive features swapped). The held-out
scenarios below were never used for design or tuning, so they test whether conclusions generalise.

    python experiments/run_protocol.py --data data/raw/creditcard.csv            # all scenarios (~2 h)
    python experiments/run_protocol.py --data data/raw/creditcard.csv --scenarios dev,k2 --seeds 0
    python experiments/run_protocol.py --aggregate-only                          # rebuild the summary

Runs already on disk are skipped, so an interrupted protocol resumes where it stopped.
Output: results/protocol/<scenario>/seed<k>/..., results/protocol_summary.csv,
results/protocol_comparisons.csv, results/figures/protocol_comparisons.png.
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.run_experiment import parse_args as experiment_args  # noqa: E402
from experiments.run_experiment import run  # noqa: E402
from src.evaluation.stats import paired_difference  # noqa: E402
from src.viz.plots import plot_comparison_intervals  # noqa: E402

# name -> (role, extra run_experiment arguments). "synthetic" runs without --data.
SCENARIOS = {
    "dev": ("development", ["--inject-drift", "0.5"]),
    "drift30": ("held-out", ["--inject-drift", "0.3"]),
    "drift70": ("held-out", ["--inject-drift", "0.7"]),
    "k2": ("held-out", ["--inject-drift", "0.5", "--swap-k", "2"]),
    "k6": ("held-out", ["--inject-drift", "0.5", "--swap-k", "6"]),
    "random_pairs": ("held-out", ["--inject-drift", "0.5", "--pair-seed", "1"]),
    "gradual": ("held-out", ["--inject-drift", "0.4", "--drift-width", "0.2"]),
    "delay1": ("label delay", ["--inject-drift", "0.5", "--label-delay", "1"]),
    "delay3": ("label delay", ["--inject-drift", "0.5", "--label-delay", "3"]),
    "synthetic": ("development", []),
}
MODELS = "rule,logreg,random_forest,gradient_boosting,online_logreg"
REFERENCE = "+ ADWIN + accumulate"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "creditcard.csv"))
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def scenario_args(name, seed, data, out_dir):
    _, extra = SCENARIOS[name]
    base = ["--seed", str(seed), "--models", MODELS, "--out", str(out_dir)]
    if name == "synthetic":
        return base + ["--batch-size", "2000"]
    return base + ["--data", data, "--batch-size", "10000"] + extra


def drift_batch_of(name, args):
    if name == "synthetic":
        return 0.5 * args.n_samples / args.batch_size
    return args.inject_drift * 284_807 / args.batch_size


def aggregate(root, scenarios, seeds):
    """Pool every run: per-scenario summary over seeds, and paired comparisons against the reference."""
    summaries, comparisons = [], []
    for name in scenarios:
        role = SCENARIOS[name][0]
        runs = []
        for seed in seeds:
            seed_dir = root / name / f"seed{seed}"
            for path in seed_dir.glob("*_summary.csv"):
                tag = path.name[: -len("_summary.csv")]
                summary = pd.read_csv(path)
                for model_name in summary["model_name"]:
                    r = pd.read_csv(seed_dir / f"{tag}_{model_name}.csv")
                    r["seed"] = seed
                    runs.append(r)
        if not runs:
            continue
        runs = pd.concat(runs, ignore_index=True)
        db = drift_batch_of(name, experiment_args(scenario_args(name, seeds[0], "x.csv", root)))
        after = runs[runs["batch_idx"] >= math.ceil(db)]
        before = runs[runs["batch_idx"] < math.floor(db)]
        # an alarm in the batch that contains the drift point is a true detection, so count it
        detect = runs[runs["batch_idx"] >= math.floor(db)]
        n_seeds = runs["seed"].nunique()
        keys = ["family", "strategy"]
        s = after.groupby(keys, sort=False).agg(
            pr_auc_after=("pr_auc", "mean"), op_recall_after=("op_recall", "mean"),
            op_precision_after=("op_precision", "mean")).reset_index()
        extra = pd.DataFrame({
            "pr_auc_before": before.groupby(keys)["pr_auc"].mean(),
            "false_alarms_per_run": before.groupby(keys)["drift_flag"].sum() / n_seeds,
            "alarms_after_drift_per_run": detect.groupby(keys)["drift_flag"].sum() / n_seeds,
        }).reset_index()
        s = s.merge(extra, on=keys, how="left")
        s.insert(0, "scenario", name)
        s.insert(1, "role", role)
        summaries.append(s)

        # paired comparison: reference strategy vs every other strategy of the same static model
        wide = after.pivot_table(index="batch_idx", columns=["family", "strategy", "seed"], values="pr_auc")
        for family in after.loc[after["strategy"] == REFERENCE, "family"].unique():
            ref = wide[family][REFERENCE]
            for other in after.loc[after["family"] == family, "strategy"].unique():
                if other == REFERENCE:
                    continue
                c = paired_difference(ref, wide[family][other])
                comparisons.append({"scenario": name, "role": role, "family": family, "vs": other, **c})
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(comparisons)


def main():
    args = parse_args()
    root = Path(args.out) / "protocol"
    scenarios = args.scenarios.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    if not args.aggregate_only:
        for name in scenarios:
            for seed in seeds:
                out_dir = root / name / f"seed{seed}"
                if list(out_dir.glob("*_summary.csv")):
                    print(f"[skip] {name} seed {seed}: already done")
                    continue
                print(f"\n=== {name} ({SCENARIOS[name][0]}), seed {seed} ===", flush=True)
                run(experiment_args(scenario_args(name, seed, args.data, out_dir)), plots=False)

    summary, comparisons = aggregate(root, scenarios, seeds)
    summary.to_csv(Path(args.out) / "protocol_summary.csv", index=False)
    comparisons.to_csv(Path(args.out) / "protocol_comparisons.csv", index=False)
    plot_comparison_intervals(comparisons, Path(args.out) / "figures" / "protocol_comparisons.png",
                              reference=REFERENCE)

    print("\nPost-drift PR-AUC, mean over seeds")
    print(summary.pivot_table(index=["scenario", "family"], columns="strategy", values="pr_auc_after",
                              sort=False).round(3).to_string())
    print(f"\n{REFERENCE} minus each alternative (block-bootstrap 95% CI)")
    for _, c in comparisons.iterrows():
        mark = "significant" if c["significant"] else "ns"
        print(f"  {c['scenario']:13s} {c['family']:30s} vs {c['vs']:22s} {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {mark}")


if __name__ == "__main__":
    main()
