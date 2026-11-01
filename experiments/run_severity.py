"""Drift severity vs performance loss (the synopsis problem statement's open question).

Severity is varied on two axes, everything else at the frozen defaults:
  ULB        number of swapped feature pairs k = 1, 2, 4, 6, 8 (the k most fraud-predictive features
             swapped with the k least predictive), drift at 50%
  synthetic  how far fraud is shifted on its signal features: 0.5 ... 3.0 (2.0 = the default stream)
For each: the frozen model's post-drift loss, ADWIN's detection delay, and accumulate-after-drift's
post-drift PR-AUC and recovery time.

    python experiments/run_severity.py --data data/raw/creditcard.csv
Output: results/severity/..., results/severity_summary.csv, results/figures/severity.png
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
from src.viz.plots import plot_severity  # noqa: E402

MODELS = "logreg,random_forest"
STRATEGIES = "base,adwin,accumulate"


def first_alarm_delay(runs, model_name, drift_batch):
    r = runs[(runs["model_name"] == model_name) & (runs["batch_idx"] >= math.floor(drift_batch))]
    alarms = r.loc[r["drift_flag"] == 1, "batch_idx"]
    return float(alarms.iloc[0] - math.floor(drift_batch)) if len(alarms) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "creditcard.csv"))
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()
    root = Path(args.out) / "severity"

    settings = [("ULB", "swapped pairs k", k, ["--data", args.data, "--batch-size", "10000",
                                              "--inject-drift", "0.5", "--swap-k", str(k)]) for k in (1, 2, 4, 6, 8)]
    settings += [("synthetic", "fraud shift", s, ["--batch-size", "2000", "--synthetic-shift", str(s)])
                 for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
    rows = []
    for dataset, axis, level, dargs in settings:
        print(f"\n=== {dataset}: {axis} = {level} ===", flush=True)
        exp = experiment_args(dargs + ["--models", MODELS, "--strategies", STRATEGIES,
                                       "--out", str(root / f"{dataset}_{level}")])
        runs, summary, _ = run(exp, plots=False)
        drift_batch = (0.5 * 284_807 / 10_000) if dataset == "ULB" else 0.5 * exp.n_samples / exp.batch_size
        for family, group in summary.groupby("family", sort=False):
            s = group.set_index("strategy")
            adwin_name = s.loc["+ ADWIN", "model_name"]
            adwin = runs[runs["model_name"] == adwin_name]
            rows.append({
                "dataset": dataset, "axis": axis, "level": level, "model": family.replace(" (static)", ""),
                "pr_auc_before": s.loc["base", "pr_auc_before"],
                "frozen_pr_auc_after": s.loc["base", "mean_pr_auc_after_drift"],
                "frozen_drop_pct": s.loc["base", "drop_pct"],
                "adwin_delay_batches": first_alarm_delay(runs, adwin_name, drift_batch),
                "adwin_false_alarms": int(adwin.loc[adwin["batch_idx"] < math.floor(drift_batch), "drift_flag"].sum()),
                "accumulate_pr_auc_after": s.loc["+ ADWIN + accumulate", "mean_pr_auc_after_drift"],
                "accumulate_recovery_batches": s.loc["+ ADWIN + accumulate", "recovery_batches"],
            })
    table = pd.DataFrame(rows)
    table.to_csv(Path(args.out) / "severity_summary.csv", index=False)
    plot_severity(table, Path(args.out) / "figures" / "severity.png")
    print("\n" + table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
