"""Hyperparameter sensitivity on the DEVELOPMENT scenarios only (synthetic, and ULB with drift at 50%).

Held-out scenarios (run_protocol.py) are never used here, so choosing settings from this sweep does not
leak into the held-out evaluation. Each setting is varied alone, others at their defaults:
  --accumulate     batches gathered after an alarm (accumulate-after-drift)
  --retrain-every  periodic retraining interval
  --adwin-delta    ADWIN confidence (smaller = fewer alarms)

    python experiments/run_sensitivity.py --data data/raw/creditcard.csv
Output: results/sensitivity/..., results/sensitivity_summary.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiments.run_experiment import parse_args as experiment_args  # noqa: E402
from experiments.run_experiment import run  # noqa: E402

SWEEPS = {
    "accumulate": (["3", "6", "10"], "accumulate"),
    "retrain-every": (["2", "3", "5"], "periodic"),
    "adwin-delta": (["0.0002", "0.002", "0.02"], "adwin,accumulate"),
}
MODELS = "logreg,random_forest,gradient_boosting"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "creditcard.csv"))
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args()
    root = Path(args.out) / "sensitivity"

    datasets = {"synthetic": ["--batch-size", "2000"],
                "ULB drift 50%": ["--data", args.data, "--batch-size", "10000", "--inject-drift", "0.5"]}
    rows = []
    for param, (values, strategies) in SWEEPS.items():
        for value in values:
            for dname, dargs in datasets.items():
                out = root / f"{param}_{value}" / dname.split()[0].lower()
                print(f"\n=== {param} = {value} on {dname} ===", flush=True)
                exp = experiment_args(dargs + ["--models", MODELS, "--strategies", strategies,
                                               f"--{param}", value, "--out", str(out)])
                _, summary, _ = run(exp, plots=False)
                for _, s in summary.iterrows():
                    rows.append({"parameter": param, "value": value, "dataset": dname, "family": s["family"],
                                 "strategy": s["strategy"], "pr_auc_after": s["mean_pr_auc_after_drift"],
                                 "recovery_batches": s["recovery_batches"], "alarms": s["n_drift_flags"],
                                 "runtime_s": s["runtime_s"]})
    table = pd.DataFrame(rows)
    table.to_csv(Path(args.out) / "sensitivity_summary.csv", index=False)
    print("\nPost-drift PR-AUC by setting (mean over the 3 models)")
    print(table.groupby(["parameter", "strategy", "dataset", "value"])["pr_auc_after"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
