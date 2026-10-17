import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.plots import plot_metric_over_time, plot_summary

ROOT = Path(__file__).resolve().parents[1]


def fake_results():
    """Contract 4 lets the plots be tested without running any model."""
    rng = np.random.default_rng(0)
    rows = []
    for family in ["Logistic regression (static)", "Logistic regression (online)"]:
        for strategy in ["base", "+ ADWIN"]:
            for b in range(1, 11):
                rows.append({"family": family, "strategy": strategy, "model_name": f"{family}{strategy}",
                             "batch_idx": b, "pr_auc": rng.random(), "recall_at_p": rng.random(),
                             "drift_flag": int(strategy == "+ ADWIN" and b == 6)})
    return pd.DataFrame(rows)


def test_plots_write_png_files(tmp_path):
    res = fake_results()
    over_time = plot_metric_over_time(res, tmp_path / "over_time.png", drift_batch=5.5)
    summary = res.groupby("model_name", as_index=False)["pr_auc"].mean().rename(columns={"pr_auc": "mean_pr_auc"})
    bars = plot_summary(summary, tmp_path / "summary.png")
    for path in (over_time, bars):
        assert path.exists() and path.stat().st_size > 10_000


def test_run_experiment_end_to_end(tmp_path):
    cmd = [sys.executable, str(ROOT / "experiments" / "run_experiment.py"),
           "--n-samples", "12000", "--batch-size", "2000",
           "--models", "logreg,online_logreg", "--detectors", "adwin", "--out", str(tmp_path)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr

    # logreg: base, periodic retrain, +adwin, +adwin+accumulate; online_logreg: base, +adwin
    runs = sorted(p.name for p in tmp_path.glob("synthetic_*.csv") if p.name != "synthetic_summary.csv")
    assert len(runs) == 6
    summary = pd.read_csv(tmp_path / "synthetic_summary.csv")
    assert len(summary) == 6 and "mean_pr_auc_after_drift" in summary.columns
    for name in ["pr_auc_over_time", "recall_over_time", "summary"]:
        assert (tmp_path / "figures" / f"synthetic_{name}.png").exists()
