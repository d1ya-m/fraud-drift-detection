"""What the drift signal can and cannot see, and whether a richer or label-free signal detects drift sooner.

A diagnostic on FROZEN models (trained on the 6 warm-up batches, then never updated; seed 42), which is
exactly the state a detector-triggered strategy is in while it waits for an alarm. Outputs:

  detector_blindspot.csv  fraud miss rate and share of legitimate transactions flagged at 0.5, before/after drift
  detector_signals.csv    for four drift signals, each watched by ADWIN (delta 0.002, reset after every alarm):
      missed frauds        one 0/1 per fraud (missed at 0.5). The signal used in all main experiments. Needs labels.
      fraud score drop     one value per fraud: 1 - score. Same frauds, but keeps HOW confident the model was.
                           Needs labels.
      alert rate           one 0/1 per transaction (flagged at 0.5), all transactions. Needs NO labels.
      mean score           the model's score of every transaction. Needs NO labels.
    The two label-based signals use the main experiments' rule (ignore alarms where the signal fell, i.e. the
    model got better); the label-free ones count a change in either direction, since either can mean drift.
    Reported per scenario: first alarm at or after the drift batch, and false alarms before it (or anywhere,
    on the undrifted data).

PRELIMINARY: no strategy uses the new signals; this only measures detection on frozen models.

    python experiments/run_detector_blindspot.py --data data/raw/creditcard.csv      (~5 minutes)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from river import drift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_experiment import parse_args as experiment_args  # noqa: E402
from experiments.run_experiment import load_data  # noqa: E402
from src.data.loader import get_feature_columns  # noqa: E402
from src.drift.detectors import make_adwin  # noqa: E402
from src.models.baselines import make_gradient_boosting, make_logreg, make_random_forest  # noqa: E402

MAKERS = {"logreg": make_logreg, "random_forest": make_random_forest, "gradient_boosting": make_gradient_boosting}
SCENARIOS = {  # name -> extra run_experiment arguments
    "as recorded": [],
    "k2": ["--inject-drift", "0.5", "--swap-k", "2"],
    "k4 (dev)": ["--inject-drift", "0.5"],
    "k6": ["--inject-drift", "0.5", "--swap-k", "6"],
    "drift30": ["--inject-drift", "0.3"],
    "drift70": ["--inject-drift", "0.7"],
    "gradual": ["--inject-drift", "0.4", "--drift-width", "0.2"],
}
SIGNALS = {  # name -> (needs labels, per-batch values from (scores, labels), rising-only rule)
    "missed frauds": (True, lambda s, y: (s[y == 1] < 0.5).astype(float), True),
    "fraud score drop": (True, lambda s, y: 1.0 - s[y == 1], True),
    "alert rate": (False, lambda s, y: (s >= 0.5).astype(float), False),
    "mean score": (False, lambda s, y: s, False),
}


class AnyChangeADWIN:
    """ADWIN that alarms on a change in either direction and restarts after each alarm."""

    def __init__(self, delta=0.002):
        self.delta = delta
        self.detector = drift.ADWIN(delta=delta, clock=1)

    def update(self, values):
        for v in values:
            self.detector.update(float(v))
            if self.detector.drift_detected:
                self.detector = drift.ADWIN(delta=self.delta, clock=1)
                return True
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()

    rates, signals = [], []
    for scenario, extra in SCENARIOS.items():
        a = experiment_args(["--data", args.data, "--batch-size", "10000"] + extra)
        df, _, drift_batch = load_data(a)
        X, y = df[get_feature_columns(df)], df["Class"].to_numpy()  # same features as the stream (no Time)
        bs, warm = a.batch_size, a.warmup_batches
        n_batches = math.ceil(len(df) / bs)
        db = None if drift_batch is None else math.floor(drift_batch)
        for name, make in MAKERS.items():
            model = make(seed=a.seed)
            model.fit(X.iloc[:warm * bs], y[:warm * bs])
            batches = []
            for b in range(warm, n_batches):
                sl = slice(b * bs, (b + 1) * bs)
                batches.append((b, np.asarray(model.predict_proba(X.iloc[sl])), y[sl]))
            if db is not None:
                for period, keep in [("before drift", lambda b: b < db),
                                     ("after drift", lambda b: b >= math.ceil(drift_batch))]:
                    sel = [(s, yy) for b, s, yy in batches if keep(b)]
                    s_all, y_all = np.concatenate([s for s, _ in sel]), np.concatenate([yy for _, yy in sel])
                    rates.append({"scenario": scenario, "model": name, "period": period,
                                  "fraud_miss_rate": float(np.mean(s_all[y_all == 1] < 0.5)),
                                  "legit_flagged_rate": float(np.mean(s_all[y_all == 0] >= 0.5))})
            for signal, (needs_labels, values, rising_only) in SIGNALS.items():
                det = make_adwin() if rising_only else AnyChangeADWIN()
                alarms = [b for b, s, yy in batches if det.update(values(s, yy))]
                signals.append({
                    "scenario": scenario, "model": name, "signal": signal, "needs_labels": needs_labels,
                    "drift_batch": db,
                    "false_alarms": len(alarms) if db is None else sum(b < db for b in alarms),
                    "first_alarm_after_drift": None if db is None else next((b for b in alarms if b >= db), None),
                    "delay_batches": None if db is None else next((b - db for b in alarms if b >= db), None)})
    out = Path(args.out)
    pd.DataFrame(rates).to_csv(out / "detector_blindspot.csv", index=False)
    pd.DataFrame(signals).to_csv(out / "detector_signals.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(signals).pivot_table(index=["model", "scenario"], columns="signal",
                                            values=["delay_batches", "false_alarms"], sort=False, dropna=False).to_string())


if __name__ == "__main__":
    main()
