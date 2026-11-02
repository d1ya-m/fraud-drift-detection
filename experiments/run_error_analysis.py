"""Error analysis on the development scenario (ULB, drift at 50%), at the OPERATIONAL threshold.

Every transaction is flagged when its score reaches the threshold learnt from the previous labelled batch
(the 90%-precision threshold of that batch), i.e. what a live system would do. For three random-forest
strategies (frozen, + ADWIN + accumulate, + DDM) it reports:
  confusion        TP / FN / FP / TN before and after the drift, alerts per 10,000 transactions, and the
                   share of fraud AMOUNT caught (a cost view: missing a large fraud costs more)
  missed frauds    recall by amount band and by quiet vs busy hours
  alarms           for + DDM: the batches where it fired, their fraud counts and missed-fraud rates

    python experiments/run_error_analysis.py --data data/raw/creditcard.csv --inject-drift 0.5
Output: results/<tag>_errors_confusion.csv, _errors_by_group.csv, _errors_alarms.csv
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.run_experiment import add_common_args, evaluate, load_data  # noqa: E402
from src.drift.detectors import make_adwin, make_ddm  # noqa: E402
from src.models.baselines import make_random_forest  # noqa: E402
from src.models.drift_retrain import AccumulateAfterDrift  # noqa: E402


def live_thresholds(results, label_delay=0):
    """Threshold in force for each batch: the latest finite 90%-precision threshold of a labelled batch."""
    thr = results.set_index("batch_idx")["threshold"].replace(np.inf, np.nan).ffill()
    return thr.shift(1 + label_delay)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()
    out = Path(args.out)
    df, tag, drift_batch = load_data(args)
    if drift_batch is None:
        raise SystemExit("needs a known drift: --inject-drift")

    strategies = {
        "frozen": (make_random_forest(seed=args.seed), None),
        "+ ADWIN + accumulate": (AccumulateAfterDrift(make_random_forest(seed=args.seed), accumulate=args.accumulate),
                                 make_adwin(delta=args.adwin_delta)),
        "+ DDM": (make_random_forest(seed=args.seed), make_ddm()),
    }
    hours = (df["Time"] // 3600).astype(int) % 24
    band = pd.cut(df["Amount"], [-0.01, 1, 100, np.inf], labels=["amount <= 1", "1 < amount <= 100", "amount > 100"])

    confusion, groups, alarms = [], [], []
    for name, (model, detector) in strategies.items():
        sink = []
        res = evaluate(model, detector, df, args, score_sink=sink, family="Random forest", strategy=name)
        scores = pd.concat(sink, ignore_index=True)
        scores["threshold"] = scores["batch_idx"].map(live_thresholds(res, args.label_delay))
        scores = scores.dropna(subset=["threshold"])
        scores["flagged"] = scores["score"] >= scores["threshold"]
        scores["period"] = np.where(scores["batch_idx"] >= math.ceil(drift_batch), "after drift",
                                    np.where(scores["batch_idx"] < math.floor(drift_batch), "before drift", "drift batch"))
        scores["amount"] = df.loc[scores["row"], "Amount"].to_numpy()
        scores["band"] = band.loc[scores["row"]].astype(str).to_numpy()
        scores["hours"] = np.where(hours.loc[scores["row"]].to_numpy() < 7, "quiet hours", "busy hours")

        for period, g in scores[scores["period"] != "drift batch"].groupby("period"):
            fraud, flag = g["y"] == 1, g["flagged"]
            confusion.append({
                "strategy": name, "period": period, "transactions": len(g),
                "TP": int((fraud & flag).sum()), "FN": int((fraud & ~flag).sum()),
                "FP": int((~fraud & flag).sum()), "TN": int((~fraud & ~flag).sum()),
                "recall": (fraud & flag).sum() / fraud.sum(),
                "precision": (fraud & flag).sum() / flag.sum() if flag.sum() else np.nan,
                "alerts_per_10k": 1e4 * flag.mean(),
                "fraud_amount_caught_pct": 100 * g.loc[fraud & flag, "amount"].sum() / g.loc[fraud, "amount"].sum(),
            })
            for col in ["band", "hours"]:
                for value, h in g[fraud].groupby(col):
                    groups.append({"strategy": name, "period": period, "group": value, "frauds": len(h),
                                   "recall": h["flagged"].mean()})

        if name == "+ DDM":
            batch = scores.groupby("batch_idx").apply(
                lambda b: pd.Series({"frauds": int(b["y"].sum()),
                                     "missed_at_0.5": float((b.loc[b["y"] == 1, "score"] < 0.5).mean())}),
                include_groups=False)
            flags = res.set_index("batch_idx")["drift_flag"]
            batch["alarm"] = flags.reindex(batch.index).fillna(0).astype(int)
            batch["before_drift"] = batch.index < math.floor(drift_batch)
            alarms.append(batch.reset_index())

    pd.DataFrame(confusion).to_csv(out / f"{tag}_errors_confusion.csv", index=False)
    pd.DataFrame(groups).to_csv(out / f"{tag}_errors_by_group.csv", index=False)
    pd.concat(alarms).to_csv(out / f"{tag}_errors_alarms.csv", index=False)
    print(pd.DataFrame(confusion).round(3).to_string(index=False))
    print(pd.DataFrame(groups).round(3).to_string(index=False))
    a = pd.concat(alarms)
    print(a[a["alarm"] == 1].to_string(index=False))
    print(f"median frauds per batch {a['frauds'].median():.0f}; median missed share {a['missed_at_0.5'].median():.2f}")


if __name__ == "__main__":
    main()
