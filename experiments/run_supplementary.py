"""SUPPLEMENTARY experiment, added after the held-out protocol: periodic retraining with an ADWIN reset.

Why: the protocol showed accumulate-after-drift never beats periodic retraining on held-out scenarios. A
per-batch analysis located the whole gap in the batches between the drift and ADWIN's alarm (accumulate is
frozen while the detector is still collecting evidence; periodic keeps adapting on schedule). Once the alarm
fires, accumulate is ahead (+0.06 PR-AUC while it accumulates) and level afterwards.

Candidate fix ("periodic + ADWIN reset"): retrain on schedule as periodic does, and on an ADWIN alarm drop the
old window and refit on every new batch until the window is full of post-drift data again.

Honest design:
  part B "fresh"     4 NEW drift scenarios, fixed before any run and never used for anything else.
                     periodic vs accumulate vs reset, 3 seeds. This is the only independent test of the fix.
  part A "existing"  the fix on the 10 protocol scenarios (seed 0), paired with the existing runs. The fix was
                     designed from these scenarios, so this is development evidence, NOT validation.
Original results are not touched. Output: results/supplementary/...

    python experiments/run_supplementary.py --data data/raw/creditcard.csv            # B, then A (~1.5 h)
    python experiments/run_supplementary.py --parts fresh --aggregate-only
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
from experiments.run_protocol import SCENARIOS as PROTOCOL_SCENARIOS  # noqa: E402
from src.evaluation.stats import paired_difference  # noqa: E402

FRESH = {  # fixed before any run; never inspected during development
    "drift40": ["--inject-drift", "0.4"],
    "drift60_k3": ["--inject-drift", "0.6", "--swap-k", "3"],
    "rand2_at35": ["--inject-drift", "0.35", "--pair-seed", "2"],
    "gradual_late": ["--inject-drift", "0.55", "--drift-width", "0.15"],
}
MODELS = "logreg,random_forest,gradient_boosting"
RESET = "periodic + ADWIN reset"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "raw" / "creditcard.csv"))
    p.add_argument("--parts", default="gap,fresh,existing")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def scenario_args(extra, seed, data, out_dir, strategies):
    base = ["--seed", str(seed), "--models", MODELS, "--strategies", strategies, "--out", str(out_dir)]
    if extra is None:  # synthetic
        return base + ["--batch-size", "2000"]
    return base + ["--data", data, "--batch-size", "10000"] + extra


def drift_batch(extra, args):
    if extra is None:
        return 0.5 * args.n_samples / args.batch_size
    return args.inject_drift * 284_807 / args.batch_size


def load_runs(folder, seed):
    runs = []
    for path in folder.glob("*_summary.csv"):
        tag = path.name[: -len("_summary.csv")]
        for name in pd.read_csv(path)["model_name"]:
            r = pd.read_csv(folder / f"{tag}_{name}.csv")
            r["seed"] = seed
            runs.append(r)
    return runs


def compare(name, role, runs, db, pairs):
    """Post-drift PR-AUC per strategy and paired CIs for each (a, b) in pairs."""
    runs = pd.concat(runs, ignore_index=True)
    after = runs[runs["batch_idx"] >= math.ceil(db)]
    detect = runs[runs["batch_idx"] >= math.floor(db)]
    keys = ["family", "strategy"]
    s = after.groupby(keys).agg(pr_auc_after=("pr_auc", "mean"), op_recall_after=("op_recall", "mean"),
                                op_precision_after=("op_precision", "mean")).reset_index()
    alarms = detect.groupby(keys)["drift_flag"].sum() / runs["seed"].nunique()
    s = s.merge(alarms.rename("alarms_after_drift_per_run").reset_index(), on=keys)
    s.insert(0, "scenario", name)
    s.insert(1, "role", role)
    wide = after.pivot_table(index="batch_idx", columns=["family", "strategy", "seed"], values="pr_auc")
    comps = []
    for family in after["family"].unique():
        for a, b in pairs:
            if a in wide[family] and b in wide[family]:
                c = paired_difference(wide[family][a], wide[family][b])
                comps.append({"scenario": name, "role": role, "family": family, "a": a, "b": b, **c})
    return s, comps


def gap_by_phase(protocol_root, scenarios, data):
    """Accumulate minus periodic, per post-drift batch, split by phase relative to accumulate's ADWIN alarm.

    Reads the original protocol runs only. Phases: 'waiting for alarm' (drift happened, refit not yet done),
    'accumulating' (refit on every batch), 'frozen after' (6 batches gathered), 'no alarm' (ADWIN never fired).
    """
    import numpy as np
    y = pd.read_csv(data, usecols=["Class"])["Class"]
    frauds = y.groupby(y.index // 10_000).sum()
    rows = []
    for name in scenarios:
        role, extra = PROTOCOL_SCENARIOS[name]
        if name == "synthetic":
            continue
        a = experiment_args(scenario_args(extra, 0, data, protocol_root, "periodic"))
        db = drift_batch(extra, a)
        for seed_dir in sorted((protocol_root / name).glob("seed*")):
            for fam in ("logreg", "random_forest", "gradient_boosting"):
                per = pd.read_csv(next(seed_dir.glob(f"*_{fam}_retrain3.csv")))
                acc = pd.read_csv(next(seed_dir.glob(f"*_{fam}_acc6+adwin.csv")))
                alarms = acc.loc[(acc["drift_flag"] == 1) & (acc["batch_idx"] >= math.floor(db)), "batch_idx"]
                alarm = alarms.min() if len(alarms) else np.nan
                m = per[["batch_idx", "pr_auc"]].merge(acc[["batch_idx", "pr_auc"]], on="batch_idx",
                                                       suffixes=("_periodic", "_accumulate"))
                m = m[m["batch_idx"] >= math.ceil(db)].copy()
                if np.isnan(alarm):
                    m["phase"] = "no alarm"
                else:  # labels of the alarm batch trigger the refit, so the next batch is the first one helped
                    m["phase"] = np.select([m["batch_idx"] <= alarm, m["batch_idx"] <= alarm + 5],
                                           ["waiting for alarm", "accumulating"], "frozen after")
                m["scenario"], m["role"], m["seed"], m["family"] = name, role, seed_dir.name, fam
                m["detection_delay"] = alarm - math.floor(db)
                m["frauds_next_3_batches"] = int(frauds.loc[math.floor(db):math.floor(db) + 2].sum())
                rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out["diff"] = out["pr_auc_accumulate"] - out["pr_auc_periodic"]
    return out


def main():
    args = parse_args()
    root = Path(args.out) / "supplementary"
    parts = args.parts.split(",")
    summaries, comparisons = [], []

    if "gap" in parts:
        gap = gap_by_phase(Path(args.out) / "protocol", [s for s in PROTOCOL_SCENARIOS if s != "synthetic"], args.data)
        gap.to_csv(Path(args.out) / "supplementary_gap_by_phase.csv", index=False)
        print(gap.groupby("phase").agg(batches=("diff", "size"), mean_diff=("diff", "mean")).round(3))
        if parts == ["gap"]:
            return

    if "fresh" in parts:
        seeds = [0, 1, 2]
        for name, extra in FRESH.items():
            runs = []
            for seed in seeds:
                out_dir = root / "fresh" / name / f"seed{seed}"
                if not args.aggregate_only and not list(out_dir.glob("*_summary.csv")):
                    print(f"\n=== fresh: {name}, seed {seed} ===", flush=True)
                    run(experiment_args(scenario_args(extra, seed, args.data, out_dir, "periodic,accumulate,periodic_reset")),
                        plots=False)
                runs += load_runs(out_dir, seed)
            if runs:
                db = drift_batch(extra, experiment_args(scenario_args(extra, 0, "x.csv", root, "periodic")))
                s, c = compare(name, "fresh (independent)", runs, db,
                               [(RESET, "periodic retrain"), (RESET, "+ ADWIN + accumulate"),
                                ("+ ADWIN + accumulate", "periodic retrain")])
                summaries.append(s)
                comparisons += c

    if "existing" in parts:
        seed = 0
        for name, (role, extra) in PROTOCOL_SCENARIOS.items():
            extra = None if name == "synthetic" else extra
            out_dir = root / "existing" / name / f"seed{seed}"
            if not args.aggregate_only and not list(out_dir.glob("*_summary.csv")):
                print(f"\n=== existing: {name}, seed {seed} ===", flush=True)
                run(experiment_args(scenario_args(extra, seed, args.data, out_dir, "periodic_reset")), plots=False)
            runs = load_runs(out_dir, seed)
            # pair with the original protocol runs of the same scenario and seed (read-only)
            runs += [r[r["strategy"].isin(["periodic retrain", "+ ADWIN + accumulate"])
                       & r["family"].str.contains("Logistic|Random forest|Gradient")]
                     for r in load_runs(Path(args.out) / "protocol" / name / f"seed{seed}", seed)]
            if runs:
                db = drift_batch(extra, experiment_args(scenario_args(extra, 0, "x.csv", root, "periodic")))
                s, c = compare(name, f"existing {role} (development for the fix)", runs, db,
                               [(RESET, "periodic retrain"), (RESET, "+ ADWIN + accumulate")])
                summaries.append(s)
                comparisons += c

    summary, comps = pd.concat(summaries, ignore_index=True), pd.DataFrame(comparisons)
    summary.to_csv(Path(args.out) / "supplementary_summary.csv", index=False)
    comps.to_csv(Path(args.out) / "supplementary_comparisons.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.pivot_table(index=["role", "scenario", "family"], columns="strategy", values="pr_auc_after",
                              sort=False).round(3).to_string())
    for _, c in comps.iterrows():
        mark = "significant" if c["significant"] else "ns"
        print(f"  {c['scenario']:13s} {c['family']:30s} {c['a']:22s} - {c['b']:22s} "
              f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] {mark}")


if __name__ == "__main__":
    main()
