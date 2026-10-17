"""Figures for the report. Plots only read results DataFrames (the CSVs in results/), never models.

Colour follows the adaptation strategy, never a line's position, so a strategy has the same
colour in every panel and every run.
"""
import math

import matplotlib

matplotlib.use("Agg")  # write files only, never open a window
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed categorical order (colour-blind-safe for adjacent lines)
STRATEGY_COLORS = {
    "base": "#2a78d6",
    "periodic retrain": "#eb6834",
    "+ ADWIN": "#1baf7a",
    "+ DDM": "#eda100",
    "+ ADWIN + accumulate": "#e87ba4",
}
SINGLE_SERIES = STRATEGY_COLORS["base"]
# DDM is drawn first and wider so that where it matches ADWIN exactly (same alarm batch) the
# ADWIN line stays visible on top of it instead of being hidden
LINE_WIDTHS = {"+ DDM": 3.4}
DRAW_ORDER = ["+ DDM", "+ ADWIN", "periodic retrain", "+ ADWIN + accumulate", "base"]

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall_at_p": "Recall at 90% precision",
    "fpr": "False-positive rate",
    "mean_pr_auc": "Mean PR-AUC across batches",
    "mean_pr_auc_after_drift": "Mean PR-AUC after the drift",
    "mean_recall_at_p": "Mean recall at 90% precision",
}


def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
    })


def plot_metric_over_time(results, out_path, metric="pr_auc", drift_batch=None, smooth=3, title=None):
    """One panel per base model ("family"), one line per strategy; dots mark batches where a detector fired.

    results needs columns family, strategy, batch_idx, drift_flag and the metric.
    smooth is a rolling-mean window in batches (1 = raw), since per-batch PR-AUC is noisy.
    """
    _style()
    families = list(dict.fromkeys(results["family"]))
    ncols = min(3, len(families))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows + 0.8),
                             sharex=True, sharey=True, squeeze=False)

    for ax, family in zip(axes.flat, families):
        fam = results[results["family"] == family]
        for z, strategy in enumerate(DRAW_ORDER):
            run = fam[fam["strategy"] == strategy].sort_values("batch_idx")
            if run.empty:
                continue
            color = STRATEGY_COLORS[strategy]
            y = run[metric].rolling(smooth, min_periods=1).mean()
            ax.plot(run["batch_idx"], y, color=color, linewidth=LINE_WIDTHS.get(strategy, 1.6),
                    solid_capstyle="round", zorder=2 + z)
            fired = run["drift_flag"] == 1
            ax.plot(run.loc[fired, "batch_idx"], y[fired], "o", color=color, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=10 + z)
        # a detector that never fires leaves its line hidden under "base", so say so
        silent = [s.lstrip("+ ") for s in ("+ ADWIN", "+ DDM")
                  if (fam["strategy"] == s).any() and fam.loc[fam["strategy"] == s, "drift_flag"].sum() == 0]
        if silent:
            ax.text(0.99, 0.03, f"no alarms from {' or '.join(silent)} (lines = base)", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        if drift_batch is not None:
            ax.axvline(drift_batch, color=MUTED, linewidth=1)
            ax.annotate("true drift", (drift_batch, 1.0), xytext=(4, -2), textcoords="offset points",
                        color=INK_SECONDARY, fontsize=8.5, va="top")
        ax.set_title(family, loc="left")
        ax.set_ylim(0, 1.02)

    for ax in axes.flat[len(families):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Batch (time order)")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    handles = [plt.Line2D([], [], color=c, linewidth=LINE_WIDTHS.get(s, 1.6)) for s, c in STRATEGY_COLORS.items()]
    handles.append(plt.Line2D([], [], marker="o", color=MUTED, markeredgecolor=SURFACE, linestyle="none", markersize=6))
    labels = list(STRATEGY_COLORS) + ["drift detected"]
    top = 1 - 0.6 / fig.get_figheight()  # leave a fixed band for title + legend
    smooth_note = f" ({smooth}-batch rolling mean)" if smooth > 1 else ""
    fig.suptitle(title or f"{METRIC_LABELS.get(metric, metric)} over time{smooth_note}",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.005, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_summary(summary, out_path, metric="mean_pr_auc", title=None):
    """Horizontal bars, one per model run, sorted best first, value printed at each bar end."""
    _style()
    data = summary.dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(data) + 1.2))
    ax.set_axisbelow(True)
    bars = ax.barh(data["model_name"], data[metric], color=SINGLE_SERIES, height=0.7)
    for bar, value in zip(bars, data[metric]):
        ax.annotate(f"{value:.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or METRIC_LABELS.get(metric, metric), loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
