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
    "DDM warning window": "#008300",
}
IMBALANCE_COLORS = {
    "no handling": "#2a78d6",
    "class weights": "#eb6834",
    "SMOTE": "#1baf7a",
    "undersampling": "#eda100",
}
SINGLE_SERIES = STRATEGY_COLORS["base"]
# DDM is drawn first and wider so that where it matches ADWIN exactly (same alarm batch) the
# ADWIN line stays visible on top of it instead of being hidden
LINE_WIDTHS = {"+ DDM": 3.4}
DRAW_ORDER = ["+ DDM", "+ ADWIN", "DDM warning window", "periodic retrain", "+ ADWIN + accumulate", "base"]

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "recall_at_p": "Recall at 90% precision (oracle threshold)",
    "op_recall": "Recall at the previous batch's 90%-precision threshold",
    "fpr": "False-positive rate",
    "mean_pr_auc": "Mean PR-AUC across batches",
    "mean_pr_auc_after_drift": "Mean PR-AUC after the drift",
    "mean_recall_at_p": "Mean recall at 90% precision (oracle threshold)",
    "mean_op_recall": "Mean operational recall",
}


def apply_style():
    """Apply the project chart style (colours, fonts, grid) to matplotlib."""
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
    apply_style()
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
        detectors = {"+ ADWIN": "ADWIN", "+ DDM": "DDM", "DDM warning window": "DDM warning window"}
        silent = [label for s, label in detectors.items()
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


def plot_grouped_bars(summary, out_path, metrics, group="family", color_by="imbalance", row_by="strategy",
                      colors=IMBALANCE_COLORS, title=None):
    """Grid of grouped horizontal bar charts: one row per `row_by` value, one column per metric;
    inside each panel, one group of bars per `group` value, one coloured bar per `color_by` level."""
    apply_style()
    rows = list(dict.fromkeys(summary[row_by]))
    groups = list(dict.fromkeys(summary[group]))
    levels = [c for c in colors if c in set(summary[color_by])]
    slot = 0.8 / len(levels)
    panel_h = 0.24 * len(groups) * len(levels) + 0.7
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=(5.4 * len(metrics), panel_h * len(rows) + 0.9),
                             squeeze=False, sharey=True)

    for i, row in enumerate(rows):
        sub = summary[summary[row_by] == row]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            ax.set_axisbelow(True)
            for g_i, g in enumerate(groups):
                for l_i, level in enumerate(levels):
                    value = sub.loc[(sub[group] == g) & (sub[color_by] == level), metric]
                    if value.empty or value.isna().all():
                        continue
                    v = float(value.iloc[0])
                    y = g_i + (l_i - (len(levels) - 1) / 2) * slot
                    ax.barh(y, v, height=slot * 0.85, color=colors[level])
                    ax.annotate(f"{v:.2f}", (v, y), xytext=(3, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK_SECONDARY)
            ax.set_yticks(range(len(groups)), groups)
            ax.set_xlim(0, 1.1)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", visible=True)
            ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
            ax.set_title(f"{row}: {METRIC_LABELS.get(metric, metric)}", loc="left", fontsize=10)
    # first group and first legend level on top; the y-axis is shared, so flip it exactly once
    axes[0, 0].invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[level]) for level in levels]
    top = 1 - 0.55 / fig.get_figheight()
    fig.suptitle(title or "", x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles, levels, loc="upper left", ncol=len(levels), fontsize=9,
               bbox_to_anchor=(0.005, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


FAMILY_COLORS = {
    "Logistic regression (static)": "#2a78d6",
    "Random forest (static)": "#eb6834",
    "Gradient boosting (static)": "#1baf7a",
}


def plot_comparison_intervals(comparisons, out_path, reference, title=None):
    """Forest plot: one panel per alternative strategy; per scenario, the reference minus that strategy
    (post-drift PR-AUC) with its 95% CI for each model. Filled marker = CI excludes 0."""
    apply_style()
    # "vs base" (the frozen model) is left to the tables: its large gaps would squash the shared scale
    others = [s for s in ["periodic retrain", "+ ADWIN", "+ DDM", "DDM warning window"]
              if s in set(comparisons["vs"])]
    scenarios = list(dict.fromkeys(comparisons["scenario"]))
    families = [f for f in FAMILY_COLORS if f in set(comparisons["family"])]
    fig, axes = plt.subplots(1, len(others), figsize=(3.6 * len(others), 0.42 * len(scenarios) + 1.8),
                             sharey=True, sharex=True, squeeze=False)
    offsets = {f: (i - (len(families) - 1) / 2) * 0.22 for i, f in enumerate(families)}
    for ax, other in zip(axes[0], others):
        ax.axvline(0, color=INK_SECONDARY, linewidth=1, zorder=1)
        ax.spines["left"].set_visible(False)
        for y, scenario in enumerate(scenarios):
            for family in families:
                row = comparisons[(comparisons["scenario"] == scenario) & (comparisons["family"] == family)
                                  & (comparisons["vs"] == other)]
                if row.empty:
                    continue
                r = row.iloc[0]
                color = FAMILY_COLORS[family]
                yy = y + offsets[family]
                ax.plot([r["lo"], r["hi"]], [yy, yy], color=color, linewidth=1.6, solid_capstyle="round")
                ax.plot(r["mean"], yy, "o", markersize=6, color=color, markeredgecolor=color,
                        markerfacecolor=color if r["significant"] else SURFACE, markeredgewidth=1.5)
        ax.set_title(f"vs {other}", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Difference in post-drift PR-AUC")
    axes[0, 0].set_yticks(range(len(scenarios)), scenarios)
    axes[0, 0].invert_yaxis()
    axes[0, 0].tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)

    handles = [plt.Line2D([], [], color=FAMILY_COLORS[f], marker="o", linewidth=1.6) for f in families]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor=MUTED, linestyle="none"),
                plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linestyle="none")]
    labels = [f.replace(" (static)", "") for f in families] + ["95% CI excludes 0", "CI includes 0"]
    top = 1 - 0.6 / fig.get_figheight()
    fig.suptitle(title or f"{reference} minus each alternative (right of 0 = {reference} better)",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.005, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


MODEL_COLORS = {"Logistic regression": "#2a78d6", "Random forest": "#eb6834", "Gradient boosting": "#1baf7a"}


def plot_severity(table, out_path, title=None):
    """Drift severity study: per dataset (row), post-drift PR-AUC of the frozen model (hollow) and of
    accumulate-after-drift (filled), and ADWIN's detection delay, against the severity level."""
    apply_style()
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.3 * len(datasets) + 0.8), squeeze=False)
    for row, dataset in zip(axes, datasets):
        d = table[table["dataset"] == dataset]
        for model, g in d.groupby("model", sort=False):
            g = g.sort_values("level")
            color = MODEL_COLORS.get(model, MUTED)
            row[0].plot(g["level"], g["frozen_pr_auc_after"], color=color, linewidth=1.6, marker="o", markersize=6,
                        markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.5)
            row[0].plot(g["level"], g["accumulate_pr_auc_after"], color=color, linewidth=1.6, marker="o",
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            row[1].plot(g["level"], g["adwin_delay_batches"], color=color, linewidth=1.6, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        axis_name = d["axis"].iloc[0]
        row[0].set_title(f"{dataset}: post-drift PR-AUC", loc="left")
        row[0].set_ylim(0, 1.02)
        row[1].set_title(f"{dataset}: ADWIN detection delay (batches)", loc="left")
        row[1].set_ylim(bottom=0)
        for ax in row:
            ax.set_xlabel(f"Drift severity: {axis_name}")
    handles = [plt.Line2D([], [], color=c, linewidth=1.6) for m, c in MODEL_COLORS.items() if m in set(table["model"])]
    labels = [m for m in MODEL_COLORS if m in set(table["model"])]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor=SURFACE, linestyle="none"),
                plt.Line2D([], [], color=MUTED, marker="o", markerfacecolor=MUTED, linestyle="none")]
    labels += ["frozen model", "+ ADWIN + accumulate"]
    top = 1 - 0.6 / fig.get_figheight()
    fig.suptitle(title or "Drift severity vs performance loss and detection", x=0.01, y=0.995, ha="left",
                 va="top", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", ncol=len(labels), fontsize=9,
               bbox_to_anchor=(0.005, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_feature_importance(tables, out_path, top=10, title=None):
    """Side-by-side horizontal bars: the top features of each importance table (e.g. before/after drift)."""
    apply_style()
    fig, axes = plt.subplots(1, len(tables), figsize=(3.9 * len(tables), 0.32 * top + 1.6), squeeze=False)
    for ax, (label, series) in zip(axes[0], tables.items()):
        s = series.sort_values(ascending=False).head(top)[::-1]
        ax.set_axisbelow(True)
        ax.barh(s.index, s.values, color=SINGLE_SERIES, height=0.7)
        for y, v in enumerate(s.values):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(3, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK_SECONDARY)
        ax.set_xlim(0, s.max() * 1.25 if len(s) else 1)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="y", length=0, labelcolor=INK_SECONDARY)
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_xlabel("mean |SHAP value|")
    fig.suptitle(title or "", x=0.01, y=0.99, ha="left", va="top", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


REPORTED_COLORS = {
    "deployed": "#2a78d6",
    "reported, random split": "#eb6834",
    "reported, time-ordered split": "#1baf7a",
}


def plot_reported_vs_deployed(raw, out_path, train_pct, drift_pct=None, metric="pr_auc", title=None):
    """One panel per model: deployed score per slice of the stream (line) against the scores the
    developer would have reported before deployment (horizontal lines over the deployment period)."""
    apply_style()
    families = list(dict.fromkeys(raw["family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(4.4 * len(families), 3.6), sharey=True, squeeze=False)
    for ax, family in zip(axes[0], families):
        fam = raw[raw["family"] == family]
        dep = fam[fam["evaluation"] == "deployed"].sort_values("position_pct")
        span = (dep["start_pct"].min(), dep["end_pct"].max())
        for evaluation in ["reported, random split", "reported, time-ordered split"]:
            level = fam.loc[fam["evaluation"] == evaluation, metric].mean()
            ax.hlines(level, *span, color=REPORTED_COLORS[evaluation], linewidth=1.6)
        ax.plot(dep["position_pct"], dep[metric], color=REPORTED_COLORS["deployed"], linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axvspan(0, train_pct, color=GRID, alpha=0.5, linewidth=0)
        ax.annotate("training data", (train_pct / 2, 1.0), ha="center", va="top", fontsize=8.5, color=INK_SECONDARY)
        if drift_pct is not None:
            ax.axvline(drift_pct, color=MUTED, linewidth=1)
            ax.annotate("true drift", (drift_pct, 0.9), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.02)
        ax.set_title(family, loc="left")
        ax.set_xlabel("Position in the stream (%)")
    axes[0, 0].set_ylabel(METRIC_LABELS.get(metric, metric))

    handles = [plt.Line2D([], [], color=c, linewidth=2 if k == "deployed" else 1.6,
                          marker="o" if k == "deployed" else None, markeredgecolor=SURFACE)
               for k, c in REPORTED_COLORS.items()]
    labels = ["deployed (scored on each later slice)", "reported: random split", "reported: time-ordered split"]
    top = 1 - 0.6 / fig.get_figheight()
    fig.suptitle(title or "", x=0.01, y=0.995, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles, labels, loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0.005, 1 - 0.3 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_summary(summary, out_path, metric="mean_pr_auc", title=None):
    """Horizontal bars, one per model run, sorted best first, value printed at each bar end."""
    apply_style()
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
