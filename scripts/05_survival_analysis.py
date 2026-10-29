#!/usr/bin/env python3
"""
Survival analysis of total session time until participants moved on to the
exit survey, stratified by experimental group, using scikit-survival.

Each participant's time is the wall-clock time from their first event (the
start of the tutorial) to their final experimental level_complete or
time_limit_reached event. The session time limit (70 minutes) applies to this
same span, so participants who timed out sit at the limit. Every participant
counts as an event at that time - whether they completed every level or hit the
time limit, that is when they "moved on" - so nobody is censored.

Produces:
  - a Kaplan-Meier plot (with 95% confidence bands) per group
  - a log-rank test comparing the groups
  - a CSV of the per-participant times used
  - a bar chart of each participant's session time in ascending order

Reads data/deidentified/*.json (produced by scripts 01-03). Only participants
with a matched exit survey are included, matching 04_generate_plots.py.

Usage:
    python scripts/05_survival_analysis.py --combined-dir data/deidentified --out analysis_plots
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

REPO_ROOT = Path(__file__).resolve().parent.parent

COLORS = {"control": "#4C72B0", "standard_ai": "#DD8452"}
LINESTYLES = ["-", "--"]
# How each group is named in figures and tables (the data keeps the internal ids)
GROUP_DISPLAY_NAMES = {"control": "Solo", "standard_ai": "Human+AI"}


def display_name(group):
    return GROUP_DISPLAY_NAMES.get(group, group)

EXPERIMENTAL_LEVELS = ["level_001", "level_002", "level_003", "level_004"]
SESSION_TIME_LIMIT_MINUTES = 70
IEEE_COLUMN_WIDTH_IN = 3.5
# Widths for two panels side by side in one column (sum + gap = 3.5 in)
SIDE_BY_SIDE_SESSION_WIDTH_IN = 2.05
SIDE_BY_SIDE_ATTEMPTS_WIDTH_IN = 1.35
# Panels of the compact session time + attempts figure: 0.62 + 0.36 of the column
COMPACT_SESSION_WIDTH_IN = 0.62 * IEEE_COLUMN_WIDTH_IN
COMPACT_ATTEMPTS_WIDTH_IN = 0.36 * IEEE_COLUMN_WIDTH_IN
COMPACT_PANEL_HEIGHT_IN = 1.85
# Match IEEEtran body text: 10 pt Times
IEEE_FONT_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                "mathtext.fontset": "stix", "font.size": 10, "ps.fonttype": 42}


def load_participants(combined_dir):
    participants = []
    for f in sorted(combined_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with f.open(encoding="utf-8") as fh:
            participants.append(json.load(fh))
    return participants


def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_survival_df(participants):
    """
    One row per participant: wall-clock minutes from their first event (start of
    the tutorial) to their final experimental level_complete or time_limit_reached
    event (when they moved on to the survey). moved_on is the event indicator and is always True.
    completed_all records whether they finished every level or timed out first.
    """
    rows = []
    for p in participants:
        events = sorted((e for e in p.get("__subcollections__", {}).get("events", {}).values()
                         if e.get("localTime")), key=lambda e: e["localTime"])
        exp = [e for e in events if e.get("isExperiment")]
        if not exp:
            continue
        ends = [e for e in exp if e.get("eventType") in ("level_complete", "time_limit_reached")]
        last = ends[-1] if ends else exp[-1]
        completed_levels = {e.get("currentLevel") for e in exp
                            if e.get("eventType") == "level_complete" and e.get("success")}
        seconds = (parse_time(last["localTime"]) - parse_time(events[0]["localTime"])).total_seconds()
        # Code runs on the experimental levels divided by the levels reached, matching
        # perf_avg_runs_per_level_by_group in 04_generate_plots.py
        exp_level_events = [e for e in exp if e.get("currentLevel") in EXPERIMENTAL_LEVELS]
        levels_reached = {e.get("currentLevel") for e in exp_level_events}
        runs = sum(e.get("eventType") == "run_simulation" for e in exp_level_events)
        rows.append({
            "participant": p.get("participantId"),
            "group": p.get("experimentalGroup"),
            "time_minutes": seconds / 60,
            "moved_on": True,
            "completed_all": set(EXPERIMENTAL_LEVELS) <= completed_levels,
            "avg_runs_per_level": runs / len(levels_reached) if levels_reached else np.nan,
        })
    return pd.DataFrame(rows)


def plot_kaplan_meier(surv_df, groups, logrank, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for gi, g in enumerate(groups):
        sub = surv_df[surv_df.group == g]
        if sub.empty:
            continue
        t, s, (ci_lo, ci_hi) = kaplan_meier_estimator(
            sub.moved_on.to_numpy(), sub.time_minutes.to_numpy(),
            conf_type="log-log")
        # start the curve at (0, 1) and extend it to the group's last observed time
        t = np.concatenate([[0], t, [sub.time_minutes.max()]])
        s = np.concatenate([[1], s, [s[-1]]])
        ci_lo = np.concatenate([[1], ci_lo, [ci_lo[-1]]])
        ci_hi = np.concatenate([[1], ci_hi, [ci_hi[-1]]])

        color = COLORS.get(g, None)
        label = f"{display_name(g)} (n={len(sub)}, {int((~sub.completed_all).sum())} timed out)"
        ax.step(t, s, where="post", color=color, linewidth=2,
                linestyle=LINESTYLES[gi % len(LINESTYLES)], label=label)
        ax.fill_between(t, ci_lo, ci_hi, step="post", color=color, alpha=0.15, linewidth=0)

        # mark the step for participants who moved on because they hit the time limit
        timed_out = sub.time_minutes[~sub.completed_all].to_numpy()
        s_before = [s[np.searchsorted(t, x, side="left") - 1] for x in timed_out]
        ax.scatter(timed_out, s_before, marker="o", s=40, facecolors="white",
                   edgecolors=color, linewidths=1.5, zorder=5)

    if logrank is not None:
        chi2, p = logrank
        ax.text(0.02, 0.04, f"Log-rank test: χ² = {chi2:.2f}, p = {p:.3f}",
                transform=ax.transAxes, fontsize=9, color="dimgray")

    ax.axvline(SESSION_TIME_LIMIT_MINUTES, color="gray", linestyle=":", linewidth=1)
    ax.annotate(f"{SESSION_TIME_LIMIT_MINUTES}-min session limit", (SESSION_TIME_LIMIT_MINUTES, 1.02),
                xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=8, color="dimgray")

    ax.plot([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="gray",
            markersize=6, label="Moved on because of time limit")
    ax.fill_between([], [], [], color="gray", alpha=0.15, label="95% CI")
    ax.set_xlim(0, SESSION_TIME_LIMIT_MINUTES + 3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Total session time since start of tutorial (minutes)")
    ax.set_ylabel("Proportion not yet done with tasks")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.08))
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "perf_survival_time_to_completion.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def plot_session_times_sorted(surv_df, out_dir, half=False):
    """
    One bar per participant: total session time (minutes) in ascending order,
    colored by group, with a black dot above participants who did not complete
    every level. Written as PNG and EPS (vector, for LaTeX/Overleaf).
    half=True writes a half-column version (for two panels side by side in one
    column) with a stacked legend and shorter labels, still at 10 pt.
    """
    df = surv_df.sort_values("time_minutes").reset_index(drop=True)
    x = np.arange(len(df))
    # Sized for one IEEE conference column (3.5 in) so it is placed at 100% scale
    with plt.rc_context(IEEE_FONT_RC):
        fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_SESSION_WIDTH_IN, 2.6) if half
                               else (IEEE_COLUMN_WIDTH_IN, 2.5))
        ax.bar(x, df.time_minutes, width=0.8, color=[COLORS.get(g) for g in df.group],
               edgecolor="black", linewidth=0.4)
        incomplete = df[~df.completed_all]
        dot_y = SESSION_TIME_LIMIT_MINUTES + 6
        ax.scatter(incomplete.index, np.full(len(incomplete), dot_y), color="black", s=8, zorder=5)
        ax.axhline(SESSION_TIME_LIMIT_MINUTES, color="black", linestyle="--", linewidth=0.8)
        limit_label = "limit" if half else "session limit"
        ax.annotate(f"{SESSION_TIME_LIMIT_MINUTES}-min {limit_label}", (-0.6, SESSION_TIME_LIMIT_MINUTES),
                    xytext=(1, 3), textcoords="offset points", color="black")

        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.4,
                         label=f"{display_name(g)} (n={(df.group == g).sum()})")
                   for g in sorted(df.group.unique())]
        # Two legend rows: groups on top, the "did not complete" marker below.
        # (Legends fill column-first, so the marker goes second in the list.)
        handles.insert(1, Line2D([], [], marker="o", linestyle="", color="black", markersize=3,
                                 label="Did not finish" if half else "Did not complete all levels"))
        if half:  # one entry per row, in reading order
            handles = handles[0:1] + handles[2:] + handles[1:2]
        fig.legend(handles=handles, loc="upper center", ncol=1 if half else 2, frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2,
                   labelspacing=0.2)

        ax.set_xticks([])
        ax.set_xlim(-0.6, len(df) - 0.4)
        ax.set_ylim(0, dot_y + 4)
        ax.set_yticks(range(0, SESSION_TIME_LIMIT_MINUTES + 1, 10))
        ax.set_ylabel("Minutes")
        ax.set_facecolor("#F2F2F2")
        ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.76 if half else 0.87))
        for ext in ("png", "eps"):
            out_path = out_dir / f"completion_time_by_participant{'_half' if half else ''}.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def plot_session_and_attempts(surv_df, groups, out_dir):
    """
    Compact single-column compound figure, written as two panels for LaTeX
    subcaptions: (a) each participant's session time in ascending order, with the
    group legend above it, and (b) the distribution of average attempts per level
    by group. Both panels share one height and the plot areas line up, so they sit
    side by side in one column at their natural size (10 pt text). The "did not
    finish" dots and the n per group go in the caption. Written as PNG and EPS.
    """
    df = surv_df.sort_values("time_minutes").reset_index(drop=True)
    x = np.arange(len(df))
    with plt.rc_context(IEEE_FONT_RC):
        fig_time, ax_time = plt.subplots(figsize=(COMPACT_SESSION_WIDTH_IN, COMPACT_PANEL_HEIGHT_IN))
        fig_runs, ax_runs = plt.subplots(figsize=(COMPACT_ATTEMPTS_WIDTH_IN, COMPACT_PANEL_HEIGHT_IN))

        ax_time.bar(x, df.time_minutes, width=0.8, color=[COLORS.get(g) for g in df.group],
                    edgecolor="black", linewidth=0.4)
        incomplete = df[~df.completed_all]
        dot_y = SESSION_TIME_LIMIT_MINUTES + 6
        ax_time.scatter(incomplete.index, np.full(len(incomplete), dot_y), color="black", s=8, zorder=5)
        ax_time.axhline(SESSION_TIME_LIMIT_MINUTES, color="black", linestyle="--", linewidth=0.8)
        ax_time.annotate(f"{SESSION_TIME_LIMIT_MINUTES}-min limit", (-0.6, SESSION_TIME_LIMIT_MINUTES),
                         xytext=(1, 3), textcoords="offset points", color="black")
        ax_time.set_xticks([])
        ax_time.set_xlim(-0.6, len(df) - 0.4)
        ax_time.set_ylim(0, dot_y + 4)
        ax_time.set_yticks(range(0, SESSION_TIME_LIMIT_MINUTES + 1, 20))
        ax_time.set_ylabel("Session time (min)")

        vals = [df[df.group == g].avg_runs_per_level.dropna().tolist() for g in groups]
        bp = ax_runs.boxplot(vals, positions=range(len(groups)), widths=0.6, showfliers=False,
                             patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                             boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                             capprops={"linewidth": 0.6})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(COLORS.get(g))
        ax_runs.set_xticks([])
        ax_runs.set_xlim(-0.5, len(groups) - 0.5)
        ax_runs.set_ylim(bottom=0)
        ax_runs.set_ylabel("Avg. runs per level")

        for ax in (ax_time, ax_runs):
            ax.set_facecolor("#F2F2F2")
            ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.4, label=display_name(g))
                   for g in groups]
        fig_time.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                        handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
        top = 1 - 0.27 / COMPACT_PANEL_HEIGHT_IN  # room for the legend above panel (a)
        fig_time.tight_layout(pad=0.3, rect=(0, 0, 1, top))
        fig_runs.tight_layout(pad=0.3)
        # line the two plot areas up vertically: same bottom and top in both panels
        bottom = max(ax_time.get_position().y0, ax_runs.get_position().y0)
        for fig, ax in ((fig_time, ax_time), (fig_runs, ax_runs)):
            pos = ax.get_position()
            ax.set_position([pos.x0, bottom, pos.width, ax_time.get_position().y1 - bottom])

        for fig, name in ((fig_time, "perf_session_time_and_attempts_a"),
                          (fig_runs, "perf_session_time_and_attempts_b")):
            for ext in ("png", "eps"):
                out_path = out_dir / f"{name}.{ext}"
                fig.savefig(out_path, dpi=300)
                print(f"  -> wrote {out_path}")
            plt.close(fig)


LEVEL_PANEL_TITLES = {
    "level_001": "L1 \u00b7 Authoring, easy",
    "level_002": "L2 \u00b7 Authoring, hard",
    "level_003": "L3 \u00b7 Debugging, easy",
    "level_004": "L4 \u00b7 Debugging, hard",
}


def build_level_survival_df(participants):
    """
    One row per (participant, level reached): wall-clock minutes from the first
    event on the level until its first successful completion (succeeded=True), or
    until the participant was stopped by the session time limit (succeeded=False,
    censored). Participants who never reached a level have no row for it.
    """
    rows = []
    for p in participants:
        by_level = {}
        for e in p.get("__subcollections__", {}).get("events", {}).values():
            if e.get("isExperiment") and e.get("localTime") and e.get("currentLevel") in EXPERIMENTAL_LEVELS:
                by_level.setdefault(e["currentLevel"], []).append(e)
        for level, events in by_level.items():
            events.sort(key=lambda e: e["localTime"])
            success = next((e for e in events if e.get("eventType") == "level_complete" and e.get("success")),
                           None)
            end = success or next((e for e in events if e.get("eventType") == "time_limit_reached"),
                                  events[-1])
            minutes = (parse_time(end["localTime"]) - parse_time(events[0]["localTime"])).total_seconds() / 60
            rows.append({"participant": p.get("participantId"), "group": p.get("experimentalGroup"),
                         "level": level, "time_minutes": minutes, "succeeded": success is not None})
    return pd.DataFrame(rows)


def plot_kaplan_meier_by_level(level_df, groups, out_dir):
    """
    Kaplan-Meier curves of the share of participants who have not yet succeeded on
    each level, by group (one panel per level, shared axes). Participants stopped
    by the session time limit are censored (+). Sized for one IEEE column, 10 pt.
    Prints a log-rank test per level.
    """
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(IEEE_COLUMN_WIDTH_IN, 2.9), sharex=True, sharey=True)
        x_max = level_df.time_minutes.max()
        for ax, level in zip(axes.flat, EXPERIMENTAL_LEVELS):
            sub = level_df[level_df.level == level]
            for g, ls in zip(groups, LINESTYLES):
                gsub = sub[sub.group == g]
                if gsub.empty:
                    continue
                t, surv = kaplan_meier_estimator(gsub.succeeded.to_numpy(), gsub.time_minutes.to_numpy())
                ax.step(np.r_[0, t, x_max], np.r_[1, surv, surv[-1]], where="post",
                        color=COLORS.get(g), linestyle=ls, linewidth=1.4)
                # censored participants: + at their time on the curve
                for ct in gsub[~gsub.succeeded].time_minutes:
                    y = np.r_[1, surv][np.searchsorted(t, ct, side="right")]
                    ax.plot(ct, y, marker="+", color=COLORS.get(g), markersize=6, markeredgewidth=1.2)
            if sub.group.nunique() == 2:
                y = Surv.from_arrays(event=sub.succeeded.to_numpy(), time=sub.time_minutes.to_numpy())
                chi2, pval = compare_survival(y, sub.group.to_numpy())
                print(f"  {level}: log-rank chi2={chi2:.3f}, p={pval:.3f}")
            ax.set_title(LEVEL_PANEL_TITLES[level], fontsize=10, pad=3)
            ax.set_facecolor("#F2F2F2")
            ax.grid(color="#8C8C8C", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_xlim(0, x_max)
        axes[0, 0].set_ylim(-0.03, 1.03)
        axes[0, 0].set_yticks([0, 0.5, 1])
        fig.supylabel("Not yet succeeded", fontsize=10, x=0.01)
        fig.supxlabel("Minutes on level", fontsize=10, y=0.01)
        handles = [Line2D([], [], color=COLORS.get(g), linestyle=ls, linewidth=1.4, label=display_name(g))
                   for g, ls in zip(groups, LINESTYLES)]
        handles.append(Line2D([], [], marker="+", linestyle="", color="black", markersize=6,
                              markeredgewidth=1.2, label="Timed out"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   handlelength=1.6, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0.02, 0.02, 1, 0.92), h_pad=0.5, w_pad=0.6)
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_survival_time_to_success_by_level.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Kaplan-Meier survival analysis of time until moving on to the survey, by group.")
    parser.add_argument("--combined-dir", default=REPO_ROOT / "data" / "deidentified",
                        help="Directory of de-identified participant JSON files (default: data/deidentified).")
    parser.add_argument("--out", default=REPO_ROOT / "analysis_plots",
                        help="Output directory for the plot and CSV (default: analysis_plots).")
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    out_dir = Path(args.out)
    if not combined_dir.is_dir():
        print(f"Error: combined dataset directory not found: {combined_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    participants = load_participants(combined_dir)
    # Keep only real participants: those with a matched exit survey.
    participants = [p for p in participants if p.get("exitSurvey")]
    surv_df = build_survival_df(participants)
    if surv_df.empty:
        print("No participants with experimental events found.", file=sys.stderr)
        sys.exit(1)

    groups = sorted(surv_df.group.dropna().unique())
    for g in groups:
        sub = surv_df[surv_df.group == g]
        print(f"{g}: n={len(sub)}, completed all levels={int(sub.completed_all.sum())}, "
              f"timed out={int((~sub.completed_all).sum())}")

    logrank = None
    if len(groups) == 2:
        y = Surv.from_arrays(event=surv_df.moved_on.to_numpy(), time=surv_df.time_minutes.to_numpy())
        chi2, p = compare_survival(y, surv_df.group.to_numpy())
        logrank = (chi2, p)
        print(f"Log-rank test: chi2={chi2:.3f}, p={p:.3f}")

    plot_kaplan_meier(surv_df, groups, logrank, out_dir)
    plot_session_times_sorted(surv_df, out_dir)
    plot_session_times_sorted(surv_df, out_dir, half=True)
    plot_session_and_attempts(surv_df, groups, out_dir)
    print("Per-level time until first success:")
    plot_kaplan_meier_by_level(build_level_survival_df(participants), groups, out_dir)
    csv_path = out_dir / "perf_survival_time_to_completion.csv"
    surv_df.to_csv(csv_path, index=False)
    print(f"  -> wrote {csv_path}")


if __name__ == "__main__":
    main()
