#!/usr/bin/env python3
"""
Generate plots comparing experimental groups on both task performance
(from Firestore event logs) and exit survey responses.

Reads data/deidentified/*.json (participant data + events + exit survey,
produced by scripts 01-03) and writes PNGs to an output folder (default:
analysis_plots/).

Usage:
    python scripts/04_generate_plots.py --combined-dir data/deidentified --out analysis_plots
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

REPO_ROOT = Path(__file__).resolve().parent.parent

COLORS = {"control": "#4C72B0", "standard_ai": "#DD8452"}
IEEE_COLUMN_WIDTH_IN = 3.5
# Widths for two panels side by side in one column (sum + gap = 3.5 in)
SIDE_BY_SIDE_SESSION_WIDTH_IN = 2.05
SIDE_BY_SIDE_ATTEMPTS_WIDTH_IN = 1.35
# Match IEEEtran body text: 10 pt Times
IEEE_FONT_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                "mathtext.fontset": "stix", "font.size": 10, "ps.fonttype": 42}
# Header text for each task type in the per-level panel figures
TASK_HEADER_LABELS = {"code authoring": "Code Authoring", "code completion": "Code Debugging/Editing"}

# How each group is named in figures and tables (the data keeps the internal ids)
GROUP_DISPLAY_NAMES = {"control": "Solo", "standard_ai": "Human+AI"}


def display_name(group):
    return GROUP_DISPLAY_NAMES.get(group, group)

EXPERIMENTAL_LEVELS = ["level_001", "level_002", "level_003", "level_004"]
LEVEL_DESCRIPTIONS = {
    "level_001": "Easy code authoring",
    "level_002": "Hard code authoring",
    "level_003": "Easy code completion",
    "level_004": "Hard code completion",
}


def load_participants(combined_dir):
    participants = []
    for f in sorted(combined_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with f.open(encoding="utf-8") as fh:
            participants.append(json.load(fh))
    return participants


def group_label(group, n_by_group):
    return f"{display_name(group)} (n={n_by_group.get(group, 0)})"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# Block types that move the robot (pick/drop, sensing, control and print blocks excluded)
MOVEMENT_BLOCK_TYPES = {"move_forward", "turn_clockwise", "turn_counter_clockwise"}


def count_movement_blocks(run_event):
    """Number of movement blocks in a run_simulation event's code snapshot."""
    snapshot = run_event.get("codeSnapshot")
    if not snapshot:
        return np.nan
    blocks = json.loads(snapshot) if isinstance(snapshot, str) else snapshot
    return sum(b.get("type") in MOVEMENT_BLOCK_TYPES for b in blocks)


def build_performance_df(participants):
    """One row per (participant, experimental level): attempts, collisions, completion, time."""
    rows = []
    for p in participants:
        pid = p.get("participantId")
        group = p.get("experimentalGroup")
        events = p.get("__subcollections__", {}).get("events", {})

        by_level = {}
        for e in events.values():
            if not e.get("isExperiment"):
                continue
            level = e.get("currentLevel")
            if level not in EXPERIMENTAL_LEVELS:
                continue
            by_level.setdefault(level, {"run_simulation": [], "collision": 0,
                                          "level_complete": [], "time_limit_reached": [],
                                          "times": []})
            bucket = by_level[level]
            if e.get("localTime"):
                bucket["times"].append(parse_time(e["localTime"]))
            et = e.get("eventType")
            if et == "run_simulation":
                bucket["run_simulation"].append(e)
            elif et == "collision":
                bucket["collision"] += 1
            elif et == "level_complete":
                bucket["level_complete"].append(e)
            elif et == "time_limit_reached":
                bucket["time_limit_reached"].append(e)

        for level in EXPERIMENTAL_LEVELS:
            bucket = by_level.get(level)
            if bucket is None:
                continue  # participant never reached this level
            completions = bucket["level_complete"]
            success = any(c.get("success") for c in completions)
            # Wall-clock time on the level, across all restarts: from its first event until
            # the participant moved on (successful completion, or the session time limit).
            # Unlike timeSpentSeconds, this doesn't reset when a level is restarted.
            end_event = next((c for c in completions if c.get("success")), None)
            if end_event is None and bucket["time_limit_reached"]:
                end_event = bucket["time_limit_reached"][0]
            if end_event is not None and end_event.get("localTime") and bucket["times"]:
                time_spent = (parse_time(end_event["localTime"]) - min(bucket["times"])).total_seconds()
            else:
                time_spent = np.nan
            # Movement blocks in the last program run on the level (for a completed
            # level, the program that succeeded)
            runs = sorted((r for r in bucket["run_simulation"] if r.get("localTime")),
                          key=lambda r: r["localTime"])
            movement_blocks = count_movement_blocks(runs[-1]) if runs else np.nan
            rows.append({
                "participant": pid,
                "group": group,
                "level": level,
                "attempts": len(bucket["run_simulation"]),
                "movement_blocks": movement_blocks,
                "collisions": bucket["collision"],
                "completed": success,
                "time_spent_seconds": time_spent,
                "timed_out": len(bucket["time_limit_reached"]) > 0,
            })
    return pd.DataFrame(rows)


PROGRAMMING_EXPERIENCE_QUESTION = "General Programming_1"  # 1 = complete beginner, 5 = expert

# Self-efficacy items, rated 1 (strongly disagree) to 5 (strongly agree)
SELF_EFFICACY_QUESTIONS = {
    "QID10_1": "Felt capable of completing the tasks",
    "QID10_2": "Confident explaining my code logic",
    "QID10_3": "Felt in control of the robot",
    "QID10_4": "Confident on a harder future task",
}
LIKERT_LEVELS = [1, 2, 3, 4, 5]
# Sequential (light -> dark) so it never reads as a group color
LIKERT_COLORS = ["#F2F0F7", "#CBC9E2", "#9E9AC8", "#756BB1", "#54278F"]

AI_PARTNER_QUESTIONS = {
    "Communication Qual_1": "Partner gave clear,\nactionable suggestions",
    "Dependency_1": "Waited for partner's input\nbefore acting",
    "Trust_1": "Relied on partner's suggestions\nwithout double-checking",
}

DIFFICULTY_QUESTIONS = {
    "QID11_1": "Authoring",
    "QID11_2": "Debugging/Editing",
}


def build_survey_df(participants):
    rows = []
    for p in participants:
        group = p.get("experimentalGroup")
        pid = p.get("participantId")
        es = p.get("exitSurvey")
        if not es:
            continue
        answers = es.get("answers", {})
        for key, value in answers.items():
            rows.append({"participant": pid, "group": group, "question": key, "value": value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def grouped_boxplot(ax, categories, group_names, data, n_by_group, ylabel, title=None):
    """
    data: dict[group][category] -> list of numeric values (one per participant).
    Draws side-by-side box plots per category, one box per group, with outliers hidden.
    """
    n_groups = len(group_names)
    x = np.arange(len(categories))
    width = 0.8 / n_groups

    for gi, group in enumerate(group_names):
        xpos = x + (gi - (n_groups - 1) / 2) * width
        vals = [[v for v in data[group].get(cat, []) if not (isinstance(v, float) and np.isnan(v))]
                for cat in categories]
        color = COLORS.get(group, None)
        ax.boxplot(vals, positions=xpos, widths=width * 0.8, showfliers=False,
                   patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": color, "alpha": 0.85},
                   medianprops={"color": "black"})
        ax.plot([], [], marker="s", linestyle="", color=color, markersize=10,
                label=group_label(group, n_by_group))

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out_dir, name):
    out_path = out_dir / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# ---------------------------------------------------------------------------
# Performance plots
# ---------------------------------------------------------------------------

def plot_success_rate_by_level(perf_df, groups, n_by_group, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(EXPERIMENTAL_LEVELS))
    width = 0.8 / len(groups)
    for gi, g in enumerate(groups):
        sub = perf_df[perf_df.group == g]
        rates = []
        for level in EXPERIMENTAL_LEVELS:
            lvl_sub = sub[sub.level == level]
            rates.append(lvl_sub.completed.mean() if len(lvl_sub) else 0)
        xpos = x + (gi - (len(groups) - 1) / 2) * width
        ax.bar(xpos, rates, width=width * 0.9, label=group_label(g, n_by_group),
               color=COLORS.get(g, None), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPERIMENTAL_LEVELS)
    ax.set_ylabel("Success rate (fraction of participants who completed)")
    ax.set_title("Level completion success rate by experimental group")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.15)
    save(fig, out_dir, "perf_success_rate_by_level.png")


def plot_completion_by_level(perf_df, groups, n_by_group, out_dir):
    """
    Compound figure (one panel per group) of 100% stacked bars: per level, how many
    participants completed it vs. did not (timed out on it or never reached it).
    Counts are printed in each segment. Written as PNG and EPS (vector, for LaTeX/Overleaf).
    """
    x = np.arange(len(EXPERIMENTAL_LEVELS))
    xlabels = [f"Level {i + 1}\n" + LEVEL_DESCRIPTIONS[lvl].replace(" code ", "\ncode ")
               for i, lvl in enumerate(EXPERIMENTAL_LEVELS)]
    with plt.rc_context({"font.size": 9, "ps.fonttype": 42}):
        fig, axes = plt.subplots(len(groups), 1, figsize=(6.5, 2.6 * len(groups)), sharex=True)
        axes = np.atleast_1d(axes)
        for i, (ax, g) in enumerate(zip(axes, groups)):
            n = n_by_group[g]
            sub = perf_df[(perf_df.group == g) & perf_df.completed]
            done = np.array([(sub.level == lvl).sum() for lvl in EXPERIMENTAL_LEVELS])
            counts = {"Completed": done, "Did not complete": n - done}
            # completed segments use the group's color, as in the other figures
            status_colors = {"Completed": COLORS.get(g), "Did not complete": "#D9D9D9"}
            bottom = np.zeros(len(x))
            for status, c in counts.items():
                pct = 100 * c / n
                ax.bar(x, pct, bottom=bottom, width=0.55, color=status_colors[status],
                       edgecolor="black", linewidth=0.6, label=status)
                for xi, b, p, k in zip(x, bottom, pct, c):
                    if k > 0:
                        ax.text(xi, b + p / 2, str(k), ha="center", va="center", fontsize=9)
                bottom += pct
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
            ax.set_ylabel("Participants (%)")
            ax.set_title(f"({chr(ord('a') + i)}) {display_name(g)} (n={n})", fontweight="bold", fontsize=10, pad=18)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                      handlelength=1, fontsize=8, borderaxespad=0.1)
            ax.grid(axis="y", color="0.9")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(xlabels)
        fig.tight_layout()
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_completion_by_level.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def plot_level_panels(values_by_level, groups, ylabel, filename, out_dir, tick_step=5):
    """
    Compound figure (the four levels side by side, sharing one y-scale) of a per-level
    measure as box plots by group, with the n under each box and a task-type header
    over each pair of levels. values_by_level: dict[level][group] -> list of values.
    Sized for one IEEE conference column with 10 pt Times text. Written as PNG and EPS.
    """
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(1, len(EXPERIMENTAL_LEVELS), figsize=(IEEE_COLUMN_WIDTH_IN, 2.5),
                                 sharey=True)
        for i, (ax, level) in enumerate(zip(axes, EXPERIMENTAL_LEVELS)):
            vals = [values_by_level[level].get(g, []) for g in groups]
            bp = ax.boxplot(vals, positions=range(len(groups)), widths=0.4, showfliers=False,
                            patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                            boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                            capprops={"linewidth": 0.6})
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(COLORS.get(g))
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([f"n={len(v)}" for v in vals])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, len(groups) - 0.5)
            difficulty = LEVEL_DESCRIPTIONS[level].split(" ", 1)[0]
            ax.set_title(f"Level {i + 1}\n{difficulty}", fontsize=10)
            ax.set_facecolor("#F2F2F2")
            ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        # one shared scale, sized to the tallest whisker across all panels
        y_max = max(line.get_ydata().max() for ax in axes for line in ax.lines)
        # start a little below 0 so near-zero boxes (e.g. Level 3) are not hidden by the axis
        axes[0].set_ylim(-0.05 * y_max, y_max * 1.05)
        axes[0].yaxis.set_major_locator(MultipleLocator(tick_step))
        axes[0].set_ylabel(ylabel)
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.6, label=display_name(g))
                   for g in groups]
        fig.legend(handles=handles, loc="upper center", ncol=len(groups), frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.83), w_pad=0.4)

        # Task-type header spanning each run of levels with the same task
        # (e.g. "Code Authoring" over Levels 1-2), with a rule underneath.
        fig.canvas.draw()
        to_fig = fig.transFigure.inverted()
        tasks = [LEVEL_DESCRIPTIONS[lvl].split(" ", 1)[1] for lvl in EXPERIMENTAL_LEVELS]
        start = 0
        for end in range(1, len(tasks) + 1):
            if end < len(tasks) and tasks[end] == tasks[start]:
                continue
            x0 = axes[start].get_position().x0
            x1 = axes[end - 1].get_position().x1
            title_top = max(to_fig.transform(axes[j].title.get_window_extent())[1, 1]
                            for j in range(start, end))
            fig.add_artist(Line2D([x0 + 0.01, x1 - 0.01], [title_top + 0.01] * 2,
                                  transform=fig.transFigure, color="black", linewidth=0.6))
            fig.text((x0 + x1) / 2, title_top + 0.02,
                     TASK_HEADER_LABELS.get(tasks[start], tasks[start].capitalize()),
                     ha="center", va="bottom")
            start = end
        for ext in ("png", "eps"):
            out_path = out_dir / f"{filename}.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def completed_values_by_level(perf_df, groups, column, scale=1.0):
    """dict[level][group] -> values of `column` (times `scale`) for completed levels only."""
    completed = perf_df[perf_df.completed & perf_df[column].notna()]
    return {level: {g: list(completed[(completed.level == level) & (completed.group == g)][column] * scale)
                    for g in groups}
            for level in EXPERIMENTAL_LEVELS}


def plot_time_spent_by_level(perf_df, groups, n_by_group, out_dir):
    """Time to complete each level (minutes), completed levels only."""
    plot_level_panels(completed_values_by_level(perf_df, groups, "time_spent_seconds", 1 / 60),
                      groups, "Time until success\n(in minutes)", "perf_time_spent_by_level", out_dir)


def plot_attempts_until_success_by_level(perf_df, groups, n_by_group, out_dir):
    """
    Attempts (code runs) until success on each level, completed levels only. No code
    runs happen on a level after it is completed, so a completed level's run count is
    its attempts until success.
    """
    plot_level_panels(completed_values_by_level(perf_df, groups, "attempts"),
                      groups, "Runs until success", "perf_attempts_until_success_by_level",
                      out_dir)


def add_task_headers(fig, level_axes, labels=None):
    """
    Task-type header (e.g. "Code Authoring" over Levels 1-2) spanning each run of
    level panels with the same task, with a rule underneath, above the panel titles.
    level_axes: one axes per level, in EXPERIMENTAL_LEVELS order. labels: optional
    task -> header text, overriding TASK_HEADER_LABELS (e.g. shorter text for
    narrow figures).
    """
    labels = {**TASK_HEADER_LABELS, **(labels or {})}
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    # rule and text offsets are 0.035 in and 0.05 in above the titles, whatever the figure height
    rule_gap, text_gap = 0.035 / fig.get_figheight(), 0.05 / fig.get_figheight()
    tasks = [LEVEL_DESCRIPTIONS[lvl].split(" ", 1)[1] for lvl in EXPERIMENTAL_LEVELS]
    start = 0
    for end in range(1, len(tasks) + 1):
        if end < len(tasks) and tasks[end] == tasks[start]:
            continue
        x0 = level_axes[start].get_position().x0
        x1 = level_axes[end - 1].get_position().x1
        title_top = max(to_fig.transform(level_axes[j].title.get_window_extent())[1, 1]
                        for j in range(start, end))
        fig.add_artist(Line2D([x0 + 0.01, x1 - 0.01], [title_top + rule_gap] * 2,
                              transform=fig.transFigure, color="black", linewidth=0.6))
        fig.text((x0 + x1) / 2, title_top + text_gap,
                 labels.get(tasks[start], tasks[start].capitalize()),
                 ha="center", va="bottom")
        start = end


def plot_time_and_attempts_grid(perf_df, groups, out_dir):
    """
    Compact single-column compound figure: time until success (top row) and attempts
    until success (bottom row) on each level (columns), completed levels only. Task
    and level headers are drawn once on top and the legend once at the bottom; the n
    per box goes in the caption. Each row shares one y-scale. Written as PNG and EPS.
    """
    rows = [
        (completed_values_by_level(perf_df, groups, "time_spent_seconds", 1 / 60),
         "Time until\nsuccess (min)"),
        (completed_values_by_level(perf_df, groups, "attempts"), "Runs until\nsuccess"),
    ]
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(len(rows), len(EXPERIMENTAL_LEVELS),
                                 figsize=(IEEE_COLUMN_WIDTH_IN, 3.0), sharey="row")
        for r, (values_by_level, ylabel) in enumerate(rows):
            for i, level in enumerate(EXPERIMENTAL_LEVELS):
                ax = axes[r, i]
                vals = [values_by_level[level].get(g, []) for g in groups]
                bp = ax.boxplot(vals, positions=range(len(groups)), widths=0.5, showfliers=False,
                                patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                                boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                                capprops={"linewidth": 0.6})
                for patch, g in zip(bp["boxes"], groups):
                    patch.set_facecolor(COLORS.get(g))
                ax.set_xticks([])
                ax.set_xlim(-0.5, len(groups) - 0.5)
                if r == 0:
                    difficulty = LEVEL_DESCRIPTIONS[level].split(" ", 1)[0]
                    ax.set_title(f"L{i + 1} · {difficulty}", fontsize=10, pad=3)
                ax.set_facecolor("#F2F2F2")
                ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
            row_axes = axes[r]
            # one scale per row, sized to its tallest whisker; start a little below 0
            # so near-zero boxes (e.g. Level 3) are not hidden by the axis
            y_max = max(line.get_ydata().max() for ax in row_axes for line in ax.lines)
            row_axes[0].set_ylim(-0.05 * y_max, y_max * 1.05)
            row_axes[0].yaxis.set_major_locator(MultipleLocator(5))
            row_axes[0].set_ylabel(ylabel)
        fig.align_ylabels(axes[:, 0])
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.6, label=display_name(g))
                   for g in groups]
        fig.legend(handles=handles, loc="lower center", ncol=len(groups), frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 0.92), w_pad=0.4, h_pad=0.6)

        add_task_headers(fig, axes[0])
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_time_and_attempts_by_level.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


TIMED_OUT_MARKER = {"marker": "+", "color": "black", "markersize": 7, "markeredgewidth": 1.2}
KM_PANEL_TITLES = {
    "level_001": "L1 · Authoring, easy",
    "level_002": "L2 · Authoring, hard",
    "level_003": "L3 · Debugging, easy",
    "level_004": "L4 · Debugging, hard",
}


def levels_with_time(perf_df):
    """Rows with a time on the level (completed, or stopped by the session limit), in minutes."""
    return perf_df[perf_df.time_spent_seconds.notna()].assign(minutes=lambda d: d.time_spent_seconds / 60)


def plot_time_to_success_km(perf_df, groups, out_dir):
    """
    Kaplan-Meier estimate of the share of each group that has succeeded on each
    level by each minute (2x2 panels, one per level, own time axis). Participants
    stopped by the session time limit are censored (+) instead of dropped; levels a
    participant never reached are left out. Prints a log-rank test per level for
    time and runs. Single IEEE column, 10 pt text. Written as PNG and EPS.
    """
    df = levels_with_time(perf_df)
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(IEEE_COLUMN_WIDTH_IN, 2.9), sharex=True, sharey=True)
        # one time scale for every level, so levels can be compared directly
        x_max = df.minutes.max() * 1.05
        print("  Log-rank tests (censored at the session time limit):")
        for ax, level in zip(axes.flat, EXPERIMENTAL_LEVELS):
            sub = df[df.level == level]
            for g, ls in zip(groups, ["-", "--"]):
                gsub = sub[sub.group == g]
                if gsub.empty:
                    continue
                t, surv = kaplan_meier_estimator(gsub.completed.to_numpy(), gsub.minutes.to_numpy())
                succeeded = 100 * (1 - np.r_[1, surv])
                ax.step(np.r_[0, t, x_max], np.r_[succeeded, succeeded[-1]], where="post",
                        color=COLORS.get(g), linestyle=ls, linewidth=1.4)
                for ct in gsub[~gsub.completed].minutes:
                    ax.plot(ct, succeeded[np.searchsorted(t, ct, side="right")], zorder=3,
                            **TIMED_OUT_MARKER)
            if sub.group.nunique() == 2:
                for label, col in (("time", "minutes"), ("runs", "attempts")):
                    y = Surv.from_arrays(event=sub.completed.to_numpy(), time=sub[col].to_numpy(dtype=float))
                    chi2, pval = compare_survival(y, sub.group.to_numpy())
                    print(f"    {level} {label}: chi2={chi2:.3f}, p={pval:.3f}")
            ax.set_xlim(0, x_max)
            ax.set_ylim(-4, 104)
            ax.yaxis.set_major_locator(MultipleLocator(25))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.set_title(KM_PANEL_TITLES[level], fontsize=10, pad=3)
            ax.set_facecolor("#F2F2F2")
            ax.grid(color="#8C8C8C", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.supylabel("Succeeded (%)", fontsize=10, x=0.0, ha="left")
        fig.supxlabel("Minutes on level", fontsize=10, y=0.0, va="bottom")
        handles = [Line2D([], [], color=COLORS.get(g), linestyle=ls, linewidth=1.4, label=display_name(g))
                   for g, ls in zip(groups, ["-", "--"])]
        handles.append(Line2D([], [], linestyle="", label="Timed out", **TIMED_OUT_MARKER))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   handlelength=1.6, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93), h_pad=0.5, w_pad=0.6)
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_time_to_success_km_by_level.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def plot_runs_until_success_by_level(perf_df, groups, out_dir):
    """
    Runs until success on each level (1x4 panels with task headers), box plots for
    participants who completed the level; runs made by participants stopped by the
    session limit are drawn as + (a lower bound). Single IEEE column, 10 pt text.
    Written as PNG and EPS.
    """
    df = levels_with_time(perf_df)
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(1, len(EXPERIMENTAL_LEVELS), figsize=(IEEE_COLUMN_WIDTH_IN, 2.2), sharey=True)
        for i, (ax, level) in enumerate(zip(axes, EXPERIMENTAL_LEVELS)):
            sub = df[df.level == level]
            completed = [list(sub[(sub.group == g) & sub.completed].attempts) for g in groups]
            bp = ax.boxplot(completed, positions=range(len(groups)), widths=0.5, showfliers=False,
                            patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                            boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                            capprops={"linewidth": 0.6})
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(COLORS.get(g))
            for j, g in enumerate(groups):
                timed_out = sub[(sub.group == g) & ~sub.completed].attempts
                ax.plot([j] * len(timed_out), timed_out, linestyle="", zorder=3, **TIMED_OUT_MARKER)
            ax.set_xticks([])
            ax.set_xlim(-0.5, len(groups) - 0.5)
            difficulty = LEVEL_DESCRIPTIONS[level].split(" ", 1)[0]
            ax.set_title(f"L{i + 1} · {difficulty}", fontsize=10, pad=3)
            ax.set_facecolor("#F2F2F2")
            ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        # sized to the tallest whisker or timed-out marker
        r_max = max(line.get_ydata().max() for ax in axes for line in ax.lines if len(line.get_ydata()))
        axes[0].set_ylim(-0.05 * r_max, r_max * 1.05)
        axes[0].yaxis.set_major_locator(MultipleLocator(5))
        axes[0].set_ylabel("Runs until success")
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.6, label=display_name(g))
                   for g in groups]
        handles.append(Line2D([], [], linestyle="", label="Timed out", **TIMED_OUT_MARKER))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0, 0.09, 1, 0.88), w_pad=0.4)
        add_task_headers(fig, axes)
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_runs_until_success_by_level.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def plot_attempts_by_level(perf_df, groups, n_by_group, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {g: {} for g in groups}
    for g in groups:
        sub = perf_df[perf_df.group == g]
        for level in EXPERIMENTAL_LEVELS:
            data[g][level] = list(sub[sub.level == level].attempts)
    grouped_boxplot(ax, EXPERIMENTAL_LEVELS, groups, data, n_by_group,
                    ylabel="Code runs (run_simulation events)",
                    title="Number of simulation attempts per level by experimental group")
    save(fig, out_dir, "perf_attempts_by_level.png")


def plot_collisions_by_level(perf_df, groups, n_by_group, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {g: {} for g in groups}
    for g in groups:
        sub = perf_df[perf_df.group == g]
        for level in EXPERIMENTAL_LEVELS:
            data[g][level] = list(sub[sub.level == level].collisions)
    grouped_boxplot(ax, EXPERIMENTAL_LEVELS, groups, data, n_by_group,
                    ylabel="Collisions",
                    title="Collisions per level by experimental group")
    save(fig, out_dir, "perf_collisions_by_level.png")


def plot_avg_runs_per_level(perf_df, groups, n_by_group, out_dir, half=False):
    """
    Box plot of each participant's average code runs (run_simulation events) per
    experimental level - total runs divided by the number of levels they reached,
    so participants who timed out are not penalized for reaching fewer levels - by
    group. Sized for one IEEE conference column with 10 pt Times text. Written as
    PNG and EPS. half=True writes a half-column version (for two panels side by
    side in one column), still at 10 pt.
    """
    per_participant = perf_df.groupby(["group", "participant"]).attempts.mean()

    with plt.rc_context(IEEE_FONT_RC):
        # heights match completion_time_by_participant(_half)
        fig, ax = plt.subplots(figsize=(SIDE_BY_SIDE_ATTEMPTS_WIDTH_IN, 2.6) if half
                               else (IEEE_COLUMN_WIDTH_IN, 2.5))
        vals = [list(per_participant[g]) if g in per_participant.index.get_level_values(0) else []
                for g in groups]
        bp = ax.boxplot(vals, positions=range(len(groups)), widths=0.55 if half else 0.4,
                        showfliers=False,
                        patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                        boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                        capprops={"linewidth": 0.6})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(COLORS.get(g))
        ax.set_xticks(range(len(groups)))
        # in the narrow version, break the group name after spaces and "+" so labels fit
        names = [display_name(g).replace(" ", "\n").replace("+", "+\n") if half else display_name(g)
                 for g in groups]
        ax.set_xticklabels([f"{name}\n($n={len(v)}$)" for name, v in zip(names, vals)])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5 if half else -0.6, len(groups) - (0.5 if half else 0.4))
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Avg. runs per level")
        ax.set_facecolor("#F2F2F2")
        ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.3)
        for ext in ("png", "eps"):
            out_path = out_dir / f"perf_avg_runs_per_level_by_group{'_half' if half else ''}.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


# Task dimensions (task type) and the levels in each, easy then hard
TASK_DIMENSIONS = [("Authoring", ["level_001", "level_002"]),
                   ("Debugging/Editing", ["level_003", "level_004"])]


def level_x_positions(gap=0.6):
    """x position of each level, with a gap between task dimensions."""
    xs, x = {}, 0.0
    for _, levels in TASK_DIMENSIONS:
        for level in levels:
            xs[level] = x
            x += 1
        x += gap
    return xs


# Short dimension names for narrow panels
SHORT_DIMENSION_NAMES = {"Authoring": "Auth.", "Debugging/Editing": "Debug."}


def label_dimension_axis(ax, xs, short=False):
    """
    Easy/Hard tick labels under each level and the dimension name under each pair.
    short=True uses E/H and abbreviated dimension names, for narrow panels.
    """
    ax.set_xticks([xs[l] for l in EXPERIMENTAL_LEVELS])
    difficulties = [LEVEL_DESCRIPTIONS[l].split(" ", 1)[0] for l in EXPERIMENTAL_LEVELS]
    ax.set_xticklabels([d[0] for d in difficulties] if short else difficulties)
    ax.tick_params(axis="x", length=0, pad=2)
    for name, levels in TASK_DIMENSIONS:
        center = (xs[levels[0]] + xs[levels[-1]]) / 2
        ax.annotate(SHORT_DIMENSION_NAMES.get(name, name) if short else name, (center, 0),
                    xycoords=("data", "axes fraction"), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top")


def style_paper_axes(ax, grid_axis="y"):
    ax.set_facecolor("#F2F2F2")
    ax.grid(axis=grid_axis, color="#8C8C8C", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_successful_participants_by_task(perf_df, groups, n_by_group, out_dir):
    """
    Share of each group who completed each task, in the same layout as the other
    per-level figures (one panel per level, task headers on top, shared y-scale,
    legend below). Each bar is 100% of the group: completers in the group color and
    everyone else (timed out on the level or never reached it) stacked in gray,
    with the number of participants written on each segment. Percent rather than
    counts because the groups differ in size. Sized for one IEEE column, 10 pt.
    """
    completed = {(g, l): int(perf_df[(perf_df.group == g) & (perf_df.level == l)].completed.sum())
                 for g in groups for l in EXPERIMENTAL_LEVELS}
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(1, len(EXPERIMENTAL_LEVELS), figsize=(IEEE_COLUMN_WIDTH_IN, 2.5),
                                 sharey=True)
        for i, (ax, level) in enumerate(zip(axes, EXPERIMENTAL_LEVELS)):
            for x, g in enumerate(groups):
                done, n = completed[(g, level)], n_by_group[g]
                done_pct = 100 * done / n
                ax.bar(x, done_pct, 0.6, color=COLORS.get(g), edgecolor="black", linewidth=0.4)
                ax.bar(x, 100 - done_pct, 0.6, bottom=done_pct, color="#D9D9D9",
                       edgecolor="black", linewidth=0.4)
                # participant counts on each segment
                for count, bottom, height in ((done, 0, done_pct), (n - done, done_pct, 100 - done_pct)):
                    if count:
                        ax.text(x, bottom + height / 2, str(count), ha="center", va="center")
            ax.set_xticks([])
            ax.set_xlim(-0.5, len(groups) - 0.5)
            difficulty = LEVEL_DESCRIPTIONS[level].split(" ", 1)[0]
            ax.set_title(f"L{i + 1} \u00b7 {difficulty}", fontsize=10, pad=3)
            style_paper_axes(ax)
        axes[0].set_ylim(0, 101)
        axes[0].yaxis.set_major_locator(MultipleLocator(25))
        axes[0].set_ylabel("Participants (%)")
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.4,
                         label=f"{display_name(g)} completed") for g in groups]
        handles.append(Patch(facecolor="#D9D9D9", edgecolor="black", linewidth=0.4, label="Did not complete"))
        # legend fills column-first: group entries in the left column
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handlelength=1.4,
                   handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1, labelspacing=0.2)
        fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 0.9), w_pad=0.4)
        add_task_headers(fig, axes)
        save_paper_figure(fig, out_dir, "perf_successful_participants_by_task")


def plot_movement_blocks_by_dimension(perf_df, groups, out_dir):
    """
    Horizontal box plots of the number of movement blocks (move forward and turns)
    in each successful program, per task dimension and group. Whiskers span the
    minimum to maximum and a white diamond marks the mean. Completed levels only.
    Sized for one IEEE column, 10 pt.
    """
    completed = perf_df[perf_df.completed & perf_df.movement_blocks.notna()]
    with plt.rc_context(IEEE_FONT_RC):
        fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 1.9))
        centers = []
        for d, (name, levels) in enumerate(TASK_DIMENSIONS):
            center = -d * 1.3  # first dimension on top
            centers.append(center)
            for i, g in enumerate(groups):
                y = center + (0.5 - i) * 0.42
                vals = completed[(completed.group == g) & completed.level.isin(levels)].movement_blocks
                bp = ax.boxplot([vals], positions=[y], widths=0.36, vert=False, whis=(0, 100),
                                showmeans=True, patch_artist=True,
                                meanprops={"marker": "D", "markerfacecolor": "white",
                                           "markeredgecolor": "black", "markersize": 3.5},
                                medianprops={"color": "black", "linewidth": 1},
                                boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                                capprops={"linewidth": 0.6})
                bp["boxes"][0].set_facecolor(COLORS.get(g))
        ax.set_yticks(centers)
        ax.set_yticklabels([name.replace("/", "/\n") for name, _ in TASK_DIMENSIONS])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(left=0)
        ax.set_xlabel("Movement blocks in successful program")
        style_paper_axes(ax, grid_axis="x")
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.6, label=display_name(g))
                   for g in groups]
        handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                              markeredgecolor="black", markersize=3.5, label="Mean"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
        save_paper_figure(fig, out_dir, "perf_movement_blocks_by_dimension")


def plot_success_rate_by_condition(perf_df, groups, n_by_group, out_dir):
    """
    Success rate (% of the group who completed the level) on each task, one panel
    per condition, sharing one y-scale. Sized for one IEEE column, 10 pt.
    """
    xs = level_x_positions()
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(1, len(groups), figsize=(IEEE_COLUMN_WIDTH_IN, 1.9), sharey=True)
        for ax, g in zip(axes, groups):
            rates = [100 * perf_df[(perf_df.group == g) & (perf_df.level == l)].completed.sum() / n_by_group[g]
                     for l in EXPERIMENTAL_LEVELS]
            ax.bar([xs[l] for l in EXPERIMENTAL_LEVELS], rates, 0.7, color=COLORS.get(g),
                   edgecolor="black", linewidth=0.4)
            label_dimension_axis(ax, xs, short=True)
            ax.set_xlim(min(xs.values()) - 0.55, max(xs.values()) + 0.55)
            ax.set_title(f"{display_name(g)} ($n={n_by_group[g]}$)", fontsize=10, pad=3)
            style_paper_axes(ax)
        axes[0].set_ylim(0, 100)
        axes[0].yaxis.set_major_locator(MultipleLocator(25))
        axes[0].set_ylabel("Success rate (%)")
        fig.tight_layout(pad=0.3, w_pad=0.6)
        save_paper_figure(fig, out_dir, "perf_success_rate_by_condition")


# Panels of the compact completion + runs compound figure: 0.69 + 0.29 of the column
COMPLETION_PANEL_WIDTH_IN = 0.69 * IEEE_COLUMN_WIDTH_IN
RUNS_PANEL_WIDTH_IN = 0.29 * IEEE_COLUMN_WIDTH_IN
COMPOUND_PANEL_HEIGHT_IN = 2.4


def plot_completion_and_runs_compound(perf_df, groups, n_by_group, out_dir):
    """
    Compact single-column compound figure, written as two panels for LaTeX
    subcaptions, side by side at their natural size (10 pt text):
    (a) share of each group who completed each level (as in
        perf_successful_participants_by_task, narrowed: two-line level titles,
        shorter task headers and legend), and
    (b) each participant's average runs until success - their runs until success
        averaged over the levels they completed (participants who completed no
        level are left out) - by group.
    Both panels are the same height and their plot areas line up.
    """
    completed = {(g, l): int(perf_df[(perf_df.group == g) & (perf_df.level == l)].completed.sum())
                 for g in groups for l in EXPERIMENTAL_LEVELS}
    runs_until_success = perf_df[perf_df.completed].groupby(["group", "participant"]).attempts.mean()
    with plt.rc_context(IEEE_FONT_RC):
        fig_a, axes = plt.subplots(1, len(EXPERIMENTAL_LEVELS), sharey=True,
                                   figsize=(COMPLETION_PANEL_WIDTH_IN, COMPOUND_PANEL_HEIGHT_IN))
        fig_b, ax_b = plt.subplots(figsize=(RUNS_PANEL_WIDTH_IN, COMPOUND_PANEL_HEIGHT_IN))

        # (a) completion by level
        for i, (ax, level) in enumerate(zip(axes, EXPERIMENTAL_LEVELS)):
            for x, g in enumerate(groups):
                done, n = completed[(g, level)], n_by_group[g]
                done_pct = 100 * done / n
                ax.bar(x, done_pct, 0.7, color=COLORS.get(g), edgecolor="black", linewidth=0.4)
                ax.bar(x, 100 - done_pct, 0.7, bottom=done_pct, color="#D9D9D9",
                       edgecolor="black", linewidth=0.4)
                for count, bottom, height in ((done, 0, done_pct), (n - done, done_pct, 100 - done_pct)):
                    if count:
                        ax.text(x, bottom + height / 2, str(count), ha="center", va="center")
            ax.set_xticks([])
            ax.set_xlim(-0.5, len(groups) - 0.5)
            difficulty = LEVEL_DESCRIPTIONS[level].split(" ", 1)[0]
            ax.set_title(f"L{i + 1}\n{difficulty}", fontsize=10, pad=3, linespacing=1.0)
            style_paper_axes(ax)
        axes[0].set_ylim(0, 101)
        axes[0].yaxis.set_major_locator(MultipleLocator(25))
        axes[0].set_ylabel("Participants (%)")
        handles = [Patch(facecolor=COLORS.get(g), edgecolor="black", linewidth=0.4, label=display_name(g))
                   for g in groups]
        handles.append(Patch(facecolor="#D9D9D9", edgecolor="black", linewidth=0.4, label="Did not complete"))
        fig_a.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handlelength=1.0,
                     handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1, labelspacing=0.2)
        bottom_in, top_in = 0.4, 0.24  # room for the legend below and the task headers above
        fig_a.tight_layout(pad=0.3, w_pad=0.15, rect=(0, bottom_in / COMPOUND_PANEL_HEIGHT_IN, 1,
                                                       1 - top_in / COMPOUND_PANEL_HEIGHT_IN))

        # (b) average runs until success
        vals = [list(runs_until_success[g]) if g in runs_until_success.index.get_level_values(0) else []
                for g in groups]
        bp = ax_b.boxplot(vals, positions=range(len(groups)), widths=0.6, showfliers=False,
                          patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                          boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                          capprops={"linewidth": 0.6})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(COLORS.get(g))
        ax_b.set_xticks([])
        ax_b.set_xlim(-0.5, len(groups) - 0.5)
        ax_b.set_ylim(bottom=0)
        ax_b.yaxis.set_major_locator(MultipleLocator(2))
        ax_b.set_ylabel("Avg. runs until\nsuccess")
        style_paper_axes(ax_b)
        fig_b.tight_layout(pad=0.3)
        # line the plot areas up: (b) takes (a)'s plot-area bottom and top
        a_pos, b_pos = axes[0].get_position(), ax_b.get_position()
        ax_b.set_position([b_pos.x0, a_pos.y0, b_pos.width, a_pos.height])

        add_task_headers(fig_a, axes, labels={"code authoring": "Authoring",
                                              "code completion": "Debug./Editing"})
        print(f"  runs until success n per group: {dict(zip(groups, map(len, vals)))}")
        save_paper_figure(fig_a, out_dir, "perf_completion_and_runs_a")
        save_paper_figure(fig_b, out_dir, "perf_completion_and_runs_b")


def save_paper_figure(fig, out_dir, name):
    for ext in ("png", "eps"):
        out_path = out_dir / f"{name}.{ext}"
        fig.savefig(out_path, dpi=300)
        print(f"  -> wrote {out_path}")
    plt.close(fig)


def plot_completion_rate_per_participant(perf_df, groups, n_by_group, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    per_participant = (
        perf_df.groupby(["group", "participant"])["completed"]
        .agg(lambda s: s.sum() / len(EXPERIMENTAL_LEVELS))
        .reset_index(name="completion_rate")
    )
    data = {g: {"Overall": list(per_participant[per_participant.group == g].completion_rate)} for g in groups}
    grouped_boxplot(ax, ["Overall"], groups, data, n_by_group,
                    ylabel="Fraction of 4 experimental levels completed",
                    title="Per-participant overall completion rate by group")
    ax.set_ylim(0, 1.15)
    save(fig, out_dir, "perf_completion_rate_per_participant.png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_completion_time_table(perf_df, groups, out_dir):
    """
    Average time (minutes) to complete each level per group, among participants who
    completed that level. Written as a LaTeX (booktabs) table sized for one IEEE
    conference column, and a CSV with the per-cell n.
    """
    completed = perf_df[perf_df.completed & perf_df.time_spent_seconds.notna()].copy()
    completed["minutes"] = completed.time_spent_seconds / 60

    rows = []
    for g in groups:
        row = {"group": g}
        for level in EXPERIMENTAL_LEVELS:
            vals = completed[(completed.group == g) & (completed.level == level)].minutes
            row[level] = vals.mean() if len(vals) else np.nan
            row[f"{level}_n"] = len(vals)
        rows.append(row)
    table = pd.DataFrame(rows)
    csv_path = out_dir / "table_completion_time_by_level.csv"
    table.round(2).to_csv(csv_path, index=False)
    print(f"  -> wrote {csv_path}")

    def fmt(v):
        return "--" if pd.isna(v) else f"{v:.2f}"

    def tex_escape(text):
        return text.replace("_", "\\_")

    # Stack "Level N / Easy / auth." in the header so the table fits a 3.5 in column
    short = {"code authoring": "auth.", "code completion": "compl."}
    headers = []
    for i, lvl in enumerate(EXPERIMENTAL_LEVELS):
        difficulty, task = LEVEL_DESCRIPTIONS[lvl].split(" ", 1)
        headers.append(f"\\makecell[r]{{Level {i + 1}\\\\{difficulty}\\\\{short.get(task, task)}}}")
    header_row = " & ".join(headers)
    body = "\n".join(
        f"    {tex_escape(display_name(r['group']))} & "
        + " & ".join(fmt(r[lvl]) for lvl in EXPERIMENTAL_LEVELS) + " \\\\"
        for r in rows)
    n_notes = "; ".join(
        f"{tex_escape(display_name(r['group']))}: "
        + ", ".join(f"L{i + 1} $n={r[f'{lvl}_n']}$" for i, lvl in enumerate(EXPERIMENTAL_LEVELS))
        for r in rows)
    # IEEE style puts table captions above the table
    tex = f"""% Requires \\usepackage{{booktabs}} and \\usepackage{{makecell}}
\\begin{{table}}[t]
  \\centering
  \\caption{{Average completion time for each level (in minutes). Only participants
  who completed the level were considered. Auth.\\ = code authoring;
  compl.\\ = code completion. ({n_notes}.)}}
  \\label{{tab:completion-time-by-level}}
  \\begin{{tabular}}{{l{"r" * len(EXPERIMENTAL_LEVELS)}}}
    \\toprule
     & {header_row} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    tex_path = out_dir / "table_completion_time_by_level.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  -> wrote {tex_path}")

# ---------------------------------------------------------------------------
# Survey plots
# ---------------------------------------------------------------------------

def plot_numeric_questions(survey_df, groups, n_by_group, question_map, out_dir, filename, title):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cats = list(question_map.values())
    data = {g: {} for g in groups}
    for g in groups:
        sub = survey_df[survey_df.group == g]
        for key, label in question_map.items():
            vals = pd.to_numeric(sub[sub.question == key].value, errors="coerce").dropna().tolist()
            data[g][label] = vals
    grouped_boxplot(ax, cats, groups, data, n_by_group,
                    ylabel="Rating (1-5 scale)", title=title)
    ax.set_ylim(0, 5.7)
    save(fig, out_dir, filename)


def plot_self_efficacy_likert(survey_df, groups, n_by_group, out_dir):
    """
    100% stacked horizontal bars of the self-efficacy Likert items: one panel per item,
    one bar per group, one segment per response value (1-5) with its count. Sized
    for one IEEE conference column with 10 pt Times text. Written as PNG and EPS.
    """
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(len(SELF_EFFICACY_QUESTIONS), 1, sharex=True,
                                 figsize=(IEEE_COLUMN_WIDTH_IN, 0.62 * len(SELF_EFFICACY_QUESTIONS) + 0.55))
        for ax, (key, label) in zip(axes, SELF_EFFICACY_QUESTIONS.items()):
            sub = survey_df[survey_df.question == key]
            y = np.arange(len(groups))[::-1]  # first group on top
            for yi, g in zip(y, groups):
                vals = pd.to_numeric(sub[sub.group == g].value, errors="coerce").dropna().astype(int)
                n = len(vals)
                left = 0.0
                for level, color in zip(LIKERT_LEVELS, LIKERT_COLORS):
                    k = int((vals == level).sum())
                    if k == 0 or n == 0:
                        continue
                    width = 100 * k / n
                    ax.barh(yi, width, left=left, height=0.75, color=color,
                            edgecolor="black", linewidth=0.4)
                    text_color = "white" if level >= 4 else "black"
                    ax.text(left + width / 2, yi, str(k), ha="center", va="center", color=text_color)
                    left += width
            ax.set_yticks(y)
            ax.set_yticklabels([f"{display_name(g)} (n={n_by_group[g]})" for g in groups])
            ax.tick_params(axis="y", length=0)
            ax.set_title(label, fontsize=10, loc="left", pad=2)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.55, len(groups) - 0.45)
            ax.spines[["top", "right", "left"]].set_visible(False)
        for ax in axes[:-1]:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", length=0)
        axes[-1].set_xticks(range(0, 101, 25))
        axes[-1].set_xticklabels([f"{v}%" for v in range(0, 101, 25)])
        axes[-1].set_xlabel("Participants")
        handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=str(level))
                   for level, c in zip(LIKERT_LEVELS, LIKERT_COLORS)]
        fig.legend(handles=handles, loc="upper center", ncol=len(LIKERT_LEVELS), frameon=False,
                   handlelength=1.0, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2,
                   title="1 = strongly disagree, 5 = strongly agree", title_fontsize=10)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.86), h_pad=0.3)
        for ext in ("png", "eps"):
            out_path = out_dir / f"survey_self_efficacy_likert.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def write_self_efficacy_table(survey_df, groups, n_by_group, out_dir, stat="mean"):
    """
    Mean or median (stat="mean" / "median") rating (1-5) of each self-efficacy Likert
    item per group. Written as a LaTeX (booktabs) table sized for one IEEE conference
    column, and a CSV with the n.
    """
    rows = []
    for key, label in SELF_EFFICACY_QUESTIONS.items():
        sub = survey_df[survey_df.question == key]
        row = {"item": label}
        for g in groups:
            vals = pd.to_numeric(sub[sub.group == g].value, errors="coerce").dropna()
            row[g] = vals.agg(stat) if len(vals) else np.nan
            row[f"{g}_n"] = len(vals)
        rows.append(row)
    table = pd.DataFrame(rows)
    name = f"table_self_efficacy_{stat}s"
    csv_path = out_dir / f"{name}.csv"
    table.round(2).to_csv(csv_path, index=False)
    print(f"  -> wrote {csv_path}")

    def fmt(v):
        if pd.isna(v):
            return "--"
        if stat == "median":  # medians of 1-5 ratings are whole or half points
            return f"{v:g}"
        return f"{v:.2f}"

    def tex_escape(text):
        return text.replace("_", "\\_")

    header = " & ".join(f"\\makecell{{{tex_escape(display_name(g))}\\\\($n={n_by_group[g]}$)}}" for g in groups)
    body = "\n".join(
        f"    {r['item']} & " + " & ".join(fmt(r[g]) for g in groups) + " \\\\"
        for r in rows)
    # IEEE style puts table captions above the table
    tex = f"""% Requires \\usepackage{{booktabs}} and \\usepackage{{makecell}}
\\begin{{table}}[t]
  \\centering
  \\caption{{{stat.capitalize()} self-efficacy rating for each item by group
  (1~=~strongly disagree, 5~=~strongly agree).}}
  \\label{{tab:self-efficacy-{stat}s}}
  \\begin{{tabular}}{{l{"c" * len(groups)}}}
    \\toprule
    Item & {header} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  -> wrote {tex_path}")


def write_demographics_table(survey_df, out_dir):
    """
    Participant background across all participants: the number and percentage who
    gave each answer. Written as a LaTeX (booktabs) table sized for one IEEE
    conference column, and a CSV.
    """
    # (section label, question key, answer order); answers not listed are appended
    sections = [
        ("Education", "Education Level", ["Some College", "Bachelor's"]),
        ("Programming experience$^{\\dagger}$", PROGRAMMING_EXPERIENCE_QUESTION,
         ["1", "2", "3", "4", "5"]),
        ("Block-based coding use", "Block-based Code_1", ["Never", "Rarely", "Occasionally", "Often"]),
        ("Programmed a robot before", "Robotics Experience", ["Yes", "No"]),
        ("LLM use for problem-solving", "AI Familiarity_1", ["Daily", "Weekly", "Monthly", "Never"]),
    ]
    category_labels = {"Some College": "Some college"}
    # 1-5 rating scales that also get a mean / median row
    LIKERT_SUMMARY_QUESTIONS = {PROGRAMMING_EXPERIENCE_QUESTION}
    n_total = survey_df.participant.nunique()

    rows = []  # (label, is_section, n, percent)
    for label, key, order in sections:
        answers = survey_df[survey_df.question == key].value.astype(str)
        observed = set(answers)
        cats = [c for c in order if c in observed] + sorted(observed - set(order))
        rows.append((label, True, "", ""))
        for c in cats:
            n = int((answers == c).sum())
            rows.append((category_labels.get(c, c), False, str(n), f"{100 * n / n_total:.0f}"))
        if key in LIKERT_SUMMARY_QUESTIONS:
            vals = pd.to_numeric(answers, errors="coerce").dropna()
            rows.append(("Mean / median", False, f"{vals.mean():.2f} / {vals.median():g}", None))

    csv_rows = [{"row": label, "n": n, "percent": pct} for label, _, n, pct in rows]
    csv_path = out_dir / "table_demographics.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  -> wrote {csv_path}")

    def tex_escape(text):
        return text.replace("_", "\\_").replace("&", "\\&")

    def tex_cells(n, pct):
        # A row with no percentage (mean / median) spans both number columns
        return f"\\multicolumn{{2}}{{r}}{{{n}}}" if pct is None else f"{n} & {pct}"

    body = "\n".join(
        f"    {'' if is_section else '\\quad '}{tex_escape(label)} & {tex_cells(n, pct)} \\\\"
        for label, is_section, n, pct in rows)
    # IEEE style puts table captions above the table
    tex = f"""% Requires \\usepackage{{booktabs}}
\\begin{{table}}[t]
  \\centering
  \\caption{{Participant background ($n={n_total}$).}}
  \\label{{tab:demographics}}
  \\begin{{tabular}}{{lrr}}
    \\toprule
     & $n$ & \\% \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}

  \\vspace{{2pt}}
  \\raggedright $^{{\\dagger}}$Self-rated from 1 (complete beginner) to 5 (expert).
\\end{{table}}
"""
    tex_path = out_dir / "table_demographics.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  -> wrote {tex_path}")


def print_programming_experience(survey_df, groups):
    """Programming experience is a background measure, so it is reported in text, not plotted."""
    sub = survey_df[survey_df.question == PROGRAMMING_EXPERIENCE_QUESTION]
    for g in groups:
        vals = pd.to_numeric(sub[sub.group == g].value, errors="coerce").dropna()
        if len(vals):
            print(f"     programming experience, {g}: median {vals.median():g} "
                  f"(range {vals.min():g}-{vals.max():g}, n={len(vals)})")


def plot_difficulty_ranking(survey_df, groups, n_by_group, out_dir):
    """
    100% stacked horizontal bars, one per group: the share of participants who ranked
    each programming type as the hardest (rank 1, assuming the survey listed hardest
    first), with counts in the segments. Compact strip sized for one IEEE conference
    column with 10 pt Times text. Written as PNG and EPS.
    """
    cats = list(DIFFICULTY_QUESTIONS.items())
    # Two shades from the Likert palette, so segments never read as group colors
    seg_colors = [LIKERT_COLORS[4], LIKERT_COLORS[1]]
    text_colors = ["white", "black"]
    with plt.rc_context(IEEE_FONT_RC):
        fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 1.25))
        y = np.arange(len(groups))[::-1]  # first group on top
        for yi, g in zip(y, groups):
            sub = survey_df[survey_df.group == g]
            n = sub[sub.question.isin(DIFFICULTY_QUESTIONS)].participant.nunique()
            left = 0.0
            for (key, _), color, text_color in zip(cats, seg_colors, text_colors):
                k = int((pd.to_numeric(sub[sub.question == key].value, errors="coerce") == 1).sum())
                if k == 0 or n == 0:
                    continue
                width = 100 * k / n
                ax.barh(yi, width, left=left, height=0.7, color=color, edgecolor="black",
                        linewidth=0.4)
                ax.text(left + width / 2, yi, str(k), ha="center", va="center", color=text_color)
                left += width
        ax.set_yticks(y)
        ax.set_yticklabels([f"{display_name(g)} ($n={n_by_group[g]}$)" for g in groups])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(groups) - 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.spines[["top", "right", "left"]].set_visible(False)
        handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=f"{label} hardest")
                   for (_, label), c in zip(cats, seg_colors)]
        fig.legend(handles=handles, loc="upper center", ncol=len(cats), frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2)
        fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.8))
        for ext in ("png", "eps"):
            out_path = out_dir / f"survey_difficulty_ranking.{ext}"
            fig.savefig(out_path, dpi=300)
            print(f"  -> wrote {out_path}")
        plt.close(fig)


def plot_categorical_breakdown(survey_df, groups, n_by_group, out_dir):
    categorical_cols = ["Education Level", "Block-based Code_1", "Robotics Experience", "AI Familiarity_1"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, categorical_cols):
        sub = survey_df[survey_df.question == col]
        counts = sub.groupby(["group", "value"]).size().unstack(fill_value=0)
        counts = counts.reindex(groups)
        cats = counts.columns.tolist()
        x = np.arange(len(cats))
        width = 0.8 / len(groups)
        for gi, g in enumerate(groups):
            xpos = x + (gi - (len(groups) - 1) / 2) * width
            ax.bar(xpos, counts.loc[g], width=width * 0.9,
                   label=group_label(g, n_by_group), color=COLORS.get(g))
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel("Participants")
        ax.set_title(col)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend()
    fig.suptitle("Participant background by experimental group", fontsize=13)
    save(fig, out_dir, "survey_background_breakdown.png")


def main():
    parser = argparse.ArgumentParser(description="Generate performance and survey comparison plots.")
    parser.add_argument("--combined-dir", default=REPO_ROOT / "data" / "deidentified",
                         help="Directory of de-identified participant JSON files (default: data/deidentified).")
    parser.add_argument("--out", default=REPO_ROOT / "analysis_plots",
                         help="Output directory for plots (default: analysis_plots).")
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    out_dir = Path(args.out)
    if not combined_dir.is_dir():
        print(f"Error: combined dataset directory not found: {combined_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

    participants = load_participants(combined_dir)
    if not participants:
        print("No participant files found.", file=sys.stderr)
        sys.exit(1)

    # Keep only real participants: those with a matched exit survey (the survey is
    # matched by Prolific ID). Drops test/abandoned sessions; 03_deidentify.py
    # has already dropped these from the released data.
    excluded = [p.get("participantId") for p in participants if not p.get("exitSurvey")]
    participants = [p for p in participants if p.get("exitSurvey")]
    print(f"Included {len(participants)} participants; excluded {len(excluded)} "
          f"without an exit survey: {excluded}")

    groups = sorted({p.get("experimentalGroup") for p in participants if p.get("experimentalGroup")})
    n_by_group = {g: sum(1 for p in participants if p.get("experimentalGroup") == g) for g in groups}
    print(f"Groups: {n_by_group}")

    perf_df = build_performance_df(participants)
    survey_df = build_survey_df(participants)

    print("Generating performance plots...")
    plot_success_rate_by_level(perf_df, groups, n_by_group, out_dir)
    plot_completion_by_level(perf_df, groups, n_by_group, out_dir)
    plot_time_spent_by_level(perf_df, groups, n_by_group, out_dir)
    plot_attempts_until_success_by_level(perf_df, groups, n_by_group, out_dir)
    plot_time_and_attempts_grid(perf_df, groups, out_dir)
    plot_time_to_success_km(perf_df, groups, out_dir)
    plot_runs_until_success_by_level(perf_df, groups, out_dir)
    plot_successful_participants_by_task(perf_df, groups, n_by_group, out_dir)
    plot_movement_blocks_by_dimension(perf_df, groups, out_dir)
    plot_success_rate_by_condition(perf_df, groups, n_by_group, out_dir)
    plot_completion_and_runs_compound(perf_df, groups, n_by_group, out_dir)
    plot_attempts_by_level(perf_df, groups, n_by_group, out_dir)
    plot_collisions_by_level(perf_df, groups, n_by_group, out_dir)
    plot_completion_rate_per_participant(perf_df, groups, n_by_group, out_dir)
    plot_avg_runs_per_level(perf_df, groups, n_by_group, out_dir)
    plot_avg_runs_per_level(perf_df, groups, n_by_group, out_dir, half=True)
    write_completion_time_table(perf_df, groups, out_dir)

    print("Generating survey plots...")
    plot_self_efficacy_likert(survey_df, groups, n_by_group, out_dir)
    write_self_efficacy_table(survey_df, groups, n_by_group, out_dir, stat="mean")
    write_self_efficacy_table(survey_df, groups, n_by_group, out_dir, stat="median")
    print_programming_experience(survey_df, groups)
    write_demographics_table(survey_df, out_dir)
    ai_groups = [g for g in groups if g != "control"] or groups
    ai_n_by_group = {g: n_by_group[g] for g in ai_groups}
    plot_numeric_questions(survey_df, ai_groups, ai_n_by_group, AI_PARTNER_QUESTIONS, out_dir,
                            "survey_ai_partner_ratings.png",
                            "AI partner ratings (AI-assisted group only)")
    plot_difficulty_ranking(survey_df, groups, n_by_group, out_dir)
    plot_categorical_breakdown(survey_df, groups, n_by_group, out_dir)

    print(f"\nDone. Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
