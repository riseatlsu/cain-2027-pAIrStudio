#!/usr/bin/env python3
"""
Reproduce the figures, table and statistics reported in the pAIrStudio paper.

Reads the de-identified participant files in data/deidentified/ (P01.json ...
P12.json) and writes:

  figs/
    perf_completion_and_runs_a.{eps,png}   Fig. 2a  completion by task
    perf_completion_and_runs_b.{eps,png}   Fig. 2b  average runs until success
    perf_level_time_by_participant.{eps,png}  Fig. 3  time to success per participant
  stats/
    demographics.csv                 Sec. V-A   participant background
    completion_by_task.csv           Fig. 2a    participants who completed each task
    runs_until_success.csv           Fig. 2b    per-participant runs until success, and group medians
    logrank_time_and_runs_by_task.csv  Sec. V   log-rank tests of time and of runs until success
    ai_chats_by_task.csv             Sec. V     messages sent to Otto, per task
    self_efficacy_means.csv          Table I    mean self-efficacy rating per item and group
    likert_mann_whitney.csv          Sec. V     Mann-Whitney U tests of the self-efficacy items
    software_versions.csv                       Python and package versions used

Usage:
    python scripts/04_analysis.py [--data-dir data/deidentified] [--figs-dir figs] [--stats-dir stats]
"""

import argparse
import itertools
import json
import os
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

# Fixed EPS creation date, so rerunning the script gives identical files
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sksurv.compare import compare_survival
from sksurv.util import Surv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Groups: internal ids in the data, names used in the paper
CONTROL, AI_GROUP = "control", "standard_ai"
GROUP_DISPLAY_NAMES = {CONTROL: "Solo", AI_GROUP: "Human+AI"}
COLORS = {CONTROL: "#4C72B0", AI_GROUP: "#DD8452"}

EXPERIMENTAL_LEVELS = ["level_001", "level_002", "level_003", "level_004"]
TASK_IDS = {level: f"T{i + 1}" for i, level in enumerate(EXPERIMENTAL_LEVELS)}
LEVEL_DIFFICULTY = {"level_001": "Easy", "level_002": "Hard", "level_003": "Easy", "level_004": "Hard"}
LEVEL_TASK_TYPE = {"level_001": "Authoring", "level_002": "Authoring",
                   "level_003": "Debug./Editing", "level_004": "Debug./Editing"}
PANEL_TITLES = {
    "level_001": "T1 · Authoring, easy",
    "level_002": "T2 · Authoring, hard",
    "level_003": "T3 · Debugging, easy",
    "level_004": "T4 · Debugging, hard",
}

# Exit survey items (keys of exitSurvey.answers)
PROGRAMMING_EXPERIENCE_QUESTION = "General Programming_1"  # 1 = complete beginner, 5 = expert
SELF_EFFICACY_QUESTIONS = {  # 1 = strongly disagree, 5 = strongly agree
    "QID10_1": "Felt capable of completing the tasks",
    "QID10_2": "Confident explaining my code logic",
    "QID10_3": "Felt in control of the robot",
    "QID10_4": "Confident on a harder future task",
}

# Figures: one IEEE column, 10 pt Times to match the body text
IEEE_COLUMN_WIDTH_IN = 3.5
IEEE_FONT_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                "mathtext.fontset": "stix", "font.size": 10, "ps.fonttype": 42}

FIGS_DIR = REPO_ROOT / "figs"
STATS_DIR = REPO_ROOT / "stats"


def display_name(group):
    return GROUP_DISPLAY_NAMES.get(group, group)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_participants(data_dir):
    """One dict per participant file (files starting with "_" are combined copies)."""
    participants = []
    for f in sorted(data_dir.glob("*.json")):
        if not f.name.startswith("_"):
            participants.append(json.loads(f.read_text(encoding="utf-8")))
    return participants


def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_performance_df(participants):
    """
    One row per (participant, task reached): code runs ("attempts"), whether the
    task was completed, whether the session time limit stopped the participant on
    it, and the wall-clock time on the task, from its first event until the
    participant moved on (successful completion, or the session time limit).
    Tasks a participant never reached have no row.
    """
    rows = []
    for p in participants:
        by_level = {}
        for e in p.get("__subcollections__", {}).get("events", {}).values():
            level = e.get("currentLevel")
            if not e.get("isExperiment") or level not in EXPERIMENTAL_LEVELS:
                continue
            bucket = by_level.setdefault(level, {"runs": 0, "complete": [], "time_limit": [], "times": []})
            if e.get("localTime"):
                bucket["times"].append(parse_time(e["localTime"]))
            et = e.get("eventType")
            if et == "run_simulation":
                bucket["runs"] += 1
            elif et == "level_complete":
                bucket["complete"].append(e)
            elif et == "time_limit_reached":
                bucket["time_limit"].append(e)

        for level in EXPERIMENTAL_LEVELS:
            bucket = by_level.get(level)
            if bucket is None:
                continue
            end_event = next((c for c in bucket["complete"] if c.get("success")), None)
            if end_event is None and bucket["time_limit"]:
                end_event = bucket["time_limit"][0]
            if end_event is not None and end_event.get("localTime") and bucket["times"]:
                minutes = (parse_time(end_event["localTime"]) - min(bucket["times"])).total_seconds() / 60
            else:
                minutes = np.nan
            rows.append({"participant": p.get("participantId"), "group": p.get("experimentalGroup"),
                         "level": level, "attempts": bucket["runs"],
                         "completed": any(c.get("success") for c in bucket["complete"]),
                         "timed_out": bool(bucket["time_limit"]), "minutes": minutes})
    return pd.DataFrame(rows)


def build_survey_df(participants):
    """One row per (participant, survey answer)."""
    return pd.DataFrame([{"participant": p.get("participantId"), "group": p.get("experimentalGroup"),
                          "question": key, "value": value}
                         for p in participants
                         for key, value in p["exitSurvey"].get("answers", {}).items()])


def survey_values(survey_df, question, group=None):
    sub = survey_df[survey_df.question == question]
    if group is not None:
        sub = sub[sub.group == group]
    return pd.to_numeric(sub.value, errors="coerce").dropna().to_numpy()


def write_stats_csv(table, name, digits=4):
    """Save one analysis as stats/<name>.csv, with tasks labelled T1-T4 as in the paper."""
    table = table.rename(columns={"level": "task"})
    if "task" in table.columns:
        table["task"] = table["task"].replace(TASK_IDS)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_DIR / f"{name}.csv"
    table.round(digits).to_csv(path, index=False)
    print(f"  -> wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")


def save_paper_figure(fig, name):
    for ext in ("png", "eps"):
        path = FIGS_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  -> wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    plt.close(fig)


def style_paper_axes(ax):
    ax.set_facecolor("#F2F2F2")
    ax.grid(axis="y", color="#8C8C8C", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Demographics (Section V-A)
# ---------------------------------------------------------------------------

def write_demographics(survey_df):
    """Number and percentage of all participants giving each background answer."""
    sections = [  # (label, question key, answer order); answers not listed are appended
        ("Education", "Education Level", ["Some College", "Bachelor's"]),
        ("Programming experience (1-5)", PROGRAMMING_EXPERIENCE_QUESTION, ["1", "2", "3", "4", "5"]),
        ("Block-based coding use", "Block-based Code_1", ["Never", "Rarely", "Occasionally", "Often"]),
        ("Programmed a robot before", "Robotics Experience", ["Yes", "No"]),
        ("LLM use for problem-solving", "AI Familiarity_1", ["Daily", "Weekly", "Monthly", "Never"]),
    ]
    n_total = survey_df.participant.nunique()
    rows = []
    for label, key, order in sections:
        answers = survey_df[survey_df.question == key].value.astype(str)
        observed = set(answers)
        rows.append({"row": label, "n": "", "percent": ""})
        for c in [c for c in order if c in observed] + sorted(observed - set(order)):
            n = int((answers == c).sum())
            rows.append({"row": c, "n": n, "percent": round(100 * n / n_total)})
    exp = survey_values(survey_df, PROGRAMMING_EXPERIENCE_QUESTION)
    rows += [{"row": "Programming experience mean", "n": f"{exp.mean():.2f}", "percent": ""},
             {"row": "Programming experience SD", "n": f"{exp.std(ddof=1):.2f}", "percent": ""}]
    write_stats_csv(pd.DataFrame(rows), "demographics")


# ---------------------------------------------------------------------------
# Fig. 2: completion by task (a) and average runs until success (b)
# ---------------------------------------------------------------------------

COMPLETION_PANEL_WIDTH_IN = 0.69 * IEEE_COLUMN_WIDTH_IN
RUNS_PANEL_WIDTH_IN = 0.29 * IEEE_COLUMN_WIDTH_IN
COMPOUND_PANEL_HEIGHT_IN = 1.85


def add_task_headers(fig, level_axes):
    """Task-type header with a rule underneath, spanning the panels of each task type."""
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    rule_gap, text_gap = 0.035 / fig.get_figheight(), 0.05 / fig.get_figheight()
    types = [LEVEL_TASK_TYPE[lvl] for lvl in EXPERIMENTAL_LEVELS]
    start = 0
    for end in range(1, len(types) + 1):
        if end < len(types) and types[end] == types[start]:
            continue
        x0, x1 = level_axes[start].get_position().x0, level_axes[end - 1].get_position().x1
        title_top = max(to_fig.transform(level_axes[j].title.get_window_extent())[1, 1]
                        for j in range(start, end))
        fig.add_artist(Line2D([x0 + 0.01, x1 - 0.01], [title_top + rule_gap] * 2,
                              transform=fig.transFigure, color="black", linewidth=0.6))
        fig.text((x0 + x1) / 2, title_top + text_gap, types[start], ha="center", va="bottom")
        start = end


def plot_completion_and_runs(perf_df, groups, n_by_group):
    """
    Fig. 2, written as two panels for LaTeX subfloats, both included at natural size:
    (a) share of each group who completed each task (participant counts on the bars;
        participants who timed out on a task or never reached it did not complete it);
    (b) each participant's runs until success averaged over the tasks they completed
        (participants who completed no task are left out), by group. Outliers hidden.
    Both panels are the same height and their plot areas line up.
    """
    completed = {(g, l): int(perf_df[(perf_df.group == g) & (perf_df.level == l)].completed.sum())
                 for g in groups for l in EXPERIMENTAL_LEVELS}
    write_stats_csv(pd.DataFrame([{"level": l, "group": g, "n": n_by_group[g], "completed": completed[(g, l)],
                                   "percent_completed": 100 * completed[(g, l)] / n_by_group[g]}
                                  for l in EXPERIMENTAL_LEVELS for g in groups]), "completion_by_task")

    runs_until_success = (perf_df[perf_df.completed].groupby(["group", "participant"])
                          .attempts.agg(["mean", "count"]).reset_index())
    runs_until_success.columns = ["group", "participant", "avg_runs_until_success", "tasks_completed"]
    runs_until_success["tasks_completed"] = runs_until_success.tasks_completed.astype("Int64")
    summary = runs_until_success.groupby("group").avg_runs_until_success.agg(["count", "median"])
    write_stats_csv(pd.concat([
        runs_until_success,
        pd.DataFrame([{"group": g, "participant": f"median (n={int(summary.loc[g, 'count'])})",
                       "avg_runs_until_success": summary.loc[g, "median"]} for g in groups]),
    ]), "runs_until_success")

    with plt.rc_context(IEEE_FONT_RC):
        fig_a, axes = plt.subplots(1, len(EXPERIMENTAL_LEVELS), sharey=True,
                                   figsize=(COMPLETION_PANEL_WIDTH_IN, COMPOUND_PANEL_HEIGHT_IN))
        fig_b, ax_b = plt.subplots(figsize=(RUNS_PANEL_WIDTH_IN, COMPOUND_PANEL_HEIGHT_IN))

        # (a) completion by task
        for i, (ax, level) in enumerate(zip(axes, EXPERIMENTAL_LEVELS)):
            for x, g in enumerate(groups):
                done, n = completed[(g, level)], n_by_group[g]
                done_pct = 100 * done / n
                ax.bar(x, done_pct, 0.7, color=COLORS[g], edgecolor="black", linewidth=0.4)
                ax.bar(x, 100 - done_pct, 0.7, bottom=done_pct, color="#D9D9D9",
                       edgecolor="black", linewidth=0.4)
                for count, bottom, height in ((done, 0, done_pct), (n - done, done_pct, 100 - done_pct)):
                    if count:
                        ax.text(x, bottom + height / 2, str(count), ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_xlim(-0.5, len(groups) - 0.5)
            ax.set_title(f"L{i + 1}\n{LEVEL_DIFFICULTY[level]}", fontsize=10, pad=3, linespacing=1.0)
            style_paper_axes(ax)
        axes[0].set_ylim(0, 101)
        axes[0].yaxis.set_major_locator(MultipleLocator(25))
        axes[0].set_ylabel("Participants (%)")
        handles = [Patch(facecolor=COLORS[g], edgecolor="black", linewidth=0.4, label=display_name(g))
                   for g in groups]
        handles.append(Patch(facecolor="#D9D9D9", edgecolor="black", linewidth=0.4, label="Did not complete"))
        fig_a.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handlelength=1.0,
                     handletextpad=0.4, columnspacing=1.0, borderaxespad=0.1, labelspacing=0.2)
        bottom_in, top_in = 0.4, 0.24  # room for the legend below and the task headers above
        fig_a.tight_layout(pad=0.3, w_pad=0.15, rect=(0, bottom_in / COMPOUND_PANEL_HEIGHT_IN, 1,
                                                       1 - top_in / COMPOUND_PANEL_HEIGHT_IN))

        # (b) average runs until success
        vals = [runs_until_success[runs_until_success.group == g].avg_runs_until_success.tolist()
                for g in groups]
        bp = ax_b.boxplot(vals, positions=range(len(groups)), widths=0.6, showfliers=False,
                          patch_artist=True, medianprops={"color": "black", "linewidth": 1},
                          boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                          capprops={"linewidth": 0.6})
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(COLORS[g])
        ax_b.set_xticks([])
        ax_b.set_xlim(-0.5, len(groups) - 0.5)
        ax_b.set_ylim(bottom=0)
        ax_b.yaxis.set_major_locator(MultipleLocator(4))
        ax_b.set_ylabel("Runs until\nsuccess")
        style_paper_axes(ax_b)
        fig_b.tight_layout(pad=0.3)
        a_pos, b_pos = axes[0].get_position(), ax_b.get_position()
        ax_b.set_position([b_pos.x0, a_pos.y0, b_pos.width, a_pos.height])

        add_task_headers(fig_a, axes)
        save_paper_figure(fig_a, "perf_completion_and_runs_a")
        save_paper_figure(fig_b, "perf_completion_and_runs_b")


# ---------------------------------------------------------------------------
# Fig. 3: time to success per participant, and log-rank tests
# ---------------------------------------------------------------------------

# 2x2 layout: easy tasks (T1, T3) on the top row, hard tasks (T2, T4) on the bottom
LEVELS_BY_DIFFICULTY = [["level_001", "level_003"], ["level_002", "level_004"]]


def draw_level_bars(ax, df, level, roster, y_max):
    """
    One bar per participant with their minutes on `level`, colored by group, left
    to right: completers (ascending), then participants stopped by the session time
    limit on this task (bar = time until the limit; black dot above), then
    participants who never reached the task (x on the axis).
    """
    reached = df[df.level == level].set_index("participant")
    slots = []  # (sort key, group, minutes or None, timed out)
    for pid, g in roster:
        if pid not in reached.index:
            slots.append(((2, 0), g, None, False))
        else:
            row = reached.loc[pid]
            slots.append(((0 if row.completed else 1, row.minutes), g, row.minutes, not row.completed))
    slots.sort(key=lambda s: s[0])
    dot_y = y_max * 1.08
    for xi, (_, g, value, timed_out) in enumerate(slots):
        if value is None:
            ax.plot(xi, 0, marker="x", color=COLORS[g], markersize=4, markeredgewidth=1.2,
                    clip_on=False, zorder=5)
            continue
        ax.bar(xi, value, width=0.8, color=COLORS[g], edgecolor="black", linewidth=0.4)
        if timed_out:
            ax.scatter(xi, dot_y, color="black", s=8, zorder=5)
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(slots) - 0.4)
    ax.set_ylim(0, dot_y * 1.06)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3))
    style_paper_axes(ax)


def plot_level_time_by_participant(perf_df, roster, groups):
    """Fig. 3: draw_level_bars for the four tasks, one shared minute scale per row."""
    df = perf_df[perf_df.minutes.notna()]
    with plt.rc_context(IEEE_FONT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(IEEE_COLUMN_WIDTH_IN, 2.55), sharey="row")
        for row_axes, row_levels in zip(axes, LEVELS_BY_DIFFICULTY):
            row_max = df[df.level.isin(row_levels)].minutes.max()
            for ax, level in zip(row_axes, row_levels):
                draw_level_bars(ax, df, level, roster, y_max=row_max)
                ax.set_title(PANEL_TITLES[level], fontsize=10, pad=2)
        fig.supylabel("Minutes on task", fontsize=10, x=0.0, ha="left")
        handles = [Patch(facecolor=COLORS[g], edgecolor="black", linewidth=0.4, label=display_name(g))
                   for g in groups]
        handles += [Line2D([], [], marker="o", linestyle="", color="black", markersize=3, label="Timed out"),
                    Line2D([], [], marker="x", linestyle="", color="#595959", markersize=5,
                           markeredgewidth=1.2, label="Not reached")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                   handlelength=0.9, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.05)
        fig.tight_layout(pad=0.2, rect=(0.02, 0, 1, 0.92), h_pad=0.3, w_pad=0.4)
        save_paper_figure(fig, "perf_level_time_by_participant")


def logrank_by_task(perf_df):
    """
    Survival analysis of time to first success (minutes on the task) and of runs
    until first success on each task, Solo vs. Human+AI: log-rank test
    (scikit-survival compare_survival). Participants stopped by the session time
    limit are right-censored; participants who never reached a task are left out.
    """
    df = perf_df[perf_df.minutes.notna()]
    rows = []
    for level in EXPERIMENTAL_LEVELS:
        sub = df[df.level == level]
        for label, col in (("time", "minutes"), ("runs", "attempts")):
            y = Surv.from_arrays(event=sub.completed.to_numpy(), time=sub[col].to_numpy(dtype=float))
            chi2, p = compare_survival(y, sub.group.to_numpy())
            rows.append({"level": level, "measure": label, "n": len(sub),
                         "df": sub.group.nunique() - 1, "chi2": chi2, "p": p})
            print(f"     log-rank {TASK_IDS[level]} {label}: chi2 = {chi2:.2f}, p = {p:.3f}")
    write_stats_csv(pd.DataFrame(rows), "logrank_time_and_runs_by_task")


# ---------------------------------------------------------------------------
# Use of Otto (Human+AI group)
# ---------------------------------------------------------------------------

def write_ai_chats_by_task(participants, perf_df):
    """
    Messages Human+AI participants sent to Otto (chat_message events with role
    "user") on the tutorial and each task: participants who reached it, how many
    sent at least one message, and messages sent (total, mean, median, max).
    The last row covers all four experimental tasks per participant.
    """
    reached = set(zip(perf_df.participant, perf_df.level))
    per_level = {lvl: [] for lvl in ["tutorial"] + EXPERIMENTAL_LEVELS}
    totals = []
    for p in participants:
        if p.get("experimentalGroup") != AI_GROUP:
            continue
        pid = p.get("participantId")
        sent = {}
        for e in p.get("__subcollections__", {}).get("events", {}).values():
            if e.get("eventType") == "chat_message" and e.get("role") == "user":
                level = e.get("levelId") if e.get("isExperiment") else "tutorial"
                sent[level] = sent.get(level, 0) + 1
        for level in per_level:
            if level == "tutorial" or (pid, level) in reached:
                per_level[level].append(sent.get(level, 0))
        totals.append(sum(sent.get(lvl, 0) for lvl in EXPERIMENTAL_LEVELS))

    def summary(label, msgs):
        msgs = pd.Series(msgs, dtype=float)
        return {"level": label, "reached": len(msgs), "used_ai": int((msgs > 0).sum()),
                "messages_total": int(msgs.sum()), "messages_mean": msgs.mean(),
                "messages_median": msgs.median(), "messages_max": msgs.max()}

    rows = [summary(lvl, m) for lvl, m in per_level.items()] + [summary("all experimental tasks", totals)]
    write_stats_csv(pd.DataFrame(rows), "ai_chats_by_task")


# ---------------------------------------------------------------------------
# Self-efficacy (Table I) and Mann-Whitney U tests
# ---------------------------------------------------------------------------

def write_self_efficacy_means(survey_df, groups):
    rows = []
    for key, label in SELF_EFFICACY_QUESTIONS.items():
        row = {"item": label}
        for g in groups:
            vals = survey_values(survey_df, key, g)
            row[g], row[f"{g}_n"] = vals.mean(), len(vals)
        rows.append(row)
    write_stats_csv(pd.DataFrame(rows), "self_efficacy_means", digits=2)


def exact_mann_whitney(x, y):
    """
    Two-sided Mann-Whitney U test with an exact permutation p-value: U of x (with
    midranks for ties) is recomputed for every split of the pooled ratings into
    groups of len(x) and len(y), so ties need no normal approximation.
    """
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled)
    n1 = len(x)
    offset = n1 * (n1 + 1) / 2
    u = ranks[:n1].sum() - offset
    null = np.array([ranks[list(idx)].sum() - offset
                     for idx in itertools.combinations(range(len(pooled)), n1)])
    mu = n1 * len(y) / 2
    return u, np.mean(np.abs(null - mu) >= abs(u - mu) - 1e-9)


def test_self_efficacy(survey_df):
    rows = []
    for q, label in SELF_EFFICACY_QUESTIONS.items():
        solo, ai = survey_values(survey_df, q, CONTROL), survey_values(survey_df, q, AI_GROUP)
        u, p = exact_mann_whitney(ai, solo)
        rows.append({"question": q, "item": label, f"{CONTROL}_n": len(solo), f"{AI_GROUP}_n": len(ai),
                     f"{CONTROL}_mean": solo.mean(), f"{AI_GROUP}_mean": ai.mean(),
                     f"U_{AI_GROUP}": u, "p_exact": p})
        print(f"     Mann-Whitney {label}: U = {u:g}, p = {p:.3f}")
    write_stats_csv(pd.DataFrame(rows), "likert_mann_whitney")


def write_software_versions():
    packages = ["numpy", "pandas", "scipy", "matplotlib", "scikit-survival"]
    write_stats_csv(pd.DataFrame([{"package": "Python", "version": platform.python_version()}]
                                 + [{"package": pkg, "version": version(pkg)} for pkg in packages]),
                    "software_versions")


def main():
    global FIGS_DIR, STATS_DIR
    parser = argparse.ArgumentParser(description="Reproduce the figures and statistics in the paper.")
    parser.add_argument("--data-dir", default=REPO_ROOT / "data" / "deidentified", type=Path,
                        help="De-identified participant files (default: data/deidentified).")
    parser.add_argument("--figs-dir", default=FIGS_DIR, type=Path, help="Figure output (default: figs).")
    parser.add_argument("--stats-dir", default=STATS_DIR, type=Path, help="Statistics output (default: stats).")
    args = parser.parse_args()
    FIGS_DIR, STATS_DIR = args.figs_dir.resolve(), args.stats_dir.resolve()
    if not args.data_dir.is_dir():
        sys.exit(f"Error: data directory not found: {args.data_dir}")
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Only participants with a matched exit survey are analysed (the released data
    # already contains only these 12).
    participants = [p for p in load_participants(args.data_dir) if p.get("exitSurvey")]
    groups = [g for g in (CONTROL, AI_GROUP) if any(p.get("experimentalGroup") == g for p in participants)]
    n_by_group = {g: sum(p.get("experimentalGroup") == g for p in participants) for g in groups}
    print(f"Participants: {n_by_group}")
    roster = [(p.get("participantId"), p.get("experimentalGroup")) for p in participants]
    perf_df = build_performance_df(participants)
    survey_df = build_survey_df(participants)

    print("Demographics")
    write_demographics(survey_df)
    print("Fig. 2: completion and runs until success")
    plot_completion_and_runs(perf_df, groups, n_by_group)
    print("Fig. 3 and survival analysis: time to success per task")
    plot_level_time_by_participant(perf_df, roster, groups)
    logrank_by_task(perf_df)
    print("Use of Otto")
    write_ai_chats_by_task(participants, perf_df)
    print("Table I and Mann-Whitney U tests: self-efficacy")
    write_self_efficacy_means(survey_df, groups)
    test_self_efficacy(survey_df)
    write_software_versions()


if __name__ == "__main__":
    main()
