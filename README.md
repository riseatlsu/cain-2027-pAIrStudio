# pAIrStudio Replication Package

Data and analysis scripts to reproduce the figures, tables and statistical tests
in the pAIrStudio paper. Participants solved block-based robot programming tasks
either alone ("Solo", internal id `control`) or with an AI partner ("Human+AI",
internal id `standard_ai`), then completed an exit survey.

## Repository layout

```
data/
  deidentified/              released data (one JSON file per participant)
    P01.json ... P12.json    Firestore record + event log + exit survey answers
    _combined_dataset.json   all participants in one file, keyed by participant ID
    _survey_question_key.json  survey answer key -> full question text
scripts/
  01_download_firestore.py   download the raw Firestore database      (needs access)
  02_combine_datasets.py     merge Firestore data with the exit survey (needs raw data)
  03_deidentify.py           strip identifiers -> data/deidentified/   (needs raw data)
  04_generate_plots.py       figures, tables and per-level tests
  05_survival_analysis.py    Kaplan-Meier / log-rank analysis of session time
figs/                        figures as they appear in the paper
requirements.txt
```

Running the scripts writes their output to `analysis_plots/` (not tracked).

## Replicating the results

Only the de-identified data is needed. Tested with Python 3.14.6:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/04_generate_plots.py
python scripts/05_survival_analysis.py
```

Both scripts read `data/deidentified/` and write to `analysis_plots/` by default,
and can be run from any directory. Statistical test results (log-rank tests etc.)
are printed to the terminal. The figures used in the paper are copied to `figs/`
for comparison.

## Data and de-identification

The raw data is not included because it contains personal data: Prolific IDs
(in the Firestore records and the survey), and IP addresses and approximate
location in the Qualtrics export. `scripts/03_deidentify.py` produces the
released data from it:

- Keeps only the 12 analysed participants: those with a Prolific ID and a matched
  exit survey. Test and abandoned sessions are dropped.
- Assigns new participant IDs `P01`-`P12` in order of each participant's first
  recorded event, replacing the app's participant ID everywhere (record and
  every event).
- Removes the Firebase user ID, Prolific ID and Qualtrics response IDs, as well
  as the survey response dates. Survey duration, progress and completion are kept.
- The IP address, location, name and email columns of the survey are never copied
  out of the raw CSV (`02_combine_datasets.py` already leaves them out).
- Refuses to write output if any removed identifier still appears anywhere in
  it, and warns about email addresses or Prolific-style IDs in survey answers or
  chat messages.

Event times (`localTime`, ISO 8601 UTC) are kept because the timing analyses
need them. Free-text survey answers and chat messages with the AI partner were
reviewed and contain no identifying information.

### Data format

Each `data/deidentified/P##.json` holds one participant:

- `participantId`, `experimentalGroup` (`control` or `standard_ai`) and
  `lastUpdated`
- `chatConversations` (AI group only): chat history with the AI partner, per level
- `__subcollections__.events`: the event log, keyed by event ID. Each event has
  an `eventType` (e.g. `run_simulation`, `level_complete`, `time_limit_reached`,
  `collision`, `chat_message`), `levelId` (`tutorial_*`, `level_001`-`level_004`),
  `isExperiment`, `localTime`, and type-specific fields such as `codeSnapshot`
  (the Blockly program run) and `runNumber`
- `exitSurvey.answers`: survey answers keyed by Qualtrics column name; see
  `_survey_question_key.json` for the question text. `exitSurvey.meta` holds the
  response duration, progress and completion flag

## Full pipeline (requires raw data access)

These steps rebuild `data/deidentified/` from the source systems and are listed
for transparency. They need the Firebase service account key and the Qualtrics
export, which are not shared.

```
python scripts/01_download_firestore.py --key path/to/serviceAccountKey.json   # -> data/raw/firestore_export/
#   place the Qualtrics CSV export at data/raw/exit_survey.csv
python scripts/02_combine_datasets.py                                          # -> data/combined/
python scripts/03_deidentify.py                                                # -> data/deidentified/
```

`data/raw/` and `data/combined/` contain identifiers and are gitignored.

### 01: Downloading Firestore

Downloads every root collection and every nested subcollection, recursively.
The service account key comes from Firebase Console -> Project Settings ->
Service Accounts -> "Generate new private key" (never commit it; it grants full
admin access to the database). Optional flags: `--out`, `--collections`,
`--combined-only`, `--no-per-doc-files`.

Output in `data/raw/firestore_export/`:
- `<collection>.json`: one file per root collection
- `<collection>/<doc_id>.json`: one file per document, including its
  subcollections nested under a `__subcollections__` key (e.g. each
  participant's `events`)
- `_combined_export.json`: every root collection in one file

Firestore-specific types are converted to JSON-safe representations:
timestamps -> ISO 8601 strings; GeoPoints, document references and bytes ->
`{"__type__": ...}` objects.

### 02: Combining with the exit survey

Merges each participant's Firestore record with their exit survey response,
matched by Prolific ID (the participant's `prolificId` field vs. the survey's
ID-confirmation column). Optional flags: `--participants-dir`, `--survey-csv`,
`--out`.

Output in `data/combined/`: one `<doc_id>.json` per participant (Firestore data
plus an `exitSurvey` block, or `null` if no response matched),
`_combined_dataset.json` and `_survey_question_key.json`. If a Prolific ID has
several submissions, the most recently recorded one is kept and the others are
listed under `exitSurvey.duplicateResponseIds`.

## Generating comparison plots

`scripts/04_generate_plots.py` reads `data/deidentified/` and produces PNG plots comparing
experimental groups (`control` vs `standard_ai`) on both task performance and
survey responses:

```
python scripts/04_generate_plots.py --combined-dir data/deidentified --out analysis_plots
```

Performance plots (built from each participant's `events` subcollection,
`level_001`-`level_004`, i.e. `isExperiment: true`):
- `perf_success_rate_by_level.png` — completion success rate per level
- `perf_completion_by_level.png` / `.eps` — compound figure (one panel per group) of 100% stacked bars: per level, participants who completed it vs. did not (timed out or never reached it), with counts in each segment. The EPS is vector, for LaTeX/Overleaf
- `perf_time_spent_by_level.png` / `.eps` — compound figure (the four levels side by side, sized for one IEEE column) of wall-clock time to complete each level across all restarts; completed levels only, with per-level n under each box. The EPS is vector, for LaTeX/Overleaf
- `perf_attempts_until_success_by_level.png` / `.eps` — same layout as the time-per-level figure, for code runs until success on each level; completed levels only, with per-level n under each box
- `perf_time_and_attempts_by_level.png` / `.eps` — compact single-column version of the two figures above: time (top row) and runs (bottom row) until success, Levels 1–4 as columns, headers and legend drawn once, n given in the caption
- `perf_time_to_success_km_by_level.png` / `.eps` — censoring-aware time to success: Kaplan-Meier share of each group that has succeeded on each level by each minute (2×2, own time axis per level); participants stopped by the session limit are censored (+). Log-rank tests for time and runs per level are printed
- `perf_runs_until_success_by_level.png` / `.eps` — runs until success per level (1×4 with task headers) for participants who completed the level, plus + for the runs made by participants who timed out (a lower bound)
- `perf_successful_participants_by_task.png` / `.eps` — percentage of each group who completed each level (percent because the groups differ in size; participant counts are written on the bars), in the same per-level panel layout as the time/runs figures (task headers, L1–L4 titles, shared y-scale). Completers are in the group color and everyone else in gray, so each bar is 100% of the group
- `perf_completion_and_runs_a.*` and `_b.*` (PNG + EPS) — compact single-column compound figure, as two panels for LaTeX subcaptions: (a) the completion-by-level figure narrowed to fit beside (b) each participant's runs until success averaged over the levels they completed (participants who completed no level are left out). Same height, plot areas aligned; include both at natural size (no `width=`)
- `perf_movement_blocks_by_dimension.png` / `.eps` — horizontal box plots of the movement blocks (move forward, turns) in each successful program, per task dimension and group; whiskers span min to max, a diamond marks the mean. Debugging/Editing counts include the provided starter code
- `perf_success_rate_by_condition.png` / `.eps` — success rate (% of the group) on each task, one panel per condition
- `perf_attempts_by_level.png` — number of simulation runs (code executions) per level
- `perf_collisions_by_level.png` — collisions per level
- `perf_completion_rate_per_participant.png` — overall fraction of levels completed, per participant
- `perf_avg_runs_per_level_by_group.png` / `.eps` — box plot of each participant's average code runs (run_simulation events) per experimental level they reached (tutorial excluded), by group; sized for one IEEE column. Also written as `perf_avg_runs_per_level_by_group_half.*` (narrow, 2.05 in wide) for two panels side by side in one column

Tables:
- `table_completion_time_by_level.tex` / `.csv` — average completion time per level and group (completers only), sized to fit one IEEE conference column. The .tex is a booktabs table (needs `booktabs` and `makecell`); the CSV includes the n behind each cell
- `table_self_efficacy_means.tex` / `.csv` and `table_self_efficacy_medians.tex` / `.csv` — mean and median rating of each self-efficacy Likert item per group, sized to fit one IEEE conference column (needs `booktabs` and `makecell`)
- `table_demographics.tex` / `.csv` — participant background across all participants: number and percentage giving each answer (education, programming experience, block-based coding use, prior robot programming, LLM use), sized to fit one IEEE conference column (needs `booktabs`)

Survey plots (built from `exitSurvey.answers`):
- `survey_self_efficacy_likert.png` / `.eps` — 100% stacked bars of the four self-efficacy Likert items by group, with response counts; sized for one IEEE column. (Programming experience is printed as a median/range summary instead of plotted.)
- `survey_ai_partner_ratings.png` — communication/dependency/trust ratings (AI-assisted group only)
- `survey_difficulty_ranking.png` / `.eps` — compact 100% stacked horizontal bars, one per group: share who ranked authoring vs. debugging/editing as the hardest (rank 1), with counts in the segments; sized for one IEEE column
- `survey_background_breakdown.png` — education, block-code familiarity, robotics/AI experience by group

Most performance and survey plots are box plots (median, IQR, whiskers; outliers
hidden); the success-rate plot is a bar of the fraction of participants who
completed each level.

Figures and tables show the groups as "Solo" (internal id `control`) and "Human+AI" (internal id `standard_ai`) (set in `GROUP_DISPLAY_NAMES` in `scripts/04_generate_plots.py` and `scripts/05_survival_analysis.py`); data files and CSVs keep the internal id.

Only participants with a matched exit survey are included (the survey is
matched by Prolific ID). Sessions without one (tests/abandoned runs) are already
dropped from `data/deidentified/`; when run on `data/combined/` they are excluded
and listed when the script runs.

## Survival analysis

`scripts/05_survival_analysis.py` uses scikit-survival to run a Kaplan-Meier analysis of
total session time until participants moved on to the exit survey, stratified by
group, and a log-rank test comparing the groups:

```
python scripts/05_survival_analysis.py --combined-dir data/deidentified --out analysis_plots
```

Each participant's time runs from their first event (the start of the tutorial)
to their final experimental `level_complete` or `time_limit_reached` event — the
same span the 70-minute session limit applies to, so timeouts sit at 70 minutes. Everyone is an event at that
time (whether they finished every level or hit the time limit, that is when they
moved on), so nothing is censored. Uses the same participant filter as
`scripts/04_generate_plots.py`.

- `perf_survival_time_to_completion.png` — Kaplan-Meier curves with 95% CIs and the log-rank result (open circles = timed out)
- `perf_survival_time_to_completion.csv` — per-participant time and event status
- `completion_time_by_participant.png` / `.eps` — one bar per participant, total session time in ascending order, colored by group; a black dot marks participants who did not complete all levels. Also written as `completion_time_by_participant_half.*` (narrow, 2.05 in wide) for two panels side by side in one column
- `perf_session_time_and_attempts_a.*` and `_b.*` (PNG + EPS) — compact single-column compound figure, as two panels for LaTeX subcaptions: (a) session time per participant (as above) with the group legend, and (b) average code runs per level reached by group. Same height, plot areas aligned; include both at natural size (no `width=`) so the text prints at 10 pt
- `perf_survival_time_to_success_by_level.png` / `.eps` — Kaplan-Meier curves of the share of participants who have not yet succeeded on each level (minutes on the level), by group, one panel per level; participants stopped by the time limit are censored (+). A log-rank test per level is printed
