# pAIrStudio: Replication Package

Replication package for the paper **"pAIrStudio: A Preliminary Study on Human-AI
Collaboration in Industrial Robot Programming."** It holds the de-identified study
data and the scripts that reproduce every figure, table and statistic in the paper.

## About the study

*pAIrStudio* is a web-based, block-based robot programming environment with an
embedded LLM pair programming partner, **Otto** (OpenAI gpt-4o, few-shot prompted
with the current task, the state of the simulated warehouse and the participant's
blocks). Participants program a mobile robot in a tile-based warehouse simulation to
pick up boxes from a pickup zone and drop them onto a conveyor in a drop-off zone.
The platform is available at <https://riseatlsu.github.io/pAIrStudio/sandbox/>.
Its full source code (JavaScript, with Phaser.js and Blockly.js) is included as
`pAIrStudio-main.zip`.

In a between-subjects study, **12 novice programmers** (undergraduate Computer
Science students) each attended an individual 70-minute lab session. They were
randomly assigned to program alone (**Solo**, *n* = 5) or with Otto (**Human+AI**,
*n* = 7). After five short tutorials, they solved four tasks in a counterbalanced
order (balanced Latin square):

| Task | Phase | Difficulty | Description |
|------|-------|------------|-------------|
| T1 | Authoring | Easy | Write a program from scratch |
| T2 | Authoring | Hard | Write a program from scratch, with obstacles and a patrolling robot |
| T3 | Debugging/Editing | Easy | Fix provided code with a bug that prevents pickup |
| T4 | Debugging/Editing | Hard | Complete provided, unfinished code, with obstacles and a patrolling robot |

Participants could not skip or revisit tasks, and could run their code any number of
times. The platform logged every code run (with a snapshot of the program), every
message exchanged with Otto, and task completions. A 20-item exit questionnaire
covered background, self-efficacy, the perceived difficulty of each phase, and
(Human+AI only) experience with Otto. The study was approved by our institutional
review board.

## Repository layout

```
data/deidentified/               released study data (12 participants)
  P01.json ... P12.json          one file per participant: group, event log, Otto chats, exit survey
  _combined_dataset.json         all 12 participants in one file
  _survey_question_key.json      survey answer key -> question text
  exit_survey.csv                exit survey answers, one row per participant
  exit_survey_questions.csv      question text for each exit_survey.csv column
figs/                            figures in the paper (EPS for LaTeX, PNG for viewing)
stats/                           every statistic reported in the paper, one CSV per analysis
scripts/
  04_analysis.py                 reproduces figs/ and stats/ from data/deidentified/
  01_download_firestore.py       data collection pipeline (needs access to the raw data;
  02_combine_datasets.py           listed for transparency, see below)
  03_deidentify.py
pAIrStudio-main.zip              full source code of the pAIrStudio platform
requirements.txt
```

## Reproducing the results

Only the de-identified data is needed. Tested with Python 3.14.6 (package versions in
`stats/software_versions.csv`):

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/04_analysis.py
```

The script rewrites `figs/` and `stats/`, and reruns produce identical files.
`git status` therefore shows no changes after a successful reproduction.
`--data-dir`, `--figs-dir` and `--stats-dir` change the input and output folders.

## Where each result comes from

All outputs are written by `scripts/04_analysis.py`. Groups appear as "Solo" and
"Human+AI" in the figures. The data and CSVs use the internal ids `control` (Solo)
and `standard_ai` (Human+AI). Tasks are labelled T1-T4 (`level_001`-`level_004` in
the data).

| Paper | Output | Content |
|-------|--------|---------|
| Fig. 1 | not included | Screenshot of the pAIrStudio interface |
| §V-A Demographics | `stats/demographics.csv` | Background answers (number and % of 12), and the programming-experience mean and SD |
| Fig. 2a | `figs/perf_completion_and_runs_a.*`, `stats/completion_by_task.csv` | Participants in each group who completed each task |
| Fig. 2b | `figs/perf_completion_and_runs_b.*`, `stats/runs_until_success.csv` | Each participant's runs until success, averaged over the tasks they completed, and the group medians (7.13 Solo vs. 6.25 Human+AI) |
| Fig. 3 | `figs/perf_level_time_by_participant.*` | Each participant's time to success on each task |
| §V log-rank tests | `stats/logrank_time_and_runs_by_task.csv` | Log-rank tests of time and of runs until first success on each task (df, χ², p) |
| §V use of Otto | `stats/ai_chats_by_task.csv` | Human+AI participants who messaged Otto, and messages sent on each task |
| Table I | `stats/self_efficacy_means.csv` | Mean self-efficacy rating per item and group |
| §V Mann-Whitney U | `stats/likert_mann_whitney.csv` | Solo vs. Human+AI on each self-efficacy item (U, exact two-sided p) |
| §V participant quotes | `data/deidentified/exit_survey.csv` | Open-ended answers, by participant ID (P01-P12) |

### Definitions

- **Task completion.** A task counts as completed when the box was delivered to the
  conveyor (`level_complete` event with `success: true`). Participants who ran out of
  session time on a task, or never reached it, did not complete it.
- **Time to success.** Wall-clock minutes from the first event on a task until its
  successful completion, or until the session time limit for participants who timed
  out.
- **Runs until success.** The number of code runs (`run_simulation` events) on a
  task.
- **Survival analysis.** Time and runs until first success are compared with a
  log-rank test (scikit-survival `compare_survival`). Participants stopped by the
  session time limit are right-censored, and participants who never reached a task
  are left out of that task.
- **Mann-Whitney U.** The p-values are exact. U is recomputed for every split of
  the 12 pooled ratings into groups of 5 and 7, using midranks for ties.
- **Messages to Otto.** Counted from `chat_message` events with role `user`.

## Data

### Format

Each `data/deidentified/P##.json` holds one participant:

- `participantId`, `experimentalGroup` (`control` or `standard_ai`) and `lastUpdated`
- `chatConversations` (Human+AI only): chat history with Otto, per task
- `__subcollections__.events`: the event log, keyed by event ID. Each event has an
  `eventType` (e.g. `run_simulation`, `level_complete`, `time_limit_reached`,
  `collision`, `chat_message`), `levelId` (`tutorial_*`, `level_001`-`level_004`),
  `isExperiment`, `localTime` (ISO 8601, UTC), and type-specific fields such as
  `codeSnapshot` (the Blockly program that was run) and `runNumber`
- `exitSurvey.answers`: survey answers keyed by Qualtrics column name (question
  text in `_survey_question_key.json`). `exitSurvey.meta` holds the response
  duration, progress and completion flag

`exit_survey.csv` holds the same survey answers as a flat table, one row per
participant. A blank cell means the question was not answered or not shown; for
example, the questions about Otto were shown only to the Human+AI group.

### De-identification

The raw data is not shared because it contains personal data: Prolific IDs (in the
platform records and the survey), and IP addresses and approximate location in the
Qualtrics export. `scripts/03_deidentify.py` produced the released data:

- Keeps only the 12 analysed participants, those with a matched exit survey. Test
  and abandoned sessions are dropped.
- Assigns new IDs `P01`-`P12` in order of each participant's first recorded event.
- Removes the Firebase user ID, Prolific ID, Qualtrics response IDs and survey
  response dates. The survey's IP address, location, name and email columns are
  never copied from the raw export.
- Refuses to write output if a removed identifier still appears anywhere in it.

Event times are kept because the timing analyses need them. Free-text survey answers
and chat messages with Otto were reviewed and contain no identifying information.

### Data collection pipeline (requires raw data access)

The steps below rebuilt `data/deidentified/` from the source systems. They are
listed for transparency, because they need the Firebase service account key and the
Qualtrics export, neither of which is shared.

```
python scripts/01_download_firestore.py --key path/to/serviceAccountKey.json   # -> data/raw/firestore_export/
#   place the Qualtrics CSV export at data/raw/exit_survey.csv
python scripts/02_combine_datasets.py   # Firestore records + exit survey, matched by Prolific ID -> data/combined/
python scripts/03_deidentify.py         # -> data/deidentified/
```

`data/raw/` and `data/combined/` contain identifiers and are gitignored.
`firebase-admin` in `requirements.txt` is needed only for step 01.
