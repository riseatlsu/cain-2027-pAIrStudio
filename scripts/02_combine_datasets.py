#!/usr/bin/env python3
"""
Combine the Firestore participants export with the pAIrStudio exit survey CSV.

Matches each participant's `prolificId` field (from data/raw/firestore_export/participants/*.json)
against the exit survey's ID-confirmation column ("End" / QID3_TEXT), and writes one
merged JSON file per participant into data/combined/. That output still holds
identifiers; run 03_deidentify.py on it before sharing.

Usage:
    python scripts/02_combine_datasets.py \
        --participants-dir data/raw/firestore_export/participants \
        --survey-csv data/raw/exit_survey.csv \
        --out data/combined

If multiple survey responses share the same prolific ID (e.g. a retake), the most
recently recorded ("RecordedDate") response is used, and the rest are noted under
exitSurvey.duplicateResponseIds.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Qualtrics export layout: row 0 = short column keys, row 1 = full question text,
# row 2 = internal ImportId metadata (not needed here), row 3+ = response data.
SURVEY_ID_COLUMN = "End"  # holds the prolific ID confirmation (QID3_TEXT)

# Columns that describe the response itself rather than an answer, kept under
# a separate "meta" key instead of being mixed in with question answers.
META_COLUMNS = {
    "StartDate", "EndDate", "Status", "IPAddress", "Progress",
    "Duration (in seconds)", "Finished", "RecordedDate", "ResponseId",
    "RecipientLastName", "RecipientFirstName", "RecipientEmail",
    "ExternalReference", "LocationLatitude", "LocationLongitude",
    "DistributionChannel", "UserLanguage", SURVEY_ID_COLUMN,
}


def load_survey(csv_path):
    """Return (question_text_by_key, responses_by_prolific_id)."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    keys, question_text, _import_ids = rows[0], rows[1], rows[2]
    data_rows = rows[3:]

    question_text_by_key = {
        k: q for k, q in zip(keys, question_text) if k not in META_COLUMNS
    }

    responses_by_prolific_id = {}
    for row in data_rows:
        record = dict(zip(keys, row))
        prolific_id = record.get(SURVEY_ID_COLUMN, "").strip()
        if not prolific_id:
            continue  # preview/test responses with no ID confirmation
        responses_by_prolific_id.setdefault(prolific_id, []).append(record)

    return question_text_by_key, responses_by_prolific_id


def pick_canonical_response(responses):
    """When a prolific ID has multiple submissions, keep the most recently recorded one."""
    return max(responses, key=lambda r: r.get("RecordedDate", ""))


def build_exit_survey_block(record, all_responses_for_id, question_text_by_key):
    answers = {
        key: value
        for key, value in record.items()
        if key not in META_COLUMNS and value.strip() != ""
    }
    block = {
        "meta": {
            "responseId": record.get("ResponseId", ""),
            "startDate": record.get("StartDate", ""),
            "endDate": record.get("EndDate", ""),
            "recordedDate": record.get("RecordedDate", ""),
            "durationSeconds": record.get("Duration (in seconds)", ""),
            "finished": record.get("Finished", ""),
            "progress": record.get("Progress", ""),
        },
        "answers": answers,
    }
    if len(all_responses_for_id) > 1:
        block["duplicateResponseIds"] = [
            r.get("ResponseId", "") for r in all_responses_for_id
            if r.get("ResponseId") != record.get("ResponseId")
        ]
    return block


def main():
    parser = argparse.ArgumentParser(
        description="Merge Firestore participants with the exit survey CSV, matched by prolific ID."
    )
    parser.add_argument("--participants-dir", default=REPO_ROOT / "data" / "raw" / "firestore_export" / "participants",
                         help="Directory of per-participant JSON files (default: data/raw/firestore_export/participants).")
    parser.add_argument("--survey-csv", default=REPO_ROOT / "data" / "raw" / "exit_survey.csv",
                         help="Path to the exit survey CSV export (default: data/raw/exit_survey.csv).")
    parser.add_argument("--out", default=REPO_ROOT / "data" / "combined",
                         help="Output directory (default: data/combined).")
    args = parser.parse_args()

    participants_dir = Path(args.participants_dir)
    survey_csv = Path(args.survey_csv)
    out_dir = Path(args.out)

    if not participants_dir.is_dir():
        print(f"Error: participants directory not found: {participants_dir}", file=sys.stderr)
        sys.exit(1)
    if not survey_csv.is_file():
        print(f"Error: survey CSV not found: {survey_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    question_text_by_key, responses_by_prolific_id = load_survey(survey_csv)

    with (out_dir / "_survey_question_key.json").open("w", encoding="utf-8") as f:
        json.dump(question_text_by_key, f, indent=2, ensure_ascii=False)

    participant_files = sorted(participants_dir.glob("*.json"))
    combined = {}
    matched, unmatched = 0, []

    for pf in participant_files:
        with pf.open(encoding="utf-8") as f:
            participant = json.load(f)

        prolific_id = (participant.get("prolificId") or "").strip()
        responses = responses_by_prolific_id.get(prolific_id)

        if responses:
            canonical = pick_canonical_response(responses)
            participant["exitSurvey"] = build_exit_survey_block(
                canonical, responses, question_text_by_key
            )
            matched += 1
        else:
            participant["exitSurvey"] = None
            unmatched.append((pf.stem, prolific_id))

        out_file = out_dir / pf.name
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(participant, f, indent=2, ensure_ascii=False)
        combined[pf.stem] = participant

    with (out_dir / "_combined_dataset.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    survey_ids_unused = set(responses_by_prolific_id) - {
        (json.load(open(pf, encoding="utf-8")).get("prolificId") or "").strip()
        for pf in participant_files
    }

    print(f"Matched {matched}/{len(participant_files)} participants to a survey response.")
    if unmatched:
        print("Participants with NO matching survey response:")
        for doc_id, pid in unmatched:
            print(f"  - {doc_id} (prolificId={pid!r})")
    if survey_ids_unused:
        print(f"Survey responses with no matching participant: {len(survey_ids_unused)}")
        for pid in survey_ids_unused:
            print(f"  - {pid!r}")
    print(f"\nWrote {len(participant_files)} file(s) to {out_dir}/")
    print(f"Combined file: {out_dir / '_combined_dataset.json'}")
    print(f"Survey question key: {out_dir / '_survey_question_key.json'}")


if __name__ == "__main__":
    main()
