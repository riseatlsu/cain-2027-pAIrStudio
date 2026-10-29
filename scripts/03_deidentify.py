#!/usr/bin/env python3
"""
De-identify the combined dataset for public release.

Reads data/combined/*.json (produced by 02_combine_datasets.py, which still
holds Firebase user IDs, Prolific IDs and Qualtrics response IDs) and writes a
de-identified copy to data/deidentified/:

  - Keeps only the analysed participants: those with a Prolific ID and a
    matched exit survey (test/abandoned sessions are dropped).
  - Gives each participant a new ID (P01, P02, ...) in order of their first
    recorded event (client `localTime`), and uses it as the file name, the `participantId` field and
    the `participantId` on every event.
  - Removes the Firebase user ID, Prolific ID, Qualtrics response IDs and the
    survey response timestamps.
  - Fails if any removed identifier still appears anywhere in the output.

Usage:
    python scripts/03_deidentify.py --combined-dir data/combined --out data/deidentified
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level participant fields that identify a person or link back to Firebase,
# Prolific or Qualtrics.
IDENTIFYING_FIELDS = {"firebaseUserId", "prolificId", "qualtricsResponseId"}

# Exit survey meta fields kept in the release (response ID and dates are dropped).
SURVEY_META_KEPT = {"durationSeconds", "finished", "progress"}

# Patterns that should never appear in text people wrote (survey answers, chat).
PII_PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "Prolific-style ID": re.compile(r"\b[0-9a-f]{24}\b"),
}

# Fields holding chat text, in chatConversations and in chat events.
CHAT_TEXT_FIELDS = {"content", "message", "userMessage", "aiResponse"}


def load_participants(combined_dir):
    participants = []
    for f in sorted(combined_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with f.open(encoding="utf-8") as fh:
            participants.append(json.load(fh))
    return participants


def events_of(p):
    return (p.get("__subcollections__") or {}).get("events") or {}


def first_event_time(p):
    # `localTime` is the client ISO time; `timestamp` is often an unresolved
    # serverTimestamp placeholder, so it can't be used for ordering.
    times = [e["localTime"] for e in events_of(p).values() if isinstance(e.get("localTime"), str)]
    return min(times) if times else ""


def identifiers_of(p):
    """Every identifier value of a participant that must not appear in the release."""
    ids = {p.get(k) for k in IDENTIFYING_FIELDS}
    ids.add(p.get("participantId"))
    survey = p.get("exitSurvey") or {}
    ids.add((survey.get("meta") or {}).get("responseId"))
    ids.update(survey.get("duplicateResponseIds") or [])
    return {str(i) for i in ids if i}


def free_text_of(p):
    """Text participants wrote or read: exit survey answers and chat messages.

    Code snapshots are left out; their random Blockly block IDs trip the patterns.
    """
    texts = list(((p.get("exitSurvey") or {}).get("answers") or {}).values())

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in CHAT_TEXT_FIELDS and isinstance(v, str):
                    texts.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(p.get("chatConversations"))
    walk(events_of(p))
    return texts


def deidentify(p, new_id):
    out = {k: v for k, v in p.items() if k not in IDENTIFYING_FIELDS}
    out["participantId"] = new_id
    for event in events_of(out).values():
        if "participantId" in event:
            event["participantId"] = new_id
    survey = out.get("exitSurvey")
    if survey:
        out["exitSurvey"] = {
            "meta": {k: v for k, v in survey["meta"].items() if k in SURVEY_META_KEPT},
            "answers": survey["answers"],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="De-identify the combined dataset for public release.")
    parser.add_argument("--combined-dir", default=REPO_ROOT / "data" / "combined",
                        help="Directory of merged participant JSON files (default: data/combined).")
    parser.add_argument("--out", default=REPO_ROOT / "data" / "deidentified",
                        help="Output directory (default: data/deidentified).")
    args = parser.parse_args()

    combined_dir = Path(args.combined_dir)
    out_dir = Path(args.out)
    if not combined_dir.is_dir():
        print(f"Error: combined dataset directory not found: {combined_dir}", file=sys.stderr)
        sys.exit(1)

    participants = load_participants(combined_dir)
    included = [p for p in participants if p.get("prolificId") and p.get("exitSurvey")]
    included.sort(key=first_event_time)
    print(f"Keeping {len(included)} of {len(participants)} participants "
          f"(dropped {len(participants) - len(included)} without a Prolific ID + exit survey).")

    # Clear JSON files from an earlier run so a dropped participant can't linger.
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()

    removed_ids = set()
    combined = {}
    width = max(2, len(str(len(included))))
    for i, p in enumerate(included, start=1):
        new_id = f"P{i:0{width}d}"
        removed_ids |= identifiers_of(p)
        combined[new_id] = deidentify(p, new_id)

    # Safety check: no removed identifier may survive anywhere in the output.
    serialized = json.dumps(combined, ensure_ascii=False)
    leaked = sorted(i for i in removed_ids if i in serialized)
    if leaked:
        print(f"Error: {len(leaked)} identifier(s) still present in the output; nothing written.",
              file=sys.stderr)
        sys.exit(1)
    for new_id, p in combined.items():
        for text in free_text_of(p):
            for label, pattern in PII_PATTERNS.items():
                for match in sorted(set(pattern.findall(text))):
                    print(f"Warning: possible {label} in {new_id}, review before release: {match!r}")

    for new_id, p in combined.items():
        with (out_dir / f"{new_id}.json").open("w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
    with (out_dir / "_combined_dataset.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    shutil.copy(combined_dir / "_survey_question_key.json", out_dir / "_survey_question_key.json")

    print(f"Wrote {len(combined)} participant file(s) to {out_dir}/")


if __name__ == "__main__":
    main()
