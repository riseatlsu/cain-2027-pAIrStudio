#!/usr/bin/env python3
"""
Download all data from a Firebase Firestore database to local JSON files.

Walks every root collection and recursively follows every subcollection on
every document, so the full database tree is captured regardless of depth.
For each root collection, also writes one JSON file per document (named
<out>/<collection>/<doc_id>.json) containing that document's fields plus
all of its nested subcollections (e.g. a participant's events).

Usage:
    python scripts/01_download_firestore.py --key path/to/serviceAccountKey.json --out data/raw/firestore_export

    # Only export specific root collections:
    python scripts/01_download_firestore.py --key key.json --out ./export --collections users orders

Auth:
    Requires a Firebase service account JSON key. Generate one at:
    Firebase Console -> Project Settings -> Service Accounts -> Generate new private key
"""

import argparse
import base64
import json
import sys
from datetime import datetime, date
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import DocumentReference, GeoPoint

REPO_ROOT = Path(__file__).resolve().parent.parent


def json_default(value):
    """Convert Firestore-specific / non-JSON-native types to JSON-safe values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, DocumentReference):
        return {"__type__": "reference", "path": value.path}
    if isinstance(value, GeoPoint):
        return {"__type__": "geopoint", "latitude": value.latitude, "longitude": value.longitude}
    if isinstance(value, bytes):
        return {"__type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_document(doc_snapshot, stats):
    """Return a JSON-safe dict for a document, including any nested subcollections."""
    data = doc_snapshot.to_dict() or {}
    stats["documents"] += 1

    subcollections = {}
    for subcol_ref in doc_snapshot.reference.collections():
        subcollections[subcol_ref.id] = export_collection(subcol_ref, stats)

    if subcollections:
        data["__subcollections__"] = subcollections

    return data


def export_collection(collection_ref, stats):
    """Return {doc_id: doc_data} for every document in a collection."""
    stats["collections"] += 1
    result = {}
    for doc_snapshot in collection_ref.stream():
        result[doc_snapshot.id] = export_document(doc_snapshot, stats)
    return result


def main():
    parser = argparse.ArgumentParser(description="Download all data from a Firestore database.")
    parser.add_argument(
        "--key", required=True,
        help="Path to the Firebase service account JSON key file.",
    )
    parser.add_argument(
        "--out", default=REPO_ROOT / "data" / "raw" / "firestore_export",
        help="Output directory for exported JSON files (default: data/raw/firestore_export).",
    )
    parser.add_argument(
        "--collections", nargs="*", default=None,
        help="Specific root collection names to export. Omit to export every root collection.",
    )
    parser.add_argument(
        "--combined-only", action="store_true",
        help="Only write the single combined export file, skip per-collection files.",
    )
    parser.add_argument(
        "--no-per-doc-files", action="store_true",
        help="Skip writing one JSON file per document (in a <collection>/ subfolder).",
    )
    args = parser.parse_args()

    key_path = Path(args.key)
    if not key_path.is_file():
        print(f"Error: service account key not found at {key_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cred = credentials.Certificate(str(key_path))
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    if args.collections:
        root_collections = [db.collection(name) for name in args.collections]
    else:
        root_collections = list(db.collections())

    if not root_collections:
        print("No root collections found (or none matched --collections).")
        sys.exit(0)

    stats = {"collections": 0, "documents": 0}
    combined = {}

    for collection_ref in root_collections:
        name = collection_ref.id
        print(f"Exporting collection '{name}'...")
        data = export_collection(collection_ref, stats)
        combined[name] = data

        if not args.combined_only:
            out_file = out_dir / f"{name}.json"
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=json_default, ensure_ascii=False)
            print(f"  -> wrote {out_file} ({len(data)} top-level docs)")

        if not args.no_per_doc_files:
            doc_dir = out_dir / name
            doc_dir.mkdir(parents=True, exist_ok=True)
            for doc_id, doc_data in data.items():
                doc_file = doc_dir / f"{doc_id}.json"
                with doc_file.open("w", encoding="utf-8") as f:
                    json.dump(doc_data, f, indent=2, default=json_default, ensure_ascii=False)
            print(f"  -> wrote {len(data)} per-document file(s) to {doc_dir}/")

    combined_file = out_dir / "_combined_export.json"
    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, default=json_default, ensure_ascii=False)

    print(
        f"\nDone. {stats['collections']} collection(s), {stats['documents']} document(s) exported."
    )
    print(f"Combined export: {combined_file}")
    if not args.combined_only:
        print(f"Per-collection files in: {out_dir}")


if __name__ == "__main__":
    main()
