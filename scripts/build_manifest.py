"""Phase B.1: build logical_cases.json — one entry per LOGICAL case (100 total:
98 plain 1:1 folders + the doubled RE-WA-2021-0077 folder split into two:
RE-WA-2021-0077 (Goldfields, farm #35) and RE-WA-2021-0077-B (Midwest, farm
#100, provisional ID pending organizer confirmation of its true RE number
per DATASET_CONSTRUCTION.md §4.2).
"""
import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "Task2" / "SFRE_cases"
MANIFEST_CSV = ROOT / "case_manifest.csv"
OUT = ROOT / "logical_cases.json"

DOUBLED_FOLDER = "RE-WA-2021-0077"
DOUBLED_SPLITS = [
    ("RE-WA-2021-0077", "Goldfields"),
    ("RE-WA-2021-0077-B", "Midwest"),
]


def tracks_for(folder_path, only_substr=None):
    files = sorted(os.listdir(folder_path))
    if only_substr:
        files = [f for f in files if only_substr in f]
    by_track = {}
    for f in files:
        m = re.match(r"(\d+)_", f)
        if m:
            by_track[int(m.group(1))] = str(folder_path / f)
    return {str(t): by_track.get(t) for t in range(1, 10)}


def flags_for(folder_id, raw_flags):
    flags = []
    if raw_flags == "MISSING_TRACK_1":
        flags.append("MISSING_TRACK_1")
    return flags


def main():
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logical_cases = []
    for row in rows:
        folder_id = row["folder_id"]
        folder_path = CASES_DIR / folder_id

        if folder_id == DOUBLED_FOLDER:
            for case_id, subset in DOUBLED_SPLITS:
                logical_cases.append({
                    "case_id": case_id,
                    "folder_id": folder_id,
                    "folder_path": str(folder_path),
                    "subset_filter": subset,
                    "farm_number": None,
                    "tracks": tracks_for(folder_path, only_substr=subset),
                    "flags": ["DOUBLED_A" if subset == "Goldfields" else "DOUBLED_B",
                              "PROVISIONAL_ID" if case_id.endswith("-B") else None],
                })
            # dedupe None flags
            for lc in logical_cases[-2:]:
                lc["flags"] = [f for f in lc["flags"] if f]
            continue

        logical_cases.append({
            "case_id": folder_id,
            "folder_id": folder_id,
            "folder_path": str(folder_path),
            "subset_filter": None,
            "farm_number": row["farm_numbers"],
            "tracks": tracks_for(folder_path),
            "flags": flags_for(folder_id, row["flags"]),
        })

    OUT.write_text(json.dumps(logical_cases, indent=2), encoding="utf-8")

    n_missing_t1 = sum(1 for lc in logical_cases if "MISSING_TRACK_1" in lc["flags"])
    n_doubled = sum(1 for lc in logical_cases if any(f.startswith("DOUBLED") for f in lc["flags"]))
    print(f"wrote {OUT}: {len(logical_cases)} logical cases "
          f"({n_missing_t1} missing-track-1, {n_doubled} from the doubled folder)")


if __name__ == "__main__":
    main()
