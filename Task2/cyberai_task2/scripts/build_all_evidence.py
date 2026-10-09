"""Phase B.2: run ingest.py's build_context() over every logical case in
logical_cases.json -> evidence/<case_id>.md (100 files).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ingest import build_context  # noqa: E402

MANIFEST = ROOT / "logical_cases.json"
OUT_DIR = ROOT / "evidence"


def main():
    logical_cases = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)

    word_counts = []
    missing_track1_notes = 0
    failures = []

    for lc in logical_cases:
        case_id = lc["case_id"]
        try:
            text = build_context(lc["folder_path"], only_substr=lc["subset_filter"])
        except Exception as e:  # noqa: BLE001
            failures.append((case_id, str(e)))
            continue

        # identity-anchor header check
        assert "Case identity anchors" in text, f"{case_id}: missing identity header"

        if "was NOT provided for this case" in text:
            missing_track1_notes += 1

        out_path = OUT_DIR / f"{case_id}.md"
        out_path.write_text(text, encoding="utf-8")
        word_counts.append(len(text.split()))

    print(f"wrote {len(word_counts)}/{len(logical_cases)} evidence files to {OUT_DIR}")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for case_id, err in failures:
            print(f"  {case_id}: {err}")
    if word_counts:
        print(f"word count: min={min(word_counts)} max={max(word_counts)} "
              f"avg={sum(word_counts)/len(word_counts):.0f}")
    print(f"cases with a 'Track N not provided' note: {missing_track1_notes} "
          f"(expected 2, from the two MISSING_TRACK_1 folders)")


if __name__ == "__main__":
    main()
