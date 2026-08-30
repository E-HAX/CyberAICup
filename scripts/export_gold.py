"""Phase A: export dev/drafts/*.json (treated as human-verified gold, per user
instruction 2026-08-15 to skip the dashboard step) into dev/gold.csv, wide
submission-shaped format: case_id, CP1..CP41.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "dev" / "drafts"
DEV_CASES = ROOT / "dev" / "dev_cases.csv"
OUT = ROOT / "dev" / "gold.csv"

CPS = [f"CP{i}" for i in range(1, 42)]


def load_dev_ids():
    with open(DEV_CASES, newline="", encoding="utf-8") as f:
        return [row["dev_id"] for row in csv.DictReader(f)]


def main():
    dev_ids = load_dev_ids()
    rows = []
    for dev_id in dev_ids:
        draft_path = DRAFTS / f"{dev_id}.json"
        if not draft_path.exists():
            raise SystemExit(f"missing draft for dev case {dev_id}: {draft_path}")
        data = json.loads(draft_path.read_text(encoding="utf-8"))
        by_cp = {v["cp"]: v["verdict"] for v in data["verdicts"]}
        missing = [cp for cp in CPS if cp not in by_cp]
        if missing:
            raise SystemExit(f"{dev_id}: missing verdicts for {missing}")
        bad = [(cp, by_cp[cp]) for cp in CPS if by_cp[cp] not in ("1", "0", "N/A")]
        if bad:
            raise SystemExit(f"{dev_id}: invalid verdict values {bad}")
        row = {"case_id": dev_id}
        row.update({cp: by_cp[cp] for cp in CPS})
        rows.append(row)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id"] + CPS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} cases x {len(CPS)} CPs = {len(rows)*len(CPS)} cells)")


if __name__ == "__main__":
    main()
