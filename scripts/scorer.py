"""Phase A: score a predictions CSV (case_id, CP1..CP41) against dev/gold.csv.
Reports overall accuracy, per-CP accuracy, per-element accuracy.
Usage: python scorer.py <predictions.csv> [--gold dev/gold.csv]
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPS = [f"CP{i}" for i in range(1, 42)]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["case_id"]: {cp: row[cp] for cp in CPS} for row in csv.DictReader(f)}


def load_cp_elements():
    ref = json.loads((ROOT / "dev" / "cp_reference.json").read_text(encoding="utf-8"))
    return {r["cp"]: r["element"] for r in ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--gold", default=str(ROOT / "dev" / "gold.csv"))
    args = ap.parse_args()

    gold = load_csv(args.gold)
    pred = load_csv(args.predictions)
    cp_element = load_cp_elements()

    missing_cases = [c for c in gold if c not in pred]
    if missing_cases:
        raise SystemExit(f"predictions missing cases: {missing_cases}")

    total = correct = 0
    per_cp = {cp: [0, 0] for cp in CPS}  # [correct, total]
    per_element = {}

    for case_id, gold_row in gold.items():
        pred_row = pred[case_id]
        for cp in CPS:
            g, p = gold_row[cp], pred_row.get(cp, "")
            is_correct = int(g == p)
            total += 1
            correct += is_correct
            per_cp[cp][0] += is_correct
            per_cp[cp][1] += 1
            el = cp_element[cp]
            per_element.setdefault(el, [0, 0])
            per_element[el][0] += is_correct
            per_element[el][1] += 1

    print(f"overall accuracy: {correct}/{total} = {correct/total:.4f}\n")

    print("per-element accuracy:")
    for el in sorted(per_element):
        c, t = per_element[el]
        print(f"  {el}: {c}/{t} = {c/t:.4f}")

    print("\nper-CP accuracy (sorted worst first):")
    ranked = sorted(per_cp.items(), key=lambda kv: kv[1][0] / kv[1][1])
    for cp, (c, t) in ranked:
        marker = " <-- below 0.7" if c / t < 0.7 else ""
        print(f"  {cp}: {c}/{t} = {c/t:.4f}{marker}")


if __name__ == "__main__":
    main()
