"""Token-efficiency: cps/cp_XX.md duplicates governing policy text per CP
(64 citation instances but only 18 unique section files -> 4.3x waste when an
agent reads many cp_XX.md files in one call). Build one deduped OKF bundle:
  okf/reference.md -- CP index (official text + citation pointers) followed by
                       every cited policy section's text, each exactly once.
Single file (not two) so an inference agent needs one Read call, not two.
Derived from cps_mapping/element*.json + dev/cp_reference.json + policy/
section_*.md -- same grounded citations as the original OKF bundle, just laid
out without duplication.
"""
import json
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPS_MAPPING = sorted(glob.glob(str(ROOT / "cps_mapping" / "element*.json")))
CP_REFERENCE = ROOT / "dev" / "cp_reference.json"
OUT_DIR = ROOT / "okf"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    cp_cites = {}
    cp_rationale = {}
    for f in CPS_MAPPING:
        for entry in json.loads(Path(f).read_text(encoding="utf-8")):
            cp_cites[entry["cp"]] = entry["citations"]
            cp_rationale[entry["cp"]] = entry.get("rationale", "")

    cp_ref = {r["cp"]: r for r in json.loads(CP_REFERENCE.read_text(encoding="utf-8"))}
    all_citations = sorted({c for cites in cp_cites.values() for c in cites})

    lines = ["# OKF reference bundle",
             "Part 1: official text for all 41 checking points, with pointers to the",
             "policy sections (Part 2) that govern each. Part 2: every cited policy",
             "section's text, included exactly once even if multiple CPs cite it.",
             "",
             "## Part 1 — Checking-point index", ""]
    cp_order = [f"CP{i}" for i in range(1, 42)]
    for cp in cp_order:
        ref = cp_ref[cp]
        cites = cp_cites.get(cp, [])
        lines.append(f"\n### {cp} ({ref['element']} — {ref['subelement']})")
        lines.append(f"**Official text:** {ref['text']}")
        lines.append(f"**Governed by:** {', '.join(cites) if cites else '(none mapped)'}")
        if cp_rationale.get(cp):
            lines.append(f"**Why these sections govern it:** {cp_rationale[cp]}")

    lines.append("\n\n## Part 2 — Unique governing policy sections\n")
    for citation in all_citations:
        path = ROOT / citation
        lines.append(f"\n{'='*72}\n### {citation}\n{'='*72}")
        lines.append(path.read_text(encoding="utf-8"))

    (OUT_DIR / "reference.md").write_text("\n".join(lines), encoding="utf-8")

    dup_bytes = sum((ROOT / c).stat().st_size for cites in cp_cites.values() for c in cites)
    unique_bytes = sum((ROOT / c).stat().st_size for c in all_citations)
    out_size = (OUT_DIR / "reference.md").stat().st_size
    print(f"unique sections: {len(all_citations)} ({unique_bytes:,} bytes)")
    print(f"duplicated citation instances: {sum(len(c) for c in cp_cites.values())} ({dup_bytes:,} bytes)")
    print(f"dedup ratio: {dup_bytes/unique_bytes:.2f}x")
    print(f"wrote {OUT_DIR}/reference.md ({out_size:,} bytes, single file)")


if __name__ == "__main__":
    main()
