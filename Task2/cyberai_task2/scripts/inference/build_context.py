"""Phase D helper: assemble the per-case, per-element context bundle that gets
handed to inference agents. Pure text assembly, zero compliance logic —
every prompt gets: the case evidence + the OKF bundle (official CP text +
governing policy text) for whichever CPs it's being asked about.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CPS_DIR = ROOT / "cps"
EVIDENCE_DIR = ROOT / "evidence"
CP_REFERENCE = ROOT / "dev" / "cp_reference.json"

ELEMENT_CPS = {}  # "Element-1" -> ["CP1", ...] in order, filled lazily


def _load_element_map():
    global ELEMENT_CPS
    if ELEMENT_CPS:
        return ELEMENT_CPS
    ref = json.loads(CP_REFERENCE.read_text(encoding="utf-8"))
    for r in ref:
        ELEMENT_CPS.setdefault(r["element"], []).append(r["cp"])
    return ELEMENT_CPS


def cp_number(cp):
    return int(re.match(r"CP(\d+)", cp).group(1))


def cp_bundle_text(cp):
    """Read cps/cp_XX.md verbatim -- official text + governing policy text."""
    path = CPS_DIR / f"cp_{cp_number(cp):02d}.md"
    return path.read_text(encoding="utf-8")


def all_cps():
    return [f"CP{i}" for i in range(1, 42)]


def elements():
    return _load_element_map()


def element_cps(element):
    return _load_element_map()[element]


def case_evidence(case_id):
    return (EVIDENCE_DIR / f"{case_id}.md").read_text(encoding="utf-8")


def element_bundle_text(element):
    """Concatenate the OKF bundle (text + citations) for every CP in one element."""
    cps = element_cps(element)
    parts = [cp_bundle_text(cp) for cp in cps]
    return f"\n\n{'#'*80}\n\n".join(parts)


def full_okf_bundle_text():
    return element_bundle_text("Element-1") + "\n\n" + element_bundle_text("Element-2") \
        + "\n\n" + element_bundle_text("Element-3") + "\n\n" + element_bundle_text("Element-4")
