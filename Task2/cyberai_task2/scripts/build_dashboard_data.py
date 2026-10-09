"""Assemble one JSON blob for the verification dashboard: per-case per-track evidence text,
41 CP reference, drafts, and auto-detected flag chips (hints only)."""
import os, re, json, csv, sys
sys.path.insert(0, os.path.dirname(__file__))
from ingest import docx_to_text, xlsx_to_text, TRACK_NAMES

ROOT = "/Users/siddhantparashar/projects/cyberai_task2"
CASES_DIR = f"{ROOT}/Task2/SFRE_cases"
DEV = list(csv.DictReader(open(f"{ROOT}/dev/dev_cases.csv")))
CPS = json.load(open(f"{ROOT}/dev/cp_reference.json"))

LANG_PAT = re.compile(r'\b(Mandarin|Cantonese|Chinese|Vietnamese|Arabic|Hindi|Punjabi|Thai|Spanish|Italian|Portuguese)\b')
DEFIC_PAT = re.compile(r'\[DEFICIENC[A-Z]*[^\]]*\]', re.I)

def case_tracks(folder, only_substr=None):
    files = sorted(os.listdir(folder))
    if only_substr:
        files = [f for f in files if only_substr in f]
    by_track = {}
    for f in files:
        m = re.match(r'(\d+)_', f)
        if m:
            by_track[int(m.group(1))] = f
    tracks = {}
    for t in range(1, 10):
        if t not in by_track:
            tracks[str(t)] = {"present": False, "filename": None, "text": ""}
            continue
        fname = by_track[t]
        p = os.path.join(folder, fname)
        text = docx_to_text(p) if p.endswith(".docx") else xlsx_to_text(p)
        tracks[str(t)] = {"present": True, "filename": fname, "text": text}
    return tracks

def find_flags(tracks):
    flags = []
    for t_str, tr in tracks.items():
        t = int(t_str)
        if not tr["present"]:
            flags.append({"type": "missing", "track": t, "label": f"Track {t} ({TRACK_NAMES[t]}) not provided"})
            continue
        text = tr["text"]
        n_defic = len(DEFIC_PAT.findall(text))
        if n_defic:
            flags.append({"type": "deficiency", "track": t, "label": f"{n_defic} explicit [DEFICIENCY] marker(s) in Track {t}"})
        for lang in set(LANG_PAT.findall(text)):
            flags.append({"type": "language", "track": t, "label": f"non-English mention ({lang}) in Track {t}"})
    return flags

cases_out = []
for row in DEV:
    dev_id = row["dev_id"]
    folder = os.path.join(CASES_DIR, row["folder_id"])
    only = row["subset"] or None
    tracks = case_tracks(folder, only)
    flags = find_flags(tracks)
    draft_path = f"{ROOT}/dev/drafts/{dev_id}.json"
    draft = {}
    if os.path.exists(draft_path):
        d = json.load(open(draft_path))
        draft = {v["cp"]: v for v in d.get("verdicts", [])}
    cases_out.append({
        "id": dev_id,
        "folder_id": row["folder_id"],
        "subset_note": only,
        "tracks": tracks,
        "flags": flags,
        "draft": draft,
    })

data = {"cases": cases_out, "cps": CPS, "track_names": TRACK_NAMES}
out_path = f"{ROOT}/dev/dashboard/data.json"
json.dump(data, open(out_path, "w"))
sz = os.path.getsize(out_path)
print(f"wrote {out_path}  ({sz/1024:.0f} KB)")
