"""Full-corpus run (100 logical cases), full-context, checkpointed case-by-case.

- Reads case_ids from logical_cases.json (order preserved).
- Per case: builds the full-context prompt (evidence/<id>.md + okf/reference.md),
  calls the model via litai_client, parses 1/0/N/A verdicts, and writes a
  per-case checkpoint to submission_run/<case_id>.json (atomic).
- Resume: cases with an existing checkpoint are skipped, so a run interrupted
  by credit exhaustion can be continued after switching LIGHTNING_API_KEY.
- On a network/auth/credit exception the loop STOPS (so you don't burn 3x
  retries per remaining case); a TRUNCATED (finish=length) response is logged
  and skipped (no checkpoint) for a later re-run.
- At the end (or whenever --assemble is run), assembles submission CSV in the
  template shape: "RE Number,CP1,...,CP41".

Usage:
  export LIGHTNING_API_KEY=...  LITAI_MODEL=google/gemini-3.5-flash
  python run_full_corpus.py                 # run + assemble
  python run_full_corpus.py --assemble      # just re-assemble from checkpoints
"""
import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import litai_client  # noqa: E402
from run_standalone import build_prompt, CPS, OKF_REFERENCE, ROOT  # noqa: E402

EVIDENCE_DIR = ROOT / "evidence"
MANIFEST = ROOT / "logical_cases.json"
CHECKPOINT_DIR = ROOT / "submission_run"
OUT_CSV = ROOT / "submission_100.csv"

PER_CALL_TIMEOUT = 240  # seconds; a hung API call is abandoned (case re-runs on resume)


def _timeout_handler(signum, frame):
    raise TimeoutError("LLM call timed out")


signal.signal(signal.SIGALRM, _timeout_handler)

JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a raw JSON array (no markdown fences, no prose before or "
    "after) matching the Output format above."
)


def case_ids():
    return [lc["case_id"] for lc in json.loads(MANIFEST.read_text(encoding="utf-8"))]


def load_checkpoint(case_id):
    p = CHECKPOINT_DIR / f"{case_id}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_checkpoint(case_id, payload):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    tmp = CHECKPOINT_DIR / f"{case_id}.json.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(CHECKPOINT_DIR / f"{case_id}.json")


def parse_verdicts(content):
    import re as _re
    try:
        arr = litai_client.extract_json_array(content)
        verd = {r.get("cp"): r.get("verdict", "") for r in arr if r.get("cp") in CPS}
        if verd:
            return verd
    except Exception:
        pass
    # regex fallback: resilient to trailing commas / prose / truncation
    pairs = _re.findall(r'"cp"\s*:\s*"(CP\d+)"[^{}]*?"verdict"\s*:\s*"(1|0|N/A)"', content, _re.DOTALL)
    return {cp: v for cp, v in pairs if cp in CPS}


def run(max_tokens):
    ids = case_ids()
    done = sum(1 for c in ids if load_checkpoint(c))
    print(f"cases total={len(ids)} already_done={done}", flush=True)

    for i, case_id in enumerate(ids, 1):
        if load_checkpoint(case_id):
            print(f"[{i}/{len(ids)}] {case_id}: skip (cached)", flush=True)
            continue

        evidence = (EVIDENCE_DIR / f"{case_id}.md").read_text(encoding="utf-8")
        prompt = build_prompt("persona_standard", evidence, OKF_REFERENCE.read_text(encoding="utf-8"), CPS) + JSON_INSTRUCTION

        t0 = time.time()
        signal.alarm(PER_CALL_TIMEOUT)
        try:
            content, pt, ct, finish = litai_client.chat(prompt, max_tokens=max_tokens)
        except TimeoutError:
            signal.alarm(0)
            print(f"[{i}/{len(ids)}] {case_id}: TIMEOUT after {PER_CALL_TIMEOUT}s — skipped, re-run later", flush=True)
            continue
        except Exception as e:  # noqa: BLE001  (credit/auth/network -> stop, resume later)
            signal.alarm(0)
            print(f"[{i}/{len(ids)}] {case_id}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            print(f"STOPPING at case {i} ({case_id}) — switch LIGHTNING_API_KEY and re-run to resume.", flush=True)
            return False
        signal.alarm(0)

        elapsed = time.time() - t0
        if finish == "length" or ct >= max_tokens:
            print(f"[{i}/{len(ids)}] {case_id}: TRUNCATED (finish={finish}, comp={ct}) — skipped, re-run later", flush=True)
            continue

        try:
            verdicts = parse_verdicts(content)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(ids)}] {case_id}: PARSE FAIL {str(e)[:120]} — skipped", flush=True)
            continue

        save_checkpoint(case_id, {
            "case_id": case_id, "verdicts": verdicts,
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "finish_reason": finish},
        })
        print(f"[{i}/{len(ids)}] {case_id}: ok {len(verdicts)}/41 verdicts "
              f"(prompt={pt} comp={ct} finish={finish} {elapsed:.0f}s)", flush=True)

    return True


def assemble():
    ids = case_ids()
    rows = []
    missing = []
    for case_id in ids:
        cp = load_checkpoint(case_id)
        if not cp:
            missing.append(case_id)
            continue
        row = {"RE Number": case_id}
        row.update({c: cp["verdicts"].get(c, "") for c in CPS})
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["RE Number"] + CPS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: {len(rows)} rows, {len(missing)} missing", flush=True)
    if missing:
        print("missing cases:", missing, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--assemble", action="store_true", help="only re-assemble CSV from checkpoints")
    args = ap.parse_args()

    if args.assemble:
        assemble()
        return

    finished = run(args.max_tokens)
    assemble()
    if not finished:
        sys.exit(1)


if __name__ == "__main__":
    main()
