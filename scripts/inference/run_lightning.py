"""Bake-off candidate: Lightning AI / Nemotron Ultra, via litai.
Reuses the exact same canonical prompts (prompts/*.md + okf/reference.md) as
run_standalone.py -- same context_block/render/build_prompt functions -- so
this is a fair like-for-like comparison, not a different pipeline.
No JSON-schema/tool-calling enforcement available for this model on Lightning,
so the prompt asks for raw JSON and we parse defensively.
Records prompt_tokens/completion_tokens per call (Lightning's total_tokens
field is broken -- always reports 0 -- so we sum ourselves).
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_standalone import build_prompt, CPS, OKF_REFERENCE, ROOT  # noqa: E402
import litai_client  # noqa: E402

MODEL = litai_client.MODEL
JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a raw JSON array (no markdown fences, no prose before or "
    "after) matching the Output format above."
)


def extract_json_array(text):
    text = text.strip()
    # strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--evidence-dir", default=str(ROOT / "dev" / "evidence"))
    ap.add_argument("--out", default=str(ROOT / "dev" / "predictions_nemotron.csv"))
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--usage-log", default=str(ROOT / "dev" / "nemotron_usage.json"))
    args = ap.parse_args()

    reference_text = OKF_REFERENCE.read_text(encoding="utf-8")
    rows = []
    usage_records = []

    for case_id in args.cases:
        evidence_path = Path(args.evidence_dir) / f"{case_id}.md"
        evidence_text = evidence_path.read_text(encoding="utf-8")
        prompt = build_prompt("persona_standard", evidence_text, reference_text, CPS) + JSON_INSTRUCTION

        t0 = time.time()
        content, prompt_tokens, completion_tokens, finish_reason = litai_client.chat(
            prompt, max_tokens=args.max_tokens
        )
        elapsed = time.time() - t0
        total_tokens = prompt_tokens + completion_tokens

        rec = {
            "case_id": case_id, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "total_tokens": total_tokens,
            "finish_reason": finish_reason, "elapsed_s": round(elapsed, 1),
        }

        try:
            parsed = extract_json_array(content)
            by = {r["cp"]: r.get("verdict", "") for r in parsed}
            rec["parsed_cps"] = len(by)
            rec["parse_ok"] = True
        except Exception as e:  # noqa: BLE001
            by = {}
            rec["parse_ok"] = False
            rec["parse_error"] = str(e)
            rec["raw_tail"] = content[-500:]

        usage_records.append(rec)
        print(f"{case_id}: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} "
              f"finish={finish_reason} parsed_cps={rec.get('parsed_cps', 0)} ok={rec['parse_ok']} "
              f"({elapsed:.1f}s)")

        row = {"case_id": case_id}
        row.update({cp: by.get(cp, "") for cp in CPS})
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + CPS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")

    total_prompt = sum(r["prompt_tokens"] for r in usage_records)
    total_completion = sum(r["completion_tokens"] for r in usage_records)
    summary = {
        "model": MODEL, "cases": len(args.cases),
        "total_prompt_tokens": total_prompt, "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "records": usage_records,
    }
    Path(args.usage_log).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.usage_log}")
    print(f"\nTOTAL: prompt={total_prompt:,} completion={total_completion:,} "
          f"grand_total={total_prompt + total_completion:,} tokens across {len(args.cases)} cases")


if __name__ == "__main__":
    main()
