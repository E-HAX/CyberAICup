"""Standalone, organizer-reproducible inference script.

Unlike the Workflow-based dev-loop (scripts/inference/freca_dev_workflow.js,
which relies on Claude Code subagents Reading files themselves), this script
has NO tool access to give a model -- it reads prompts/*.md and
okf/reference.md itself and embeds the content directly into the request, so
it can be run standalone against a real provider API with just a pinned
model name + an API key. This is what organizers would run to reproduce a
submission.

Every request is content-addressed (MD5 of provider|model|temperature|prompt)
and cached under scripts/inference/cache/ -- a repeat run with unchanged
inputs never re-calls the API. Every request (hit or miss) is appended to
scripts/inference/transcripts/transcript.jsonl for the verification bundle.

--dry-run builds the exact prompt and computes its hash without calling any
API or spending any credits -- use this to verify prompt construction is
deterministic before spending real calls.

Usage:
  python run_standalone.py --dry-run --cases RE-TAS-2021-0006 RE-NT-2022-0008
  python run_standalone.py --provider anthropic --cases RE-TAS-2021-0006 --out dev/predictions_standalone.csv
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = ROOT / "prompts"
OKF_REFERENCE = ROOT / "okf" / "reference.md"
CACHE_DIR = ROOT / "scripts" / "inference" / "cache"
TRANSCRIPT_DIR = ROOT / "scripts" / "inference" / "transcripts"

# Pinned model snapshots -- fill in the exact dated snapshot string once the
# model choice for submission is locked. An alias like "claude-sonnet-5"
# resolves to whatever is current at call time and is NOT reproducible on
# its own; this constant is what actually gets declared to organizers.
PINNED_MODELS = {
    "anthropic": None,  # e.g. "claude-sonnet-4-5-20250929" -- TODO pin before submission
    "gemini": None,      # e.g. "gemini-2.5-pro-XXXXXXXX" -- TODO pin once Gemini key confirmed
}
TEMPERATURE = 0.0
CPS = [f"CP{i}" for i in range(1, 42)]

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cp": {"type": "string"},
                    "reasoning": {"type": "string", "maxLength": 400},
                    "verdict": {"type": "string", "enum": ["1", "0", "N/A"]},
                    "confidence": {"type": "number"},
                    "cited_evidence": {"type": "string", "maxLength": 300},
                },
                "required": ["cp", "reasoning", "verdict", "confidence"],
            },
        },
    },
    "required": ["results"],
}


def load_template(name):
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(tpl, vars_):
    out = tpl
    for k, v in vars_.items():
        out = out.replace("{" + k + "}", v)
    return out


def context_block(evidence_text, reference_text):
    return (
        "## Context\n\n"
        "### Case evidence\n\n" + evidence_text + "\n\n"
        "### OKF reference bundle (official CP text + governing policy sections)\n\n" + reference_text
    )


def scope_text(cps):
    if len(cps) >= 41:
        return "all 41 checking points (CP1 through CP41)"
    return f"these {len(cps)} checking point(s): " + ", ".join(cps)


def build_prompt(persona, evidence_text, reference_text, cps):
    tpl = load_template(persona)
    return render(tpl, {
        "CONTEXT_BLOCK": context_block(evidence_text, reference_text),
        "CP_SCOPE_TEXT": scope_text(cps),
    })


def request_hash(provider, model, temperature, prompt):
    h = hashlib.md5()
    h.update(f"{provider}|{model}|{temperature}|{prompt}".encode("utf-8"))
    return h.hexdigest()


def write_transcript(entry):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_DIR / "transcript.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def call_anthropic(model, prompt):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"name": "submit_verdicts", "input_schema": PERSONA_SCHEMA}],
        tool_choice={"type": "tool", "name": "submit_verdicts"},
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("no tool_use block in Anthropic response")


def call_gemini(model, prompt):
    import google.generativeai as genai
    genai.configure()  # reads GOOGLE_API_KEY / GEMINI_API_KEY
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
            "response_schema": PERSONA_SCHEMA,
        },
    )
    return json.loads(resp.text)


def run_persona(provider, evidence_text, reference_text, case_id, persona, cps, dry_run):
    prompt = build_prompt(persona, evidence_text, reference_text, cps)
    model = PINNED_MODELS[provider]
    h = request_hash(provider, model, TEMPERATURE, prompt)
    cache_file = CACHE_DIR / f"{h}.json"

    transcript = {
        "case_id": case_id, "persona": persona, "cps": cps,
        "provider": provider, "model": model, "temperature": TEMPERATURE,
        "prompt_hash": h, "prompt_chars": len(prompt),
        "dry_run": dry_run,
    }

    if dry_run:
        write_transcript(transcript)
        return {"_dry_run": True, "prompt_hash": h, "prompt_chars": len(prompt)}

    if cache_file.exists():
        transcript["cache_hit"] = True
        write_transcript(transcript)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if model is None:
        raise SystemExit(f"no pinned model configured for provider={provider} -- set PINNED_MODELS in this script")

    if provider == "anthropic":
        result = call_anthropic(model, prompt)
    elif provider == "gemini":
        result = call_gemini(model, prompt)
    else:
        raise SystemExit(f"unknown provider {provider}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    transcript["cache_hit"] = False
    write_transcript(transcript)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--evidence-dir", default=str(ROOT / "dev" / "evidence"))
    ap.add_argument("--out", default=str(ROOT / "dev" / "predictions_standalone.csv"))
    ap.add_argument("--dry-run", action="store_true", help="build + hash prompts, no API call, no cost")
    args = ap.parse_args()

    reference_text = OKF_REFERENCE.read_text(encoding="utf-8")
    rows = []

    for case_id in args.cases:
        evidence_path = Path(args.evidence_dir) / f"{case_id}.md"
        evidence_text = evidence_path.read_text(encoding="utf-8")
        res = run_persona(args.provider, evidence_text, reference_text, case_id, "persona_standard", CPS, args.dry_run)

        if args.dry_run:
            print(f"{case_id}: dry-run OK  hash={res['prompt_hash']}  {res['prompt_chars']:,} chars")
            continue

        by = {r["cp"]: r["verdict"] for r in res.get("results", [])}
        row = {"case_id": case_id}
        row.update({cp: by.get(cp, "") for cp in CPS})
        rows.append(row)

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case_id"] + CPS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
