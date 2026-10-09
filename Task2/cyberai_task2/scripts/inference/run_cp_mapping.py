"""Standalone, organizer-reproducible OKF grounding pass (Phase 1).

Fixes the same gap the inference side had: the original cps_mapping/element*.json
files were produced by 4 ad-hoc Agent tool calls with instructions typed
directly into the tool call, not saved as a template, with no pinned model,
no temperature control, and no transcript. This script reconstructs that step
as a real, reproducible pipeline: canonical prompt (prompts/cp_mapping.md),
pinned model + temperature=0, MD5-cached responses, full transcript log, and
a --dry-run mode that verifies deterministic prompt construction at zero cost.

One call per element (matches the original structure): each call embeds ALL
184 policy sections (~268KB / ~67K tokens -- well within a long-context
model's window) + that element's CP official texts, and asks for citations
+ rationale per CP.

Usage:
  python run_cp_mapping.py --dry-run --elements Element-1 Element-2 Element-3 Element-4
  python run_cp_mapping.py --provider anthropic --elements Element-1 --out cps_mapping/element1_v2.json
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_standalone import render, write_transcript, PINNED_MODELS, TEMPERATURE, CACHE_DIR  # noqa: E402
import litai_client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = ROOT / "prompts"
POLICY_DIR = ROOT / "policy"
CP_REFERENCE = ROOT / "dev" / "cp_reference.json"

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cp": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": ["cp", "citations", "rationale"],
            },
        },
    },
    "required": ["results"],
}


def load_elements():
    ref = json.loads(CP_REFERENCE.read_text(encoding="utf-8"))
    elements = {}
    for r in ref:
        elements.setdefault(r["element"], []).append(r)
    return elements


def policy_sections_block():
    files = sorted(glob.glob(str(POLICY_DIR / "section_*.md")))
    parts = []
    for f in files:
        name = Path(f).relative_to(ROOT)
        parts.append(f"\n{'='*72}\n### policy/{name.name}\n{'='*72}\n" + Path(f).read_text(encoding="utf-8"))
    return "".join(parts)


def cp_list_block(cps):
    parts = []
    for r in cps:
        parts.append(f"\n### {r['cp']} ({r['element']} — {r['subelement']})\n{r['text']}")
    return "".join(parts)


def build_mapping_prompt(cps, sections_block):
    tpl = (PROMPTS_DIR / "cp_mapping.md").read_text(encoding="utf-8")
    return render(tpl, {
        "POLICY_SECTIONS_BLOCK": sections_block,
        "CP_LIST_BLOCK": cp_list_block(cps),
    })


def request_hash(provider, model, temperature, prompt):
    h = hashlib.md5()
    h.update(f"{provider}|{model}|{temperature}|{prompt}".encode("utf-8"))
    return h.hexdigest()


def call_anthropic(model, prompt):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"name": "submit_mapping", "input_schema": MAPPING_SCHEMA}],
        tool_choice={"type": "tool", "name": "submit_mapping"},
    )
    for block in resp.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("no tool_use block in Anthropic response")


def call_lightning(prompt):
    """Nemotron via Lightning (no tool-calling) -> {"results": [...]}."""
    content, pt, ct, finish = litai_client.chat(prompt + JSON_INSTRUCTION, max_tokens=8000)
    parsed = litai_client.extract_json_array(content)
    return {"results": parsed, "_prompt_tokens": pt, "_completion_tokens": ct, "_finish_reason": finish}


JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a raw JSON array (no markdown fences, no prose before or "
    "after), one object per CP: {\"cp\": \"<CPn>\", \"citations\": [\"policy/section_X-Y.md\", ...], "
    "\"rationale\": \"<1-3 sentences>\"}."
)


def run_element(provider, element, cps, sections_block, dry_run):
    prompt = build_mapping_prompt(cps, sections_block)
    model = litai_client.MODEL if provider == "lightning" else PINNED_MODELS[provider]
    h = request_hash(provider, model, TEMPERATURE, prompt)
    cache_file = CACHE_DIR / f"mapping_{h}.json"

    transcript = {
        "step": "cp_mapping", "element": element, "cps": [r["cp"] for r in cps],
        "provider": provider, "model": model, "temperature": TEMPERATURE,
        "prompt_hash": h, "prompt_chars": len(prompt), "dry_run": dry_run,
    }

    if dry_run:
        write_transcript(transcript)
        return {"_dry_run": True, "prompt_hash": h, "prompt_chars": len(prompt)}

    if cache_file.exists():
        transcript["cache_hit"] = True
        write_transcript(transcript)
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if model is None:
        raise SystemExit(f"no pinned model configured for provider={provider} -- set PINNED_MODELS in run_standalone.py")

    if provider == "anthropic":
        result = call_anthropic(model, prompt)
    elif provider == "lightning":
        result = call_lightning(prompt)
    else:
        raise SystemExit(f"provider {provider} not wired up for cp_mapping yet")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    transcript["cache_hit"] = False
    write_transcript(transcript)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "lightning"], default="anthropic")
    ap.add_argument("--elements", nargs="+", default=["Element-1", "Element-2", "Element-3", "Element-4"])
    ap.add_argument("--out-dir", default=str(ROOT / "cps_mapping"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    elements = load_elements()
    sections_block = policy_sections_block()
    print(f"policy sections block: {len(sections_block):,} chars (~{len(sections_block)//4:,} tokens)")

    for element in args.elements:
        cps = elements[element]
        res = run_element(args.provider, element, cps, sections_block, args.dry_run)

        if args.dry_run:
            print(f"{element}: dry-run OK  hash={res['prompt_hash']}  {res['prompt_chars']:,} chars  ({len(cps)} CPs)")
            continue

        out_path = Path(args.out_dir) / f"{element.lower().replace('-', '')}_v2.json"
        out_path.write_text(json.dumps(res.get("results", []), indent=2), encoding="utf-8")
        print(f"{element}: wrote {out_path} ({len(res.get('results', []))} CPs)")


if __name__ == "__main__":
    main()
