"""Hybrid-RAG inference pipeline (comparison vs full-context).

Per case: chunk evidence -> hybrid retrieve (BM25 + dense -> weighted fusion
-> ColBERT rerank) per CP -> per-element batched Nemotron call -> verdicts.

Token-sane by design: governing policy sections are included ONCE per element
(not once per CP that cites them), and retrieved evidence chunks are
deduplicated across CPs. Reuses canonical persona_standard.md Rules/Output.

Resumable: per-(case,element) results cached under dev/rag_cache/.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
import litai_client  # noqa: E402
from retrieval import HybridRetriever, chunk_evidence, build_query  # noqa: E402
from ingest import TRACK_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
CPS_REFERENCE = ROOT / "dev" / "cp_reference.json"
CPS_MAPPING_DIR = ROOT / "cps_mapping"
POLICY_DIR = ROOT / "policy"
PROMPTS_DIR = ROOT / "prompts"
EVIDENCE_DIR = ROOT / "dev" / "evidence"
DEV_CASES = ROOT / "dev" / "dev_cases.csv"
CACHE_DIR = ROOT / "dev" / "rag_cache"

CPS = [f"CP{i}" for i in range(1, 42)]


def load_cp_reference():
    ref = json.loads(CPS_REFERENCE.read_text(encoding="utf-8"))
    by_cp = {r["cp"]: r for r in ref}
    elements = {}
    for r in ref:
        elements.setdefault(r["element"], []).append(r["cp"])
    return by_cp, elements


def load_citations():
    cites = {}
    # glob original mapping only (exclude _v2 regenerated variants)
    for f in sorted(CPS_MAPPING_DIR.glob("element[0-9].json")):
        for entry in json.loads(f.read_text(encoding="utf-8")):
            cites[entry["cp"]] = entry.get("citations", [])
    return cites


def section_text(citation):
    p = ROOT / citation
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_rag_context_block(cp_list, by_cp, cites, retrieved):
    """Dedup'd context: CP index -> unique policy sections -> unique evidence
    chunks (labeled E#) -> per-CP evidence pointer list."""
    # --- CP index (text + citation names only) ---
    parts = ["## Context\n", "### Checking points (official text + governing citations)\n"]
    for cp in cp_list:
        r = by_cp[cp]
        cnames = ", ".join(cites.get(cp, []) or ["(none)"])
        parts.append(f"\n**{cp}** — {r['subelement']}: {r['text']}\n  _governed by: {cnames}_")

    # --- unique policy sections, each once ---
    seen_sec = {}
    sec_order = []
    for cp in cp_list:
        for c in cites.get(cp, []):
            if c not in seen_sec:
                seen_sec[c] = True
                sec_order.append(c)
    parts.append("\n\n### Governing policy sections (full text, each once)\n")
    for c in sec_order:
        parts.append(f"\n{'='*60}\n#### {c}\n{'='*60}\n{section_text(c)}")

    # --- dedup'd evidence chunks ---
    chunks_seen = {}
    chunk_labels = []
    cp_evidence = {}
    for cp in cp_list:
        cp_evidence[cp] = []
        for track, text in retrieved.get(cp, []):
            key = (track, text[:120])
            if key not in chunks_seen:
                label = f"E{len(chunk_labels)+1}"
                chunks_seen[key] = label
                chunk_labels.append((label, track, text))
            cp_evidence[cp].append(chunks_seen[key])

    parts.append("\n\n### Retrieved evidence (deduplicated)\n")
    for label, track, text in chunk_labels:
        tname = TRACK_NAMES.get(track, f"Track {track}")
        parts.append(f"\n**[{label}] {tname}:** {text}")

    parts.append("\n\n### Evidence relevant to each checking point\n")
    for cp in cp_list:
        refs = ", ".join(cp_evidence[cp]) if cp_evidence[cp] else "(none retrieved)"
        parts.append(f"- **{cp}**: {refs}")

    return "\n".join(parts)


def render_persona(context_block, cp_list):
    tpl = (PROMPTS_DIR / "persona_standard.md").read_text(encoding="utf-8")
    scope = f"these {len(cp_list)} checking point(s): {', '.join(cp_list)}"
    out = tpl.replace("{CONTEXT_BLOCK}", context_block).replace("{CP_SCOPE_TEXT}", scope)
    return out + (
        "\n\nRespond with ONLY a raw JSON array (no markdown fences, no prose), one object per CP: "
        '{"cp": "<CPn>", "reasoning": "<2-3 sentences>", "verdict": "1"|"0"|"N/A", '
        '"confidence": 0.0-1.0, "cited_evidence": "<short quote>"}.'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--fuse-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--out", default=str(ROOT / "dev" / "predictions_rag.csv"))
    ap.add_argument("--usage-log", default=str(ROOT / "dev" / "rag_usage.json"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    by_cp, elements = load_cp_reference()
    cites = load_citations()

    if args.cases:
        case_ids = args.cases
    else:
        import csv as _csv
        with open(DEV_CASES) as f:
            case_ids = [r["dev_id"] for r in _csv.DictReader(f)]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    usage = []
    rows = []

    for case_id in case_ids:
        evidence = (EVIDENCE_DIR / f"{case_id}.md").read_text(encoding="utf-8")
        chunks = chunk_evidence(evidence, case_id)
        retriever = HybridRetriever(chunks, alpha=args.alpha)

        retrieved = {}
        for cp in CPS:
            ctext = by_cp[cp]["text"]
            gs = [section_text(c) for c in cites.get(cp, []) if section_text(c)]
            query = build_query(ctext, gs)
            hits = retriever.retrieve(query, top_k=args.top_k, fuse_k=args.fuse_k)
            retrieved[cp] = [(h[0], h[2]) for h in hits]

        case_verdicts = {}
        for element, cp_list in elements.items():
            cache_file = CACHE_DIR / f"{case_id}__{element}__a{args.alpha}.json"
            if cache_file.exists() and not args.no_resume:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                for cp, v in cached.get("verdicts", {}).items():
                    case_verdicts[cp] = v
                usage.append(cached.get("usage", {}))
                continue

            ctx = build_rag_context_block(cp_list, by_cp, cites, retrieved)
            prompt = render_persona(ctx, cp_list)
            t0 = time.time()
            content, pt, ct, finish = litai_client.chat(prompt, max_tokens=args.max_tokens)
            elapsed = time.time() - t0

            rec = {
                "case_id": case_id, "element": element, "prompt_tokens": pt,
                "completion_tokens": ct, "total_tokens": pt + ct,
                "finish_reason": finish, "elapsed_s": round(elapsed, 1),
                "prompt_chars": len(prompt),
            }
            try:
                parsed = litai_client.extract_json_array(content)
                verd = {r.get("cp"): r.get("verdict", "") for r in parsed}
                rec["parsed_cps"] = len(verd)
                rec["parse_ok"] = True
            except Exception as e:  # noqa: BLE001
                verd = {}
                rec["parse_ok"] = False
                rec["parse_error"] = str(e)
                rec["raw_tail"] = content[-400:]

            cache_file.write_text(json.dumps({
                "case_id": case_id, "element": element,
                "verdicts": verd, "usage": rec, "raw_content": content,
            }), encoding="utf-8")

            for cp, v in verd.items():
                case_verdicts[cp] = v
            usage.append(rec)
            print(f"{case_id} {element}: prompt={pt} comp={ct} total={pt+ct} finish={finish} "
                  f"parsed={rec.get('parsed_cps',0)} ok={rec['parse_ok']} ({elapsed:.0f}s)", flush=True)

        row = {"case_id": case_id}
        row.update({cp: case_verdicts.get(cp, "") for cp in CPS})
        rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id"] + CPS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")

    tp = sum(r["prompt_tokens"] for r in usage)
    tc = sum(r["completion_tokens"] for r in usage)
    summary = {
        "alpha": args.alpha, "top_k": args.top_k, "fuse_k": args.fuse_k,
        "cases": len(case_ids), "calls": len(usage),
        "total_prompt_tokens": tp, "total_completion_tokens": tc, "total_tokens": tp + tc,
        "records": usage,
    }
    Path(args.usage_log).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.usage_log}")
    print(f"\nTOTAL: prompt={tp:,} completion={tc:,} grand_total={tp+tc:,} tokens, {len(usage)} calls, {len(case_ids)} cases")


if __name__ == "__main__":
    main()
