"""Grid search over hybrid-fusion weight (alpha) x dense model (bge-small vs
bge-base), measured purely on RETRIEVAL quality against gold — no LLM calls.

Metric (computed per CP, averaged over all 10 dev cases):
  - track-hit@k : does the retrieved top-k include a track that gold's cited
                  evidence references?  (coarse)
  - phrase-hit@k: does the retrieved top-k contain the distinctive tokens of
                  gold's quoted evidence snippet?  (fine — "did we retrieve the
                  decisive sentence")

Same query construction as the RAG pipeline (CP text + short governing-policy
expansion), so this directly predicts RAG downstream behavior.
"""
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))  # scripts/inference/
sys.path.insert(0, str(Path(__file__).resolve().parent))                  # scripts/
from retrieval import chunk_evidence, tokenize, build_query  # noqa: E402
from retrieval import BGE_QUERY_PREFIX  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT / "dev" / "evidence"
DRAFTS_DIR = ROOT / "dev" / "drafts"
CP_REFERENCE = ROOT / "dev" / "cp_reference.json"
CPS_MAPPING_DIR = ROOT / "cps_mapping"
DEV_CASES = ROOT / "dev" / "dev_cases.csv"

CPS = [f"CP{i}" for i in range(1, 42)]
TRACK_RE = re.compile(r"Track\s+(\d+)", re.I)
QUOTE_RE = re.compile(r"['\"]([^'\"]{6,})['\"]")
STOP = set("a an the and or of to in on for with at by from as is are was were be been it its this that these those not no do does did has have had shall must may can will would should".split())


def load_meta():
    by_cp = {r["cp"]: r for r in json.loads(CP_REFERENCE.read_text(encoding="utf-8"))}
    cites = {}
    for f in sorted(CPS_MAPPING_DIR.glob("element[0-9].json")):
        for e in json.loads(f.read_text(encoding="utf-8")):
            cites[e["cp"]] = e.get("citations", [])
    return by_cp, cites


def gold_evidence(case_id):
    d = json.loads((DRAFTS_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    return {v["cp"]: v.get("evidence", "") for v in d["verdicts"]}


def gold_tracks(ev):
    return set(int(m.group(1)) for m in TRACK_RE.finditer(ev))


def gold_quote_tokens(ev):
    quotes = QUOTE_RE.findall(ev)
    if not quotes:
        return None
    best = max(quotes, key=len)
    toks = [t for t in tokenize(best) if t not in STOP]
    return set(toks) if len(toks) >= 3 else None


def main():
    by_cp, cites = load_meta()
    with open(DEV_CASES) as f:
        case_ids = [r["dev_id"] for r in csv.DictReader(f)]

    from fastembed import TextEmbedding
    dense_small = TextEmbedding("BAAI/bge-small-en-v1.5")
    dense_base = TextEmbedding("BAAI/bge-base-en-v1.5")

    ALPHAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    DENSE = {"small": dense_small, "base": dense_base}

    # accumulators: (dense_name, alpha, k) -> [track_hits, phrase_hits, denom]
    stats = defaultdict(lambda: [0, 0, 0, 0, 0])  # track_hit, phrase_hit, track_denom, phrase_denom, n

    for case_id in case_ids:
        evidence = (EVIDENCE_DIR / f"{case_id}.md").read_text(encoding="utf-8")
        chunks = chunk_evidence(evidence, case_id)
        corpus = [c["text"] for c in chunks]
        tracks = [c["track"] for c in chunks]
        tok = [tokenize(t) for t in corpus]
        bm25 = BM25Okapi(tok)

        # precompute dense doc embeddings for both models
        doc_embs = {}
        for name, model in DENSE.items():
            doc_embs[name] = np.asarray(list(model.embed(corpus)), dtype="float32")
            doc_embs[name] /= (np.linalg.norm(doc_embs[name], axis=1, keepdims=True) + 1e-9)

        gev = gold_evidence(case_id)

        for cp in CPS:
            ctext = by_cp[cp]["text"]
            gs = [ (ROOT / c).read_text(encoding="utf-8") for c in cites.get(cp, []) if (ROOT / c).exists() ]
            query = build_query(ctext, gs)

            b = np.asarray(bm25.get_scores(tokenize(query)), dtype="float32")
            bn = (b - b.min()) / (b.max() - b.min() + 1e-9)

            dsc = {}
            for name, model in DENSE.items():
                q = np.asarray(list(model.embed([BGE_QUERY_PREFIX + query]))[0], dtype="float32")
                q = q / (np.linalg.norm(q) + 1e-9)
                s = (doc_embs[name] @ q).astype("float32")
                dsc[name] = (s - s.min()) / (s.max() - s.min() + 1e-9)

            gt = gold_tracks(gev.get(cp, ""))
            gq = gold_quote_tokens(gev.get(cp, ""))

            for name in DENSE:
                for alpha in ALPHAS:
                    fused = alpha * dsc[name] + (1 - alpha) * bn
                    for k in (3, 5):
                        top = np.argsort(-fused)[:k]
                        top_tracks = {tracks[i] for i in top}
                        st = stats[(name, alpha, k)]
                        if gt:
                            st[2] += 1
                            if top_tracks & gt:
                                st[0] += 1
                        if gq:
                            st[3] += 1
                            retrieved_text = " ".join(corpus[i] for i in top).lower()
                            if all(t in retrieved_text for t in gq):
                                st[1] += 1
                        st[4] += 1

    # print results
    for k in (3, 5):
        print(f"\n===== hit@{k} =====")
        print(f"{'dense':6} {'alpha':>5} {'track-hit':>11} {'phrase-hit':>12}")
        for name in ("small", "base"):
            for alpha in ALPHAS:
                th, ph, td, pd, n = stats[(name, alpha, k)]
                tr = th / td if td else 0
                pr = ph / pd if pd else 0
                print(f"{name:6} {alpha:>5} {tr:>10.3f} ({th}/{td}) {pr:>10.3f} ({ph}/{pd})")
        # best per model
        for name in ("small", "base"):
            best = max(ALPHAS, key=lambda a: (stats[(name, a, k)][0] / stats[(name, a, k)][2]) if stats[(name, a, k)][2] else 0)
            bth, _, btd, _, _ = stats[(name, best, k)]
            print(f"  -> {name} best alpha={best} track-hit={bth/btd:.3f} ({bth}/{btd})")


if __name__ == "__main__":
    main()
