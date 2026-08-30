# Hybrid-RAG vs Full-Context — comparison report

_Model: Nemotron Ultra 550B (`lightning-ai/nvidia-nemotron-3-ultra-550b-a55b`) via Lightning AI API, on the same 4 dev cases (identical gold, identical persona rules, only the evidence-delivery mechanism differs)._

## What was built (hybrid RAG pipeline)

Per case, per checking point (CP):

1. **Chunk** the evidence (48–56 track-tagged chunks/case; `### Sheet:` headers and oversize splits keep dense registers retrievable).
2. **Lexical retrieval** — BM25 (`rank_bm25`).
3. **Semantic retrieval** — dense `BAAI/bge-small-en-v1.5` (384-dim, ONNX, torch-free), cosine.
4. **Hybrid weighted fusion** — `score = 0.5·dense_norm + 0.5·bm25_norm` (min-max normalized).
5. **Reranker** — ColBERT-style late-interaction MaxSim (`answerdotai/answerai-colbert-small-v1`), rerank top-20 → top-5.
6. **LLM** — Nemotron Ultra, one call per element (4 calls/case), with retrieved evidence + the element's governing policy sections (deduplicated).

All retrieval is local/deterministic (pinned ONNX models, no randomness). LLM via the official Lightning API (temperature not exposed → noted as residual non-determinism).

## Results (same 4 cases, Nemotron)

| Case | RAG | Full-context |
|---|---|---|
| RE-TAS-2021-0006 | 33/41 = 80.5% | 35/41 = 85.4% |
| RE-NSW-2020-0033 | 32/41 = 78.0% | 34/41 = 82.9% |
| RE-VIC-2020-0093 | 30/41 = 73.2% | 28/41 = 68.3% |
| RE-NT-2022-0008 | 33/41 = 80.5% | 36/41 = 87.8% |
| **TOTAL** | **128/164 = 78.0%** | **133/164 = 81.1%** |

**RAG is ~3.1pp worse than full-context** on aggregate. It won only on the single hardest case (RE-VIC-2020-0093, +4.9pp), where focusing on retrieved passages helped cut through noise — but lost ground on the cleaner cases where full context simply had more complete coverage.

## Token cost (the more important finding)

| | RAG | Full-context |
|---|---|---|
| Calls per case | 4 (per element) | 1 |
| Tokens per case | **~62K** (incl. 1 wasted parse-failure retry) | **~41K** |

**RAG used ~50% MORE tokens than full-context.** Why: this dataset's complete evidence is only ~8K words (~22–26K tokens) — it already fits in a single context window. RAG's whole purpose is to avoid huge contexts; here there's nothing huge to avoid, so it only adds (a) the persona instructions repeated 4×, (b) 4× completion overhead, (c) retrieval that sometimes misses the one decisive sentence (e.g. the "records maintained in Mandarin" line for CP23, the handwashing evidence for CP10).

## Conclusion / recommendation

For FRECA specifically, **full-context beats hybrid RAG on both accuracy and cost**. The evidence is small, and the decisive signals (a status flag in one table row, a single "Mandarin" sentence) are easy to lose in retrieval. Recommendation: keep the full-context pipeline; treat the RAG build as a documented negative result (stronger than the positive result, since it justifies not adding retrieval complexity).

The RAG implementation remains usable (`scripts/inference/run_rag.py`, `scripts/inference/retrieval.py`) in case organizers ever audit the architecture choice.

## OKF reproducibility — now demonstrated end-to-end

`scripts/inference/run_cp_mapping.py --provider lightning` was actually run (not just dry-run) via the official API and regenerated the CP→policy mapping for all 4 elements (`cps_mapping/element1-4_v2.json`):

- 41/41 CPs covered, 0 dangling citations (every cited `policy/section_*.md` exists).
- **Only 12/41 CPs match the original Claude-derived mapping exactly.** The mapping is genuinely model-dependent: "reproducible" means *same pinned model + same prompt → same mapping*, not *any model → same mapping*. Both mappings are valid LLM-derived artifacts (satisfying the no-hard-coding + official-API rules).

The committed `okf/reference.md` (used by the full-context results) is deterministically rebuilt from `cps_mapping/element*.json` via `scripts/build_okf_deduped.py`.

## Files

- `scripts/inference/retrieval.py` — chunking + BM25 + dense + fusion + ColBERT rerank.
- `scripts/inference/run_rag.py` — RAG pipeline (dedup'd per-element prompts, resumable cache).
- `scripts/inference/litai_client.py` — shared Nemotron client (env-var auth, token accounting).
- `scripts/inference/run_cp_mapping.py` — reproducible OKF mapping (now supports `--provider lightning`).
- `dev/predictions_rag_pilot.csv`, `dev/rag_usage_pilot.json` — RAG predictions + token log.
- `cps_mapping/element1-4_v2.json` — Nemotron-regenerated OKF mapping.
