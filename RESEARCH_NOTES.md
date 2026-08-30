# FRECA Task 2 — Research Notes (papers & techniques)

_Compiled 2026-08-12. Full annotated report. See FRECA_PLAN.md for how these feed the architecture._

## Recommended pipeline shape (literature consensus)
Structure-then-reason, not one monolithic prompt:
1. Decompose policy into atomic compliance units; pre-map each CP → governing section(s).
2. Hybrid retrieval (BM25 + dense) + cross-encoder rerank, section-aware chunking (policy side often small enough to skip retrieval).
3. Cite-then-decide with post-hoc NLI/entailment verification of the citation.
4. Self-consistency (majority vote over N traces) + verbalized confidence.
5. Reason in free text first, then emit constrained JSON. Fixed seed/temp, pinned model.

## Theme 1 — LLMs for regulatory/legal compliance
- **GraphCompliance** (arXiv 2510.26309, 2025) [MUST-HAVE pattern]: Policy Graph of `{subject, constraint, context, condition}` + Context Graph; deterministic "compliance gate" before the semantic LLM call. +4.1–7.2 micro-F1 over LLM-only/RAG on 300 GDPR scenarios. → model each of 41 CPs as a compliance unit; deterministic N/A gating, LLM only for 1/0.
- **Compliance-to-Code** (2505.19804, 2025): turn objectively-checkable requirements into executable predicates. → split CPs into rule-computable (present/absent, date-in-window, English-ness, retention ≥2y) vs judgment-required.
- **CALLM / LLM+SMT** [nice-to-have]: formalize multi-condition rules, verify with a solver. Only for CPs with genuine boolean logic.
- **LLM-Driven GDPR Compliance** (FSE 2025), **LLM extraction of GDPR reqs** (RE 2025): LLMs good at extracting/structuring requirements, weak at end-to-end verdicts without scaffolding. → use LLM offline to author the CP→section map.
- **AI/NLP in Regulatory Compliance** (ACL 2025 Findings): field standard is extract → structure → match → explain.

## Theme 2 — RAG for legal/regulatory text
- **Section-aware chunking** (Legal Chunking 2024; "A New HOPE" 2505.02171) [MUST-HAVE]: whole section/clause = one chunk, summary-augmented, metadata (section id) attached. Keeps conditions+exceptions together; makes citation exact.
- **Hybrid + rerank** (TREC-2024 RAG; hybrid-search refs) [MUST-HAVE]: BM25 + dense, RRF (k≈60), cross-encoder rerank. BM25 often beats dense on keyword-heavy legal text. Our Rules instrument is small → may put whole governing section in-context and skip policy-side retrieval; retrieve over evidence only.
- **CoCoLex** (2508.05534) [nice]: confidence-guided copy decoding → quote exact rule text.
- **LegalBench-RAG / hallucination studies**: even commercial legal-RAG hallucinates; semantic similarity ≠ legal relevance → motivates the verification layer.

## Theme 3 — Hallucination & citation verification
- **Chain-of-Verification / CoVe** (ACL 2024 Findings) [MUST-HAVE]: draft → self-generated verification questions → revised answer. Cheap borderline-CP precision boost.
- **Claim-entailment / post-hoc NLI attribution** (2412.11404) [MUST-HAVE]: check the cited section+evidence logically entail the label. On failure, downgrade confidence — do NOT blanket-flip verdict.
- **SelfCheckGPT** (EMNLP 2023): inconsistency across samples signals hallucination → disagreement = uncertainty flag.
- **Faithfulness vs factuality**: our task is a faithfulness problem (world = provided evidence + Rules). Constrain reasoning to supplied context only.

## Theme 4 — Self-consistency, ensembling, LLM-as-judge, CoT
- **Self-Consistency** (Wang et al., ICLR 2023) [MUST-HAVE]: N diverse CoT traces → majority vote. Vote fraction = free confidence. (Reproducibility tension: normally needs T>0; we sidestep via persona-diversity at T=0.)
- **MSLR-Bench / IRAC** (2511.07979, 2025) [MUST-HAVE structure]: model-generated CoT > hand-crafted; structure as Issue–Rule–Application–Conclusion; IRAC recall correlates with expert judgement.
- **Multi-persona / LLM-as-judge** [nice, caveat]: prosecutor/defender/judge helps hard CPs but LLM judges are self-inconsistent — not sole high-stakes deciders. Use disagreement as a flag.
- **Reasoning-consensus DAG** (2607.27783) [nice/advanced]: weight conclusions by independent-trace support.

## Theme 5 — Confidence calibration  ⚠️ WE HAVE NO LABELS
- Temperature scaling ≫ Platt/isotonic on small sets; isotonic overfits ~100 points. ECE via equal-width bins.
- Verbalized confidence (2512.11998, 2603.09309): prompt for 0–100 confidence, calibrate post-hoc.
- **BUT: no labels are provided in this competition** → no supervised calibration is possible at all. Use self-consistency agreement as the (uncalibrated) confidence signal; any threshold tuning must use our own hand-labeled dev set, not hidden labels.

## Theme 6 — Small-sample / few-shot
- **Few-shot dilemma** (2509.13196): more exemplars can HURT (67%→54% with 100 in-context). TF-IDF selection of ≈2–4 relevant examples/class beats random/semantic. → don't stuff exemplars.
- Don't fine-tune on ~100 points (collapses toward uniform).
- Per-CP base-rate priors as tie-breakers [nice] — but we have no labels to estimate them, so only usable from our dev set with caution.

## Theme 7 — Structured generation & reproducibility
- **Let Me Speak Freely?** (EMNLP 2024, 2408.02442) [MUST-HAVE]: forcing JSON during reasoning degrades reasoning → reason free-text first, format second (or put `reasoning` before `verdict` in the JSON).
- **Constrained decoding** (Outlines/LM-Format-Enforcer/vLLM/XGrammar): guarantee valid JSON; enum-constrain to {1,0,N/A}. Validity ≠ correctness — apply only to the final extraction.
- **Determinism**: fixed seed + temp 0 + pinned model snapshot + fixed index + logged prompts/traces. Report run-to-run variance (2601.02370).

## Theme 8 — Agricultural/biosecurity compliance
- Little direct LLM work. Closest: "As You Wish: Mission Planning w/ Formal Verification in Precision Agriculture" (2606.18519); USDA Export Verification background. → our edge is transferring GDPR/financial/building-code compliance methods (Themes 1–3).

## Must-have vs nice-to-have
MUST: CP→section mapping as compliance units; rule-computable vs judgment split; IRAC free-text reasoning w/ citations; self-consistency majority vote; entailment citation check + CoVe on borderline; reason-then-format + enum-constrained JSON; full determinism harness.
NICE: multi-persona hard-CP pass; full policy/context KG + deterministic gate; LLM+SMT for boolean CPs; DAG consensus; base-rate priors.
AVOID: many few-shot exemplars; fine-tuning on the small set; isotonic calibration; forcing JSON during reasoning; trusting retrieval as grounding; treating valid JSON as correct.

## Key papers quick-ref
GraphCompliance 2510.26309 · Compliance-to-Code 2505.19804 · RAG compliance framework (COLING 2025) · LegalBench 2308.11462 · MSLR-Bench 2511.07979 · Self-Consistency (ICLR 2023) · CoVe (ACL 2024 Findings) · SelfCheckGPT (EMNLP 2023) · fine-grained attribution 2412.11404 · CoCoLex 2508.05534 · TREC-2024 RAG 2411.09607 · Let Me Speak Freely 2408.02442 · Few-shot Dilemma 2509.13196 · Variance-Aware Annotation 2601.02370.
_(Newest 2026-dated IDs are leads to verify; the load-bearing 2023–2025 techniques are established.)_
