# FRECA — Task 2 EDA, Research & Architecture Plan

_CyberAI Cup 2026 · Task 2 · Farm Registered Establishment Compliance Audit_
_Prepared 2026-08-12. Owner: Task-2 lead + Mahek._

---

## 0. TL;DR

- The dataset is a **blind test set**: 99 case folders provided (spec says 100), **no ground-truth labels**, and the `submission_template.xlsx` is **empty** (header only, no RE numbers pre-filled). We must derive everything unsupervised and fill in the RE-Number column ourselves.
- **This kills FRECA v4 Phase 4 outright.** Isotonic Regression / GroupKFold / ECE gating all require labels to fit and validate. There are none. Any "trained calibrator" phase is impossible and must be replaced with **unsupervised** disagreement/verification logic.
- The core difficulty is **signal vs. noise**: each case mixes deliberately planted deficiencies (both explicit and subtle) with heavy synthetic template-noise (the "commodity" field is randomised per document in *every* case). Winning = reading substantive compliance content while ignoring cosmetic mismatches.
- FRECA v4 has genuinely strong bones (OKF policy mapping, applicability gate, prompt-persona ensemble at T=0, determinism harness). We keep ~70% of it. We cut the supervised calibration, fix the citation-verification gate (it's biased toward "0" and will cost accuracy), and right-size the ETL and call-count.
- Recommended: **FRECA v5** — a structure-then-reason, per-CP, policy-grounded pipeline with prompt-ensemble self-consistency, entailment-based (not cosine-threshold) citation checks, and unsupervised conservative arbitration. Details in §5.

---

## 1. What the task actually is

- **Input per case:** 9 evidence tracks (Registration form, HACCP plan, Pest Control Record, Farm Management Plan, Farm Site Plan, Hygiene & Sanitation Plan, Bait Station Map, Phytosanitary Security Procedure, Traceability Records). Tracks 1/2/4/5/6/7/8 are `.docx`; tracks 3 and 9 are `.xlsx`.
- **Output per case:** 41 verdicts CP1–CP41, each ∈ {`1` compliant, `0` non-compliant, `N/A` not-applicable}. 99×41 ≈ 4,059 decisions (spec: 100×41 = 4,100).
- **Scoring:** plain accuracy at three granularities — overall, per-CP (×41), per-element (×4). **All three are accuracy, so every cell matters equally; there is no partial credit and no class-weighting.** Maximise raw correct-cell count.
- **Hard constraints:**
  1. **Human-involvement / no hard-coding:** prompts must NOT encode CP pass/fail logic (e.g. "CP3 requires X"). Reasoning must be derived from the policy PDF + evidence. Violations → disqualified at method verification.
  2. **Reproducibility:** at verification you submit the exact prompt(s) + model name/version; the run must reproduce. Determinism is mandatory.

---

## 2. EDA — ground truth about the data (verified, not assumed)

### 2.1 Inventory
- `SFRE_cases/` contains **99** `RE-<STATE>-<YEAR>-<NNNN>` folders (spec says 100 → confirm whether one case is intentionally withheld or missing before submitting; the submission must match provided identifiers exactly).
- Every folder has exactly 9 files, numbered `1_…` … `9_…`, so lexicographic sort = track order. Good — deterministic ingestion is trivial.
- `checkingpoints_all_elements_onesheet.xlsx`: one sheet, 3 rows × 41 cols. Row 1 = element/sub-element grouping, **Row 2 = the official CP text (one per column)**, Row 3 = CP labels `CP1..CP41`. We have all 41 CP definitions verbatim (extracted).
- Policy PDF: **132 pages**, ~266k chars, the *Export Control (Plants and Plant Products) Rules 2021*, Compilation No. 3 (4 Aug 2023). Sections are numbered `X-Y` (e.g. `4-2`, `4-3`, `4-7A`) → cleanly chunkable by section.
- `submission_template.xlsx`: header `RE Number, CP1…CP41`, **zero data rows**. We generate the 99 (or 100) rows.

### 2.2 The two things that will decide our score

**(A) Deficiencies are deliberately planted — at two difficulty levels.**
- **Explicit:** literal in-text markers like `[DEFICIENCY — NOT DOCUMENTED]` (e.g. HACCP plan missing waste-management section; missing phytosanitary-security section). These are near-free points if we read the whole doc.
- **Subtle / semantic:** buried facts that violate a specific CP, e.g.
  - "Records are maintained in **Mandarin**…" → violates CP22/CP23/CP40 ("records must be in English").
  - Door "threshold seals … **worn with gaps of 8–22 mm**" → pest-exclusion / design CP (2.2).
  - Retention "**Jan 2025 – current**" with a June-2025 record date, target "≥ 2 years", status "migration in progress" → **<2-year** retention → CP22/CP38 non-compliant.
  - "Substitution controls rely on **single-operator checks** for late-shift dispatches", "flow chart **omits one alternate transfer path**", "temporary **mesh** segregation pending permanent barrier", "pesticide cartons held in **open shelf** during stocktake" → operational/phytosanitary/chemical-storage CPs.
  These require the model to *reason against the specific CP obligation*, not pattern-match keywords. This is where accuracy is won or lost.

**(B) Massive synthetic template-noise that is NOT a compliance signal.**
- The **"Registered Commodity" field is randomised per document in every case.** In one folder the seven docx tracks claimed: Barley / Field Peas / Mangoes / Chickpeas / Rice&Pulses across tracks — and this happens in *all* sampled cases. Some tracks even name a *different establishment and RE number* in their HACCP header (e.g. "Longreach Cattle & Grain", "RE-QLD-3089").
- **Interpretation:** these are generation artifacts, not planted traceability violations. The *consistent* identity anchors are the **folder RE number** and the **establishment name in tracks 4–8**. The commodity/name-per-track is noise.
- **Consequence (critical):** a naïve auditor that flags "documents disagree on commodity/establishment" as a traceability failure would mark almost every case non-compliant on the 4.1 CPs and tank accuracy. The pipeline must **anchor case identity on the folder RE number and NOT penalise cross-track commodity/name mismatch.** We should state this explicitly in the prompt as context ("evidence tracks are independently authored; treat header commodity/name fields as non-authoritative") — note this is *evidence-handling guidance, not CP logic*, so it stays within the no-hard-coding rule.

### 2.3 Format specifics for the ETL
- The "Farm Site Plan" (Track 5) and "Bait Station Map" (Track 7) are **mostly prose + a small ASCII box**, not real coordinate grids. FRECA v4's "parse ASCII grid → coordinate list" step is over-engineered for this data — plain text extraction (paragraphs + tables) is sufficient. Don't build a grid parser.
- `.xlsx` tracks (3, 9) are multi-sheet registers (Pest Activity Log, Bait Station Register, Chemical Storage Register, Establishment Condition; Receival/Movement/Treatment/Dispatch/Transport/Rejected/Archive). These carry the hard evidence for Elements 3 & 4 (dates, English-ness, retention windows, station condition). Flatten each sheet to labelled text rows; **preserve dates and the literal cell values** — the subtle violations live here.

---

## 3. Research synthesis (full report saved separately)

Literature converges on a **structure-then-reason** pipeline. The load-bearing, well-established techniques we should adopt:

- **Compliance-unit decomposition + CP→section pre-mapping** (GraphCompliance 2025; Compliance-to-Code 2025): formalise each CP as `{subject, obligation, condition, exception}` tied to specific Rule sections. Reserve the LLM for the semantic 1/0 call; let deterministic logic handle N/A gating. → validates FRECA's OKF idea.
- **Section-aware chunking + hybrid (BM25 + dense) retrieval + cross-encoder rerank** for the policy side. Legal text is keyword-heavy, so BM25 matters. Because the Rules PDF is a *single small instrument*, we can often just place the whole governing section(s) in-context per CP and skip retrieval on the policy side — retrieving only helps on the evidence side.
- **IRAC reasoning (Issue–Rule–Application–Conclusion), model-generated CoT** (MSLR-Bench, LegalBench): auditable and correlates with expert judgement.
- **Self-consistency / majority vote** (Wang et al., ICLR 2023): highest-ROI accuracy lever; vote fraction is a free confidence proxy.
- **Chain-of-Verification + entailment/NLI citation checks** (CoVe 2024; attribution work): verify the cited section actually *entails* the verdict. Crucially — on failure, **downgrade confidence / re-examine, do NOT blanket-flip to non-compliant.**
- **Reason-then-format** (Let Me Speak Freely, EMNLP 2024): reason in free text, emit constrained JSON in a separate/late step; enum-constrain to {1,0,N/A} so the 100×41 matrix never has a parse error.
- **Determinism harness:** pinned model snapshot, temperature 0 (or fixed seed), cached prompts+responses, logged retrievals/traces → reproducibility + audit trail.
- **Calibration caveats (important given our reality):** temperature scaling ≫ isotonic on ~100 points; **but we have zero labels**, so *no* supervised calibration is available at all. Few-shot caveat: stuffing many exemplars *hurts* accuracy.

Full annotated report with venues, arXiv IDs, and per-technique "how it applies" notes is in `RESEARCH_NOTES.md`.

---

## 4. Critical analysis of the proposed FRECA v4 architecture

**Verdict: strong skeleton, one fatal phase, two dangerous details, some over-engineering. Keep ~70%.**

### 4.1 What is genuinely good — keep it
- **OKF knowledge bundle (Phase 1):** having an LLM map each CP → governing policy sections/definitions, stored as per-CP markdown with a `citations:` list, is the right way to satisfy the no-hard-coding rule *and* get grounded retrieval. The mapping is derived from the policy, not authored by us. ✅ Keep.
- **Applicability gate with "assume applicable unless explicit documented exemption" (Phase 3.2):** correct instinct — prevents N/A being used as an escape hatch. Given scoring is pure accuracy and N/A is a real third class, over-using N/A is costly. ✅ Keep, but see §5.4 for how N/A should actually be decided.
- **Prompt-persona ensemble at temperature 0 (Phase 3.3):** clever. Standard/Adversarial/Step-by-Step personas give *diversity for self-consistency while staying deterministic* (normal self-consistency needs T>0, which breaks reproducibility; persona-diversity at T=0 sidesteps that). ✅ Keep — this is one of the best ideas in v4.
- **Determinism/verification packaging (Phase 5):** MD5-keyed response cache, pinned `requirements.txt`, declared prompts, hardware/determinism note. ✅ Exactly what method-verification needs.
- **Deterministic, lexicographic, OS-agnostic ETL (Phase 2):** right principle. ✅ Keep the sorting/flattening; drop the grid parser (§4.3).

### 4.2 Fatal flaw — Phase 4 cannot exist
- **There are no labels.** Isotonic Regression needs `(features → true label)`; GroupKFold needs `y` to split; ECE needs `acc(B_m)`. With zero labels, none of this can be fit or evaluated. **Delete Phase 4 as written.**
- Replace with **unsupervised arbitration** (§5.5): combine the 3 persona verdicts + the applicability gate + the citation-entailment check into a rule-based decision, using agreement as confidence. No fitting required. If we want *any* calibration, the only honest source is a **small dev set we hand-label ourselves** (see §7) — used strictly to choose prompts/thresholds, never as training data baked into the submission logic.

### 4.3 Dangerous details / over-engineering — fix these
1. **Citation-verification gate defaulting to `0` on low cosine similarity (Phase 3.4) is actively harmful.** Two problems: (a) cosine similarity between free-text reasoning and policy text is a weak proxy for "the citation supports the verdict" — a correct *compliant* verdict often paraphrases and scores low; (b) auto-flipping to non-compliant systematically biases the whole submission toward `0`, which will *lose* accuracy on the many genuinely-compliant CPs. **Fix:** replace cosine-threshold with an **entailment check** ("does cited section + cited evidence entail this verdict?"); on failure, **re-run / lower confidence / send to arbitration — never blanket-set 0.** A blind global 0.78 threshold with no labels to tune it is a guess that can only hurt.
2. **Call-count blow-up.** Per-CP × 3 personas × (gate + generation) × 99 cases ≈ 16k+ LLM calls plus embeddings. Slow, expensive, and fragile. **Fix options (pick per budget):** (a) batch the *applicability gate* for all 41 CPs in one call per case; (b) group CP generation by element (4 calls × 3 personas) or evaluate all 41 CPs in one structured call per persona per case (≈ 99×3 ≈ 300 calls) using long context — modern long-context models handle the whole policy + 9 tracks + 41 CPs at once. Start with **per-element batching** (good accuracy/cost balance), reserve **per-CP** calls only for CPs that come back low-confidence.
3. **ASCII-grid → coordinate parser (Phase 2.2):** unnecessary — the maps are prose. Extract text; skip the geometry engine.
4. **Model = Gemini is a *choice*, not a requirement.** Nothing in the task mandates Gemini. Pick on merit: long-context reasoning quality + reproducible pinned snapshots. Candidates: Gemini 2.5 Pro (huge context, cheap), or Claude / GPT long-context. Recommend a **quick 5-case bake-off** on a hand-labeled dev set before committing. (`text-embedding-004` is only needed if we keep an embedding step — with entailment-based verification we may not.)

### 4.4 Net changes v4 → v5
- ➖ Delete supervised calibration (Phase 4). Replace with unsupervised arbitration.
- 🔧 Replace cosine citation-gate with entailment; never auto-flip to 0.
- 🔧 Batch calls (element-level default) to cut cost ~50×.
- ➖ Drop ASCII-grid parser.
- ➕ Add explicit "template-noise handling" guidance (anchor on folder RE; ignore per-track commodity/name mismatch).
- ➕ Add a small self-labeled dev set for prompt/threshold selection.
- ✅ Keep OKF bundle, applicability gate, persona ensemble @T=0, determinism/packaging.

---

## 5. Recommended architecture — FRECA v5

### Phase 0 — Dev harness (do this first, half a day)
- Hand-label a **dev set of 8–10 cases** across states/commodities (≈ 400 cells). This is the ONLY thing that lets us measure whether a prompt change helps. Store as `dev/labels.csv`. It is used for *evaluation of our own prompts*, never encoded into submission logic.
- Build the scorer: overall / per-CP / per-element accuracy against dev labels.

### Phase 1 — OKF knowledge bundle (offline, once)
1. `PyMuPDF` → extract the Rules PDF, chunk by section `X-Y(...)`, one markdown file per section under `policy/`, tagged with citation id.
2. Load the 41 official CP texts from the onesheet (already extracted).
3. One heavy grounded LLM pass: for each CP, map it to its governing section(s)/definitions → `cps/cp_XX.md` with YAML frontmatter `citations: [policy/section_4-2.md, …]` and the verbatim CP text. **Human-review the mapping** (allowed — it's derived from policy, not pass/fail logic). This bundle is a declared, reproducible artifact.

### Phase 2 — Deterministic evidence ETL (per case)
- `sorted(os.listdir(case))` ingestion. Parse docx (paragraphs + tables) and xlsx (all sheets → labelled rows), preserving dates and literal values.
- Emit one `evidence_context.md` per case: clearly delimited `## Track N — <name>` sections. Prepend a short factual note: identity = folder RE number; per-track commodity/name fields are non-authoritative.
- No thresholds, no compliance logic in the ETL. Pure text normalisation.

### Phase 3 — Grounded inference (per case)
- **OKF resolver:** for the target CP(s), concatenate the cited policy sections.
- **Applicability gate:** one batched call over all 41 CPs → `{cp: applicable bool}`. Rule in prompt: assume applicable unless an explicit, documented exemption exists.
- **Persona ensemble @ T=0:** for applicable CPs, run 3 deterministic persona prompts — *Standard*, *Adversarial* (actively hunts for the planted deficiency), *Step-by-Step* (IRAC). Default batching = **per element** (4 calls × 3 personas per case). Output JSON per CP: `reasoning` (free text, generated *before* the verdict field), `verdict` ∈ {1,0}, `confidence` 0–1, `cited_sections`.
- **Citation entailment check:** verify cited section + cited evidence entail the verdict. On failure → mark low-confidence for arbitration (do **not** force 0).

### Phase 4 — Unsupervised arbitration (replaces v4 Phase 4)
- Combine per-CP: applicability gate → if not applicable and no contrary evidence, `N/A`.
- Else majority vote of the 3 persona verdicts.
- **Confidence = f(agreement, entailment pass, persona confidences).** On 2–1 splits or entailment failure, trigger a **4th tie-breaker call** (full-context, per-CP, all tracks) rather than defaulting to a class.
- Thresholds (e.g. when a split defaults which way) are **selected on the dev set**, not fit on hidden labels.

### Phase 5 — Formatting & verification package
- Write `submission.xlsx` from the template: generate the RE-Number rows to match folder identifiers exactly; cells ∈ {`1`,`0`,`N/A`}; never reorder/rename columns; enum-constrained so no stray values.
- Verification bundle: MD5-keyed prompt→response cache (SQLite/JSON), pinned `requirements.txt`, the declared prompt templates (applicability + 3 personas + tie-breaker + OKF-mapping), model name/version, and `HARDWARE_AND_DETERMINISM.md`.

---

## 6. Compliance & reproducibility guardrails
- **No-hard-coding:** prompts reference the policy sections and the official CP text only; they never say "CP_k requires X". Evidence-handling notes (identity anchoring, template noise) are about *how to read the files*, not about pass/fail thresholds — keep them phrased that way. Have a second person read every prompt against the rule before submission.
- **Determinism:** pin the exact model snapshot; temperature 0; cache every call keyed by MD5(prompt+context); log retrievals and votes. Document the residual non-determinism of cloud inference.

## 7. Open decisions / risks
1. **99 vs 100 cases** — confirm with organizers whether one case is withheld; the submission RE-Number set must match exactly. **(Blocker to final submission — resolve early.)**
2. **Model choice** — run the 5-case bake-off (Gemini 2.5 Pro vs Claude vs GPT long-context) on dev labels; pick on accuracy + reproducible snapshot + cost.
3. **Batching granularity** — start per-element; escalate to per-CP only for low-confidence cells. Measure accuracy delta on dev set.
4. **N/A discipline** — the base rate of true N/A is unknown; over- or under-calling it is a pure accuracy risk. Validate the applicability-gate behaviour hard on the dev set.
5. **No labels** — accept we are fully unsupervised; the dev set is our only measuring stick, so invest in making it representative.

## 8. Build order
1. Phase 0 dev harness + hand-labeled dev set + scorer.
2. Phase 2 ETL (docx/xlsx → `evidence_context.md`).
3. Phase 1 OKF bundle (PDF chunk + CP→section mapping) with human review.
4. Phase 3 inference on the dev set only; iterate prompts against the scorer.
5. Model bake-off; lock model + prompts.
6. Phase 4 arbitration; tune thresholds on dev.
7. Full 99-case run; Phase 5 packaging; confirm case count with organizers.
