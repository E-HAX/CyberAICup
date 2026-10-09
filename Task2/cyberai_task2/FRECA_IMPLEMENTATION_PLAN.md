# FRECA v5 — Implementation Plan

_Status snapshot 2026-08-15. Builds on FRECA_PLAN.md (architecture decision), EDA_FINDINGS.md, DATASET_CONSTRUCTION.md._

## 0. What already exists (done, verified)

| Phase | Artifact | State |
|---|---|---|
| Phase A (gold + scorer) | `dev/gold.csv` (10×41 cells, drafts accepted as gold per user instruction), `scripts/scorer.py` | Done. Self-check scores 1.0. |
| Phase B (logical-case manifest + full ETL) | `logical_cases.json` (100 logical cases), `evidence/*.md` (100 files, full corpus) | Done. 0 parse failures, 2 missing-track-1 notes as expected. |
| Phase 1 (OKF bundle, original per-CP form) | `policy/` (184 section files), `cps/cp_01..41.md`, `cps_mapping/element*.json` | Done, validated. Superseded operationally by the deduped `okf/reference.md` (§Phase D) for inference calls — kept as the audit-trail artifact. |
| Phase C (orchestration + dev model) | Workflow tool (Haiku, via `agent()`), `scripts/inference/freca_dev_workflow.js` | Decided + built. Gemini provider still pending the API key. |
| Phase D (inference engine) | `okf/reference.md`, `prompts/persona_*.md`, `prompts/tiebreaker.md`, `scripts/inference/freca_dev_workflow.js` | Built and cost-validated on real runs (see §Phase D below — this went through two redesigns after real-run token/cost data). |
| Phase E (arbitration) | Folded into the same Workflow script's stage 3 (JS, deterministic) | Built as part of Phase D's script, not a separate file. |
| Phase F (iterate on dev set) | `RESULTS_DEV_SONNET_V1.md`, `dev/predictions_sonnet_v1.csv` | 91.0% (373/410) on Sonnet single-pass. CP37/CP39 boilerplate + gold-inconsistency issues documented, not chased further per user call to move on. |
| Phase H (reproducibility half) | `scripts/inference/run_standalone.py`, `HARDWARE_AND_DETERMINISM.md` | Standalone, organizer-runnable script built: reads `prompts/*.md` + `okf/reference.md` directly (no drift from the dev-loop harness), MD5-cached responses, full transcript log, `--dry-run` mode verified deterministic (byte-identical prompt hashes across 2 runs, 0 API cost). **Submission-packaging half (xlsx writer, variant A/B) not started.** |
| Phase G (full-corpus run) | — | Not started. |
| Model/API access | Haiku/Sonnet via Workflow confirmed working; Gemini key **not ready yet**; no pinned model snapshot chosen yet (`PINNED_MODELS` in `run_standalone.py` is `None` for both providers) | Blocks any *real* (non-dry-run) standalone call and Phase G, not Phase D/F (those ran via Workflow). |

---

## Phase A — DONE. `dev/gold.csv` (10×41), `scripts/scorer.py` (overall/per-CP/per-element accuracy).

## Phase B — DONE. `scripts/build_manifest.py` → `logical_cases.json` (100 logical cases, doubled folder split into `RE-WA-2021-0077`/`RE-WA-2021-0077-B`). `scripts/build_all_evidence.py` → `evidence/*.md` (100 files, 0 failures).

---

## Phase C — Orchestration + dev iteration model

**Decision (2026-08-15):** production model = **Gemini API** (key not ready yet — production run deferred, see §7). Dev/prompt-iteration model = **Claude Haiku**, called via the **Workflow tool** (`agent()`, `model: 'haiku'`) rather than a standalone Python SDK client — no separate API key needed, and Workflow's `pipeline()`/`parallel()` give free fan-out concurrency for the ~13-300+ calls this pipeline needs at dev/full-run scale. This is an explicit, user-approved opt-in (Workflow requires it).

Gemini is NOT reachable through Workflow (Workflow/Agent only run Claude models). So the real split is: **prompt templates + the OKF reference bundle are provider-agnostic text/files**; the Workflow script drives Haiku now; a standalone `scripts/inference/providers/gemini.py` (Gemini SDK, same templates/files) gets wired in once the key lands, for the Phase G production run only.

**Deliverable:** `scripts/inference/freca_dev_workflow.js` (Workflow script), `okf/reference.md` (deduped provider-agnostic grounding bundle, see Phase D).

---

## Phase D — Grounded inference engine (per case)

This went through two redesigns after real-run data — both driven by cost problems the first two versions actually hit, not theoretical:

**v1 (per-element, 13 calls/case):** applicability gate (1 call, all 41 CPs) + 3 personas × 4 elements (12 calls) = 13 calls/case, each re-reading the full case evidence and its slice of `cps/cp_XX.md` files. Measured at ~1.16M chars for ONE case's worth of prompts when rendered inline — completely unworkable (would have meant typing megabytes into a tool call). Root cause: `cps/cp_XX.md` embeds each CP's *full* governing policy text, and adjacent CPs in the same element frequently cite the same section, so per-element batching re-reads duplicated policy text on top of re-reading full evidence 13 times.

**v2 (whole-case, 3 calls/case, first real run):** collapsed to one call per persona covering all 41 CPs, applicability folded into the verdict itself (`1`/`0`/`N/A` in one shot, no separate gate call). Deduped the policy grounding text: 41 CPs cite only 18 *unique* policy sections (64 citation instances → 18 unique = 4.3x duplication), built once via `scripts/build_okf_deduped.py`. Piloted at 1 case × 1 persona = **71.7K tokens, 1 agent, matches estimate** — but a majority-vote bug (`count >= 2` hardcoded instead of scaled to ensemble size) meant a 1-persona pilot forced *every one of the 41 CPs* into an unnecessary tiebreak call: **84 agents, ~1M tokens for what should have been 2 calls.** Fixed: majority threshold is now `floor(personas.length / 2) + 1`.

**v3 (current, confidence-gated escalation):** even at 3 calls/case, most CPs come back high-confidence on a single pass (pilot: 33+ of 41 CPs at confidence ≥0.85). Running the full 3-persona ensemble on every CP regardless is wasted spend on the easy majority. Current design:
1. **Stage 1 — standard pass, all 41 CPs, 1 call.** Reads `dev/evidence/<case>.md` + `okf/reference.md` (single merged file: CP index + deduped policy sections — was two files, merged to save a Read round-trip).
2. **Stage 2 — escalation, 0-2 calls.** CPs with confidence < 0.85 (configurable) get re-assessed by the adversarial and step-by-step (IRAC) personas, **scoped only to the escalated CP subset** (not all 41) — cuts ensemble output tokens roughly in proportion to how few CPs actually need it.
3. **Stage 3 — arbitration (plain JS, no agent call).** Non-escalated CPs keep the standard pass's verdict directly. Escalated CPs get majority vote across the 3 persona verdicts; genuine 3-way splits get one more tiebreak call (single CP, single call).
4. **Output discipline:** `reasoning` capped at 400 chars and `cited_evidence` at 300 chars via JSON Schema `maxLength` (structural cap, not just a prompt suggestion) — reasoning/evidence-quote text was the dominant *output*-token cost.
5. **No `agentType` override** — dropped the earlier `general-purpose` choice (full tool access including the `Agent` tool, i.e. capable of recursively spawning more agents) in favor of the Workflow default subagent, which does the Read + structured-output job without carrying Bash/Edit/Agent/Artifact tool-schema overhead in its system prompt.

Prompt templates: `prompts/persona_standard.md`, `prompts/persona_adversarial.md`, `prompts/persona_stepbystep.md`, `prompts/tiebreaker.md` — all reference `{EVIDENCE_FILE}` and `{REFERENCE_FILE}` as file paths the agent Reads itself (not inlined text), and a generic `{CP_SCOPE_TEXT}` so the same template serves both the full-41 standard pass and an escalation subset. **No CP-specific pass/fail logic hard-coded in any template** — verify against the no-hard-coding rule before locking (FRECA_PLAN.md §6).

**Deliverable:** `okf/reference.md`, `scripts/build_okf_deduped.py`, `prompts/persona_*.md` + `prompts/tiebreaker.md`, `scripts/inference/freca_dev_workflow.js` (Workflow script implementing stages 1-3 above — arbitration/Phase E is stage 3 of this same script, not a separate file).

**Cost, measured:** ~72K tokens for a 1-persona, all-41-CP pass on one case. Estimated full 3-persona-with-escalation pass: roughly 72K (standard) + escalation-subset-sized adversarial/stepbystep calls (proportional to how many CPs miss the confidence bar, typically well under 41) — materially less than the flat 3× (~215K/case) the v2 always-run-all-3 design would have cost, and far less than v1's unworkable ~1.16M chars/case.

---

## Phase F — Iterate on dev set until scorer plateaus

Loop: run Phase D (v3 script) on the 10 dev cases → export CSV → score against `dev/gold.csv` → adjust prompts/confidence threshold → re-score. Stop when accuracy gains flatten or time budget is spent. This is the only place prompt engineering happens. Not started yet — next step after this plan update.

---

## Phase G — Full-corpus run

1. Run the locked Phase D pipeline over all 100 logical cases (`evidence/*.md` from Phase B, not `dev/evidence/`). Call-count estimate with escalation: ~100 standard-pass calls + escalation calls proportional to how many CPs miss the confidence bar per case (pilot suggests well under 41/case) + rare tiebreak calls — nowhere near the old flat ~1,300-call (v1) or even ~300-call (v2) estimate.
2. Needs the Gemini provider wired in (§7, still blocked on the key) unless the decision is made to just run Phase G on Haiku too.
3. Spot-check a sample against manual read (not the dev set, to catch generalization issues).

---

## Phase H — Packaging (submission + verification bundle)

1. `scripts/make_submission.py`: from arbitration output + `Task2/submission_template.xlsx` header, write:
   - **Variant A** — 99 rows (doubled folder counted once, Goldfields set, per DATASET_CONSTRUCTION.md §3).
   - **Variant B** — 100 rows (split, provisional 2nd ID for Midwest).
   Cells enum-validated ∈ {1,0,N/A} before write; column order untouched.
2. Verification bundle: pinned `requirements.txt`, all prompt templates, model name/version, MD5-keyed cache dump, `HARDWARE_AND_DETERMINISM.md` describing residual cloud-inference non-determinism.
3. Send the 4 blocker questions from DATASET_CONSTRUCTION.md §4 to organizers if not already sent (row count, Midwest RE number, missing-Track-1 intent, N/A-vs-0 scoring) — re-key submission fast once answered.

---

## Build order

1. ~~Phase A~~ — done.
2. ~~Phase B~~ — done.
3. ~~Phase C~~ — orchestration decided (Workflow + Haiku); Gemini provider still pending the key.
4. ~~Phase D~~ — v3 (confidence-gated escalation) built and cost-validated on single-case pilots. Not yet run across the full 10-case dev set.
5. **Phase F (next)** — run v3 across all 10 dev cases with the full persona ensemble, export to submission-shaped CSV, score against `dev/gold.csv`, iterate.
6. **Phase G** — full 100-case run, blocked on Gemini key (or a decision to run it on Haiku instead).
7. **Phase H** — packaging.

---

## §7. Resolved / open

**Resolved (2026-08-15):**
- Model: **Gemini** for production (key not ready — see below), **Claude Haiku via the Workflow tool** for dev iteration.
- Dev gold set: drafts accepted as gold per explicit user instruction (dashboard verification step skipped) — `dev/gold.csv` built directly from `dev/drafts/*.json`.
- Token/cost architecture: three redesign rounds (documented in Phase D above) took per-case inference from an unworkable ~1.16M chars/case (v1) to a measured ~72K tokens for a single high-confidence pass, with escalation reserved for genuinely low-confidence CPs (v3, current).
- Root cause of the 84-agent/~1M-token pilot blowup identified and fixed: majority-vote threshold was hardcoded to `>= 2`, which is unsatisfiable with a 1-persona ensemble and forced all 41 CPs into unnecessary tiebreak calls. Now scales with ensemble size.

**Still open:**
1. **Organizer blockers** (carried over, unanswered) — 99 vs 100 rows, Midwest's true RE number, missing-Track-1 intent, N/A-vs-0 scoring. Not build-blocking (we produce both submission variants regardless) but should go out now given submission deadline pressure.
2. **Gemini API key** — blocks Phase G (and the Gemini-vs-Haiku bake-off) but not Phase D/F, which run entirely on Haiku via Workflow.
3. **Confidence threshold (0.85 default)** for escalation — not yet tuned against `dev/gold.csv`; first thing to check once Phase F runs across the full dev set (does escalating more/fewer CPs change accuracy, not just cost).
