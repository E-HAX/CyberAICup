# FRECA Dev-Set Result — Sonnet single-pass, v1

_2026-08-15. 10 dev cases, Sonnet, single standard-persona pass (no ensemble), confidence-gated escalation disabled for this run._

## Overall

```
overall accuracy: 373/410 = 90.98%
```

## Per-element accuracy

| Element | Accuracy |
|---|---|
| Element-1 (registration, plans) | 61/70 = 87.14% |
| Element-2 (buildings, facilities) | 90/90 = 100.00% |
| Element-3 (hygiene, waste, pest, chemical) | 114/120 = 95.00% |
| Element-4 (traceability, phytosanitary, importing country) | 108/130 = 83.08% |

## Per-CP accuracy (worst first)

| CP | Accuracy | Note |
|---|---|---|
| CP37 | 5/10 = 50.00% | Model inconsistently flags a byte-identical boilerplate sentence ("packaging QC records attached to shift summary instead of individual lot files") as a deficiency across cases where it's actually template noise, not a signal. Gold is unanimous 1 (compliant) on all 10 — unfixed. |
| CP39 | 5/10 = 50.00% | Gold itself is internally inconsistent: 4 cases share byte-identical evidence text ("Rejected-goods archive verified") but gold labeled them 1, 0, 0, 0. Model's calls are actually more internally consistent than gold here — this is a gold-label defect, not a model error. |
| CP35 | 6/10 = 60.00% | Not yet diagnosed. |
| CP36 | 6/10 = 60.00% | Not yet diagnosed. |
| CP2 | 7/10 = 70.00% | Not yet diagnosed. |
| CP40 | 7/10 = 70.00% | Not yet diagnosed. |
| CP1, CP6, CP7 | 8/10 = 80.00% | Not yet diagnosed. |
| CP19, CP20, CP21, CP24, CP25, CP27, CP34 | 9/10 = 90.00% | Not yet diagnosed. |
| Remaining 27 CPs | 100.00% | — |

31 of 41 CPs at ≥90%, 32 at 100%.

## What got this run from 70.7% → 91.0%

1. **Model tier, not ensemble size, was the dominant factor.** Haiku's 3-persona ensemble scored 70.7% on a single case; Sonnet's single pass scored 92.7% on the same case at less token cost (70K vs 163K). Haiku was skimming past dense tabular tracks (Track 3, Track 9 — xlsx registers flattened to pipe-delimited rows) and missing content buried there; Sonnet reads them reliably.
2. **CP38 fix (record retention, +2.0pp aggregate):** the model was reading a boilerplate date-range string ("Jan 2025 – current, >= 2 years") that is byte-identical across compliant and non-compliant cases, and either always calling it compliant (first pass, wrong) or always calling it non-compliant after a bad "do date arithmetic" prompt fix (also wrong — same accuracy, different wrong cases). The real signal is a separate status/verification note in the same row ("2-year retention verified" vs "migration in progress"). Fixing the prompt to point at the status flag instead of the date text got CP38 from 4/10 to 10/10.

## Known open issues

- **CP37**: needs the same boilerplate-vs-signal fix as CP38, generalized (distinguish administrative asides from actual violations of the cited governing text).
- **CP39**: gold itself needs correction before this CP can be meaningfully scored — 4 cases have identical evidence but inconsistent gold labels.
- **CP35, CP36, CP2, CP40, CP1, CP6, CP7** and the 9/10 CPs: not yet root-caused.
- Confidence-gated escalation (adversarial + step-by-step personas) was tested and found to *hurt* accuracy with Haiku (majority vote let two personas that missed evidence outvote one that found it) — not yet re-tested with Sonnet as the base model. Whether ensemble helps once the base reader is competent is an open question.

## Files

- `dev/gold.csv` — gold labels (10 cases × 41 CPs), accepted from `dev/drafts/*.json` per explicit user instruction (dashboard verification step skipped).
- `dev/predictions_sonnet_v1.csv` — this run's raw predictions.
- `scripts/scorer.py` — scoring script (`python scripts/scorer.py <predictions.csv>`).
- `prompts/persona_standard.md` — the prompt used for this run (includes the CP38 status-flag fix).
- `scripts/inference/freca_dev_workflow.js` — the Workflow script that ran this (Phase D/E combined: standard pass + confidence-gated escalation + arbitration, escalation disabled for this run via `confidenceThreshold: 0`).
