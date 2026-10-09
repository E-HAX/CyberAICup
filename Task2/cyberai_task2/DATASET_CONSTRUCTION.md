# FRECA Task 2 — Test-Dataset Construction Plan

_How to turn the 99 messy `SFRE_cases/` folders into a clean, canonical case set and a valid submission. Driven by `case_manifest.csv` and `EDA_FINDINGS.md`._

## 0. Principle
We build a **"logical case" layer** on top of the raw folders. The pipeline never touches raw folders directly — it consumes a canonical manifest of logical cases, each with a stable ID, an ordered list of evidence files, and irregularity flags. This makes ingestion deterministic and irregularity-proof, and lets us regenerate the submission in whatever row-shape the organizers finally require.

## 1. Canonical case model
For each logical case store:
```
case_id            # canonical submission RE Number
folder_path        # source folder
establishment      # stable name (from tracks 4-8) — identity anchor, NOT commodity
farm_number        # 1..100 (from Track-1 filename where present)
tracks: {1..9 -> filepath or None}
flags: [OK | MISSING_TRACK_1 | DOUBLED_A | DOUBLED_B]
```
Ingestion rule: `sorted(os.listdir(folder))`, map each file to its track by the leading `N_`. Never assert exactly 9 files.

## 2. Handling each irregularity

### 2.1 Missing Track 1 (RE-QLD-2022-0077, RE-SA-2021-0066)
- Keep the case; `tracks[1]=None`; flag `MISSING_TRACK_1`.
- Feed tracks 2–9 normally. In `evidence_context.md`, insert an explicit line: *"Track 1 (Establishment Registration Application Form) was not provided for this case."* — a factual statement, not a verdict.
- Let the model decide Element-1 CPs from remaining evidence. Where a CP genuinely cannot be assessed because its sole source (the registration form) is absent, the applicability/evidence-sufficiency logic should lean **N/A ("cannot be determined from provided evidence")** rather than fabricating a 0 or 1 — but this is a judgement to validate on the dev set, and we should confirm with organizers whether the missing file is intentional.

### 2.2 The doubled folder (RE-WA-2021-0077 → Goldfields #35 + Midwest #100)
- Split into **two logical cases**:
  - `RE-WA-2021-0077` → the `*_Goldfields_*_035` 9-file set (flag `DOUBLED_A`).
  - a **provisional second ID** for the `*_Midwest_*_100` set (flag `DOUBLED_B`) — e.g. `RE-WA-2021-0077-B` as a placeholder until the true ID is known.
- Both are stamped `RE-WA-2021-0077` internally, so we **cannot infer the 100th establishment's true RE Number**. This is a blocker that **needs organizer confirmation** (see §4).
- Run inference on **both** sets regardless, so whichever row-shape is required, predictions are ready.
- ⚠️ Do NOT merge the two sets into one context — they are different establishments; merging would create genuine (not synthetic) contradictions and wreck the traceability CPs.

## 3. Two submission variants (produce both; pick on organizer answer)
- **Variant A — 99 rows**, one per folder ID (doubled folder counted once, using the Goldfields set). Safe if the platform keys strictly on the 99 provided folder identifiers.
- **Variant B — 100 rows**, splitting the doubled folder into two, using the provisional ID for the Midwest set. Matches the spec's "100 cases".
- Both written from `submission_template.xlsx`: header untouched, columns `RE Number, CP1..CP41` in order, cells ∈ {`1`,`0`,`N/A`}, no added/removed/reordered columns. Enum-validate every cell before writing.
- Keep the mapping `case_id → row` in a sidecar so we can re-key instantly if organizers supply the official 100-row template with the real second ID.

## 4. Questions to send organizers NOW (blockers)
1. **Row count:** should the submission have **99 or 100** rows? The provided `submission_template.xlsx` is empty (0 data rows) — is there an official populated template with the RE-Number column pre-filled?
2. **Doubled folder:** `RE-WA-2021-0077` contains two establishments (Goldfields #35 and Midwest #100), both internally labelled `RE-WA-2021-0077`. What is the correct RE Number for the second (Midwest) establishment, and should it be its own row?
3. **Missing registration forms:** `RE-QLD-2022-0077` and `RE-SA-2021-0066` have no Track 1 — is this intentional (audit realism) or a packaging error? Affects Element-1 scoring.
4. For missing/unverifiable evidence, do they score the "expected" verdict as N/A or as 0?

Until answered, proceed with the logical-case model + both variants; nothing is blocked except the final row-keying.

## 5. Dev set for prompt iteration (unchanged, but pick representative cases)
Hand-label **8–10 logical cases** as our only measuring stick (no labels are provided). Deliberately include:
- at least one `MISSING_TRACK_1` case and the doubled case,
- a spread of states (include NSW+WA since they dominate; include one QLD/SA/VIC),
- cases with explicit `[DEFICIENCY]` markers AND cases with subtle violations (non-English records, <2yr retention),
- if possible one case that looks fully clean (to test false-positive rate on the `1` class).
Store as `dev/labels.csv` (`case_id, CP1..CP41`). Used only to evaluate our prompts — never encoded into submission logic.

## 6. Build steps
1. `build_manifest.py` → `case_manifest.csv` (done) + logical-case JSON (splits doubled, flags missing).
2. `ingest.py` → per logical case, parse tracks (docx paragraphs+tables, xlsx sheets→rows), emit `evidence_context.md` with track delimiters, missing-track notes, and the identity-anchor header. Preserve dates/literal values.
3. `make_submission.py` → write Variant A and Variant B from the template + a result matrix.
4. Send §4 questions to organizers in parallel.
