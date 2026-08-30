# FRECA Task 2 — Detailed EDA Findings

_CyberAI Cup 2026 · Task 2. Verified programmatically across all 99 folders on 2026-08-12._
_Companion artifact: `case_manifest.csv` (per-case inventory + irregularity flags)._

---

## A. Corpus shape (verified)
- **99 folders** named `RE-<STATE>-<YEAR>-<NNNN>`. Spec says 100 cases → see §B (the 100th establishment is physically inside a doubled folder).
- **State distribution:** NSW 27, WA 25, QLD 16, SA 14, VIC 14, TAS 2, NT 1. **Year:** 2020×18, 2021×31, 2022×31, 2023×18, 2024×1. Heavily NSW/WA-weighted; NT/TAS barely present.
- **Files per case:** 96 folders = 9 files; **2 folders = 8 files; 1 folder = 18 files.**
- **9 tracks** = Registration form (1), HACCP (2), Pest Control Record (3, xlsx), Farm Management Plan (4), Farm Site Plan (5), Hygiene & Sanitation (6), Bait Station Map (7), Phytosanitary Security (8), Traceability Records (9, xlsx). Lexicographic filename sort = track order (deterministic ingestion).
- **File health:** 0 empty / unreadable / truncated files. All docx/xlsx parse cleanly.
- **Context size per case (all 9 tracks):** ~7,100–16,200 words (median 8,193) ≈ **11k tokens avg, ~21k max**. A whole case fits comfortably in one long-context window — no chunking of evidence strictly required.

## B. The three structural irregularities (THIS is what "create the test dataset" must solve)

### B1. One doubled folder → the missing 100th establishment
`RE-WA-2021-0077/` contains **18 files = two complete 9-track sets** for two different establishments:
- **Farm 35 — Goldfields Grain Storage Pty Ltd** (`..._Goldfields_..._035`)
- **Farm 100 — Midwest Grain Holdings Pty Ltd** (`..._Midwest_..._100`)

Both sets' Track-1 registration forms are **internally stamped `RE-WA-2021-0077`** — so the registration form does NOT disambiguate them. Farm numbers run 1–100; every other folder maps to exactly one farm. **Conclusion: there are 100 real establishments but only 99 folder IDs; the 100th (Midwest, farm 100) has no folder/RE identifier of its own and was dumped into the Goldfields folder.** This is the source of the 99-vs-100 discrepancy.

### B2. Two folders missing Track 1 (the registration form)
- `RE-QLD-2022-0077/` — 8 files, **missing Track 1** (this is farm 80, Condamine Valley Grain Co-operative).
- `RE-SA-2021-0066/` — 8 files, **missing Track 1** (farm 24, Mid North Lentil Growers).
- These are exactly the two "missing" farm numbers (24, 80) — they exist as folders but lack the registration form. Track 1 is the primary source for Element-1 CPs (registered operations/scope CP1–2, change notifications CP6–7), so those CPs are partly unverifiable for these two cases. Ingestion must **tolerate missing tracks without crashing**.

### B3. Net accounting
`case_manifest.csv` flags: 96 `OK`, 2 `MISSING_TRACK_1`, 1 `DOUBLED`. All 100 farm numbers are represented once we account for B1/B2.

## C. Data-quality artifacts (synthetic-generation noise — do NOT mistake for compliance signals)

### C1. "Registered Commodity" is randomised per document — quantified
Each case's docx tracks name on average **5.6 distinct commodities** (distribution: 5 distinct×30 cases, 6 distinct×63 cases). E.g. one case: Barley / Field Peas / Mangoes / Chickpeas / Rice across its tracks. **This is generation noise, not a traceability violation.** A pipeline that flags "documents disagree on commodity" would wrongly fail almost every case on the Element-4 traceability CPs.
- **Stable identity anchors (use these):** the **folder RE number** and the **establishment name** — establishment name is consistent across tracks 4–8 in **98/99** folders (the sole exception is the doubled folder B1). Track-1 RE number matches the folder name in **96/96** folders that have a Track 1.
- **Noisy fields (ignore):** per-track commodity; internal RE numbers and establishment names inside Track 2 (HACCP) and the xlsx tracks (they frequently name a completely different establishment / RE number).

### C2. Uppercase text corruption from a global state-code find/replace
The generator replaced uppercase 2-letter state codes (**`SA`, `NT`, `WA`, …**) with the folder's own state code, but **only in ALL-CAPS text** (titles, section headings, and the all-caps deficiency markers). Examples: `PHYTOSANITARY→PHYTOQLDNITARY`, `SAMPLING→QLDMPLING`, `AUTHORISATION→AUTHORINSWTION`, `DOCUMENTED→DOCUMENSWED/DOCUMEWAED`.
- **Confirmed deterministic:** SA-folders keep `PHYTOSANITARY` intact 14/14; a control keyword check showed the corruption is confined to uppercase tokens.
- **Lower-case body text is INTACT** (99/99 for phytosanitary, sampling, sanitation, documented, quarantine, maintenance, centre…). **So for an LLM reading full context, impact is negligible.** It only bites **deterministic keyword/regex/BM25 matching on headings** — e.g. a naïve search for the literal `[DEFICIENCY — NOT DOCUMENTED]` marker misses the ~14 corrupted variants (`NOT DOCUMENSWED`, etc.). If we ever do lexical retrieval on headings, normalise/repair first; otherwise leave it.

### C3. Correction to my earlier (first-pass) claim
My initial scan over-counted "corruption" because SA legitimately appears in words like phytoSAnitary, and over-counted mid-word state substrings (SERVICE contains "VIC", control contains "NT"). The refined, verified result is C2: corruption is real but **limited to uppercase tokens**, body text is clean. Reporting this so the team doesn't over-engineer a text-repair step that isn't needed for an LLM pipeline.

## D. The compliance signal — where the answers actually live

### D1. Explicit planted-deficiency markers
**267 bracketed `[DEFICIENCY …]` markers across 93/99 cases.** Variants: `[DEFICIENCY — NOT DOCUMENTED]` ×196, `[DEFICIENCY]` ×32, `[DEFICIENCY: No procedure documented for …]` ×12, `[DEFICIENCY — NOT SITE-SPECIFIC …]`, `[DEFICIENCY — INADEQUATE …]`, plus corrupted spellings (§C2). **Concentrated in Track 2 (HACCP, ~547 broad-pattern hits), Track 6 (Hygiene, ~70), Track 3 (13).** These are near-free non-compliance points *if* the whole document is read (they're explicit) — but mapping each marker to the right CP(s) is the reasoning task, and they must NOT be regex-harvested into hard-coded rules (that would violate the no-hard-coding constraint and also miss the corrupted/implicit ones).

### D2. Subtle / semantic violations (the hard points)
Buried facts that violate a specific CP without any marker, e.g.:
- **Non-English records:** 20 cases mention a non-English language for records/workforce — **Vietnamese** (Track 3 pest records, ×~7), **Mandarin** (Track 6 hygiene, ×3, with actual CJK characters in 4 files), **Arabic** (Track 1, ×2). Violates the "records must be in English" CPs (CP22/23/40).
- **<2-year retention:** date ranges like "Jan 2025–current" against a "≥2 years" target + "migration in progress" status → retention CPs (CP22/38) fail.
- **Physical/operational:** door-seal gaps "8–22 mm" (pest exclusion, CP11/12), "single-operator checks" for late-shift dispatch (substitution, CP36), flow chart "omits one alternate transfer path" (CP33), "temporary mesh segregation pending permanent barrier", "pesticide cartons on open shelf during stocktake" (chemical storage CP27/28).
These require reading a specific fact and reasoning it against the specific CP obligation — pure keyword/marker search will not catch them.

## E. Implications for the architecture (deltas to FRECA_PLAN.md)
1. **Ingestion must be irregularity-aware:** handle 8-file (missing Track 1) and 18-file (doubled) folders without crashing; drive off `case_manifest.csv`, not `assert len==9`.
2. **Identity anchoring in the prompt:** "case identity = folder RE number + establishment name; per-track commodity and sub-document RE/establishment fields are non-authoritative and must not be treated as compliance evidence." (Evidence-handling guidance — stays within the no-hard-coding rule.)
3. **No text-repair needed for the LLM path** (body is clean). Only add heading normalisation if we adopt lexical retrieval.
4. **Read the WHOLE document per track** — deficiencies (explicit and subtle) are scattered, not in headers. Full-context per case (≈11k tokens) is cheap enough to always include all tracks.
5. **N/A discipline matters:** the two missing-Track-1 cases legitimately make some Element-1 CPs unverifiable — decide N/A vs 0 deliberately (see test-dataset plan).

---
_See `DATASET_CONSTRUCTION.md` for exactly how we turn these 99 messy folders into the clean case set + submission rows._
