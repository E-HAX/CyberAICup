# FRECA — Farm Registered Establishment Compliance Audit

**CyberAI Cup 2026 · Task 2**

A fully reproducible, API-only system that audits 100 Australian farm export-establishment cases against 41 checking points (CPs) drawn from the *Export Control (Plants and Plant Products) Rules 2021*, emitting a verdict per CP — `1` (compliant), `0` (non-compliant), or `N/A` (not applicable).

**Final output:** [`submission_100.csv`](submission_100.csv) — 100 rows × 41 CPs, one row per Registered Establishment.

**Final model:** `google/gemini-3.5-flash` via the Lightning AI `litai` SDK (86.1% micro-accuracy on a 10-case held-out dev set).

---

## 1. Architecture

The pipeline is four deterministic, separable phases:

```
┌─ ① Open Knowledge Format (offline, once) ──────────────────────────────┐
│  Export Control Rules 2021 (PDF)  →  section chunker (184 sections)     │
│       →  CP→section grounding pass (LLM reads policy text)              │
│       →  okf/reference.md  (41 CPs + 18 unique governing sections)      │
└──────────────────────────────────────────────────────────────────────────┘
                              │
┌─ ② Evidence ETL (per case) ─────────────────────────────────────────────┐
│  99 case folders (9 tracks each)  →  logical-case manifest (100 cases)  │
│       →  docx/xlsx → track-tagged markdown  →  evidence/<case_id>.md     │
└──────────────────────────────────────────────────────────────────────────┘
                              │
┌─ ③ Full-context inference (per case) ───────────────────────────────────┐
│  persona_standard prompt + OKF bundle + FULL evidence                    │
│       →  LLM (Gemini 3.5 Flash, temperature=0)                          │
│       →  41 × {1,0,N/A} verdicts (checkpointed per case)                │
└──────────────────────────────────────────────────────────────────────────┘
                              │
┌─ ④ Formatting ──────────────────────────────────────────────────────────┐
│  verdicts → submission_100.csv  (RE Number, CP1…CP41)                    │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why this shape:** (i) the OKF bundle satisfies the *no-hard-coding* rule by mapping each CP to its governing legislation *via an LLM reading the actual text* — not hand-authored; (ii) the ETL is pure, threshold-free text extraction; (iii) a single full-context pass beats both a multi-persona ensemble and hybrid RAG on accuracy *and* cost when the evidence already fits one context window (see `RAG_VS_FULLCONTEXT.md`, `GRID_SEARCH_RESULTS.md`).

---

## 2. Repository layout

| Path | Purpose |
|---|---|
| `Task2/SFRE_cases/` | Raw competition dataset (99 case folders, 9 tracks each) |
| `Task2/1-Export Control ….pdf` | The governing policy (132 pp.) |
| `Task2/checkingpoints_all_elements_onesheet.xlsx` | The 41 official CP texts |
| `policy/` | PDF chunked into 184 per-section markdown files |
| `cps_mapping/` | CP → policy-section citations (per element, committed) |
| `okf/reference.md` | **Deduplicated knowledge bundle** consumed at inference |
| `logical_cases.json` | 100 logical cases (doubled folder split into two) |
| `evidence/` | 100 flattened, track-tagged evidence documents |
| `prompts/` | Canonical prompt templates (personas, tie-breaker, CP-mapping) |
| `scripts/` | All pipeline scripts (see §4) |
| `submission_run/` | 100 per-case inference checkpoints |
| `submission_100.csv` | **Final submission** (reproduced by `--assemble`) |
| `dev/` | 10-case dev set, gold labels, model-bake-off predictions, scorer |
| `*.md` | EDA, plan, results, ablation, reproducibility, and report docs |

---

## 3. Prerequisites

- **Python 3.14**
- A **Lightning AI API key** with access to `google/gemini-3.5-flash`, plus your **teamspace name** and **teamspace ID** (ULID).

```bash
# 1. Create a venv (the project used Task2/.venv)
python3 -m venv Task2/.venv
source Task2/.venv/bin/activate

# 2. Install dependencies
pip install python-docx openpyxl pymupdf pandas litai lightning-sdk
# Optional (RAG / grid-search ablation only):
# pip install fastembed rank_bm25

# 3. Set credentials
export LIGHTNING_API_KEY="<your-key>"
export LIGHTNING_TEAMSPACE="<your-teamspace-name>"
export LIGHTNING_CLOUD_PROJECT_ID="<your-teamspace-ulid>"
export LIGHTNING_ORG="<your-org-name>"          # optional
export LITAI_MODEL="google/gemini-3.5-flash"
```

> `LIGHTNING_CLOUD_PROJECT_ID` is the 26-char ULID found in the Lightning dashboard URL under `teamspaceSettings=<ULID>%3Amembers`.

---

## 4. Recreating `submission_100.csv`

### Step 0 — clone and enter

```bash
git clone https://github.com/sidd20228/task2.git
cd task2
```

The committed repo already contains the **outputs** of every phase (`policy/`, `cps_mapping/`, `okf/reference.md`, `logical_cases.json`, `evidence/`, `submission_run/`, `submission_100.csv`), so the pipeline can be verified at any level.

### Step 1 — Open Knowledge Format (offline)

```bash
# 1a. Chunk the policy PDF (deterministic, no LLM)
Task2/.venv/bin/python scripts/parse_policy.py
#   → policy/section_*.md (184 files) + policy/_index.json

# 1b. (OPTIONAL) Regenerate the CP→section mapping via LLM.
#     NOTE: this step is model-dependent; the committed cps_mapping/*.json
#     is authoritative for exact reproduction. Regenerating with a different
#     model yields a different (still valid) mapping.
Task2/.venv/bin/python scripts/inference/run_cp_mapping.py \
    --provider lightning --elements Element-1 Element-2 Element-3 Element-4
#   → cps_mapping/element1-4_v2.json

# 1c. Build the deduplicated OKF bundle (deterministic)
Task2/.venv/bin/python scripts/build_okf_deduped.py
#   → okf/reference.md  (41 CPs + 18 unique policy sections, 4.31× dedup)
```

### Step 2 — Evidence ETL (deterministic, no LLM)

```bash
# 2a. Build the 100-case logical manifest (splits the doubled folder)
Task2/.venv/bin/python scripts/build_manifest.py
#   → logical_cases.json

# 2b. Flatten every case to a track-tagged markdown document
Task2/.venv/bin/python scripts/build_all_evidence.py
#   → evidence/<case_id>.md  (100 files)
```

### Step 3 — Full-context inference (checkpointed, resumable)

```bash
Task2/.venv/bin/python scripts/inference/run_full_corpus.py --max-tokens 40000
#   → submission_run/<case_id>.json  (one checkpoint per case)
#   → submission_100.csv             (assembled automatically at the end)
```

The runner processes **one case at a time**, writing a checkpoint after each. It:
- **skips** cases whose checkpoint already exists (resume),
- **abandons** a call that hangs past 240 s and moves on,
- **stops gracefully** on a credit/network error (so you can switch keys and re-run — nothing is lost),
- **re-runs** truncated / parse-failed cases on the next invocation.

To resume after a credit stop, set the new key and run the same command again.

### Step 4 — Assemble the exact CSV (no LLM)

```bash
Task2/.venv/bin/python scripts/inference/run_full_corpus.py --assemble
#   → submission_100.csv  (100 rows, RE Number + CP1…CP41)
```

This is **byte-for-byte deterministic**: it reads only the committed `submission_run/*.json` checkpoints. **This is the exact-reproduction path** — it does not call the LLM at all.

---

## 5. Reproducibility contract

| Layer | Determinism |
|---|---|
| Policy chunking (`parse_policy.py`) | ✅ Deterministic (regex, no LLM) |
| CP→section mapping (`run_cp_mapping.py`) | ⚠️ Model-dependent — *same model + prompt → same mapping*; committed artifact is authoritative |
| OKF bundle (`build_okf_deduped.py`) | ✅ Deterministic |
| Manifest + ETL (`build_manifest.py`, `build_all_evidence.py`) | ✅ Deterministic (lexicographic sort, no thresholds) |
| Prompt construction (`run_standalone.py --dry-run`) | ✅ Verified byte-identical hashes across runs (MD5) |
| LLM inference (`run_full_corpus.py`) | ⚠️ Temperature 0 + pinned model snapshot, but hosted inference has residual run-to-run variance |
| CSV assembly (`--assemble`) | ✅ **Byte-identical** (reads committed checkpoints) |

**Two ways to "reproduce":**

1. **Exact** — run Steps 0, 1c, 2, then `run_full_corpus.py --assemble` (or just run `--assemble` on the cloned repo). Because it reads the committed checkpoints, it produces a **byte-for-byte identical** `submission_100.csv`.
2. **End-to-end** — run all of §4 fresh. Steps 1a/1c/2 are deterministic; the LLM step (3) will produce a *statistically equivalent* CSV (~86% accuracy) but not bit-identical cells, owing to the inherent non-determinism of hosted LLM inference (documented in `HARDWARE_AND_DETERMINISM.md`). The full prompt, model string, and every hyper-parameter are pinned and logged.

---

## 6. Model & results

Final model **`google/gemini-3.5-flash`** (a reasoning model — `--max-tokens 40000` is required so hidden reasoning + answer both fit; lower budgets truncate the JSON).

Model bake-off on the 10-case dev set (full-context, identical prompts):

| Model | Accuracy | Tokens (10 cases) |
|---|---|---|
| Nemotron Ultra 550B | 76.6% | 441K |
| DeepSeek V4 Pro | 81.5% | 300K |
| **Gemini 3.5 Flash** | **86.1%** | 442K |
| Sonnet 5 (Claude) | 91.0% | ~710K |

Cross-fold (10-fold case-wise) for the final model: **86.1% ± 4.3%**. Full details, ablations (RAG, fusion-weight grid search, token/cost engineering), and the N/A-class macro-averaging caveat are in [`FRECA_TECHNICAL_REPORT.pdf`](FRECA_TECHNICAL_REPORT.pdf) / `.md`.

---

## 7. Validating a run against gold (dev set only)

The 10-case dev set has hand-validated gold labels:

```bash
Task2/.venv/bin/python scripts/scorer.py dev/predictions_gemini35flash_v1.csv
# → overall / per-CP / per-element accuracy vs dev/gold.csv
```

---

*All prompts, model identifiers, and hyper-parameters are pinned in `prompts/`, `scripts/inference/litai_client.py`, and `scripts/inference/run_standalone.py`. No API keys are stored in the repository.*
