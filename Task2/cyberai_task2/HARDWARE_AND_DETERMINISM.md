# Reproducibility & Determinism

_How FRECA satisfies the competition's "submit exact prompts + model name/version; the run must reproduce" requirement._

## What organizers run

`scripts/inference/run_standalone.py` — a standalone script with no dependency on Claude Code or any orchestration harness. It:

1. Reads `prompts/persona_standard.md` (and `persona_adversarial.md`, `persona_stepbystep.md`, `tiebreaker.md` for the escalation path) and `okf/reference.md` directly from disk — these are the single canonical source of the prompts, not hand-copied text.
2. Embeds the case evidence and OKF reference bundle inline into the prompt (the standalone script has no file-reading tool to hand the model, unlike the dev-loop harness — see below).
3. Calls the pinned provider API at `temperature=0`.
4. Caches every response by `MD5(provider|model|temperature|prompt)` under `scripts/inference/cache/` — an unchanged input never triggers a second API call.
5. Logs every request (hit or miss) to `scripts/inference/transcripts/transcript.jsonl`.

```
python scripts/inference/run_standalone.py --provider anthropic --cases <case_id ...> --out predictions.csv
```

## Pinned model version

**Not yet finalized** — `PINNED_MODELS` in `run_standalone.py` currently has both providers set to `None` and will raise clearly if a real (non-dry-run) call is attempted before being filled in. Once the production model is locked (Gemini per the current plan, pending an API key — see FRECA_IMPLEMENTATION_PLAN.md §7), the exact dated snapshot string goes here and gets declared in the submission. Model *aliases* (e.g. "latest", "sonnet") are explicitly not used for the pinned constant, because they resolve to whatever is current at call time and are not reproducible on their own.

## Temperature

`0.0`, hardcoded, both providers.

## Two execution paths — why, and why they stay in sync

- **Dev-loop (`scripts/inference/freca_dev_workflow.js`, via the Workflow tool):** subagents Read `dev/evidence/<case>.md` and `okf/reference.md` themselves via their own Read tool. Cheap to orchestrate (short prompts, no inlined multi-hundred-KB text) and fast to iterate prompts against `dev/gold.csv`.
- **Standalone (`run_standalone.py`, for organizers):** no tool access available to a raw API call, so the full evidence + reference text is embedded directly in the prompt (~100K chars / ~25K tokens per call, confirmed by the dry-run below).

Both paths render the exact same `prompts/*.md` files. The templates were deliberately restructured so the compliance-reasoning content (the `## Rules` / `## Task` / `## Output` sections) is delivery-mechanism-agnostic — only a `{CONTEXT_BLOCK}` placeholder differs between "here are the file paths, Read them" (Workflow) and "here is the content inline" (standalone). This closes a real gap that existed earlier in this project: prompt text was being hand-copied into Workflow calls and had drifted from the canonical `.md` files at least once before this restructure.

## Verified: prompt construction is deterministic

Ran `run_standalone.py --dry-run` twice over all 10 dev cases (builds the exact prompt, computes its hash, logs it — makes zero API calls, spends zero provider credits). Every one of the 10 `MD5(prompt)` hashes was byte-identical between the two runs:

```
RE-WA-2021-0077__Goldfields: 088a2f3344ec586906bfc753efe3b195  101,438 chars
RE-QLD-2022-0077:            55900bbe0d9ee3692f5580e49d87298f   97,964 chars
RE-SA-2021-0066:              074a1b3e68d6e03fb39373bec9c38321   96,100 chars
RE-QLD-2022-0144:            63769915d5976f5ae4da06e9c65086b4  102,717 chars
RE-NSW-2021-0044:            f270b293a61ca4f1b3e5c4cc2f36fd70  103,740 chars
RE-VIC-2020-0093:            01edd40fbf8c21c8256c8350e3ec4257  104,666 chars
RE-SA-2020-0120:              31ee7e995e78c6859908b33af691c838  103,133 chars
RE-NSW-2020-0033:            5ebf017bf2690f17e5955a24703550f1  102,627 chars
RE-TAS-2021-0006:            434bb24bbe085e99e7aeb9326a3c06f7  102,857 chars
RE-NT-2022-0008:              12e6946085372fe365def3a4d2d986e2  103,470 chars
```

20 transcript entries written to `scripts/inference/transcripts/transcript.jsonl`.

## Residual non-determinism (honest caveat)

Even at `temperature=0` with a pinned model snapshot, cloud LLM inference is not guaranteed bit-for-bit reproducible run to run (batching effects, floating-point non-associativity across hardware, provider-side routing). What IS reproducible and is what we control: the exact prompt (hash-verified above), the exact model version (once pinned), the exact temperature, and the response cache (so a second run against the same cache reproduces exactly, since it never re-calls the API). We do not claim byte-identical model output across independent API calls — no team can, on a hosted model — only that our inputs and process are exactly reproducible.

## OKF grounding pass (Phase 1) — the same fix, one step earlier

The CP→policy-section mapping (`cps_mapping/element*.json`) that seeds `okf/reference.md` was originally produced by 4 ad-hoc Agent tool calls with instructions typed directly into the tool call — not saved as a prompt file, no pinned model, no temperature, no transcript. Same three gaps the inference side had, fixed the same way:

- `prompts/cp_mapping.md` — canonical prompt template (one call per element: embeds the full 184-section policy corpus + that element's CP official texts, asks for grounded citations + rationale, explicitly forbidden from stating pass/fail logic).
- `scripts/inference/run_cp_mapping.py` — standalone script, same MD5-cache + transcript + `--dry-run` pattern as `run_standalone.py`.

Verified deterministic the same way: `--dry-run` across all 4 elements, twice, byte-identical hashes both times, zero API cost:

```
Element-1: 79890bb10ac72c755d6f19efda33e23b  282,872 chars  (7 CPs)
Element-2: 7709b0022a14eb4fffa4305c9fe3e981  283,148 chars  (9 CPs)
Element-3: 87a27c15bcb6989f505dc2ce07e03f95  283,506 chars  (12 CPs)
Element-4: 50b101da515de09315a452fe016c6d07  283,828 chars  (13 CPs)
```

The full 184-section policy corpus (`policy/*.md`, deterministically chunked by `scripts/parse_policy.py` — pure regex, no LLM, already reproducible) is ~279,636 chars / ~69,909 tokens and gets embedded whole in every element call, since it comfortably fits a long-context model's window and lets the model ground citations against the actual text rather than a pre-filtered candidate set.

This step is a one-time offline artifact (produces the committed `cps_mapping/*.json` / `okf/reference.md`), not part of the per-case scoring loop — it does not need to be re-run at submission time, only re-runnable on demand if asked.

## Environment

`Task2/.venv/` (Python 3.14) — see `Task2/.venv` for the pinned package set used for ETL (`python-docx`, `openpyxl`, `PyMuPDF`, `pandas`). Standalone inference additionally needs `anthropic` and/or `google-generativeai`, imported lazily in `run_standalone.py` only when a real (non-dry-run) call is made — not required at all for the dry-run/prompt-verification path above.
