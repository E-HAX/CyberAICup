# Code handover — encrypted RTC application identification

CyberAI Cup 2026 (ICSDS) Task 3. Given the sizes and inter-arrival times of the
**first five packets** of an encrypted UDP media flow, predict one of ten
classes: {Discord, GoogleMeet, Messenger, WhatsApp, Zoom} × {voice, video}.
1,285 training flows, 327 test flows, ten numbers per flow.

**Shipped result: grouped-CV accuracy 0.8342, macro-F1 0.8292.** The model is
TabICL over 241 engineered features, in two views (flat ten-class and
hierarchical application × mode) averaged, followed by one derived decision
rule. `submission.csv` in the repository root is that model's output.

**Ceiling: accuracy 0.9377, macro-F1 0.9364.** Not an estimate — 160 of the
1,285 training flows are provably unwinnable. Read §6 before proposing
improvements.

---

## 1. The two commands that matter

```bash
# train the shipped model, writing out-of-fold and test probabilities
RTC_WORK=/tmp/rtcwork python scripts/train.py --model tabicl --repeats 2

# turn those probabilities into submission.csv, with out-of-fold scores printed
RTC_WORK=/tmp/rtcwork python scripts/predict.py \
    --runs /tmp/rtcwork/oof/train_tabicl.npz --score-oof
```

`RTC_WORK` is the working root; it must contain `data/Training_set.csv` and
`data/Testing_set.csv`. On Modal it is the `/work` Volume mount and is set for
you.

The shipped `submission.csv` reproduces bit-identically from the stored
artifact without retraining:

```bash
RTC_WORK=/tmp/rtcwork python scripts/predict.py \
    --runs artifacts/hier/hier_tabicl_2803104002.npz --score-oof
```

## 2. Repository map

```
scripts/            entry points
  train.py            fit the shipped design, store probability matrices
  predict.py          blend stored matrices -> submission.csv
  spawn.py            fire-and-forget launcher against the deployed Modal app

src/
  config.py           paths, label vocabulary, CV protocol, seeds, protocol bands
  io_utils.py         load/validate CSVs, label <-> id, pseudo-group keys
  predict.py          submission writing and format validation
  reporting.py        model card, SHAP, figures

  features/
    base.py           feature blocks F1-F8, 241 columns
    likelihood.py     label-aware blocks F9-F11 (measured harmful, off by default)
    pipeline.py       build-once cache shared by every worker container

  models/
    registry.py       name + dict -> estimator, for every family used
    cv.py             fold construction (grouped and plain), stored once
    grids.py          hyperparameter grids as explicit specification lists
    evaluate.py       the fan-out unit: score one (model, params, features) point
    optuna_search.py  ask/tell tuning with a journal on the Volume
    ensemble.py       hill-climb, stacking, rank-average, nested scoring
    calibrate.py      temperature / vector / Dirichlet calibration, ECE
    decision.py       fitted decision-layer experiments (did not generalise)
    hier.py           SHIPPED: hierarchical heads, mixture view, the Zoom rule

  analysis/
    geometry.py       SHIPPED DIAGNOSIS: ambiguity, separability, the ceiling
    zoom_tsne.py      t-SNE panels, Zoom diagnostics, error budget
    tuple_lookup.py   exact-tuple lookup experiments (closed)
    controls.py       negative controls N1/N2 - the round is void if these fail
    compare.py        McNemar comparisons between families
    call_recovery.py  same-call scoring, clustering, aggregation (closed)

  transformer/        PacketFormer pretraining and linear probes (closed branch)
    model.py, pretrain.py, finetune.py, dataset_rtc.py
    probe.py, probe_eval.py, mirage_stream.py

  eda/run_eda.py      the exploratory analysis behind the feature design
  modal_app.py        every stage as a Modal function; images, Volumes, fan-out

artifacts/hier/     probability matrices for six base learners, both views
experiments/round3/ scratch scripts kept as evidence (see its README)
```

## 3. Documents

| File | What it holds |
|---|---|
| `REPORT.md` | full methodology then results; start here for the whole story |
| `PLAN.md` | the original Modal-native implementation plan, plus an execution log |
| `PLAN2.md` | phase-2 plan, ranked strategies, paper references |
| `ABLATION2.md` | rounds 1 and 2: tuning, probes, call recovery — all closed |
| `ABLATION3.md` | round 3: the label geometry, the ceiling, the shipped model |
| `HANDOVER.md` | this file |

## 4. How the shipped model is built

`src/models/hier.py` is the whole design and is short enough to read in one
sitting. Three functions do the work.

**`heads_predict`** returns two ten-class probability matrices from one fit
call:

* the *flat* view — one ten-class model, the conventional formulation;
* the *hierarchical* view — one five-class application head, plus one binary
  voice/video head per application, composed as P(app) × P(mode | app).

The two label dimensions have different difficulty (application ≈ 0.92, mode
≈ 0.89) and different failure modes, so the two views make different mistakes
and their average beats either.

**`mixture_predict`** is a third view, kept because it is cheap and diverse: it
splits each video class by whether the window is audio-only, fits fifteen
classes, and sums the two video sub-classes back together. The split is a
deterministic function of the features, so it adds no label information — what
it changes is the loss decomposition.

**`apply_zoom_rule`** is the one decision rule that ships. See §6.

`scripts/train.py` runs the first two views over the stored grouped folds and
refits on all training rows for the test matrices. `scripts/predict.py`
normalises, averages the views, blends across runs if given several, applies
the rule, and writes the file through `src/predict.py`'s validator.

## 5. Evaluation protocol — the part that is easy to get wrong

No call identifier is published, but flows from one source call are near
duplicates, so a random split scores a memorised call rather than a
generalising model. `io_utils.group_keys` builds a pseudo-group key from the
8-byte-rounded length tuple plus the duration decade, and `models/cv.py` uses
`StratifiedGroupKFold` over it. **The grouped scheme is the decisive metric
everywhere in this repository.** Two repeats of grouped 5-fold are stored;
asking for more silently gives you empty probability slabs, which is why
`scripts/train.py` clamps the request.

Every headline number in the documents is a grouped-CV number. The optimistic
"plain" scheme is reported alongside for contrast, never on its own.

## 6. What you must know before trying to improve this

Every record is the **first five packets** of a flow. A video call whose
opening packets carry no video fragment produces a record that is physically
identical to a voice-call record. Measured per application, over flows whose
every packet is under 300 bytes:

| Application | audio-only flows | of which video | grouped-CV AUC video vs voice |
|---|---|---|---|
| Discord | 290 | 52 | 0.885 |
| Messenger | 111 | 19 | 0.876 |
| GoogleMeet | 85 | 45 | 0.827 |
| **Zoom** | **160** | **80** | **0.547** |

The Zoom subset is an exact 80/80 split at coin-flip AUC, and the same subset
appears at every threshold from 200 to 500 bytes. Those 160 flows cap the whole
task at **accuracy 0.9377 / macro-F1 0.9364**. A macro-F1 of 0.90 would need
the other eight classes to average F1 ≥ 0.954 against a current 0.868 whose
residual is the same ambiguity in milder form. Reproduce all of this with
`python -m analysis.geometry` or the Modal function `run_geometry`.

**The Zoom rule.** On an evenly split, inseparable subset every assignment
scores the same accuracy, so the choice is free in accuracy and not free in
macro-F1: sending the whole subset to the smaller class rescues its recall at
no cost to the larger one. Hence — a flow predicted Zoom whose window is
audio-only is labelled `Zoom_voice`. It has no fitted parameter (the direction
follows from the 80/80 balance, the threshold is `config.BAND_AUDIO`), and it
raises macro-F1 for **all six** model families tested, in every view. It is
applied to labels rather than probabilities on purpose: it is a decision about
an inseparable subset, not a change of belief.

## 7. Closed branches — do not re-open without new evidence

| Branch | Measurement that closed it |
|---|---|
| Label-aware features F9-F11 | −0.02 accuracy even with an inner out-of-fold loop |
| MIRAGE pretraining and linear probes (F12) | −0.0105 with the best base learner |
| MIRAGE supervised transfer | 0.318 application accuracy vs a 0.384 majority baseline |
| MIRAGE activity labels | not present in the published archive at all |
| Exact-tuple lookup overrides | overrides reduce accuracy |
| Call-structure recovery | recovery works (AUC 0.998, purity 0.997); aggregation is negative even with an oracle grouping |
| Seed-bagging TabICL | 0.8350 against 0.8389 unbagged |
| Fitted macro-F1 decision layers | gained in sample, lost nested, at both 10 and 1 parameters |
| Class-balanced weights in the heads | ensemble −0.003 |

The pattern across all of them: at 1,285 rows, anything with a fitted parameter
that is chosen on the same rows it is scored on will look like a gain and will
not be one. The Zoom rule is the exception precisely because nothing about it
is fitted.

## 8. Modal

Everything heavy runs on Modal, app `rtc-cyberai`. Volumes: `rtc-work`
(features, folds, probability matrices, submission), `rtc-mirage` (auxiliary
corpus, now only of historical interest), `rtc-cache` (foundation-model
weights).

```bash
modal deploy src/modal_app.py
python scripts/spawn.py hier_stage '{"models":["tabicl","lgbm"],"repeats":2}'
python scripts/spawn.py --result fc-XXXXXXXX
modal volume get rtc-work submission.csv ./submission.csv
```

Use `modal deploy` plus `scripts/spawn.py`, never `modal run --detach`: an
ephemeral app is torn down when the local entrypoint returns, which cancels
everything it spawned. Roughly a thousand evaluated grid points were lost that
way once.

Useful functions: `hier_stage` / `hier_point` (the shipped design),
`hier_submit` (blend and write), `run_geometry` (the ceiling analysis),
`search_stage` and `finalists_stage` (the original hyperparameter sweep),
`run_controls` (leakage controls), `compare_families` (McNemar),
`rebuild_cv_results` (recover results from stored per-point files).

## 9. Environment

Python 3.12. Training needs `numpy pandas scikit-learn lightgbm xgboost
catboost scipy`; the shipped model additionally needs `tabicl` (and a GPU to be
quick about it — the L4 configuration in `modal_app.py` is what was used). The
transformer branch needs `torch`. The repository carries local virtual
environments (`.venv` for the Modal client, `.venv312` for local analysis)
which are not part of the deliverable.

## 10. Where the remaining headroom is

Google Meet, at F1 0.759 (voice) and 0.765 (video) on 152 flows — the smallest
application in the corpus and the one whose audio-only windows are least
separable among the three that are separable at all. Zoom is already at
about nine tenths of its structural maximum and is not worth further attention.
