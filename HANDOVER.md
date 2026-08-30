# Handover — Packaging Material Difference Mining

Everything needed to continue this work: what the task is, what the data actually looks like,
what was built, what the numbers are, what failed and why, and what to do next.

**Current result: pooled 5-fold out-of-fold global F1 = 0.9465** (precision 0.963, recall 0.930,
threshold 0.43, measured over all 1407 boxes). Classical baseline floor is 0.03.
`submission.csv` is produced: 668 boxes over the 100 test images.

**Read `RESULTS.md` first** — it supersedes the fold-0 numbers throughout this file, which
describe the earlier single-model state (0.905). The architecture sections below are still
accurate; the results sections are historical.

---

## 1. The task

Given 100 (template, photo) image pairs, predict bounding boxes of every real content difference
between a clean design template and a photo of the printed packaging. Differences are semantic
changes to text, graphics or layout — not printing or scanning artefacts.

Scoring is **global F1**: TP/FP/FN are accumulated across all images *before* precision and recall
are computed, so one image with many false positives damages the whole score. A prediction is a TP
if its IoU with an unmatched ground-truth box is ≥ 0.5. Submission is a CSV with the same header as
`train.csv`, one row per predicted box.

Source: `Task1/Task1 Description.docx`. Data: `Task1/PackagingMaterialDifferenceMiningDataset/`
(200 annotated train pairs, 100 unlabelled test pairs, 2.5 GB).

---

## 2. What the data actually looks like

Every claim below is reproducible with `scripts/analyze_dataset.py` (~2 min, CPU).
These measurements drove every architectural decision, so re-run them before changing direction.

| Property | Measurement | Why it matters |
|---|---|---|
| Annotations | 1407 boxes over 200 pairs; 3–11 per image, median 7 | Expect ~7 predictions per test image |
| Image size | Template and photo are **identical in size** for all 200 pairs; 55 distinct sizes, mostly 1700×2200 and 1654×2339 | Full-resolution tiling, no resizing |
| Alignment | Block phase-correlation residual p95 ≤ 0.26 px. In the real Modal run, 284/300 pairs needed no warp, 16 a sub-pixel translation, 0 needed ECC | **Registration is not a problem in this dataset.** Do not spend effort on SIFT/LoFTR matching |
| Box size | Median 22×22 px (~1.5% of image width); 419/1407 have both sides under 16 px; 83 are exactly 8×8 | Small-object regime; output stride 2, not 4 |
| Change polarity | 842 additions, 558 modifications, **0 pure deletions**; 96.5% of boxes have more ink in the photo | Signed difference channel; the polarity filter in `decode.py` |
| Degradation | Photo is a blurred, noisy, tone-shifted render; Laplacian variance 2–6× lower than the template. ~18% of pairs carry heavy shadow | Blur matching and local background normalization in `preprocess.py` |
| Blur match found | σ=0.4 for 189 pairs, σ=0.6 for 68, σ=0 for 41 (from the real preprocessing run) | Confirms the blur gap is real but small and consistent |

**The two hard parts, quantified.**

1. *Precision.* A naive normalized-difference baseline gets recall 0.57 at precision **0.015**
   (F1 0.03) — every glyph edge fires because the photo is blurred. Blur matching before
   differencing is what makes the problem tractable.
2. *Localization.* An 8×8 box at IoU ≥ 0.5 tolerates only ~2.7 px of shift. Thresholded difference
   blobs match ground truth at median IoU 0.58, with only 34% within ±2 px on all sides. This is why
   the model regresses box centers and sizes (CenterNet-style) instead of thresholding a mask.

---

## 3. Architecture

Three stages. Stage 1 is trained and working; stage 2 is built and measured but currently
ineffective (see §5); stage 3 is deterministic post-processing.

### Stage 0 — preprocessing (`src/pmdm/preprocess.py`)
1. Verify alignment by phase correlation; sub-pixel translate if needed; ECC fallback (never
   triggered so far).
2. **Blur matching**: search a σ grid, blur the template until its Laplacian variance matches the
   photo's. This removes the dominant false-positive source.
3. **Local background normalization** (`img − GaussianBlur(img, σ=31)`) producing an "ink map" that
   is invariant to shadow and tone shift.
4. Cache four PNGs per pair to the volume: blur-matched template, aligned photo, and both ink maps.

### Stage 1 — siamese change detector (`src/pmdm/model.py`)
- Shared-weight `convnext_tiny` encoder over two 4-channel streams (BGR + ink map).
- `concat(a, b, |a−b|)` fusion at every scale — the U-Net SiamDiff/SiamConc shape that a
  reality-check study found still competitive with heavier change-detection transformers.
- U-Net decoder to **output stride 2**, with a stride-2 stem taken straight from the stacked input
  so fine detail is not invented by upsampling.
- CenterNet heads: center heatmap, width/height, sub-pixel offset, plus an auxiliary change mask.
- Loss: Gaussian focal + masked L1 on wh/offset + Dice/BCE on the mask.
- Trained on 768 px tiles, 60% sampled around an annotated difference.

### Stage 2 — patch verifier (`src/pmdm/verifier.py`)
96×96 crops around each candidate through a ResNet-18 with an 8-channel stem, producing a
real-difference probability and a 4-coordinate box delta. **Currently buys almost nothing** — see §5.

### Stage 3 — post-processing (`src/pmdm/decode.py`)
- Weighted Boxes Fusion across overlapping tiles (averages coordinates; better than NMS for tiny boxes).
- **Polarity filter**: drop candidates where the template has ink and the photo does not — 0 of 1404
  ground-truth boxes look like that.
- **Ink snapping**: refit the box to the local ink-difference blob, then apply the measured constant
  margin (+0, +0, +1, +1), with a ±6 px sanity bound.
- **Global threshold sweep**: one threshold for all images, chosen to maximize global F1 on
  out-of-fold predictions. Never guess this value.

---

## 4. Results

### Stage 1, fold 0 (160 train pairs / 40 validation pairs, no synthetic data)

Two full runs were done. Run 1 hit a NaN bug from epoch 26 (fixed, see §6); run 2 is the clean one.

| Epoch | Run 1 F1 (NaN bug) | Run 2 F1 (fixed) |
|---|---|---|
| 4 | 0.762 | 0.789 |
| 9 | 0.819 | 0.840 |
| 14 | 0.849 | 0.850 |
| 19 | 0.881 | **0.905** |
| 24 | 0.900 | 0.896 |
| 29 | **0.911** | 0.889 |
| 34 | 0.896 | 0.889 |
| 39 | 0.903 | not run (cancelled at ~36) |

**Read this carefully before drawing conclusions**: the two runs agree within ±0.02 at every
checkpoint, which is the run-to-run noise on a 40-pair, 268-box validation set where a single box is
worth ~0.004 F1. The NaN fix was a genuine bug fix but produced **no measurable F1 gain**. Both runs
plateau from epoch ~19–24 onward while training loss keeps falling — the model has saturated on 160
real pairs.

Run 1's 0.911 checkpoint was **deleted during cleanup** (see §6, mistake 3). The surviving artifact
is run 2's `best.pt` at 0.905.

### Out-of-fold verification (`predict_oof`, run 2 `best.pt`)

```
TP 238  FP 20  FN 30   precision 0.9225  recall 0.8881  F1 0.9049  threshold 0.31
```

Identical to the training-time evaluation, confirming the checkpoint and the inference path agree.

### Error breakdown (`scripts/error_analysis.py`, all 40 validation pairs, 268 boxes)

```
matched at IoU >= 0.5:        252
proposed but IoU < 0.5:         5      <- box tightness
never proposed (IoU == 0):     11      <- detection
recall ceiling with perfect boxes: 0.959
```

By box size:

| Longest side | n | matched | never proposed |
|---|---|---|---|
| <12 px | 64 | 0.828 | 10 |
| 12–24 px | 47 | 0.979 | 1 |
| >24 px | 157 | 0.975 | 0 |

**This is the single most important result in the handover.** Boxes above 12 px are essentially
solved at 97–98%. Ten of the eleven total misses are tiny marks under 12 px. Box regression is
nearly exhausted as a lever (only 5 boxes lost to loose coordinates), so effort spent on better
coordinate refinement will return almost nothing.

### Stage 2 verifier, honest A/B (`verify_ab`)

Validation pairs split by index parity: verifier trained on 16 pairs' candidates, measured on the
other 24, with the pre-verifier sweep on those same 24 as the control.

| | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| before | 155 | 18 | 17 | 0.896 | 0.901 | 0.8986 |
| after | 152 | 14 | 20 | 0.916 | 0.884 | 0.8994 |

**Delta +0.0009 — no effect.** It traded 4 false positives for 3 true positives.

The cause is a pipeline design error, not a modelling one: `predict_oof` applies the full
post-processing before saving candidates, so the verifier received 137 training records of which 97
were already positive. A verifier trained on 40 negatives cannot learn what artefact noise looks
like. **Fix before retrying**: dump raw candidates at a much lower score threshold (~0.01) with
`use_snap=False, use_polarity=False`, so the verifier sees hundreds of negatives per image and has
something to discriminate.

---

## 5. What to do next, in priority order

1. **Synthetic data targeting tiny marks.** The generator (`src/pmdm/synth.py`) is written and
   smoke-tested but **has never been run at scale**. It reproduces the dataset's own construction:
   paste edits into a clean template, then degrade into a "photo" (blur, noise, JPEG, tone curve,
   shadow field, sub-pixel warp). Its three edit types are `swap` (0.40), `insert` (0.30) and
   `mark` (0.30); `mark` at 6–16 px is exactly the failing category. Raise that weight, generate
   ~6400 pairs (`modal run modal_app.py::synth --shards 16 --per-shard 400`, CPU, ~20 min), then
   retrain fold 0 with `--n-synth 4000` and compare against the 0.905 control.
   *Expected effect, extrapolated from the error breakdown and not yet measured*: closing the tiny-box
   gap moves recall toward the 0.959 ceiling, putting F1 around 0.93–0.94.
2. **Retry the verifier with proper negatives** (see §4). Only worth doing after step 1, since more
   candidates make the verifier's job meaningful.
3. **Train folds 1–4** (`modal run modal_app.py::train_all`) once the configuration is settled.
   Do not do this before, it multiplies cost by five for no information.
4. **Ensemble and submit**: `predict --folds 0,1,2,3,4 --threshold <from the OOF sweep>`.
   The threshold must come from `/ckpt/oof/fold*.json`, never a guess.

Ideas deliberately **not** pursued, with reasons: image registration (data is already aligned);
higher-capacity backbones (the failure is tiny-object recall, not representational capacity); more
epochs (curve flat since epoch 19); test-time flips (text is chiral).

---

## 6. Mistakes made, so they are not repeated

1. **bf16 NaN in the focal loss.** Under bfloat16 autocast, `1 − 1e-4` rounds to exactly 1.0, so
   `log(1 − pred)` became `-inf` and the zero-weighted term became NaN from epoch 26 of run 1.
   GradScaler skipped those steps, so weights were never corrupted, but heatmap updates were lost.
   Fixed by forcing float32 inside `gaussian_focal` and `dice_bce`. Verified: zero NaN in run 2.
2. **`train_verifier` path bug.** It appended `"stage2"` to the caller's `out_dir`, so checkpoints
   landed in `/ckpt/stage2_ab/stage2/` while the loader looked in `/ckpt/stage2_ab/`. Fixed, plus a
   guard that raises if the DataLoader ends up empty instead of silently training zero steps.
   (The stale `/ckpt/stage2_ab/stage2/` directory can be deleted.)
3. **A checkpoint was destroyed.** `modal volume cp` refuses directories but still **exits 0**, so a
   `cp && rm` chain deleted run 1's 0.911 weights without a backup. Always verify a copy landed
   before deleting; prefer `modal volume get` to a local path for backups.
4. **Launch pattern.** `nohup modal run --detach ... &` is unreliable — if the client is killed
   during app startup, before the function call is enqueued, nothing runs at all and the log looks
   like a successful start. One resume attempt silently trained zero epochs this way. Launch through
   a supervised background process and confirm training lines appear.
5. **Client disconnects kill non-detached runs.** A local DNS blip killed run 1 at epoch ~1. Use
   `--detach`, and note that `train_fold` now commits the volume after every epoch so an
   interruption costs at most one epoch.

---

## 7. Code map

```
TASK1_CYBER/
├── HANDOVER.md              <- this file
├── IMPLEMENTATION_PLAN.md   <- original architecture rationale
├── RUNBOOK.md               <- exact commands
├── modal_app.py             <- all Modal functions and entrypoints (398 lines)
├── pyproject.toml
├── checkpoints/             <- trained weights, mirrored off Modal; see checkpoints/README.md
│   ├── stage1_fold0_convnext_tiny/best.pt   the 0.905 model, 128 MB
│   ├── stage1_fold0_convnext_tiny/last.pt   resumable state at epoch 35, 384 MB
│   ├── oof/                                 out-of-fold candidates and sweep
│   └── stage2_ab/                           verifier from the A/B; reference only
├── runs/                    <- raw evidence from every run; see runs/README.md
│   ├── fold0_run2_fixed.log        the primary run, full step-level losses
│   ├── fold0_run2_history.json     per-epoch structured metrics
│   ├── fold0_run1_nanbug.log       first run, epochs 19-31 only, shows NaN onset
│   ├── oof_fold0_candidates.npz    out-of-fold boxes and scores, 40 pairs
│   ├── oof_fold0_sweep.json        the 0.9049 headline result
│   ├── error_analysis_output.txt   the size breakdown
│   ├── analyze_dataset_output.txt  full dataset analysis output
│   ├── verifier_ab_raw.txt         verifier A/B, including record counts
│   └── prep_info.json              per-pair alignment mode and blur sigma
├── scripts/
│   ├── local_smoke.py       <- CPU pre-flight; run before any Modal spend
│   ├── train_local.py       <- run training without Modal, on any GPU box
│   ├── analyze_dataset.py   <- reproduces every claim in §2
│   └── error_analysis.py    <- reproduces the breakdown in §4
└── src/pmdm/                <- 1500 lines total
    ├── config.py            <- paths (env-overridable) and hyperparameters
    ├── preprocess.py        <- alignment, blur match, ink maps
    ├── dataset.py           <- tiling dataset, CenterNet target encoding, GroupedSampler
    ├── model.py             <- SiamCenterNet
    ├── losses.py            <- focal / masked L1 / Dice
    ├── decode.py            <- heatmap decode, WBF, polarity filter, ink snapping
    ├── train.py             <- stage-1 loop with per-epoch resume
    ├── infer.py             <- tiled full-image inference
    ├── verifier.py          <- stage-2 model, training, application
    ├── synth.py             <- synthetic pair generator (never run at scale)
    ├── metric.py            <- global F1 scorer and threshold sweep
    ├── folds.py             <- deterministic 5-fold split
    ├── tiles.py             <- tiling helpers
    └── baseline.py          <- classical floor
```

### How training is actually invoked

There is one training loop, `src/pmdm/train.py::train_fold`, reached two ways:

| Path | Command | When to use |
|---|---|---|
| Modal (used for all results here) | `modal run --detach modal_app.py::train --fold 0 --epochs 40` | Normal path. `modal_app.py::train_stage1` is a thin wrapper that pins A100-40GB, mounts both volumes, and passes `on_epoch_end=ckpt_vol.commit` so each epoch is durable |
| Local / any GPU box | `python scripts/train_local.py --fold 0 --epochs 40` | No Modal account needed. Same loop, same checkpoints, same resume behaviour. Requires stage 0 first: `python scripts/train_local.py --preprocess-only` |

Both write to `$PMDM_CKPT/stage1_fold<N>_<backbone>/` and resume from `last.pt` automatically.
The exact command that produced the current 0.905 model was
`modal run --detach modal_app.py::train --fold 0 --epochs 40` with no synthetic data.

**Conventions worth knowing before editing.**
- All paths come from `config.py` and are environment-overridable (`PMDM_DATA`, `PMDM_WORK`,
  `PMDM_CKPT`), which is how the same code runs locally and on Modal volumes.
- Ground-truth boxes are keyed by `(split, idx)` tuples inside the dataset, but `load_gt()` returns
  them keyed by integer index — `train.py::build_gt` bridges the two.
- `GroupedSampler` keeps a pair's tile samples adjacent so the dataset's one-entry cache serves them;
  without it every sample decodes four full-page PNGs.
- The metric module is the authority on scoring. If you change decoding, re-check against
  `metric.global_f1`, not against intuition.

---

## 8. Infrastructure state (Modal account `bigbalak`)

**Volume `pmdm-data`**
```
/raw/train, /raw/test            uploaded dataset (2.5 GB)
/work/prep/{train,test}/NNN/     preprocessed pairs, all 300 done
/work/prep_info.json             per-pair alignment mode and blur sigma
```

**Volume `pmdm-ckpt`**
```
/stage1_fold0_convnext_tiny/best.pt       <- THE model, F1 0.905, epoch 19
/stage1_fold0_convnext_tiny/last.pt       <- epoch 35 state (optimizer + scheduler, resumable)
/stage1_fold0_convnext_tiny/history.json  <- per-epoch losses and evaluations
/oof/fold0_convnext_tiny.{npz,json}       <- out-of-fold candidates and sweep result
/stage2_ab/                               <- verifier A/B artifacts (weak result, see §4)
/hf/                                      <- cached timm weights
```

Nothing is running. Fold 0 stopped at epoch 35 of 40; the missing 4 epochs are not worth running
given the flat curve.

**Hugging Face**: the whole thing — weights, code, docs, `runs/`, and the dataset — is pushed to the
private repo <https://huggingface.co/siddhant20/task1>. That is the single link to hand a teammate;
they need a collaborator invite on the `siddhant20` account, nothing else. Clone with
`git clone https://huggingface.co/siddhant20/task1` (needs git-lfs) or
`hf download siddhant20/task1 --local-dir task1`.

Everything on `pmdm-ckpt` except the timm cache and one duplicate directory has been mirrored into
`checkpoints/` in this repo, so the weights survive without access to the Modal account. The
dataset on `pmdm-data` is not mirrored — it is the 2.5 GB competition download plus its
preprocessed derivatives, both re-creatable from `Task1/` with `prep`.

**Hardware and rough timings**: preprocessing is a CPU fan-out over 300 pairs (a few minutes);
stage-1 training is A100-40GB at ~52 s/epoch plus ~60 s per evaluation, so a 40-epoch fold is
roughly 45 minutes; inference and verifier work run on L4.

---

## 9. Getting started as the new owner

```bash
# 1. Local environment
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . torch torchvision timm opencv-python-headless "numpy<2" pandas

# 2. Confirm the pipeline works end to end on CPU, before spending anything
PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
PMDM_WORK=/tmp/pmdm_work PMDM_CKPT=/tmp/pmdm_ckpt \
.venv/bin/python scripts/local_smoke.py

# 3. Confirm the data claims for yourself
PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
.venv/bin/python scripts/analyze_dataset.py

# 4. Reproduce the current headline number (needs Modal access to the volumes)
modal run modal_app.py::oof --fold 0        # expect F1 0.9049, threshold 0.31

# 5. Then start on §5 step 1
modal run modal_app.py::synth --shards 16 --per-shard 400
modal run modal_app.py::train --fold 0 --epochs 40 --n-synth 4000
```

Modal access requires the `bigbalak` account's token (`~/.modal.toml`) or re-uploading the dataset
to your own volumes with the commands in `RUNBOOK.md`. Without either, step 4 still works offline
against the local mirror:

```bash
PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
.venv/bin/python scripts/error_analysis.py checkpoints/oof/fold0_convnext_tiny.npz
```

and `checkpoints/stage1_fold0_convnext_tiny/best.pt` loads directly into `SiamCenterNet` for
inference on any machine (see `checkpoints/README.md`).

---

## 10. Reference reading

The design draws on these; the first two are the most directly relevant.

- [Spot the Difference by Object Detection (arXiv:1801.01051)](https://arxiv.org/pdf/1801.01051) — printed book cover vs digital design, the closest published analogue
- [A Change Detection Reality Check (arXiv:2402.06994)](https://arxiv.org/abs/2402.06994) — simple siamese U-Nets remain competitive with elaborate architectures
- [Changer: Feature Interaction is What You Need (arXiv:2209.08290)](https://arxiv.org/abs/2209.08290) — stream interaction design
- [Self-Pair: Synthesizing Changes from Single Source (arXiv:2212.10236)](https://arxiv.org/abs/2212.10236) — the synthetic pair strategy
- [SAHI: Slicing Aided Hyper Inference (arXiv:2202.06934)](https://arxiv.org/abs/2202.06934) — tiled training and inference for small objects
- [Weighted Boxes Fusion (arXiv:1910.13302)](https://arxiv.org/abs/1910.13302) — coordinate-averaging box merge
- [Open-CD toolbox](https://github.com/likyoo/open-cd) — reference implementations of change-detection baselines
- [OmniDocBench (arXiv:2412.07626)](https://arxiv.org/abs/2412.07626) — the corpus the dataset was derived from, and a source of extra templates
