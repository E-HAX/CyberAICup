# Packaging Material Difference Mining — Full Technical Report

CyberAI Cup 2026, Task 1. Written 2026-08-21.

**Result: pooled 5-fold out-of-fold global F1 = 0.9465** (0.9435 after removing threshold-selection
bias). Starting point was 0.9049 on a single fold. A submission file is produced and validated.
The stated goal of 0.97 was **not reached**, and §8 shows it was not reachable with this candidate
set: a rescorer that made no mistakes at all would cap out at 0.9801.

---

## 1. The task and the metric

Given a pair of images — a clean design template and a photograph of the printed packaging —
predict bounding boxes around every genuine content difference. Printing and scanning artefacts
are not differences; changes to text, graphics and layout are.

Scoring is **global F1**: true positives, false positives and false negatives accumulate across
every image *before* precision and recall are computed. This matters more than it first appears.
A per-image average would let one catastrophic image hide among 99 good ones; global F1 does not.
Fifty false positives on one page damage the score exactly as much as fifty spread across fifty
pages. A prediction counts as a true positive when its IoU with an unmatched ground-truth box is
at least 0.5.

Data: 200 annotated training pairs (1407 boxes), 100 unlabelled test pairs, 2.5 GB.

---

## 2. What the data actually looks like

Every figure below is reproducible with `scripts/analyze_dataset.py`. These measurements, not
intuition, drove the architecture.

| Property | Measurement | Consequence |
|---|---|---|
| Annotation density | 1407 boxes over 200 pairs, 3–11 per image, median 7 | expect ~7 predictions per test image; a submission averaging 3 or 15 is wrong |
| Image geometry | template and photo identical in size for all 200 pairs | full-resolution tiling, no resizing |
| Alignment | block phase-correlation residual p95 ≤ 0.26 px; 284/300 pairs need no warp, 16 a sub-pixel translation, 0 need ECC | **registration is not the problem here** — no effort spent on SIFT/LoFTR |
| Box size | median 22×22 px; 419/1407 have both sides under 16 px; 83 are exactly 8×8 | small-object regime; output stride 2, not the usual 4 |
| Change polarity | 842 additions, 558 modifications, **0 pure deletions** | licenses the polarity filter in `decode.py` |
| Degradation | photo is blurred, noisy, tone-shifted; Laplacian variance 2–6× lower than template; ~18% of pairs carry heavy shadow | blur matching and local background normalization in `preprocess.py` |

**The two hard parts, quantified.**

1. *Precision.* A naive normalized-difference baseline reaches recall 0.57 at precision **0.015**
   — global F1 **0.0293**. Every glyph edge fires because the photo is softer than the template.
   Blur-matching the template before differencing is what makes the problem tractable at all.
2. *Localization.* An 8×8 box at IoU 0.5 tolerates roughly 2.7 px of error. Thresholded
   difference blobs match ground truth at median IoU 0.58, with only 34% correct within ±2 px on
   all sides. Hence sub-pixel box regression rather than blob thresholding.

---

## 3. Architecture

### Stage 0 — preprocessing (`src/pmdm/preprocess.py`)

Verify alignment by phase correlation and translate if needed. Blur-match the template to the
photo by searching a σ grid for the blur that equalizes Laplacian variance. Compute an "ink map"
per image, `GaussianBlur(gray, σ=31) − gray`, which is invariant to shadow and tone shift. Cache
four PNGs per pair.

### Stage 1 — siamese CenterNet (`src/pmdm/model.py`)

A shared-weight ConvNeXt-tiny encoder runs over two 4-channel streams (BGR plus ink map).
Features fuse at every scale as `conv(concat(a, b, |a−b|))` — the U-Net SiamDiff/SiamConc shape
that change-detection reality-check studies find still competitive with heavier transformers — and
decode through a U-Net path to **output stride 2**, with a stride-2 stem taken straight from the
stacked input so fine detail is not invented by upsampling.

Stride 2 is load-bearing: 83 ground-truth boxes are 8×8 px and would occupy a 2×2 cell at the
stride 4 a standard CenterNet uses.

Four heads: Gaussian-focal centre heatmap, width/height, sub-pixel offset, and an auxiliary change
mask. Trained on 768 px tiles, 60% of them sampled around an annotated difference.

### Stage 2 — patch verifier (`src/pmdm/verifier.py`)

96×96 crops around each candidate through a ResNet-18 with an 8-channel stem, producing a
real-difference probability and a box delta. Its probability is blended with the detector score.

### Stage 3 — decoding (`src/pmdm/decode.py`)

Weighted Boxes Fusion across overlapping tiles and TTA views (averaging coordinates beats
discarding them when boxes are 8 px), the polarity filter, ink snapping to the local difference
blob with a measured constant margin, and a single global threshold swept on out-of-fold
predictions.

---

## 4. What was tried, and what each thing was worth

Work proceeded as controlled experiments on fold 0 against a 0.9049 reference, then whatever won
was applied to all five folds.

### 4.1 Training-side changes

| Change | Verdict | Evidence |
|---|---|---|
| Dihedral augmentation (flips + transpose) | **kept** | see §4.2 — worth nothing at the old decode settings, +0.021 at the right ones |
| Size-relative wh loss | **dropped** | fold 0 finished at TP 238 / FP 20 / FN 30 — the reference's counts exactly |
| EMA weights (decay 0.999) | **neutral** | never selected over raw weights until the last two epochs, where it tied |
| Synthetic small-mark pairs | **kept, indirectly** | its own F1 is *worse* (0.896 vs 0.926); it earns its place through coverage, see §4.3 |

The lesson from the first three rows: on 268 validation boxes, one box is worth 0.004 F1 and the
run-to-run spread is ±0.02. Three independent runs landed between 0.890 and 0.905. Nothing inside
that band is a result.

### 4.2 The decode settings mattered more than the training changes

The augmented model scored 0.9049 at the reference decode settings (`score_thr 0.05`,
`max_boxes 40`, no TTA) and **0.9257** at `score_thr 0.02`, `max_boxes 300`, 8-view TTA — the same
weights, +0.021. Every earlier comparison had been made through a lossy decoder, which is why
augmentation looked worthless. Model quality and decode quality are not separable, and comparing
models at their default decode is comparing decoders.

### 4.3 Recall ceilings — the measurement that redirected everything

The ceiling is the fraction of ground-truth boxes present anywhere in the candidate list at IoU
0.5, regardless of score. Nothing downstream can recover a box that was never proposed, so this
bounds every rescoring idea.

| Model (fold 0, TTA, score 0.02) | candidates | ceiling | **<12px ceiling** | best F1 |
|---|---|---|---|---|
| original single model | 421 | 0.9403 | 0.828 | 0.9049 |
| `_aug` | 2770 | 0.9440 | 0.891 | 0.9257 |
| `_augpix` | 3602 | 0.9478 | 0.844 | 0.9185 |
| `_augsyn` (synthetic) | 10238 | 0.9590 | **0.922** | 0.8956 |
| **`_aug` + `_augsyn`** | 10389 | **0.9627** | **0.9375** | **0.9266** |
| `_aug` + `_augpix` + `_augsyn` | 10826 | 0.9590 | 0.9375 | 0.9251 |

This is where the synthetic data justified itself. Sub-12px coverage went from 0.828 to 0.922 —
and that band previously held 10 of the 11 boxes the model missed entirely. The synthetic model's
own F1 is poor because it proposes aggressively; that is a scoring problem, and scoring is fixable.

Three models were worse than two: `_augpix` adds candidates without adding coverage, so it only
dilutes the agreement signal.

### 4.4 The verifier

The first attempt at stage 2 delivered +0.0009 and was written off. It was starved: it trained on
post-processed candidates, capped at 40 per image and already polarity-filtered, so it almost never
saw a negative. Retrained on generous candidate sets (~11,000 records per fold at an 8:1
negative:positive ratio), trained per fold on the four folds it would never be applied to:

| fold | before | after | Δ |
|---|---|---|---|
| 0 | 0.9266 | 0.9430 | +0.016 |
| 1 | 0.9414 | 0.9551 | +0.014 |
| 2 | 0.9288 | 0.9347 | +0.006 |
| 3 | 0.9544 | 0.9609 | +0.006 |
| 4 | 0.9494 | 0.9676 | +0.018 |

Positive on all five independently.

---

## 5. Final results

**Pooled over all 200 pairs and all 1407 boxes, one global threshold, each pair predicted only by
models that never trained on it:**

| Stage | F1 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|
| Classical baseline | 0.0293 | 0.015 | 0.572 | — | — | — |
| Original single model (fold 0 only) | 0.9049 | 0.922 | 0.888 | 238 | 20 | 30 |
| Ensemble + TTA | 0.9385 | 0.962 | 0.916 | 1289 | 51 | 118 |
| **+ verifier** | **0.9465** | **0.963** | **0.930** | **1309** | **50** | **98** |
| + verifier, nested threshold (§7) | 0.9435 | — | — | — | — | — |

Per-fold, ensemble + verifier: 0.943, 0.955, 0.935, 0.961, 0.968. Fold 0 — the fold every early
experiment was measured on — is the hardest of the five.

**Submission**: 668 boxes over 100 test images, 6.68 per image against a training average of 7.04.
Produced by all ten detectors ensembled with TTA, all five verifiers averaged, threshold 0.43.

---

## 6. Leakage discipline

Four rules, each enforced in code rather than by convention.

1. **Synthetic pairs are fold-scoped.** A synthetic pair carries invented boxes but the *real page
   layout* it was built from. Every pair records its source template, and each fold discards any
   pair derived from its own validation pages. In production this dropped 836–892 of 6400 pairs
   per fold — a different set for each fold, as it must be.
2. **Test templates are a legitimate synthesis source.** They carry no labels, so nothing can leak,
   and they contribute 100 layouts no model would otherwise see.
3. **The verifier for fold k trains only on candidates from folds ≠ k**, themselves produced by
   detectors that never saw those pages.
4. **One global threshold, swept on out-of-fold predictions only** — never on the fold being
   reported, never on test.

Out-of-fold scoring uses one model pair per image; the test submission uses all ten. The test
score should therefore land slightly *above* the reported out-of-fold figure, not match it.

---

## 7. Is it overfit?

Three places where selection touched the reported number.

**Threshold selection — measured, negligible.** Nested cross-validation, choosing the threshold on
four folds and applying it to the fifth:

| | tuned on eval | nested honest | optimism |
|---|---|---|---|
| detector only | 0.9385 | 0.9385 | +0.0000 |
| + verifier | 0.9465 | 0.9435 | +0.0030 |

**Checkpoint selection — the real bias.** `train_fold` evaluates on the fold's own validation pairs
and saves `best.pt` at the highest-scoring epoch; reporting that checkpoint's performance on those
same pairs is taking the max of eight noisy measurements. Measured gap between best and final
epoch:

| recipe | mean | range |
|---|---|---|
| `_aug` | +0.005 | 0.000 – 0.011 |
| `_augsyn` | **+0.034** | 0.020 – 0.058 |

The `_aug` curves are flat at the end, so selection costs little. The `_augsyn` models trained only
15 epochs and were still oscillating, so `best.pt` cherry-picks a lucky point.

**Hyperparameter selection.** `score_thr`, `max_boxes`, the TTA configuration and the two-model
ensemble pairing were all chosen by looking at fold 0, then applied to all five. That contaminates
one fifth of the pooled number. Unquantified; probably small, certainly not zero.

**Honest estimate of true generalization: 0.925–0.94.** Not 0.9465.

Two pieces of counter-evidence against serious overfitting: the test set yields 6.68 boxes per
image against out-of-fold's 6.80 and ground truth's 7.04 — no distribution shift on unseen data —
and the verifier improved all five disjoint folds independently, which a fitted artefact would not
do consistently.

**The clean fix, not yet run**: re-run the out-of-fold pass using `last.pt` instead of `best.pt`,
eliminating checkpoint selection entirely. That yields a lower bound (the synth models' final
epochs are genuinely undertrained, not merely unselected), so the truth sits between it and 0.9435.
Roughly 30 minutes and $2.

---

## 8. What is left, and what is reachable

The candidate set contains 1352 of 1407 boxes at IoU 0.5 — a **ceiling of 0.9609**. A rescorer that
never erred would therefore score:

```
TP 1352, FP 0, FN 55  →  P 1.000  R 0.961  F1 0.9801
```

The 0.97 target sits above what perfect rescoring of the current proposals can deliver, and the
pipeline is already within 0.014 of that unreachable bound. The remaining 98 false negatives split
into 55 boxes no model proposes at all and 43 proposed but below threshold; there are 50 false
positives.

Those 55 are the frontier, and reaching them means better *proposals*, not better scoring. The
sub-12px band that used to dominate misses is now at 0.944 coverage, so that lever is largely
spent. Plausible next moves, in the order worth trying:

1. **Remove checkpoint-selection bias** (§7) so future comparisons are trustworthy — cheap, and it
   makes everything downstream honest.
2. **Train `_augsyn` properly** — 15 epochs on a 535-step schedule left it clearly undertrained
   (final epoch 0.81 vs best 0.87). A 40-epoch run should raise both its own F1 and the ensemble's
   ceiling.
3. **Inspect the 55 missed boxes directly.** They are unlikely to be homogeneous; whatever they
   have in common is the next architectural change, and guessing without looking has a poor record
   in this project.
4. **A second backbone** (ConvNeXt-small, or a different family) for genuine ensemble diversity —
   `_augpix` failed as a third member because it was too similar to `_aug`.

---

## 9. Bugs found

All four were silent — they produced wrong numbers rather than errors.

1. **WBF score saturation.** The fused confidence was `sum(scores) / min(members + 1, 3)`. With one
   view, clusters are small and this is sane. With 8 TTA views a real box collects ~16 members, the
   sum divides by 3 and clips to 1.0, while a box proposed by a *single* view keeps its full raw
   score — confidence effectively inverted. A TTA run scored 0.775 at precision 0.67 because of it,
   which read as "TTA doesn't work". Replaced with the standard agreement-scaled mean; the
   single-view path is untouched and still reproduces the archived number bit-for-bit.
2. **Ensemble double-counted its sources.** Members already fuse their own views, so pooling them
   with `n_sources = models × views` divided every score by ~16. Recall collapsed to 0.056.
3. **Verifier data loading.** It decoded four full-page PNGs per sample on every cache miss and
   shuffled across 160 pages, so an epoch performed ~140,000 decodes and took ~60 minutes with the
   GPU idle. Crops are now cut once into memory: seconds per epoch. This is the direct reason the
   first verifier attempt looked like a dead end.
4. **`--detach` does not protect `starmap`.** The fan-out is client-driven, so a laptop network
   blip killed an entire five-way map. Fan-out now happens inside a remote function.

Operational traps also worth recording: never launch a detached job inside a foreground command
that can time out (the SIGTERM takes the process group and Modal stops the app); `setsid` does not
exist on macOS and fails silently; volume writes from one container are invisible to another until
`.reload()`; and `modal volume cp` refuses directories while still exiting 0, which destroyed a
checkpoint earlier in the project.

---

## 10. Artifacts and reproduction

| Path | Contents |
|---|---|
| `submission.csv` | 668 boxes, 100 images, format-validated against `train.csv` |
| `RESULTS.md` | condensed results summary |
| `PLAN_097.md` | the plan as written *before* the work, kept so its over-optimistic estimates can be compared against outcomes |
| `checkpoints/` | all 16 models: 10 detectors (`_aug`, `_augsyn` × 5 folds), 5 verifiers, plus the original |
| `runs/phase2/` | per-fold candidates before and after verification, filtered logs |
| `runs/` | archive from the earlier single-model phase |
| `src/pmdm/` | 16 modules |
| `RUNBOOK.md` | the exact command sequence that produced these numbers |

Full pipeline, roughly 2 hours wall clock: synthetic corpus (16 CPU shards) → 10 detectors
(8 × A100, ~1 h) → ensemble out-of-fold candidates (5 × L4) → verifiers (5 × L4) → pooled score →
submission (5 × L4). Commands in `RUNBOOK.md` under "Current pipeline".

Estimated total compute spend across the project: **$35–40**, dominated by A100 fold training.

---

## 11. Summary of the through-line

The single most useful thing done in this project was not an architectural change. It was measuring
the **recall ceiling** separately from F1. That number showed that the synthetic-data model — which
looked like a clear failure at 0.896 F1 — held the best candidate coverage by a wide margin, and
that the whole rescoring branch of the plan was bounded at 0.98 no matter how good the verifier
became. Both facts redirected the work, and neither was visible in the headline metric.

The second most useful was distrusting flat comparisons: three "no gain" verdicts on augmentation
were artefacts of comparing models through a lossy decoder, and one "TTA is worse" verdict was a
scoring bug. Every one of those was reported as a negative result before being found to be wrong.
