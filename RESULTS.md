# Results — Packaging Material Difference Mining

Final measured state. Every number here is out-of-fold and leakage-free; see §4 for exactly what
that means and §5 for the bugs that produced wrong numbers along the way.

## 1. Headline

| Stage | Pooled F1 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|
| Starting point (fold 0 only, 268 boxes) | 0.9049 | 0.922 | 0.888 | 238 | 20 | 30 |
| Ensemble + TTA, pooled over all 5 folds | 0.9385 | 0.962 | 0.916 | 1289 | 51 | 118 |
| **+ cross-fold verifier** | **0.9465** | **0.963** | **0.930** | **1309** | **50** | **98** |

Pooled means all 1407 ground-truth boxes across all 200 training pairs, each predicted by models
that never trained on it, scored with a single global threshold — the same way the competition
scores a submission. The starting point is quoted on fold 0 alone because that is all that
existed; fold 0 turns out to be the hardest of the five, so the true like-for-like starting
figure is a little higher than 0.9049.

Target was 0.97. **We reached 0.9465.** Section 3 explains precisely what stands between the
two, and why most of the remainder is not reachable by tuning.

## 2. Where the gain came from

| Change | Contribution | Evidence |
|---|---|---|
| Test-time augmentation (8 dihedral views) + low score threshold + a generous box cap | ~+0.021 | fold 0: 0.9049 -> 0.9257 with the same weights |
| Dihedral training augmentation | folded into the above | model trained with it scores 0.9049 at the old decode settings and 0.9257 at the new ones |
| Synthetic small-mark pairs | sub-12px ceiling 0.828 -> 0.922 | see §3; its own F1 is *worse* (0.896), it contributes only through the ensemble |
| Two-model ensemble (plain + synthetic-trained) | +0.001 F1, +0.011 recall | fold 0: 0.9257 -> 0.9266, ceiling 0.944 -> 0.963 |
| Cross-fold verifier | +0.008 pooled | positive on all 5 folds independently: +0.016, +0.014, +0.006, +0.006, +0.018 |

Two changes that were expected to help and measurably did not: the size-relative wh loss (fold 0
finished at exactly the baseline's TP/FP/FN) and EMA weights (never selected over the raw weights
until the final epochs, where they tied). Both are still in the code behind flags, defaulted off
where they lost.

## 3. What is left, and what is reachable

The candidate set the verifier scores contains 1352 of 1407 ground-truth boxes at IoU 0.5 — a
**ceiling of 0.9609**. A rescorer that made no mistakes at all would therefore score:

```
TP 1352, FP 0, FN 55  ->  P 1.000  R 0.961  F1 0.9801
```

So 0.97 is above what perfect rescoring of the current proposals can deliver, and we are 0.014
short of that already-unreachable bound. Closing the rest means proposing the 55 boxes no model
currently finds, not scoring the existing ones better. Concretely:

- 98 false negatives = 55 never proposed by any model + 43 proposed but below threshold
- 50 false positives

The 55 are the real frontier. The sub-12px band, which used to own most misses, is now at 0.944
coverage after the synthetic data — that lever has largely been spent.

## 4. Leakage discipline

Four rules, each enforced in code rather than by convention:

1. **Synthetic pairs are fold-scoped.** Each records the template it was built from, and each
   fold drops any pair derived from its own validation pages. In production this dropped
   836–892 of 6400 pairs per fold, a different set for each fold.
2. **Test templates are a legitimate synthesis source** — they carry no labels, so nothing can
   leak; they add 100 unseen layouts.
3. **The verifier for fold k trains only on candidates from folds != k**, which were themselves
   produced by detectors that never saw those pages.
4. **One global threshold, swept on out-of-fold predictions only.**

The reported 0.9465 uses one model pair per image. The submission uses all ten models, so the
test score should land slightly above this figure rather than matching it.

## 5. Bugs found (all previously silent)

1. **WBF score saturation.** The fused confidence divided by a cap of 3, so any box agreed on by
   more than three views saturated at 1.0 while a box proposed by a single view kept its full raw
   score — confidence was effectively inverted. Harmless with one view, catastrophic with TTA or
   ensembling: it dropped a TTA run to F1 0.775 at precision 0.67. Fixed to the standard
   agreement-scaled mean; the single-view path is untouched and still reproduces bit-for-bit.
2. **Ensemble double-counted its sources.** Members already fuse their own views, so pooling them
   with `n_sources = models x views` divided every score by ~16. Recall collapsed to 0.056.
3. **Verifier data loading.** It re-decoded four full-page PNGs per sample on every cache miss and
   shuffled across 160 pages, so one epoch did ~140,000 PNG decodes and took ~60 minutes. Crops
   are now cut once into memory: seconds per epoch. This is why the first verifier attempt looked
   like a dead end.
4. **`--detach` does not protect `starmap`.** The fan-out is client-driven, so a laptop network
   blip killed the whole map. Fan-out now happens inside a remote function.

## 6. Artifacts

- `submission.csv` — 668 boxes over 100 test images, 6.68 per image against a training average of
  7.04. All ten models ensembled with TTA and all five verifiers averaged, at threshold 0.43.
- `runs/phase2/` — per-fold candidates before and after verification, plus filtered logs.
- Modal volume `pmdm-ckpt` — 10 detectors (`_aug` and `_augsyn` x 5 folds) and 5 verifiers
  (`stage2_cv/fold*`).
