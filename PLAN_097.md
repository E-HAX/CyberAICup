# Plan: 0.905 → 0.97 global F1, no leakage

**Outcome: 0.9465 pooled, not 0.97. See `RESULTS.md` for the final measurements.** The target was
not reached and, as §0's arithmetic below already hinted, was not reachable: the candidate set's
recall ceiling caps a *perfect* rescorer at 0.9801, and the realistic remainder sits in boxes no
model proposes at all. This document is kept as written, before the work, so the estimates in it
can be compared against what actually happened — most were too optimistic.

Status header. Written 2026-08-20. Control number to beat: **fold-0 out-of-fold F1 0.9049**
(TP 238, FP 20, FN 30 on 268 boxes, threshold 0.31).

## 0. What has to change, quantified

Reaching 0.970 on fold 0 means roughly TP 260, FP 8, FN 8. Against today's numbers that is
22 recovered misses and 12 fewer false positives. The 30 current misses decompose as:

| Cause | Count | Lever |
|---|---|---|
| Correct box proposed and well localized, but scored below the 0.31 threshold | 14 | score calibration / verifier rescoring |
| Never proposed at all (10 of 11 are under 12 px) | 11 | synthetic data weighted to tiny marks |
| Proposed but IoU < 0.5 | 5 | box refinement |

The first row is the largest and the cheapest: those boxes already exist in the candidate set at
the right coordinates. Nothing needs to be detected better — the scores need to separate.

Expected contributions, deliberately conservative:

| Lever | Expected | Why that number |
|---|---|---|
| Geometric augmentation (currently absent) | +0.010 – 0.020 | 160 training pairs with photometric jitter only; flips and transposes are free label-preserving multiplicity |
| Synthetic pairs weighted to sub-12px marks | +0.015 – 0.030 | directly targets the 11 never-proposed, and both runs plateaued at epoch ~20 which is a data-quantity signature |
| Verifier rescoring, trained on raw candidates | +0.015 – 0.030 | attacks the 14 low-score TPs and the 20 FPs simultaneously |
| Test-time augmentation (offsets + flips) with WBF | +0.005 – 0.010 | already plumbed for offsets; averaging coordinates helps 8px boxes |
| Multi-seed / multi-backbone ensemble | +0.010 – 0.020 | standard, and the run-to-run spread of ±0.02 means ensembling recovers real signal |
| Tiny-box target and loss tuning | +0.005 – 0.015 | minimum Gaussian radius and wh loss scale are currently size-blind |

Optimistic sum lands above 0.97; realistic sum, since these overlap heavily, is **0.94 – 0.96**.
0.97 is the goal and is reachable only if the verifier lever pays out near the top of its range.
This plan is ordered so the highest-information experiment runs first and each step is measured
against a control rather than assumed.

## 1. Non-negotiable: no leakage

Four rules, each enforced in code rather than by discipline:

1. **Synthetic pairs are fold-scoped.** `generate()` currently samples from all 200 training
   templates. A synthetic pair built from a validation-fold template puts that page's layout in
   the training set. Every synthetic pair now records its source template index, and the loader
   drops any whose source is in the current fold's validation set.
2. **Test templates are fair game.** The 100 test templates carry no labels, so synthesizing
   edits onto them leaks nothing. They add 100 unseen layouts and are used for every fold.
3. **The verifier trains only on candidates from folds it did not see.** Stage-2 records for
   fold *k* come from models that never trained on fold *k*'s validation pairs.
4. **One threshold, swept on out-of-fold predictions only.** Never on the fold being reported,
   never on test.

The reported number is the full 5-fold out-of-fold F1 over all 1407 boxes, not fold 0's 268.
A single box is worth 0.004 F1 on fold 0 and 0.0007 across all folds — fold-0-only measurement
cannot distinguish 0.95 from 0.97.

## 2. Phases

**A — Instrumentation and cheap wins** (local, CPU, no spend)
- A1 fold-scoped synthetic generation with source tracking; test templates as an extra source
- A2 geometric augmentation: horizontal/vertical flip, transpose, mild scale jitter
- A3 EMA weights and a longer cosine schedule
- A4 raw candidate dumping (pre-postprocess, low threshold) so the verifier sees negatives
- A5 tiny-box target tuning: minimum Gaussian radius, size-normalized wh loss
- A6 local smoke test over the whole pipeline before any GPU spend

**B — Synthetic corpus** (Modal CPU fan-out, ~25 min)
6000 pairs, edit mix reweighted toward marks under 12 px, sources tagged.

**C — Controlled fold-0 experiments** (2 × A100, parallel, ~1 h)
- C1 control: current recipe + geometric augmentation, no synthetic
- C2 treatment: C1 + synthetic
Keep whichever wins by more than the ±0.02 noise band; if neither does, keep the simpler.

**D — Full training** (5 × A100 parallel, ~1 h)
Winning recipe on all five folds.

**E — Verifier, done properly** (L4, ~30 min)
Raw candidates from every fold, trained leakage-free, measured as a delta against the
detector-only out-of-fold score. Kept only if positive.

**F — Inference stack** (L4, ~45 min)
TTA offsets and flips, ensemble fusion by WBF, single threshold swept on the full out-of-fold
candidate set. This is where the reported number is produced.

**G — Submission** (L4, ~15 min)
Test prediction with the frozen threshold, written to submission.csv.

## 3. Stopping rule

Every phase reports out-of-fold F1 against the previous best. A change that does not clear the
noise band is reverted, not kept "because it should help". If the ceiling lands below 0.97, the
handover states the achieved number and what the remaining gap is made of — an overstated score
is worse than a short one, because the competition scores the test set, not this validation.
