# Packaging Material Difference Mining — Implementation Plan

> **Status note.** This document is the original design rationale, written before any training run.
> It is kept as-is so the reasoning behind each choice stays readable. For current results, what was
> actually built, what failed, and what to do next, read **`HANDOVER.md`** — that file supersedes the
> phase plan in §4 below.

Target metric: global F1 (TP/FP/FN accumulated over all 100 test pairs, IoU ≥ 0.5 matching).

## 0. Measured dataset facts driving the design

| Fact | Value | Design consequence |
|---|---|---|
| Pairs | 200 train (1407 boxes, 3–11 per image, median 7) / 100 test | Data synthesis is the main lever |
| Image size | template and photo identical per pair; mostly 1654×2339, 1700×2200 | Work at native resolution with tiling |
| Alignment | block phase-correlation residual \|Δ\| p95 ≤ 0.26 px | No registration stage required |
| Box size | median 22×22 px (~1% of width); 419/1407 under 16 px; 83 exactly 8×8 | Small-object regime; sub-pixel box regression |
| Polarity | 842 additions, 558 modifications, **0 deletions**; 96.5% more ink in photo | Signed difference channel + polarity filter |
| Degradation | photo is blur + noise + tone shifted (Laplacian variance 2–6× lower); ~18% of pairs carry heavy global shadow | Blur matching + local background normalization |
| Naive diff baseline | recall 0.57, precision 0.015, F1 0.03 | Precision is problem #1 |
| Threshold-blob box vs GT | median IoU 0.58, 34% within ±2 px | Box tightness is problem #2 |

An 8×8 box at IoU ≥ 0.5 tolerates only ~2.7 px of shift, so localization accuracy is worth as much as detection accuracy.

## 1. Architecture

Three stages plus metric-aware post-processing.

**Stage 0 — preprocessing (CPU, cached to volume)**
1. Per-pair alignment check with phase correlation; ECC homography fallback if the response is low.
2. Blur matching: estimate the photo's blur from the Laplacian variance ratio and apply the matched Gaussian to the template. Removes the largest false-positive class (sharp-vs-soft glyph edges).
3. Local background normalization (`img − GaussianBlur(img, σ=31)`) on both images to remove shadows and tone shift.
4. Stack an 8-channel tensor: template RGB (3), photo RGB (3), signed difference (1), gradient-magnitude difference (1).

**Stage 1 — siamese change detector**
- Shared-weight `convnext_tiny` encoder (timm, ImageNet initialization) over both streams, with concat + absolute-difference fusion at every scale and Changer-style stream exchange.
- U-Net decoder to output stride 2, since the objects are tiny.
- CenterNet-style heads: center heatmap, width/height, sub-pixel offset, plus an auxiliary change mask.
- Loss: Gaussian focal on the heatmap + L1 on width/height and offset + Dice/BCE on the auxiliary mask.
- Tiles of 768 px with 25% overlap, sampled around ground truth plus hard-mined negatives.

**Stage 2 — patch verifier and refiner**
- 96×96 crops from both images around each stage-1 candidate, ResNet-18 with a 6-channel stem.
- Two outputs: real-difference probability and a 4-coordinate box delta.
- Trained on stage-1 out-of-fold predictions (matched candidates positive, unmatched negative).

**Stage 3 — post-processing**
- Weighted Boxes Fusion to merge tile predictions (better tiny-box coordinates than NMS).
- Polarity filter: drop candidates where the template has ink and the photo does not (0 of 1404 ground-truth boxes look like that).
- Ink snapping: fit the box to the photo-side ink connected component, then apply the measured constant margin (+0, +0, +1, +1).
- Global threshold sweep over out-of-fold predictions, optimizing global F1 with a single threshold shared by all images.

## 2. Data strategy

1. **Synthetic pair generator** replicating the dataset's own construction: clean template → paste a synthetic difference (glyph swap, character insertion, 8×8 mark, region content change, sized to match the observed distribution) → degradation chain (Gaussian blur, sensor noise, JPEG, tone curve, shadow map, ±0.3 px warp). Target 10k+ pairs from the 200 templates plus additional OmniDocBench pages.
2. **Copy-paste augmentation** of real annotated difference patches into other pairs.
3. **5-fold cross-validation** on the 200 real pairs; synthetic data is added to every training fold, never to validation.

## 3. Modal execution plan

| Modal function | Hardware | Purpose |
|---|---|---|
| `preprocess` | CPU ×8 | Alignment check, blur match, cache preprocessed pairs and tile index to the volume |
| `synth` | CPU ×16, parallel map | Generate synthetic pairs into the volume |
| `train_stage1` | A100-40GB | Train the siamese detector for one fold |
| `predict_oof` | A10G | Out-of-fold tiled inference, produces verifier training data |
| `train_verifier` | A10G | Train the stage-2 patch verifier |
| `predict_test` | A10G | Tiled test inference, WBF, verifier rescoring |
| `sweep` | CPU | Global-F1 threshold and post-processing calibration |

Two Modal volumes: `pmdm-data` (raw + preprocessed + synthetic) and `pmdm-ckpt` (checkpoints, predictions, submissions). Code ships via `add_local_python_source`, so edits take effect without rebuilding the image.

## 4. Phases

**Phase 1 — harness (no GPU)**
- Official-equivalent scorer, 5-fold split, preprocessing, tiling, and the naive diff baseline as a floor.
- Exit criterion: the scorer reproduces the F1 definition in the task description and the baseline reports F1 ≈ 0.03.

**Phase 2 — stage-1 model**
- Train one fold on Modal, decode, sweep the threshold, measure out-of-fold global F1.
- Exit criterion: out-of-fold F1 substantially above the baseline, with the precision/recall split logged.

**Phase 3 — synthetic data**
- Build the generator, pretrain on synthetic pairs, fine-tune on real folds.
- Exit criterion: measurable out-of-fold F1 gain over Phase 2 at equal decode settings.

**Phase 4 — verifier and calibration**
- Train the stage-2 verifier on out-of-fold candidates, add ink snapping and the margin calibration, re-sweep.
- Exit criterion: precision gain without a matching recall loss.

**Phase 5 — ensemble and submission**
- 5 folds × 2 backbones (ConvNeXt-UNet, SegFormer), WBF merge, tile-offset and multi-scale test-time augmentation (no flips — text is chiral), final threshold sweep, write `submission.csv`.

## 5. Repository layout

```
TASK1_CYBER/
├── IMPLEMENTATION_PLAN.md
├── modal_app.py            # Modal app: image, volumes, all remote entrypoints
├── pyproject.toml
└── src/pmdm/
    ├── config.py           # paths, hyperparameters
    ├── metric.py           # global F1 scorer, greedy IoU matching
    ├── folds.py            # 5-fold split
    ├── preprocess.py       # alignment, blur match, local normalization, channel stack
    ├── tiles.py            # tiling and box bookkeeping
    ├── dataset.py          # torch datasets, augmentation, target encoding
    ├── model.py            # siamese ConvNeXt-UNet with CenterNet heads
    ├── losses.py           # focal, L1, Dice
    ├── decode.py           # heatmap decode, WBF, ink snapping, polarity filter
    ├── train.py            # stage-1 training loop
    ├── verifier.py         # stage-2 model, training, application
    ├── synth.py            # synthetic pair generator
    ├── infer.py            # tiled inference, out-of-fold and test
    └── baseline.py         # classical diff baseline (floor)
```

## 6. Cost control

- Development runs on a single fold with a reduced tile budget before any 5-fold sweep.
- Every Modal function has an explicit timeout and writes checkpoints to the volume after each epoch, so a preempted run resumes rather than restarts.
- GPU functions are invoked explicitly; nothing trains on import.
