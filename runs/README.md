# Run archive

Raw evidence behind every number in `HANDOVER.md`. These files are checked into the repo because
the original logs lived in an ephemeral session directory and the rest lived only on Modal volumes,
which the next owner may not have access to.

The training *code* is not here — it is `src/pmdm/train.py` (the loop) driven by `modal_app.py`
(the entrypoints). This directory holds only what those runs produced.

| File | What it is | Caveat |
|---|---|---|
| `fold0_run2_fixed.log` | **The primary run.** Fold 0, 40 epochs requested, ConvNeXt-tiny, no synthetic data, after the bfloat16 loss fix. Full step-level losses, all 7 evaluations (epochs 4–34). | Modal cancelled the run at ~epoch 36, so epoch 39 never evaluated. `grep -c "hm=nan"` returns 0 — this is the evidence the loss fix worked. |
| `fold0_run2_history.json` | Per-epoch structured record from the same run: losses, timings, and evaluation metrics with the swept threshold. | 36 records (epochs 0–35). Written by `train_fold`, downloaded from the volume. |
| `fold0_run1_nanbug.log` | The first run, which carried the NaN bug from epoch 26. | **Partial: epochs 19–31 only.** Log streaming was attached mid-run, so epochs 0–18 and 32–39 were never captured to a file. Their metrics survive in the `HANDOVER.md` results table. Shows the NaN onset directly. |
| `oof_fold0_candidates.npz` | Out-of-fold predictions from run 2's `best.pt` across all 40 validation pairs: boxes and scores per pair. | Input to `scripts/error_analysis.py`. |
| `oof_fold0_sweep.json` | Global-F1 sweep result on those candidates: TP 238, FP 20, FN 30, F1 0.9049, threshold 0.31. | The headline number. |
| `error_analysis_output.txt` | Error breakdown by cause and box size — the finding that sets the current priority. | Regenerate: `python scripts/error_analysis.py runs/oof_fold0_candidates.npz` |
| `analyze_dataset_output.txt` | Full output of the dataset analysis: alignment, box sizes, polarity, blur, baseline floor. | Regenerate: `python scripts/analyze_dataset.py` (~2 min) |
| `verifier_ab_raw.txt` | Raw output of the leakage-free verifier A/B, including the training record counts that explain why it failed. | Delta was +0.0009. |
| `prep_info.json` | Per-pair preprocessing record for all 300 pairs: alignment mode, phase-correlation response, chosen blur sigma, dimensions. | Evidence for "no registration needed": 284 `none`, 16 `translate`, 0 ECC. |

## Reading the training logs

Loss keys: `hm` center heatmap (Gaussian focal), `wh` box size (masked L1), `off` sub-pixel offset,
`seg` auxiliary change mask (Dice+BCE), `total` weighted sum.

Per-step values are running averages within the epoch, which is why a single NaN batch poisons the
rest of the epoch's display while the model itself keeps training — GradScaler skips the NaN steps.

Evaluation lines carry the full metric dict including `threshold`, the single global score cutoff
chosen by sweeping to maximize global F1 on that fold's 40 validation pairs.

## Reproducing the headline number

```bash
modal run modal_app.py::oof --fold 0     # expect F1 0.9049, threshold 0.31
```

Or offline from the archived candidates, without touching Modal:

```bash
PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
.venv/bin/python scripts/error_analysis.py runs/oof_fold0_candidates.npz
```
