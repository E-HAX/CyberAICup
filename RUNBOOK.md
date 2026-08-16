# Runbook

Read `HANDOVER.md` first for context and current results. This file is commands only.

## Local setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . torch torchvision timm opencv-python-headless "numpy<2" pandas
```

Common environment prefix for local runs:

```bash
export PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset
export PMDM_WORK=/tmp/pmdm_work
export PMDM_CKPT=/tmp/pmdm_ckpt
```

| Script | Purpose | Runtime |
|---|---|---|
| `scripts/local_smoke.py` | CPU pre-flight: preprocess, dataset, model forward/backward, decode, scorer, generator | ~2 min |
| `scripts/analyze_dataset.py` | Reproduces every dataset claim in HANDOVER §2 | ~2 min |
| `scripts/error_analysis.py <oof.npz>` | Error breakdown by cause and box size | seconds |
| `scripts/train_local.py --preprocess-only` | Stage 0 over all 300 pairs, no Modal | ~10 min |
| `scripts/train_local.py --fold 0 --epochs 40` | Same training loop as Modal, on a local GPU | ~45 min on A100 |

Archived logs and artifacts from every run so far are in `runs/` — see `runs/README.md`.

## Modal

Data upload (once per account, ~2.5 GB):

```bash
modal volume create pmdm-data
modal volume create pmdm-ckpt
modal volume put pmdm-data Task1/PackagingMaterialDifferenceMiningDataset/train /raw/train
modal volume put pmdm-data Task1/PackagingMaterialDifferenceMiningDataset/test  /raw/test
```

| Step | Command | Hardware | Measured time |
|---|---|---|---|
| Stage 0 preprocessing | `modal run modal_app.py::prep` | CPU fan-out ×300 | few minutes |
| Shape/loss check | `modal run modal_app.py::smoke` | L4 | ~2 min |
| Synthetic data | `modal run modal_app.py::synth --shards 16 --per-shard 400` | CPU fan-out | ~20 min (not yet run) |
| Train one fold | `modal run --detach modal_app.py::train --fold 0 --epochs 40` | A100-40GB | ~45 min |
| Train with synthetic | `modal run --detach modal_app.py::train --fold 0 --epochs 40 --n-synth 4000` | A100-40GB | ~60 min |
| All folds | `modal run --detach modal_app.py::train_all --epochs 40` | 5 × A100-40GB | ~45 min wall |
| Out-of-fold candidates | `modal run modal_app.py::oof --fold 0` | L4 | ~3 min |
| Verifier A/B (honest) | `modal run modal_app.py::verify_ab --fold 0` | L4 | ~5 min |
| Verifier (all folds) | `modal run modal_app.py::verify` | L4 | ~15 min |
| Submission | `modal run modal_app.py::predict --folds 0,1,2,3,4 --threshold 0.31` | L4 | ~10 min |

Fetch results:

```bash
modal volume get pmdm-ckpt /submission.csv ./submission.csv
modal volume get pmdm-ckpt /stage1_fold0_convnext_tiny/history.json ./history.json
modal volume get pmdm-ckpt /oof/fold0_convnext_tiny.npz /tmp/oof.npz
```

The trained weights are already mirrored into `checkpoints/` in this repo — see
`checkpoints/README.md`. Re-mirror after a new run, or push a local checkpoint back up:

```bash
modal volume get pmdm-ckpt /stage1_fold0_convnext_tiny checkpoints/stage1_fold0_convnext_tiny
modal volume put pmdm-ckpt checkpoints/stage1_fold0_convnext_tiny /stage1_fold0_convnext_tiny
```

Use `modal volume get` for backups, never `modal volume cp` — it refuses directories while still
exiting 0, which is how one checkpoint was lost (`HANDOVER.md` §6).

## Launch discipline

- **Always `--detach` for training.** A local network blip otherwise kills the run.
- **Confirm training actually started** — look for `[fold N] epoch 0 step 0/160` in the logs. An app
  that reports "Created function ..." and then stops has trained nothing.
- Stream logs of a detached run: `modal app logs <app-id>`.
- `train_fold` commits the volume after every epoch and resumes automatically from `last.pt`, so an
  interruption costs at most one epoch. To force a fresh run, delete the checkpoint directory first —
  and back it up with `modal volume get`, not `modal volume cp` (which refuses directories while
  still exiting 0).

## Order of operations

1. `prep`, then `smoke`. The smoke test must show `hm_shape` at output stride 2 and finite losses.
2. `train --fold 0` and read the out-of-fold F1 in `history.json`. Current reference: **0.905**.
3. `synth`, then retrain fold 0 with `--n-synth 4000`. Keep the change only if F1 beats 0.905.
4. `oof` per trained fold, then `verify_ab` to measure the verifier honestly before trusting it.
5. `train_all`, then `predict`.

`--threshold` for `predict` comes from the `threshold` field in `/ckpt/oof/fold*.json`, not a guess.
