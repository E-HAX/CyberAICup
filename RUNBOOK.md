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

## Hugging Face mirror

The full working state lives in the private repo <https://huggingface.co/siddhant20/task1>: code,
docs, `runs/`, and all checkpoints. Give a teammate a collaborator invite on `siddhant20` and this
one link replaces both the repo tarball and Modal access.

```bash
hf download siddhant20/task1 --local-dir task1        # everything
hf download siddhant20/task1 checkpoints/stage1_fold0_convnext_tiny/best.pt --local-dir .
```

The dataset is uploaded separately because 600 page-scale PNGs at 2.5 GB take hours on a home
uplink. Run it when the machine can stay awake; it is resumable, so a killed run continues rather
than restarting:

```bash
.venv/bin/python scripts/push_dataset_to_hf.py
```

Re-push code and weights after a new run:

```bash
python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_large_folder(
    repo_id="siddhant20/task1", repo_type="model", folder_path=".",
    ignore_patterns=[".venv/**", "__pycache__/**", "**/__pycache__/**",
                     ".DS_Store", "**/.DS_Store", "*.zip", ".git/**", "Task1/**"])
PY
```

## Current pipeline (the one that produced 0.9465)

```bash
# 0. synthetic corpus, once (16 CPU shards, ~25 min)
modal run --detach modal_app.py::synth --shards 16 --per-shard 400 --small-bias 0.6
modal run modal_app.py::merge_synth        # separate step: shards commit in their own containers

# 1. two recipes x five folds (8 x A100, ~1 h wall)
modal run --detach modal_app.py::train_all --folds "0,1,2,3,4" --epochs 40 --tag _aug --n-synth 0
modal run --detach modal_app.py::train_all --folds "0,1,2,3,4" --epochs 15 --tag _augsyn \
    --n-synth 1500 --samples-per-pair 2 --real-repeat 4 --eval-every 2 --no-wh-relative

# 2. ensemble out-of-fold candidates (5 x L4, ~10 min)
modal run --detach modal_app.py::oof_ens_all --folds "0,1,2,3,4" --tags "_aug,_augsyn" \
    --flips --score-thr 0.02 --max-boxes 300 --suffix _ens

# 3. cross-fold verifier (5 x L4, ~10 min)
modal run --detach modal_app.py::verify_cv --suffix _ens --epochs 8 --folds "0,1,2,3,4"

# 4. pooled score, locally
PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset .venv/bin/python -m pmdm.evaluate_cv \
    --suffix _ens_verified --root /tmp/pool

# 5. submission (5 x L4, ~15 min)
modal run --detach modal_app.py::predict --folds "0,1,2,3,4" --tags "_aug,_augsyn" --flips \
    --score-thr 0.02 --max-boxes 300 --use-verifier --threshold 0.43 --n-shards 5
modal volume get pmdm-ckpt /submission.csv ./submission.csv
```

`--threshold` comes from the pooled sweep in step 4, never a guess. Check the reported
`boxes_per_image` against the training average of 7.04 — a large mismatch means the threshold did
not transfer from 2-model out-of-fold scoring to 10-model test scoring.

### Traps worth knowing

- **`--detach` does not protect `starmap` entrypoints.** The fan-out is client-driven; a network
  blip kills the map. `train_all`, `oof_ens_all`, `verify_cv` and `predict` now fan out inside
  remote functions for this reason.
- **Never launch a detached run inside a foreground command that can time out** — the SIGTERM
  takes the whole process group and Modal stops the app.
- **`setsid` does not exist on macOS.** A launch wrapped in it silently does nothing.
- **Volume writes from one container are invisible to another until `.reload()`.**

## Order of operations

1. `prep`, then `smoke`. The smoke test must show `hm_shape` at output stride 2 and finite losses.
2. `train --fold 0` and read the out-of-fold F1 in `history.json`. Current reference: **0.905**.
3. `synth`, then retrain fold 0 with `--n-synth 4000`. Keep the change only if F1 beats 0.905.
4. `oof` per trained fold, then `verify_ab` to measure the verifier honestly before trusting it.
5. `train_all`, then `predict`.

`--threshold` for `predict` comes from the `threshold` field in `/ckpt/oof/fold*.json`, not a guess.
