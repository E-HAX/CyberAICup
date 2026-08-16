"""Paths and hyperparameters.

Every path can be overridden with an environment variable so the same code runs
locally against the unpacked dataset and remotely against Modal volumes.
"""
from __future__ import annotations

import os
from pathlib import Path

# Raw dataset root: contains train/ and test/.
DATA = Path(os.environ.get("PMDM_DATA", "/data/raw"))
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
TRAIN_CSV = TRAIN_DIR / "train.csv"

# Derived artifacts (preprocessed pairs, synthetic pairs).
WORK = Path(os.environ.get("PMDM_WORK", "/data/work"))
PREP = WORK / "prep"
SYNTH = WORK / "synth"

# Checkpoints, predictions, submissions.
CKPT = Path(os.environ.get("PMDM_CKPT", "/ckpt"))

N_TRAIN = 200
N_TEST = 100
N_FOLDS = 5

# Preprocessing.
BG_SIGMA = 31.0          # local background estimation for shadow removal
BLUR_SIGMAS = [0.0, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]  # template blur-match grid
ALIGN_MIN_RESPONSE = 0.35  # below this, fall back to ECC

# Tiling.
TILE = 768
STRIDE = 576             # 25% overlap
OUT_STRIDE = 2           # detector output stride

# Detector decode.
SCORE_THR = 0.30         # starting point; the real value comes from the sweep
TOPK_PER_TILE = 64
WBF_IOU = 0.55
MAX_BOXES_PER_IMAGE = 40

# Box calibration measured on train (median ground-truth minus ink-blob margin).
BOX_MARGIN = (0, 0, 1, 1)

# Training.
BATCH = 8
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
IN_CHANS_PER_STREAM = 4  # rgb + local-normalized gray

CSV_HEADER = ["template_image", "photo_image", "left_x", "top_y", "right_x", "bottom_y"]


def pair_names(split: str, idx: int) -> tuple[str, str]:
    """Relative csv-style names for a pair index."""
    return (
        f"template/{split}_template_{idx:03d}.png",
        f"photo/{split}_photo_{idx:03d}.png",
    )
