"""Shared configuration: paths, class vocabulary, CV protocol, seeds.

All remote paths refer to Modal Volume mount points, so the same module works
inside every container without conditional logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths ----

# Modal Volume mount points (see src/modal_app.py).
WORK = Path(os.environ.get("RTC_WORK", "/work"))
MIRAGE = Path(os.environ.get("RTC_MIRAGE", "/mirage"))
CACHE = Path(os.environ.get("RTC_CACHE", "/cache"))

DATA_DIR = WORK / "data"
FEATURES_DIR = WORK / "features"
OOF_DIR = WORK / "oof"
MODELS_DIR = WORK / "models"
REPORTS_DIR = WORK / "reports"
EDA_DIR = REPORTS_DIR / "eda"

TRAIN_CSV = DATA_DIR / "Training_set.csv"
TEST_CSV = DATA_DIR / "Testing_set.csv"

# Copy of the competition CSVs baked into the image, used to seed the Volume.
SEED_DIR = Path("/root/data_seed")


def ensure_dirs() -> None:
    for d in (DATA_DIR, FEATURES_DIR, OOF_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- classes ----

APPS = ["Discord", "GoogleMeet", "Messenger", "WhatsApp", "Zoom"]
MODES = ["voice", "video"]

# Case-sensitive label vocabulary required by the submission format.
LABELS = [f"{app}_{mode}" for app in APPS for mode in MODES]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for lab, i in LABEL_TO_ID.items()}

N_PACKETS = 5
TIME_COLS = [f"relative_time_{i}" for i in range(N_PACKETS)]
LEN_COLS = [f"packet_length_{i}" for i in range(N_PACKETS)]
FEATURE_COLS = [c for pair in zip(TIME_COLS, LEN_COLS) for c in pair]

N_TEST_ROWS = 327
N_TRAIN_ROWS = 1285

# Source-call counts stated in the task description, used in EDA to derive the
# implied flows-per-call prior (40 voice + 40 video calls per app in train).
CALLS_PER_CLASS_TRAIN = 40
CALLS_PER_CLASS_TEST = 10

# ------------------------------------------------------------------ CV ----

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

# Rounding used to build the pseudo-group key that keeps near-duplicate flows
# (very likely originating from the same source call) inside the same fold.
GROUP_LEN_ROUND = 8  # bytes
GROUP_LOG_TIME_ROUND = 1  # decades of log10(duration)

# --------------------------------------------------------- domain bands ----

# Packet-length bands with a protocol meaning for SRTP-over-DTLS media flows.
BAND_TINY = (0, 100)  # STUN binding, RTCP, DTX/comfort noise
BAND_AUDIO = (100, 300)  # Opus/AAC frame + RTP/SRTP overhead
BAND_MID = (300, 1100)
BAND_MTU = (1100, 1600)  # video fragment hitting the path MTU

# Inter-arrival thresholds used for burst / frame-group segmentation.
BURST_THRESHOLDS = (0.0005, 0.001, 0.005, 0.020)
