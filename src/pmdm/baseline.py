"""Classical difference baseline — the floor every learned model must beat.

Measured on the first 40 training pairs: recall 0.57, precision 0.015, F1 0.03.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import load_prepared


def baseline_pair(split: str, idx: int, diff_thr: float = 25.0, min_area: int = 12,
                  prep_root=None):
    _, _, tn, pn = load_prepared(split, idx, prep_root)
    diff = np.abs(pn.astype(np.float32) - tn.astype(np.float32))
    mask = (diff > diff_thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes, scores = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([x, y, x + w, y + h])
        scores.append(min(1.0, area / 200.0))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    return np.asarray(boxes, np.float32), np.asarray(scores, np.float32)
