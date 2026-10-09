"""Aggregate out-of-fold evaluation across all five folds.

Fold 0 alone holds 268 boxes, where a single box moves global F1 by 0.004 and the run-to-run
spread is +-0.02. Any claim finer than that has to be made on the union of all five folds —
1407 boxes, where one box is worth 0.0007. This module loads the per-fold candidate files
written by `predict_oof`, pools them, and sweeps one global threshold over the union, which
is exactly how the competition scores a submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CKPT
from .dataset import load_gt
from .metric import sweep_threshold


def load_fold_candidates(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read one fold's npz back into {pair key: (boxes, scores)}."""
    z = np.load(path)
    keys = sorted({k.rsplit("__", 1)[0] for k in z.files})
    return {k: (z[f"{k}__boxes"], z[f"{k}__scores"]) for k in keys}


def pool_folds(folds, backbone: str = "convnext_tiny", suffix: str = "",
               root: Path | None = None) -> dict:
    """Union of every fold's out-of-fold candidates.

    Each pair appears exactly once, predicted by the one model that never trained on it.
    """
    root = Path(root or CKPT) / "oof"
    pooled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in folds:
        path = root / f"fold{f}_{backbone}{suffix}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        part = load_fold_candidates(path)
        overlap = set(part) & set(pooled)
        if overlap:
            raise ValueError(f"fold {f} repeats pairs already predicted: {sorted(overlap)[:3]}")
        pooled.update(part)
    return pooled


def evaluate_pooled(pooled: dict, gt_raw: dict | None = None) -> tuple[float, dict]:
    gt_raw = gt_raw if gt_raw is not None else load_gt()
    gt = {}
    for key in pooled:
        idx = int(key.split("_")[-1])
        gt[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))
    thr, best = sweep_threshold(pooled, gt)
    best["threshold"] = thr
    best["pairs"] = len(pooled)
    best["gt_boxes"] = int(sum(len(v) for v in gt.values()))
    return thr, best


def cross_validated_score(folds=(0, 1, 2, 3, 4), backbone: str = "convnext_tiny",
                          suffix: str = "", root: Path | None = None) -> dict:
    pooled = pool_folds(folds, backbone, suffix, root)
    _, best = evaluate_pooled(pooled)
    return best


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    folds = [int(f) for f in a.folds.split(",") if f.strip()]
    print(json.dumps(cross_validated_score(folds, a.backbone, a.suffix,
                                           Path(a.root) if a.root else None), indent=2))
