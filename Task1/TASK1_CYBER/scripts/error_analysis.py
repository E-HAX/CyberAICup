"""Break down out-of-fold errors by cause and by box size.

This is the script that decided the current priority: it separates "the model
never proposed this box" from "the model proposed it but the coordinates were
too loose", and reports the recall ceiling that perfect box regression would
reach.

    modal volume get pmdm-ckpt /oof/fold0_convnext_tiny.npz /tmp/oof.npz
    PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
    .venv/bin/python scripts/error_analysis.py /tmp/oof.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from pmdm.dataset import load_gt  # noqa: E402
from pmdm.metric import iou_matrix  # noqa: E402

SIZE_BINS = [(0, 12, "<12px"), (12, 24, "12-24px"), (24, 10 ** 9, ">24px")]


def main(npz_path: str) -> None:
    z = np.load(npz_path)
    gt = load_gt()
    keys = sorted({k.split("__")[0] for k in z.files})

    best_iou, sizes = [], []
    n_candidates = 0
    for key in keys:
        idx = int(key.split("_")[-1])
        boxes = z[f"{key}__boxes"]
        g = gt[idx]
        n_candidates += len(boxes)
        m = iou_matrix(boxes, g) if len(boxes) and len(g) else np.zeros((len(boxes), len(g)))
        for j in range(len(g)):
            best_iou.append(float(m[:, j].max()) if m.size else 0.0)
            sizes.append(max(g[j][2] - g[j][0], g[j][3] - g[j][1]))

    best_iou, sizes = np.array(best_iou), np.array(sizes)
    print(f"pairs: {len(keys)}, ground-truth boxes: {len(best_iou)}, "
          f"candidates per image: {n_candidates / max(1, len(keys)):.1f}")
    print(f"matched at IoU >= 0.5:        {int((best_iou >= 0.5).sum())}")
    print(f"proposed but IoU < 0.5:       {int(((best_iou > 0) & (best_iou < 0.5)).sum())}")
    print(f"never proposed (IoU == 0):    {int((best_iou == 0).sum())}")
    print(f"recall ceiling with perfect boxes: {float((best_iou > 0).mean()):.4f}")

    print("\nby box size (longest side):")
    for lo, hi, name in SIZE_BINS:
        m = (sizes >= lo) & (sizes < hi)
        if not m.sum():
            continue
        print(f"  {name:9s} n={int(m.sum()):4d}  "
              f"matched={float((best_iou[m] >= 0.5).mean()):.3f}  "
              f"never_proposed={int((best_iou[m] == 0).sum())}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/oof.npz")
