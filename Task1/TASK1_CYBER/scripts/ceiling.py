"""Recall ceiling of a candidate set: what fraction of ground truth was proposed at all.

No rescoring stage can recover a box the detector never emitted, so this number bounds
everything downstream. It is deliberately score-blind — a box counts if it appears anywhere in
the candidate list at IoU >= 0.5, however low its confidence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+")
    a = ap.parse_args()

    from pmdm.dataset import load_gt
    from pmdm.evaluate_cv import load_fold_candidates, evaluate_pooled
    from pmdm.metric import iou_matrix

    gt_all = load_gt()
    for path in a.npz:
        cands = load_fold_candidates(Path(path))
        total = matched = 0
        n_cand = 0
        small_total = small_matched = 0
        for key, (boxes, scores) in cands.items():
            g = gt_all.get(int(key.split("_")[-1]), np.zeros((0, 4), np.float32))
            total += len(g)
            n_cand += len(boxes)
            if not len(g):
                continue
            sides = np.maximum(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1])
            small_total += int((sides < 12).sum())
            if not len(boxes):
                continue
            best = iou_matrix(boxes, g).max(0) if len(boxes) else np.zeros(len(g))
            hit = best >= 0.5
            matched += int(hit.sum())
            small_matched += int((hit & (sides < 12)).sum())
        _, best_f1 = evaluate_pooled(cands, gt_all)
        print(f"{Path(path).name}")
        print(f"  candidates {n_cand:>6}  ceiling {matched}/{total} = {matched / max(1, total):.4f}"
              f"   <12px {small_matched}/{small_total} = {small_matched / max(1, small_total):.4f}")
        print(f"  best F1 at a single global threshold: {best_f1['f1']:.4f} "
              f"(P {best_f1['precision']:.3f} R {best_f1['recall']:.3f} thr {best_f1['threshold']:.2f})")


if __name__ == "__main__":
    main()
