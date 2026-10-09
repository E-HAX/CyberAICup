"""Global F1 scorer.

Matches the definition in the task description: TP/FP/FN are accumulated across
all images before precision, recall and F1 are computed. A prediction is a TP if
its IoU with an unmatched ground-truth box is >= 0.5.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import CSV_HEADER

IOU_THR = 0.5


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred (N,4), gt (M,4) in x1,y1,x2,y2 -> (N,M) IoU."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float32)
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def match_image(pred: np.ndarray, gt: np.ndarray, scores: np.ndarray | None = None):
    """Greedy score-ordered matching. Returns (tp, fp, fn)."""
    if scores is not None and len(pred):
        pred = pred[np.argsort(-scores)]
    ious = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    tp = fp = 0
    for i in range(len(pred)):
        best, bj = 0.0, -1
        for j in range(len(gt)):
            if used[j]:
                continue
            if ious[i, j] > best:
                best, bj = float(ious[i, j]), j
        if best >= IOU_THR and bj >= 0:
            used[bj] = True
            tp += 1
        else:
            fp += 1
    return tp, fp, int((~used).sum())


def boxes_by_pair(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, list] = defaultdict(list)
    for r in df.itertuples(index=False):
        out[r.template_image].append(
            [int(r.left_x), int(r.top_y), int(r.right_x), int(r.bottom_y)]
        )
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def global_f1(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """Score a submission-shaped dataframe against ground truth."""
    gt = boxes_by_pair(gt_df)
    pr = boxes_by_pair(pred_df)
    scores = None
    if "score" in pred_df.columns:
        scores = defaultdict(list)
        for r in pred_df.itertuples(index=False):
            scores[r.template_image].append(float(r.score))

    tp = fp = fn = 0
    for key in set(gt) | set(pr):
        p = pr.get(key, np.zeros((0, 4), np.float32))
        g = gt.get(key, np.zeros((0, 4), np.float32))
        s = np.asarray(scores[key], np.float32) if scores is not None and key in scores else None
        a, b, c = match_image(p, g, s)
        tp, fp, fn = tp + a, fp + b, fn + c

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def sweep_threshold(
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    gt: dict[str, np.ndarray],
    grid: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Pick the single global score threshold that maximizes global F1.

    candidates maps pair key -> (boxes (N,4), scores (N,)).
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)
    best_thr, best = float(grid[0]), {"f1": -1.0}
    for thr in grid:
        tp = fp = fn = 0
        for key in set(gt) | set(candidates):
            boxes, scores = candidates.get(key, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
            keep = scores >= thr
            a, b, c = match_image(boxes[keep], gt.get(key, np.zeros((0, 4), np.float32)), scores[keep])
            tp, fp, fn = tp + a, fp + b, fn + c
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        if f1 > best["f1"]:
            best = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
            best_thr = float(thr)
    return best_thr, best


def write_submission(candidates: dict[str, tuple[np.ndarray, np.ndarray]], thr: float, path):
    rows = []
    for template_name, (boxes, scores) in sorted(candidates.items()):
        photo_name = template_name.replace("template/", "photo/").replace("_template_", "_photo_")
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s < thr:
                continue
            rows.append([template_name, photo_name, int(round(x1)), int(round(y1)),
                         int(round(x2)), int(round(y2))])
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    df.to_csv(path, index=False)
    return df
