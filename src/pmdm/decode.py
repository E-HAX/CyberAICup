"""Heatmap decoding, box fusion and the metric-aware post-processing.

Two dataset facts are exploited here:
  * no ground-truth box is a pure deletion (0 of 1404), so a candidate whose
    template side has ink and whose photo side does not is dropped;
  * the ink-blob box sits within (0, 0, +1, +1) of the ground-truth box, so a
    constant margin is added after snapping.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import BOX_MARGIN, MAX_BOXES_PER_IMAGE, OUT_STRIDE, TOPK_PER_TILE, WBF_IOU
from .metric import iou_matrix


def decode_heatmap(hm_logits: torch.Tensor, wh: torch.Tensor, off: torch.Tensor,
                   score_thr: float = 0.05, topk: int = TOPK_PER_TILE,
                   stride: int = OUT_STRIDE):
    """Single-image decode. Returns (boxes (N,4) in tile pixels, scores (N,))."""
    hm = torch.sigmoid(hm_logits)
    keep = (F.max_pool2d(hm, 3, stride=1, padding=1) == hm).float()
    hm = hm * keep
    scores, flat = hm.reshape(-1).topk(min(topk, hm.numel()))
    size = hm.shape[-1]
    ys = (flat // size).float()
    xs = (flat % size).float()

    off_flat = off.reshape(2, -1)[:, flat]
    wh_flat = wh.reshape(2, -1)[:, flat]
    cx = (xs + off_flat[0]) * stride
    cy = (ys + off_flat[1]) * stride
    w, h = wh_flat[0].clamp(min=1.0), wh_flat[1].clamp(min=1.0)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)

    m = scores >= score_thr
    return boxes[m].cpu().numpy(), scores[m].cpu().numpy()


def wbf(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = WBF_IOU):
    """Weighted Boxes Fusion. Averages coordinates instead of discarding them,
    which matters at IoU 0.5 on 8x8 boxes."""
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    clusters: list[list[int]] = []
    fused: list[np.ndarray] = []
    fused_scores: list[float] = []

    for i in range(len(boxes)):
        placed = False
        if fused:
            ious = iou_matrix(boxes[i:i + 1], np.asarray(fused, np.float32))[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                clusters[j].append(i)
                members = clusters[j]
                w = scores[members]
                fused[j] = (boxes[members] * w[:, None]).sum(0) / w.sum()
                fused_scores[j] = float(w.sum() / min(len(members) + 1, 3))
                placed = True
        if not placed:
            clusters.append([i])
            fused.append(boxes[i].copy())
            fused_scores.append(float(scores[i]))

    f = np.asarray(fused, np.float32)
    s = np.clip(np.asarray(fused_scores, np.float32), 0, 1)
    order = np.argsort(-s)
    return f[order], s[order]


def polarity_filter(boxes: np.ndarray, scores: np.ndarray, tn: np.ndarray, pn: np.ndarray,
                    ink_thr: float = 3.0):
    """Drop candidates that look like a deletion (template inked, photo blank)."""
    if len(boxes) == 0:
        return boxes, scores
    keep = np.ones(len(boxes), bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(tn.shape[1], x2), min(tn.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            keep[i] = False
            continue
        t_ink = float(tn[y1:y2, x1:x2].mean())
        p_ink = float(pn[y1:y2, x1:x2].mean())
        if t_ink > ink_thr and p_ink < ink_thr:
            keep[i] = False
    return boxes[keep], scores[keep]


def snap_to_ink(boxes: np.ndarray, tn: np.ndarray, pn: np.ndarray, pad: int = 6,
                diff_thr: float = 25.0, margin: tuple[int, int, int, int] = BOX_MARGIN,
                max_shift: int = 6) -> np.ndarray:
    """Refit each box to the local ink-difference blob, then apply the measured margin."""
    if len(boxes) == 0:
        return boxes
    diff = np.abs(pn.astype(np.float32) - tn.astype(np.float32))
    out = boxes.copy()
    h, w = diff.shape
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        a, b = int(max(0, y1 - pad)), int(min(h, y2 + pad))
        c, d = int(max(0, x1 - pad)), int(min(w, x2 + pad))
        if b <= a or d <= c:
            continue
        m = diff[a:b, c:d] > diff_thr
        if m.sum() < 3:
            continue
        ys, xs = np.nonzero(m)
        nx1, ny1 = xs.min() + c, ys.min() + a
        nx2, ny2 = xs.max() + 1 + c, ys.max() + 1 + a
        cand = np.array([nx1 - margin[0], ny1 - margin[1], nx2 + margin[2], ny2 + margin[3]],
                        np.float32)
        if np.abs(cand - boxes[i]).max() <= max_shift:
            out[i] = cand
    return out


def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    b = boxes.copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    return b


def postprocess(boxes: np.ndarray, scores: np.ndarray, tn: np.ndarray, pn: np.ndarray,
                use_snap: bool = True, use_polarity: bool = True):
    if len(boxes) == 0:
        return boxes, scores
    h, w = tn.shape[:2]
    boxes = clip_boxes(boxes, h, w)
    if use_polarity:
        boxes, scores = polarity_filter(boxes, scores, tn, pn)
    if use_snap and len(boxes):
        boxes = snap_to_ink(boxes, tn, pn)
    keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) > MAX_BOXES_PER_IMAGE:
        order = np.argsort(-scores)[:MAX_BOXES_PER_IMAGE]
        boxes, scores = boxes[order], scores[order]
    return boxes, scores
