"""Tiled full-image inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import MAX_BOXES_PER_IMAGE, PREP, STRIDE, TILE
from .decode import decode_heatmap, postprocess, wbf
from .preprocess import load_prepared, stream_tensors
from .tiles import pad_to, tile_origins


# (hflip, vflip, transpose) — the same eight-element dihedral group the training
# augmentation samples from, so the model has seen every one of these views.
VIEWS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def _apply_view(x: torch.Tensor, view: tuple[int, int, int]) -> torch.Tensor:
    hflip, vflip, transpose = view
    if hflip:
        x = torch.flip(x, dims=[-1])
    if vflip:
        x = torch.flip(x, dims=[-2])
    if transpose:
        x = x.transpose(-1, -2)
    return x.contiguous()


def _unview_boxes(boxes: np.ndarray, view: tuple[int, int, int], tile: int) -> np.ndarray:
    """Map boxes decoded in a transformed view back to tile coordinates.

    The inverse of the dihedral element, applied in reverse order: transpose first, because
    it was applied last.
    """
    hflip, vflip, transpose = view
    b = boxes.copy()
    if transpose:
        b = b[:, [1, 0, 3, 2]]
    if vflip:
        y1 = tile - b[:, 3]
        b[:, 3] = tile - b[:, 1]
        b[:, 1] = y1
    if hflip:
        x1 = tile - b[:, 2]
        b[:, 2] = tile - b[:, 0]
        b[:, 0] = x1
    return b


@torch.no_grad()
def predict_pair(model, split: str, idx: int, device: str = "cuda",
                 prep_root: Path | None = None, tile: int = TILE, stride: int = STRIDE,
                 batch: int = 4, score_thr: float = 0.05, offsets: tuple[int, ...] = (0,),
                 use_snap: bool = True, use_polarity: bool = True,
                 max_boxes: int = MAX_BOXES_PER_IMAGE, flips: bool = False,
                 n_sources: int | None = None):
    """Returns (boxes (N,4) in template pixels, scores (N,)).

    `flips` runs the dihedral test-time augmentation the model was trained under: each tile is
    also passed through horizontally flipped, vertically flipped and transposed, and the boxes
    are mapped back before fusion. Four views of every tile means the coordinate averaging in
    WBF has more to work with, which is where 8 px boxes gain or lose the IoU 0.5 threshold.
    """
    t, p, tn, pn = load_prepared(split, idx, prep_root or PREP)
    h, w = t.shape[:2]
    t, p = pad_to(t, tile), pad_to(p, tile)
    tnp, pnp = pad_to(tn, tile), pad_to(pn, tile)
    a_full, b_full = stream_tensors(t, p, tnp, pnp)

    all_boxes, all_scores = [], []
    for shift in offsets:                      # tile-offset test-time augmentation
        origins = tile_origins(t.shape[0], t.shape[1], tile, stride)
        if shift:
            origins = [(max(0, min(t.shape[1] - tile, x + shift)),
                        max(0, min(t.shape[0] - tile, y + shift))) for x, y in origins]
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            a = np.stack([a_full[:, y:y + tile, x:x + tile] for x, y in chunk])
            b = np.stack([b_full[:, y:y + tile, x:x + tile] for x, y in chunk])
            ta = torch.from_numpy(a).to(device)
            tb = torch.from_numpy(b).to(device)
            for view in (VIEWS if flips else VIEWS[:1]):
                va, vb = _apply_view(ta, view), _apply_view(tb, view)
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                    dtype=torch.bfloat16, enabled="cuda" in device):
                    out = model(va, vb)
                for k, (x0, y0) in enumerate(chunk):
                    boxes, scores = decode_heatmap(
                        out["hm"][k].float(), out["wh"][k].float(), out["off"][k].float(),
                        score_thr=score_thr,
                    )
                    if not len(boxes):
                        continue
                    boxes = _unview_boxes(boxes, view, tile)
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes.append(boxes)
                    all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    # One view already yields ~2 overlapping tiles per box, so the expected source count is
    # two per view unless the caller knows better.
    if n_sources is None and (flips or len(offsets) > 1):
        n_sources = 2 * len(offsets) * (len(VIEWS) if flips else 1)
    boxes, scores = wbf(boxes, scores, n_sources=n_sources)
    boxes, scores = postprocess(boxes, scores, tn, pn, use_snap, use_polarity,
                                max_boxes=max_boxes)
    keep = (boxes[:, 0] < w) & (boxes[:, 1] < h)
    return boxes[keep], scores[keep]


@torch.no_grad()
def predict_pair_ensemble(models, split: str, idx: int, device: str = "cuda", **kwargs):
    """Fuse several models on one pair by pooling their raw candidates before WBF.

    Fusing before WBF rather than after means the coordinate averaging sees every model's
    opinion of the same box, instead of averaging already-averaged boxes. Post-processing runs
    once, at the end, on the pooled set.
    """
    if len(models) == 1:
        return predict_pair(models[0], split, idx, device=device, **kwargs)

    from .config import MAX_BOXES_PER_IMAGE as _MB
    prep_root = kwargs.get("prep_root")
    max_boxes = kwargs.pop("max_boxes", _MB)
    use_snap = kwargs.pop("use_snap", True)
    use_polarity = kwargs.pop("use_polarity", True)

    pooled_boxes, pooled_scores = [], []
    for m in models:
        # each member returns pre-postprocess candidates; postprocess is applied once below
        b, sc = predict_pair(m, split, idx, device=device, max_boxes=10 ** 6,
                            use_snap=False, use_polarity=False, **kwargs)
        if len(b):
            pooled_boxes.append(b)
            pooled_scores.append(sc)
    if not pooled_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    boxes = np.concatenate(pooled_boxes, 0)
    scores = np.concatenate(pooled_scores, 0)
    # Each member already ran WBF over its own tiles and TTA views, so what is pooled here is
    # one fused box per model per cluster. The source count is therefore the number of models,
    # not models x views — using the latter divides every score by ~16 and wipes out recall.
    n_src = kwargs.get("n_sources") or len(models)
    boxes, scores = wbf(boxes, scores, n_sources=n_src)
    t, p, tn, pn = load_prepared(split, idx, prep_root or PREP)
    h, w = tn.shape[:2]
    boxes, scores = postprocess(boxes, scores, tn, pn, use_snap, use_polarity,
                                max_boxes=max_boxes)
    keep = (boxes[:, 0] < w) & (boxes[:, 1] < h)
    return boxes[keep], scores[keep]


def load_model(ckpt_path: Path, device: str = "cuda", backbone: str = "convnext_tiny"):
    from .model import SiamCenterNet

    model = SiamCenterNet(backbone=backbone, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()
