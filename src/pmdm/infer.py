"""Tiled full-image inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import PREP, STRIDE, TILE
from .decode import decode_heatmap, postprocess, wbf
from .preprocess import load_prepared, stream_tensors
from .tiles import pad_to, tile_origins


@torch.no_grad()
def predict_pair(model, split: str, idx: int, device: str = "cuda",
                 prep_root: Path | None = None, tile: int = TILE, stride: int = STRIDE,
                 batch: int = 4, score_thr: float = 0.05, offsets: tuple[int, ...] = (0,),
                 use_snap: bool = True, use_polarity: bool = True):
    """Returns (boxes (N,4) in template pixels, scores (N,))."""
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
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                dtype=torch.bfloat16, enabled="cuda" in device):
                out = model(ta, tb)
            for k, (x0, y0) in enumerate(chunk):
                boxes, scores = decode_heatmap(
                    out["hm"][k].float(), out["wh"][k].float(), out["off"][k].float(),
                    score_thr=score_thr,
                )
                if len(boxes):
                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0
                    all_boxes.append(boxes)
                    all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    boxes, scores = wbf(boxes, scores)
    boxes, scores = postprocess(boxes, scores, tn, pn, use_snap, use_polarity)
    keep = (boxes[:, 0] < w) & (boxes[:, 1] < h)
    return boxes[keep], scores[keep]


def load_model(ckpt_path: Path, device: str = "cuda", backbone: str = "convnext_tiny"):
    from .model import SiamCenterNet

    model = SiamCenterNet(backbone=backbone, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()
