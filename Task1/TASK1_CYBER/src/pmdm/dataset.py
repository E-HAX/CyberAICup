"""Torch datasets and CenterNet-style target encoding."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import OUT_STRIDE, PREP, SYNTH, TILE, TRAIN_CSV
from .preprocess import load_prepared, stream_tensors
from .tiles import boxes_in_tile, pad_to


def load_gt(csv_path: Path | None = None) -> dict[int, np.ndarray]:
    """Ground-truth boxes keyed by train pair index."""
    df = pd.read_csv(csv_path or TRAIN_CSV)
    out: dict[int, list] = defaultdict(list)
    for r in df.itertuples(index=False):
        idx = int(str(r.template_image).split("_")[-1].split(".")[0])
        out[idx].append([r.left_x, r.top_y, r.right_x, r.bottom_y])
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def gaussian_radius(h: float, w: float, min_overlap: float = 0.5) -> float:
    """CenterNet radius: keeps a shifted box above min_overlap IoU."""
    a1, b1, c1 = 1, (h + w), w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / (2 * a1)
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / (2 * a2)
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (-b3 + np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / (2 * a3)
    return max(1.0, float(min(r1, r2, r3)))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma ** 2))
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if right <= -left or bottom <= -top:
        return
    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked = g[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(region, masked, out=region)


def encode_targets(boxes: np.ndarray, tile: int = TILE, stride: int = OUT_STRIDE):
    """Boxes in tile coordinates -> heatmap, wh, offset, regression mask, seg mask."""
    size = tile // stride
    hm = np.zeros((1, size, size), np.float32)
    wh = np.zeros((2, size, size), np.float32)
    off = np.zeros((2, size, size), np.float32)
    reg = np.zeros((1, size, size), np.float32)
    seg = np.zeros((1, size, size), np.float32)

    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < size and 0 <= iy < size):
            continue
        draw_gaussian(hm[0], ix, iy, int(gaussian_radius(bh / stride, bw / stride)))
        wh[:, iy, ix] = [bw, bh]
        off[:, iy, ix] = [cx - ix, cy - iy]
        reg[0, iy, ix] = 1.0
        sx1, sy1 = int(np.floor(x1 / stride)), int(np.floor(y1 / stride))
        sx2, sy2 = int(np.ceil(x2 / stride)), int(np.ceil(y2 / stride))
        seg[0, max(0, sy1):min(size, sy2), max(0, sx1):min(size, sx2)] = 1.0
    return hm, wh, off, reg, seg


def geometric_aug(imgs: tuple, boxes: np.ndarray, tile: int, rng: np.random.RandomState):
    """Dihedral augmentation of a square tile and its boxes.

    The training set is 160 pages, and until now the only augmentation was photometric — the
    same page geometry was seen every epoch, which is a plausible part of why both runs
    plateaued at epoch ~20. Flips and transposes are label-preserving here because a
    difference is a difference in any orientation, and they multiply the effective page count
    by eight at no I/O cost. Scale jitter is deliberately not included: box size is what the
    wh head must regress in original pixels, and the metric is size-sensitive at IoU 0.5.
    """
    t, p, tn, pn = imgs
    b = boxes.copy()

    if rng.rand() < 0.5:                                    # horizontal flip
        t, p = t[:, ::-1], p[:, ::-1]
        tn, pn = tn[:, ::-1], pn[:, ::-1]
        if len(b):
            x1 = tile - b[:, 2]
            b[:, 2] = tile - b[:, 0]
            b[:, 0] = x1
    if rng.rand() < 0.5:                                    # vertical flip
        t, p = t[::-1], p[::-1]
        tn, pn = tn[::-1], pn[::-1]
        if len(b):
            y1 = tile - b[:, 3]
            b[:, 3] = tile - b[:, 1]
            b[:, 1] = y1
    if rng.rand() < 0.5:                                    # transpose
        t, p = t.transpose(1, 0, 2), p.transpose(1, 0, 2)
        tn, pn = tn.T, pn.T
        if len(b):
            b = b[:, [1, 0, 3, 2]]

    return (np.ascontiguousarray(t), np.ascontiguousarray(p),
            np.ascontiguousarray(tn), np.ascontiguousarray(pn)), b


def photometric_jitter(photo: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Extra print/scan-like noise on the photo stream only."""
    out = photo.astype(np.float32)
    out = out * rng.uniform(0.92, 1.08) + rng.uniform(-12, 12)
    if rng.rand() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))
    if rng.rand() < 0.7:
        out += rng.normal(0, rng.uniform(1.0, 5.0), out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class GroupedSampler(torch.utils.data.Sampler):
    """Shuffle pairs, keep a pair's samples adjacent.

    Every sample decodes four full-page PNGs, so random access would decode the
    same pair `samples_per_pair` times per epoch. Grouping lets the one-entry
    cache in the dataset serve the rest of the group.
    """

    def __init__(self, n_pairs: int, samples_per_pair: int, seed: int = 0):
        self.n_pairs = n_pairs
        self.samples_per_pair = samples_per_pair
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_pairs * self.samples_per_pair

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        for pair in rng.permutation(self.n_pairs):
            base = int(pair) * self.samples_per_pair
            for k in rng.permutation(self.samples_per_pair):
                yield base + int(k)


class PairTileDataset(Dataset):
    """Random tiles from prepared pairs, biased towards annotated differences."""

    def __init__(self, items: list[tuple[str, int]], gt: dict, samples_per_pair: int = 8,
                 tile: int = TILE, positive_ratio: float = 0.6, augment: bool = True,
                 prep_root: Path | None = None):
        self.items = items
        self.gt = gt
        self.samples_per_pair = samples_per_pair
        self.tile = tile
        self.positive_ratio = positive_ratio
        self.augment = augment
        self.prep_root = prep_root or PREP
        self._cache_key = None
        self._cache_val = None

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_pair

    def _key(self, split: str, idx: int):
        return (split, idx)

    def _load(self, split: str, idx: int):
        key = (split, idx)
        if key != self._cache_key:
            root = self.prep_root if split != "synth" else SYNTH
            self._cache_val = load_prepared(split, idx, root)
            self._cache_key = key
        return self._cache_val

    def __getitem__(self, i: int):
        split, idx = self.items[i // self.samples_per_pair]
        rng = np.random.RandomState((i * 9973 + idx * 7919) % (2 ** 31))
        t, p, tn, pn = self._load(split, idx)
        boxes = self.gt.get(self._key(split, idx), np.zeros((0, 4), np.float32))

        t, p = pad_to(t, self.tile), pad_to(p, self.tile)
        tn, pn = pad_to(tn, self.tile), pad_to(pn, self.tile)
        h, w = t.shape[:2]

        if len(boxes) and rng.rand() < self.positive_ratio:
            bx = boxes[rng.randint(len(boxes))]
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            x0 = int(np.clip(cx - self.tile / 2 + rng.randint(-200, 201), 0, max(0, w - self.tile)))
            y0 = int(np.clip(cy - self.tile / 2 + rng.randint(-200, 201), 0, max(0, h - self.tile)))
        else:
            x0 = rng.randint(0, max(1, w - self.tile + 1))
            y0 = rng.randint(0, max(1, h - self.tile + 1))

        sl = (slice(y0, y0 + self.tile), slice(x0, x0 + self.tile))
        tc, pc, tnc, pnc = t[sl], p[sl], tn[sl], pn[sl]
        local = boxes_in_tile(boxes, x0, y0, self.tile)
        if self.augment:
            (tc, pc, tnc, pnc), local = geometric_aug((tc, pc, tnc, pnc), local, self.tile, rng)
            pc = photometric_jitter(pc, rng)

        a, b = stream_tensors(tc, pc, tnc, pnc)
        hm, wh, off, reg, seg = encode_targets(local, self.tile)
        return {
            "a": torch.from_numpy(a),
            "b": torch.from_numpy(b),
            "hm": torch.from_numpy(hm),
            "wh": torch.from_numpy(wh),
            "off": torch.from_numpy(off),
            "reg": torch.from_numpy(reg),
            "seg": torch.from_numpy(seg),
        }
