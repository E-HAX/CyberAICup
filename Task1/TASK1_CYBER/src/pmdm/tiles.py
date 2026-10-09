"""Tiling helpers shared by training and inference."""
from __future__ import annotations

import numpy as np

from .config import STRIDE, TILE


def tile_origins(h: int, w: int, tile: int = TILE, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Top-left corners of overlapping tiles covering the image."""
    def axis(n: int) -> list[int]:
        if n <= tile:
            return [0]
        pos = list(range(0, n - tile + 1, stride))
        if pos[-1] != n - tile:
            pos.append(n - tile)
        return pos

    return [(x, y) for y in axis(h) for x in axis(w)]


def pad_to(img: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Pad the bottom/right so an image smaller than one tile still works."""
    h, w = img.shape[:2]
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, [(0, ph), (0, pw)] + [(0, 0)] * (img.ndim - 2), mode="edge")


def boxes_in_tile(boxes: np.ndarray, x0: int, y0: int, tile: int = TILE,
                  min_visible: float = 0.7) -> np.ndarray:
    """Boxes clipped into tile-local coordinates, keeping mostly-visible ones."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    b = boxes.astype(np.float32).copy()
    area = np.maximum(1.0, (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    b[:, [0, 2]] -= x0
    b[:, [1, 3]] -= y0
    c = b.copy()
    c[:, [0, 2]] = np.clip(c[:, [0, 2]], 0, tile)
    c[:, [1, 3]] = np.clip(c[:, [1, 3]], 0, tile)
    vis = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    return c[vis / area >= min_visible]
