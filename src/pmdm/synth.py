"""Synthetic pair generator.

Reproduces the dataset's own construction: take a clean template, paste
differences into a copy of it, then degrade that copy into a "photo" (blur,
noise, JPEG, tone curve, shadow field, sub-pixel warp). The generated pairs are
written through the same preprocessing path as the real ones, so training code
cannot tell them apart.

Edit types and sizes follow the measured training statistics: median 22x22 px,
30% of boxes under 16 px, additions and modifications only, never deletions.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import DATA, SYNTH
from .preprocess import local_norm, match_blur

EDIT_WEIGHTS = {"swap": 0.40, "insert": 0.30, "mark": 0.30}

# Error analysis on fold 0: boxes under 12 px match 82.8% of the time against 97.5% for
# everything larger, and 10 of the 11 completely undetected boxes are in that band. The
# small-biased mix exists to feed that band specifically; SMALL_BIAS=0 reproduces the
# original distribution for a controlled comparison.
MARK_SIZES_DEFAULT = [6, 8, 8, 10, 12, 16, 24]
MARK_SIZES_SMALL = [4, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 16, 20]
SMALL_EDIT_WEIGHTS = {"swap": 0.25, "insert": 0.25, "mark": 0.50}


def ink_components(img: np.ndarray, min_area: int = 12, max_area: int = 8000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area and 3 <= w <= 160 and 3 <= h <= 160:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def blank_spot(ink: np.ndarray, size: int, rng: np.random.RandomState, tries: int = 40):
    """Find a location whose neighbourhood is empty but which sits near content."""
    h, w = ink.shape
    for _ in range(tries):
        x = rng.randint(4, max(5, w - size - 4))
        y = rng.randint(4, max(5, h - size - 4))
        patch = ink[y - 2:y + size + 2, x - 2:x + size + 2]
        if patch.size and patch.max() < 8:
            near = ink[max(0, y - 60):y + size + 60, max(0, x - 60):x + size + 60]
            if near.size and near.max() > 30:
                return x, y
    return None


def apply_edits(template: np.ndarray, rng: np.random.RandomState,
                n_edits: tuple[int, int] = (3, 11), small_bias: float = 0.0):
    """Return (edited image, boxes). Edits only add or change ink, never remove it."""
    edited = template.copy()
    ink = local_norm(template)
    comps = ink_components(template)
    boxes: list[list[int]] = []
    n = rng.randint(*n_edits)
    weights = SMALL_EDIT_WEIGHTS if rng.rand() < small_bias else EDIT_WEIGHTS
    mark_sizes = MARK_SIZES_SMALL if weights is SMALL_EDIT_WEIGHTS else MARK_SIZES_DEFAULT
    kinds = list(weights)
    probs = np.array([weights[k] for k in kinds], np.float32)
    probs /= probs.sum()

    for _ in range(n):
        kind = kinds[int(rng.choice(len(kinds), p=probs))]

        if kind == "swap" and len(comps) > 2:
            x, y, w, h = comps[rng.randint(len(comps))]
            for _ in range(10):
                sx, sy, sw, sh = comps[rng.randint(len(comps))]
                if abs(sw - w) <= max(4, w // 2) and abs(sh - h) <= max(4, h // 2):
                    break
            else:
                continue
            src = template[sy:sy + sh, sx:sx + sw]
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            edited[y:y + h, x:x + w] = src
            boxes.append([x, y, x + w, y + h])

        elif kind == "insert" and comps:
            sx, sy, sw, sh = comps[rng.randint(len(comps))]
            spot = blank_spot(ink, max(sw, sh), rng)
            if spot is None:
                continue
            x, y = spot
            patch = template[sy:sy + sh, sx:sx + sw]
            region = edited[y:y + sh, x:x + sw]
            if region.shape != patch.shape:
                continue
            edited[y:y + sh, x:x + sw] = np.minimum(region, patch)
            boxes.append([x, y, x + sw, y + sh])

        else:  # mark
            size = int(rng.choice(mark_sizes))
            spot = blank_spot(ink, size, rng)
            if spot is None:
                continue
            x, y = spot
            colour = int(rng.randint(0, 90))
            if rng.rand() < 0.7:
                cv2.rectangle(edited, (x, y), (x + size - 1, y + size - 1),
                              (colour, colour, colour), -1)
            else:
                cv2.circle(edited, (x + size // 2, y + size // 2), size // 2,
                           (colour, colour, colour), -1)
            boxes.append([x, y, x + size, y + size])

    return edited, np.asarray(boxes, np.float32) if boxes else np.zeros((0, 4), np.float32)


def shadow_field(shape: tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Smooth multiplicative illumination field with occasional dark patches."""
    h, w = shape
    small = rng.uniform(0.75, 1.05, (rng.randint(3, 7), rng.randint(3, 7))).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if rng.rand() < 0.35:
        mask = np.zeros((h, w), np.float32)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        axes = (rng.randint(w // 6, max(w // 6 + 1, w // 2)),
                rng.randint(h // 6, max(h // 6 + 1, h // 2)))
        cv2.ellipse(mask, (cx, cy), axes, rng.randint(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(w, h) / 30.0)
        field = field * (1.0 - mask * rng.uniform(0.15, 0.55))
    return np.clip(field, 0.2, 1.2)


def degrade(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.rand() < 0.5:                                   # sub-pixel warp
        dx, dy = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, m, (out.shape[1], out.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 2.2))
    out = out * shadow_field(out.shape[:2], rng)[..., None]
    gamma = rng.uniform(0.8, 1.25)
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    out = out + rng.normal(0, rng.uniform(1.5, 7.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    quality = int(rng.randint(55, 96))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


def _template_pool(source_splits: tuple[str, ...]) -> list[tuple[str, int, Path]]:
    """(split, page index, path) for every template we are allowed to synthesize from."""
    pool: list[tuple[str, int, Path]] = []
    for sp in source_splits:
        for path in sorted((DATA / sp / "template").glob("*.png")):
            idx = int(path.stem.split("_")[-1])
            pool.append((sp, idx, path))
    if not pool:
        raise FileNotFoundError(f"no templates under {DATA} for splits {source_splits}")
    return pool


def generate(n: int, seed: int = 0, out_root: Path | None = None,
             source_split: str = "train", max_side: int = 2400,
             source_splits: tuple[str, ...] | None = None,
             small_bias: float = 0.0) -> dict:
    """Write n synthetic pairs in prepared form.

    Returns {index: {"boxes": [...], "src_split": str, "src_idx": int}}. The source page is
    recorded so a fold can exclude synthetic pairs derived from its own validation templates —
    without that, a validation page's layout reaches the training set and the fold score is
    no longer honest. Test templates carry no labels, so synthesizing onto them leaks nothing
    and adds 100 layouts the model would otherwise never see.
    """
    out_root = Path(out_root or SYNTH) / "synth"
    out_root.mkdir(parents=True, exist_ok=True)
    pool = _template_pool(source_splits or (source_split,))

    boxes_by_index: dict[int, dict] = {}
    for i in range(n):
        rng = np.random.RandomState(seed * 100003 + i)
        src_split, src_idx, src_path = pool[rng.randint(len(pool))]
        src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if src is None:
            continue
        if max(src.shape[:2]) > max_side:
            scale = max_side / max(src.shape[:2])
            src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        edited, boxes = apply_edits(src, rng, small_bias=small_bias)
        if len(boxes) == 0:
            continue
        photo = degrade(edited, rng)
        template_matched, _ = match_blur(src, photo)

        index = seed * 1000000 + i
        d = out_root / f"{index:03d}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "t.png"), template_matched)
        cv2.imwrite(str(d / "p.png"), photo)
        cv2.imwrite(str(d / "tn.png"), local_norm(template_matched))
        cv2.imwrite(str(d / "pn.png"), local_norm(photo))
        boxes_by_index[index] = {"boxes": boxes.tolist(),
                                 "src_split": src_split, "src_idx": src_idx}

    return boxes_by_index


def merge_box_index(out_root: Path | None = None) -> int:
    """Merge per-shard box files into the single boxes.json the loader reads."""
    root = Path(out_root or SYNTH) / "synth"
    root.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list] = {}
    for shard in sorted(root.glob("boxes_shard_*.json")):
        merged.update(json.loads(shard.read_text()))
    (root / "boxes.json").write_text(json.dumps(merged))
    return len(merged)
