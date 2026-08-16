"""Stage 0: alignment check, blur matching, local background normalization.

The pairs in this dataset are already sub-pixel aligned (measured block phase
correlation residual, p95 <= 0.26 px), so alignment is a verification step with
an ECC fallback rather than a real registration stage.

The dominant false-positive source is that the photo is a blurred, noisy render
of the same content: every glyph edge differs. Matching the template's blur to
the photo before differencing removes most of it. Local background subtraction
removes the shadow and tone shifts.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ALIGN_MIN_RESPONSE, BG_SIGMA, BLUR_SIGMAS, DATA, PREP, pair_names


def read_pair(split: str, idx: int, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = root or DATA
    t_name, p_name = pair_names(split, idx)
    sub = root / split
    t = cv2.imread(str(sub / t_name), cv2.IMREAD_COLOR)
    p = cv2.imread(str(sub / p_name), cv2.IMREAD_COLOR)
    if t is None or p is None:
        raise FileNotFoundError(f"{sub / t_name} or {sub / p_name}")
    return t, p


def align(template: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, dict]:
    """Warp the photo onto the template frame if needed. Usually a no-op."""
    tg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pg = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h = min(tg.shape[0], pg.shape[0])
    w = min(tg.shape[1], pg.shape[1])
    (dx, dy), response = cv2.phaseCorrelate(tg[:h, :w].copy(), pg[:h, :w].copy())
    info = {"dx": float(dx), "dy": float(dy), "response": float(response), "mode": "none"}

    if abs(dx) < 0.5 and abs(dy) < 0.5 and response >= ALIGN_MIN_RESPONSE:
        return photo, info

    if abs(dx) < 8 and abs(dy) < 8:
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        info["mode"] = "translate"
        return cv2.warpAffine(photo, m, (template.shape[1], template.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), info

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        _, warp = cv2.findTransformECC(tg, pg, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        info["mode"] = "ecc"
    except cv2.error:
        info["mode"] = "ecc_failed"
        return photo, info
    return cv2.warpAffine(photo, warp, (template.shape[1], template.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE), info


def match_blur(template: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, float]:
    """Blur the template so its high-frequency energy matches the photo's."""
    tg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    target = cv2.Laplacian(pg, cv2.CV_64F).var()
    best_sigma, best_err, best_img = 0.0, None, template
    for sigma in BLUR_SIGMAS:
        blurred = template if sigma <= 0 else cv2.GaussianBlur(template, (0, 0), sigma)
        v = cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        err = abs(v - target)
        if best_err is None or err < best_err:
            best_sigma, best_err, best_img = sigma, err, blurred
    return best_img, best_sigma


def local_norm(img: np.ndarray, sigma: float = BG_SIGMA) -> np.ndarray:
    """Ink map: how much darker than the local background, as uint8."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), sigma)
    ink = np.clip(bg - gray, 0, 255)
    return ink.astype(np.uint8)


def preprocess_pair(split: str, idx: int, root: Path | None = None, out: Path | None = None) -> dict:
    """Write blur-matched template, aligned photo and both ink maps to PREP."""
    out = out or PREP
    template, photo = read_pair(split, idx, root)
    photo, info = align(template, photo)
    template_matched, sigma = match_blur(template, photo)
    info["blur_sigma"] = sigma

    d = out / split / f"{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "t.png"), template_matched)
    cv2.imwrite(str(d / "p.png"), photo)
    cv2.imwrite(str(d / "tn.png"), local_norm(template_matched))
    cv2.imwrite(str(d / "pn.png"), local_norm(photo))
    info.update({"split": split, "idx": idx, "h": template.shape[0], "w": template.shape[1]})
    return info


def load_prepared(split: str, idx: int, out: Path | None = None):
    d = (out or PREP) / split / f"{idx:03d}"
    t = cv2.imread(str(d / "t.png"), cv2.IMREAD_COLOR)
    p = cv2.imread(str(d / "p.png"), cv2.IMREAD_COLOR)
    tn = cv2.imread(str(d / "tn.png"), cv2.IMREAD_GRAYSCALE)
    pn = cv2.imread(str(d / "pn.png"), cv2.IMREAD_GRAYSCALE)
    if t is None:
        raise FileNotFoundError(str(d))
    return t, p, tn, pn


def stream_tensors(t: np.ndarray, p: np.ndarray, tn: np.ndarray, pn: np.ndarray):
    """Two 4-channel streams (BGR + ink map), float32 in [0,1], CHW."""
    a = np.concatenate([t.astype(np.float32) / 255.0, (tn.astype(np.float32) / 255.0)[..., None]], -1)
    b = np.concatenate([p.astype(np.float32) / 255.0, (pn.astype(np.float32) / 255.0)[..., None]], -1)
    return a.transpose(2, 0, 1), b.transpose(2, 0, 1)
