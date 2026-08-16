"""Reproduce every dataset claim made in HANDOVER.md.

    PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
    .venv/bin/python scripts/analyze_dataset.py

Runs in about two minutes on a laptop. Prints alignment residuals, box size
distribution, change polarity, blur mismatch and the classical baseline floor.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pmdm import config  # noqa: E402
from pmdm.dataset import load_gt  # noqa: E402
from pmdm.metric import match_image  # noqa: E402

SAMPLE_ALIGN = 10       # pairs used for the block phase-correlation probe
SAMPLE_BASELINE = 40    # pairs used for the classical baseline floor


def pair_paths(idx: int):
    t = config.TRAIN_DIR / "template" / f"train_template_{idx:03d}.png"
    p = config.TRAIN_DIR / "photo" / f"train_photo_{idx:03d}.png"
    return t, p


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    gt = load_gt()
    n_pairs = len(gt)

    section("1. Size and annotation counts")
    sizes = collections.Counter()
    same_size = 0
    for i in range(config.N_TRAIN):
        t, p = pair_paths(i)
        ti = cv2.imread(str(t), cv2.IMREAD_COLOR)
        pi = cv2.imread(str(p), cv2.IMREAD_COLOR)
        sizes[ti.shape[:2]] += 1
        same_size += int(ti.shape == pi.shape)
    counts = sorted(len(v) for v in gt.values())
    print(f"annotated pairs: {n_pairs}, boxes: {sum(len(v) for v in gt.values())}")
    print(f"boxes per image: min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}")
    print(f"template and photo same size: {same_size}/{config.N_TRAIN}")
    print(f"distinct sizes: {len(sizes)}; most common: {sizes.most_common(4)}")

    section("2. Alignment (is registration needed?)")
    for i in range(0, config.N_TRAIN, config.N_TRAIN // SAMPLE_ALIGN):
        t, p = pair_paths(i)
        T = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        P = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = T.shape
        bs, offs = 256, []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                a, b = T[y:y + bs, x:x + bs], P[y:y + bs, x:x + bs]
                if a.std() < 5:
                    continue
                (dx, dy), _ = cv2.phaseCorrelate(a.copy(), b.copy())
                offs.append((abs(dx), abs(dy)))
        if offs:
            o = np.array(offs)
            print(f"pair {i:3d} {T.shape} blocks={len(o):3d} "
                  f"|dx| p50/p95 {np.percentile(o[:, 0], [50, 95]).round(2)} "
                  f"|dy| p50/p95 {np.percentile(o[:, 1], [50, 95]).round(2)}")

    section("3. Box size distribution")
    ws, hs, rel = [], [], []
    for idx, boxes in gt.items():
        t, _ = pair_paths(idx)
        H, W = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE).shape
        for x1, y1, x2, y2 in boxes:
            ws.append(x2 - x1)
            hs.append(y2 - y1)
            rel.append(((x2 - x1) / W, (y2 - y1) / H))
    ws, hs, rel = np.array(ws), np.array(hs), np.array(rel)
    print(f"width  percentiles [1,25,50,75,99]: {np.percentile(ws, [1, 25, 50, 75, 99]).astype(int)}")
    print(f"height percentiles [1,25,50,75,99]: {np.percentile(hs, [1, 25, 50, 75, 99]).astype(int)}")
    print(f"relative width median: {np.median(rel[:, 0]) * 100:.2f}% of image width")
    print(f"boxes with both sides < 16 px: {int(((ws < 16) & (hs < 16)).sum())} / {len(ws)}")
    print(f"boxes exactly 8x8: {int(((ws == 8) & (hs == 8)).sum())}")

    section("4. Change polarity (additions vs deletions)")
    t_ink, p_ink = [], []
    for idx, boxes in gt.items():
        t, p = pair_paths(idx)
        T = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        P = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        nT = np.clip(cv2.GaussianBlur(T, (0, 0), 31) - T, 0, None)
        nP = np.clip(cv2.GaussianBlur(P, (0, 0), 31) - P, 0, None)
        for x1, y1, x2, y2 in boxes.astype(int):
            a, b = nT[y1:y2, x1:x2], nP[y1:y2, x1:x2]
            if a.size:
                t_ink.append(a.mean())
                p_ink.append(b.mean())
    t_ink, p_ink = np.array(t_ink), np.array(p_ink)
    print(f"boxes analysed: {len(t_ink)}")
    print(f"addition   (template blank, photo inked): {int(((t_ink < 3) & (p_ink > 3)).sum())}")
    print(f"deletion   (photo blank, template inked): {int(((p_ink < 3) & (t_ink > 3)).sum())}")
    print(f"modification (both inked):                {int(((t_ink >= 3) & (p_ink >= 3)).sum())}")
    print(f"fraction with more ink in photo: {float((p_ink > t_ink).mean()):.3f}")

    section("5. Blur and shadow mismatch")
    shadow = 0
    for i in range(0, config.N_TRAIN, 4):
        t, p = pair_paths(i)
        T = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE)
        P = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        shadow += int(P.mean() < T.mean() - 20)
        if i < 12:
            print(f"pair {i:3d} Laplacian variance template {cv2.Laplacian(T, cv2.CV_64F).var():8.1f} "
                  f"photo {cv2.Laplacian(P, cv2.CV_64F).var():8.1f}")
    print(f"shadow-heavy pairs (photo mean 20+ darker): {shadow} / {config.N_TRAIN // 4} sampled")

    section("6. Classical baseline floor")
    tp = fp = fn = 0
    for idx in sorted(gt)[:SAMPLE_BASELINE]:
        t, p = pair_paths(idx)
        T = cv2.imread(str(t), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        P = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        nT = T / (cv2.GaussianBlur(T, (0, 0), 25) + 1)
        nP = P / (cv2.GaussianBlur(P, (0, 0), 25) + 1)
        mask = (np.abs(nT - nP) > 0.18).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        preds = np.array([[x, y, x + w, y + h] for x, y, w, h, a in stats[1:] if a >= 12],
                         np.float32).reshape(-1, 4)
        a, b, c = match_image(preds, gt[idx])
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(f"{SAMPLE_BASELINE} pairs: TP {tp} FP {fp} FN {fn}")
    print(f"precision {precision:.4f} recall {recall:.4f} F1 {f1:.4f}")


if __name__ == "__main__":
    main()
