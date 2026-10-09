"""Stage 2: patch verifier and box refiner.

Stage 1 is tuned for recall; most of its output is print-artefact noise. This
model looks at a single candidate at high resolution and answers two questions:
is this a real content difference, and where exactly are its corners. The second
answer is what converts a 0.4-IoU candidate into a true positive at the 0.5
threshold, which matters because the median box is 22 px wide.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import CKPT, PREP
from .metric import iou_matrix
from .preprocess import load_prepared

CROP = 96
CONTEXT = 2.5          # crop side = CONTEXT * max(box side), clamped
POS_IOU = 0.35
NEG_IOU = 0.20


def crop_window(box: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = float(np.clip(max(x2 - x1, y2 - y1) * CONTEXT, 32, 256))
    x0 = int(round(np.clip(cx - side / 2, 0, max(0, w - side))))
    y0 = int(round(np.clip(cy - side / 2, 0, max(0, h - side))))
    return x0, y0, int(round(side)), int(round(side))


def make_crop(t, p, tn, pn, box: np.ndarray):
    """8-channel crop plus the box in normalized crop coordinates."""
    import cv2

    h, w = t.shape[:2]
    x0, y0, side, _ = crop_window(box, h, w)
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    stack = np.concatenate([t[sl], p[sl], tn[sl][..., None], pn[sl][..., None]], -1)
    stack = cv2.resize(stack, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    scale = CROP / side
    local = np.array([(box[0] - x0) * scale, (box[1] - y0) * scale,
                      (box[2] - x0) * scale, (box[3] - y0) * scale], np.float32)
    return stack.astype(np.float32).transpose(2, 0, 1) / 255.0, local, (x0, y0, side)


class Verifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=8,
                                          num_classes=0)
        feat = self.backbone.num_features
        self.head_cls = nn.Linear(feat + 4, 1)
        self.head_box = nn.Linear(feat + 4, 4)

    def forward(self, x: torch.Tensor, box: torch.Tensor):
        f = torch.cat([self.backbone(x), box / CROP], 1)
        return self.head_cls(f).squeeze(1), self.head_box(f)


class CandidateDataset(Dataset):
    """Candidates produced by stage 1, labelled against ground truth."""

    def __init__(self, records: list[dict], prep_root: Path | None = None, jitter: bool = True):
        self.records = records
        self.prep_root = prep_root or PREP
        self.jitter = jitter
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.records)

    def _pair(self, split: str, idx: int):
        key = (split, idx)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = load_prepared(split, idx, self.prep_root)
        return self._cache[key]

    def __getitem__(self, i: int):
        r = self.records[i]
        t, p, tn, pn = self._pair(r["split"], r["idx"])
        box = np.asarray(r["box"], np.float32)
        if self.jitter:
            box = box + np.random.uniform(-2, 2, 4).astype(np.float32)
        stack, local, _ = make_crop(t, p, tn, pn, box)
        target = np.zeros(4, np.float32)
        if r["label"] > 0:
            gt = np.asarray(r["gt"], np.float32)
            x0, y0, side, _ = crop_window(box, t.shape[0], t.shape[1])
            scale = CROP / side
            gt_local = np.array([(gt[0] - x0) * scale, (gt[1] - y0) * scale,
                                 (gt[2] - x0) * scale, (gt[3] - y0) * scale], np.float32)
            target = gt_local - local
        return {
            "x": torch.from_numpy(stack),
            "box": torch.from_numpy(local),
            "label": torch.tensor(float(r["label"])),
            "delta": torch.from_numpy(target),
        }


def build_records(candidates: dict, gt: dict, max_neg_per_pos: int = 8,
                  seed: int = 0) -> list[dict]:
    """candidates: key -> (split, idx, boxes, scores); gt: key -> boxes.

    Negatives outnumber positives ~26:1 in a generous candidate set, which costs a lot of
    epoch time for very little extra signal — `pos_weight` in the loss already rebalances the
    gradient. Capping the ratio per image keeps the hard negatives near real content (they are
    the ones the detector actually proposed) while cutting the record count several-fold.
    """
    records = []
    for key, (split, idx, boxes, _scores) in candidates.items():
        g = gt.get(key, np.zeros((0, 4), np.float32))
        ious = iou_matrix(boxes, g) if len(boxes) and len(g) else np.zeros((len(boxes), len(g)))
        negatives, neg_index, n_pos_here = [], [], 0
        for i in range(len(boxes)):
            best = float(ious[i].max()) if ious.size else 0.0
            j = int(ious[i].argmax()) if ious.size else -1
            if best >= POS_IOU:
                n_pos_here += 1
                records.append({"split": split, "idx": idx, "box": boxes[i].tolist(),
                                "label": 1, "gt": g[j].tolist()})
            elif best < NEG_IOU:
                negatives.append({"split": split, "idx": idx, "box": boxes[i].tolist(),
                                  "label": 0, "gt": boxes[i].tolist()})
                neg_index.append(i)
        keep = max_neg_per_pos * max(1, n_pos_here)
        if len(negatives) > keep:
            # highest-scoring negatives first: those are the ones the detector was most
            # confident about, and the only ones that survive the global threshold anyway
            order = np.argsort(-np.asarray([_scores[i] for i in neg_index], np.float32))
            negatives = [negatives[j] for j in order[:keep]]
        records += negatives
    return records


class PrecomputedCrops(Dataset):
    """Cuts every candidate's crop once, in page order, and serves them from memory."""

    def __init__(self, records: list[dict], prep_root: Path | None = None):
        import time

        root = prep_root or PREP
        order = sorted(range(len(records)), key=lambda i: (records[i]["split"],
                                                           records[i]["idx"]))
        n = len(records)
        self.x = np.zeros((n, 8, CROP, CROP), np.uint8)
        self.box = np.zeros((n, 4), np.float32)
        self.label = np.zeros(n, np.float32)
        self.delta = np.zeros((n, 4), np.float32)

        t0 = time.time()
        cache_key, cache_val = None, None
        for slot, i in enumerate(order):
            r = records[i]
            key = (r["split"], r["idx"])
            if key != cache_key:
                cache_val = load_prepared(r["split"], r["idx"], root)
                cache_key = key
            t, p, tn, pn = cache_val
            box = np.asarray(r["box"], np.float32)
            stack, local, _ = make_crop(t, p, tn, pn, box)
            self.x[slot] = np.clip(stack * 255.0, 0, 255).astype(np.uint8)
            self.box[slot] = local
            self.label[slot] = float(r["label"])
            if r["label"] > 0:
                gt = np.asarray(r["gt"], np.float32)
                x0, y0, side, _ = crop_window(box, t.shape[0], t.shape[1])
                scale = CROP / side
                gt_local = np.array([(gt[0] - x0) * scale, (gt[1] - y0) * scale,
                                     (gt[2] - x0) * scale, (gt[3] - y0) * scale], np.float32)
                self.delta[slot] = gt_local - local
        print(f"[verifier] cut {n} crops in {time.time() - t0:.0f}s", flush=True)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, i: int):
        return {
            "x": torch.from_numpy(self.x[i].astype(np.float32) / 255.0),
            "box": torch.from_numpy(self.box[i]),
            "label": torch.tensor(self.label[i]),
            "delta": torch.from_numpy(self.delta[i]),
        }


def train_verifier(records: list[dict], epochs: int = 8, batch: int = 64, lr: float = 3e-4,
                   device: str = "cuda", out_dir: Path | None = None,
                   prep_root: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir is not None else Path(CKPT) / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(json.dumps(records[:50000]))
    print(f"[verifier] {len(records)} records, "
          f"{sum(r['label'] for r in records)} positive", flush=True)

    # Crops are cut once, up front, in page order, then training reads them from memory.
    # The previous arrangement decoded four full-page PNGs per sample on every cache miss, and
    # shuffling tens of thousands of records across 160 pages meant the cache almost never hit
    # — an epoch took ~60 minutes instead of ~3. A 96x96x8 uint8 crop is 74 KB, so even 20k
    # records fit comfortably in the container's memory.
    ds = PrecomputedCrops(records, prep_root=prep_root)
    batch = min(batch, max(1, len(records)))
    drop_last = len(records) >= 2 * batch      # never leave the loader empty
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, drop_last=drop_last)
    if len(dl) == 0:
        raise ValueError(f"verifier loader is empty for {len(records)} records")
    model = Verifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl),
                                                pct_start=0.1)
    n_pos = sum(r["label"] for r in records)
    pos_weight = torch.tensor([max(1.0, (len(records) - n_pos) / max(1, n_pos))]).to(device)

    for epoch in range(epochs):
        running = 0.0
        for step, b in enumerate(dl):
            x = b["x"].to(device)
            box = b["box"].to(device)
            logit, delta = model(x, box)
            loss_cls = F.binary_cross_entropy_with_logits(logit, b["label"].to(device),
                                                          pos_weight=pos_weight)
            mask = b["label"].to(device) > 0
            loss_box = (F.smooth_l1_loss(delta[mask], b["delta"].to(device)[mask])
                        if mask.any() else delta.sum() * 0)
            loss = loss_cls + 0.5 * loss_box
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach())
            if step % 100 == 0:
                print(f"[verifier] epoch {epoch} step {step}/{len(dl)} "
                      f"loss={running / (step + 1):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")
    return {"records": len(records), "positives": int(n_pos)}


@torch.no_grad()
def apply_verifier(model, split: str, idx: int, boxes: np.ndarray, scores: np.ndarray,
                   device: str = "cuda", prep_root: Path | None = None, batch: int = 128,
                   refine: bool = True, blend: float = 0.5):
    if len(boxes) == 0:
        return boxes, scores
    t, p, tn, pn = load_prepared(split, idx, prep_root or PREP)
    crops, locals_, windows = [], [], []
    for box in boxes:
        stack, local, win = make_crop(t, p, tn, pn, box)
        crops.append(stack)
        locals_.append(local)
        windows.append(win)

    out_scores, out_boxes = [], []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(np.stack(crops[i:i + batch])).to(device)
        b = torch.from_numpy(np.stack(locals_[i:i + batch])).to(device)
        logit, delta = model(x, b)
        probs = torch.sigmoid(logit).cpu().numpy()
        new_local = (b + delta).cpu().numpy()
        for k, prob in enumerate(probs):
            x0, y0, side = windows[i + k]
            scale = side / CROP
            nb = new_local[k] * scale + np.array([x0, y0, x0, y0], np.float32)
            out_boxes.append(nb if refine else boxes[i + k])
            out_scores.append(prob)

    fused = (np.asarray(out_scores, np.float32) ** blend) * (scores ** (1 - blend))
    return np.asarray(out_boxes, np.float32), fused


def load_verifier(path: Path, device: str = "cuda"):
    model = Verifier(pretrained=False)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()
