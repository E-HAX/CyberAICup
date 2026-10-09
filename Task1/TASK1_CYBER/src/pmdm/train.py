"""Stage-1 training loop."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import BATCH, CKPT, EPOCHS, LR, PREP, SYNTH, WEIGHT_DECAY
from .dataset import GroupedSampler, PairTileDataset, load_gt
from .folds import split as fold_split
from .losses import total_loss
from .metric import sweep_threshold
from .model import SiamCenterNet


def build_gt(items: list[tuple[str, int]], synth_gt: dict | None = None) -> dict:
    """Ground truth keyed by (split, idx) for both real and synthetic pairs."""
    real = load_gt()
    gt: dict = {}
    for split, idx in items:
        if split == "synth":
            gt[(split, idx)] = synth_gt[idx] if synth_gt else np.zeros((0, 4), np.float32)
        else:
            gt[(split, idx)] = real.get(idx, np.zeros((0, 4), np.float32))
    return gt


def load_synth_gt(root: Path | None = None) -> tuple[dict, dict]:
    """Returns (boxes by synth index, source page by synth index).

    Two on-disk formats are accepted. The current one records the template each pair was
    synthesized from, which is what lets a fold exclude pairs built from its own validation
    pages. The older list-of-boxes format has no source, so those pairs report source None
    and are treated as unsafe — see `_fold_safe_synth`.
    """
    root = root or SYNTH
    path = root / "synth" / "boxes.json"
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text())
    boxes, source = {}, {}
    for k, v in raw.items():
        i = int(k)
        if isinstance(v, dict):
            boxes[i] = np.asarray(v["boxes"], np.float32)
            source[i] = (v.get("src_split"), v.get("src_idx"))
        else:
            boxes[i] = np.asarray(v, np.float32)
            source[i] = (None, None)
    return boxes, source


def _fold_safe_synth(indices: list[int], source: dict, val_idx: list[int]) -> list[int]:
    """Drop synthetic pairs derived from this fold's validation templates.

    A synthetic pair carries invented boxes, but it carries the *real page layout* it was
    built from. Training on one built from a validation template shows the model that page
    and inflates the fold score. Pairs from the test split are always safe (no labels exist
    there); pairs with no recorded source are dropped rather than trusted.
    """
    val = set(val_idx)
    keep = []
    for i in indices:
        sp, sidx = source.get(i, (None, None))
        if sp == "test":
            keep.append(i)
        elif sp == "train" and sidx is not None and sidx not in val:
            keep.append(i)
    return keep


def evaluate(model, val_idx: list[int], device: str, gt: dict, **kwargs) -> dict:
    from .infer import predict_pair

    model.eval()
    candidates, gt_eval = {}, {}
    for idx in val_idx:
        boxes, scores = predict_pair(model, "train", idx, device=device, **kwargs)
        key = f"train_{idx:03d}"
        candidates[key] = (boxes, scores)
        gt_eval[key] = gt[("train", idx)]
    thr, best = sweep_threshold(candidates, gt_eval)
    best["threshold"] = thr
    model.train()
    return best


def train_fold(fold: int, epochs: int = EPOCHS, backbone: str = "convnext_tiny",
               batch: int = BATCH, lr: float = LR, use_synth: bool = True,
               n_synth: int = 0, samples_per_pair: int = 8, device: str = "cuda",
               out_dir: Path | None = None, num_workers: int = 8,
               eval_every: int = 5, resume: bool = True, on_epoch_end=None,
               use_ema: bool = True, ema_decay: float = 0.999,
               tag: str = "", real_repeat: int = 1, wh_relative: bool = True) -> dict:
    out_dir = Path(out_dir or CKPT) / f"stage1_fold{fold}_{backbone}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_idx, val_idx = fold_split(fold)

    # Each entry in `items` draws `samples_per_pair` tiles per epoch, so a large synthetic
    # corpus would otherwise stretch an epoch by 40x and drown the 160 real pages. Repeating
    # the real entries keeps their share of the epoch up while the synthetic pairs are each
    # visited less often — which is what we want, since there are thousands of them and they
    # are the approximate half of the distribution.
    items = [("train", i) for i in train_idx] * max(1, real_repeat)
    synth_gt, synth_src = load_synth_gt() if use_synth else ({}, {})
    if use_synth and synth_gt:
        safe = _fold_safe_synth(sorted(synth_gt), synth_src, val_idx)
        dropped = len(synth_gt) - len(safe)
        chosen = safe if n_synth <= 0 else safe[:n_synth]
        print(f"[fold {fold}] synth available {len(synth_gt)}, fold-safe {len(safe)} "
              f"(dropped {dropped} from validation templates), using {len(chosen)}", flush=True)
        items += [("synth", i) for i in chosen]
    n_real = len(train_idx) * max(1, real_repeat)
    print(f"[fold {fold}] items: {n_real} real entries "
          f"({len(train_idx)} pages x{max(1, real_repeat)}), {len(items) - n_real} synthetic",
          flush=True)
    gt = build_gt(items + [("train", i) for i in val_idx], synth_gt)

    ds = PairTileDataset(items, gt, samples_per_pair=samples_per_pair, prep_root=PREP)
    sampler = GroupedSampler(len(items), samples_per_pair)
    dl = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=num_workers > 0)

    model = SiamCenterNet(backbone=backbone, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled="cuda" in device)

    # EMA weights. The heatmap head is the noisy part of this model — the current failure mode
    # is 14 correct boxes scoring just under the global threshold — and an averaged model gives
    # noticeably steadier scores than any single step's weights.
    ema = None
    if use_ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
                            use_buffers=True)

    start_epoch, history = 0, []
    ckpt_path = out_dir / "last.pt"
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        history = state.get("history", [])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        print(f"resumed from epoch {start_epoch}", flush=True)

    # The wh head's loss form is an experiment variable, so it is threaded through rather
    # than hard-coded: relative L1 puts small and large boxes on equal footing, pixel L1 is
    # what produced the 0.905 reference.
    loss_kwargs = {"wh_relative": wh_relative} if wh_relative else {"wh_relative": False,
                                                                    "w_wh": 0.2}
    best_f1 = max([h.get("f1", 0.0) for h in history], default=0.0)
    for epoch in range(start_epoch, epochs):
        model.train()
        sampler.set_epoch(epoch)
        t0, running = time.time(), {}
        for step, batch_data in enumerate(dl):
            a = batch_data["a"].to(device, non_blocking=True)
            b = batch_data["b"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch_data.items()
                       if k not in ("a", "b")}
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                dtype=torch.bfloat16, enabled="cuda" in device):
                out = model(a, b)
                losses = total_loss(out, targets, **loss_kwargs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if step % 50 == 0:
                msg = " ".join(f"{k}={running[k] / (step + 1):.3f}" for k in sorted(running))
                print(f"[fold {fold}] epoch {epoch} step {step}/{len(dl)} {msg}", flush=True)

        record = {"epoch": epoch, "seconds": round(time.time() - t0, 1),
                  **{k: running[k] / max(1, len(dl)) for k in running}}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            record.update(evaluate(model, val_idx, device, gt))
            best_state, best_which = model.state_dict(), "raw"
            if ema is not None:
                ema_record = evaluate(ema.module, val_idx, device, gt)
                record["ema"] = ema_record
                if ema_record["f1"] > record["f1"]:
                    best_state, best_which = ema.module.state_dict(), "ema"
                    record = {**record, **{k: v for k, v in ema_record.items() if k != "ema"}}
            record["selected"] = best_which
            print(f"[fold {fold}] epoch {epoch} val {record}", flush=True)
            if record.get("f1", 0.0) >= best_f1:
                best_f1 = record["f1"]
                torch.save({"model": best_state, "epoch": epoch, "metric": record},
                           out_dir / "best.pt")
        history.append(record)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "history": history,
                    "ema": ema.state_dict() if ema is not None else None},
                   ckpt_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if on_epoch_end is not None:
            on_epoch_end()   # commit the volume so a killed run resumes from here

    return {"fold": fold, "best_f1": best_f1, "history": history[-1] if history else {}}
