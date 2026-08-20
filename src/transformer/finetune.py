"""Supervised training of PacketFormer on the competition flows.

Produces exactly the same artifact contract as every other model in the search:
an .npz in the out-of-fold directory holding `oof_group`, `oof_plain`, an
optional `test` matrix and a JSON meta blob. That is what lets the transformer
join the ensemble without any special-casing.

Optionally starts from an encoder pretrained on the auxiliary MIRAGE corpus;
the pretrained and from-scratch variants are scored identically so the transfer
can be accepted or rejected on evidence.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss

import config as C
import io_utils as IO
from models import cv as cvmod
from transformer.dataset_rtc import FlowDataset, frame_to_arrays
from transformer.model import PacketFormer, cosine_schedule

N_CLASSES = len(C.LABELS)

DEFAULT_CFG = {
    "d_model": 96,
    "n_layers": 3,
    "n_heads": 4,
    "dim_ff": 192,
    "dropout": 0.15,
    "epochs": 140,
    "batch_size": 64,
    "lr": 1.5e-3,
    "weight_decay": 0.05,
    "label_smoothing": 0.05,
    "mixup_alpha": 0.4,
    "mixup_prob": 0.5,
    "swa_last": 5,
    "class_weighted": False,
    "augment": True,
    "seed": C.SEED,
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_model(cfg: dict, pretrained: str | None) -> PacketFormer:
    model = PacketFormer(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg["dropout"],
        n_classes=N_CLASSES,
    )
    if pretrained:
        state = torch.load(pretrained, map_location="cpu")
        encoder_state = {
            k: v for k, v in state["model"].items() if not k.startswith("head.")
        }
        missing = model.load_state_dict(encoder_state, strict=False)
        print(f"loaded pretrained encoder ({len(encoder_state)} tensors); {missing}")
    return model


def train_one(
    cfg: dict,
    L_tr: np.ndarray,
    T_tr: np.ndarray,
    y_tr: np.ndarray,
    L_va: np.ndarray,
    T_va: np.ndarray,
    pretrained: str | None = None,
) -> np.ndarray:
    torch.manual_seed(cfg["seed"])
    device = _device()
    model = _make_model(cfg, pretrained).to(device)

    ds = FlowDataset(L_tr, T_tr, y_tr, augment=cfg["augment"], seed=cfg["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False
    )
    va_ds = FlowDataset(L_va, T_va, augment=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=512, shuffle=False)

    weight = None
    if cfg["class_weighted"]:
        counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        weight = torch.tensor(
            len(y_tr) / (N_CLASSES * counts), dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(
        weight=weight, label_smoothing=cfg["label_smoothing"]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = cfg["epochs"] * max(1, len(loader))
    step = 0
    rng = np.random.RandomState(cfg["seed"])
    snapshots: list[np.ndarray] = []

    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in loader:
            cont = batch["cont"].to(device)
            bins = batch["bins"].to(device)
            pad = batch["pad"].to(device)
            y = batch["y"].to(device)

            lam, perm = None, None
            if cfg["mixup_alpha"] > 0 and rng.rand() < cfg["mixup_prob"]:
                lam = float(rng.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
                perm = torch.randperm(len(y), device=device)

            logits = model(cont, bins, pad, mixup_lam=lam, mixup_perm=perm)
            if lam is None:
                loss = criterion(logits, y)
            else:
                loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[perm])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * cosine_schedule(step, total_steps, warmup=total_steps // 20)
            opt.step()
            step += 1

        # Averaging the predictions of the final epochs is a cheap, variance
        # reducing substitute for a proper snapshot ensemble.
        if epoch >= cfg["epochs"] - cfg["swa_last"]:
            snapshots.append(_predict(model, va_loader, device))

    return np.mean(snapshots, axis=0)


@torch.no_grad()
def _predict(model: PacketFormer, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        logits = model(
            batch["cont"].to(device), batch["bins"].to(device), batch["pad"].to(device)
        )
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run_cv(
    cfg: dict | None = None,
    pretrained: str | None = None,
    repeats: dict | None = None,
    with_test: bool = True,
    tag: str = "tfm",
) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    repeats = repeats or {"group": 2, "plain": 1}
    train, test = IO.load_train(), IO.load_test()
    y = IO.labels_to_ids(train["label"])
    L, T = frame_to_arrays(train)
    L_te, T_te = frame_to_arrays(test)
    folds = cvmod.load_folds()

    result = {
        "model": tag,
        "hash": f"{tag}_{abs(hash(json.dumps({**cfg, 'pretrained': pretrained}, sort_keys=True))) % (10**10)}",
        "features": "sequence",
        "target_feats": False,
        "spec": {"params": {k: v for k, v in cfg.items() if k != "seed"},
                 "pretrained": bool(pretrained), "model": tag},
        "pretrained": bool(pretrained),
    }
    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()

    for scheme, n_rep in repeats.items():
        splits = folds[scheme][: n_rep * C.N_SPLITS]
        probs = np.zeros((n_rep, len(y), N_CLASSES), dtype=np.float32)
        for i, (tr_idx, va_idx) in enumerate(splits):
            rep = i // C.N_SPLITS
            probs[rep, va_idx] = train_one(
                cfg, L[tr_idx], T[tr_idx], y[tr_idx], L[va_idx], T[va_idx], pretrained
            )
            print(f"[{scheme}] fold {i + 1}/{len(splits)} done", flush=True)
        accs = [accuracy_score(y, p.argmax(axis=1)) for p in probs]
        f1s = [f1_score(y, p.argmax(axis=1), average="macro") for p in probs]
        mean_p = probs.mean(axis=0)
        result[f"{scheme}_acc"] = float(np.mean(accs))
        result[f"{scheme}_acc_std"] = float(np.std(accs))
        result[f"{scheme}_macro_f1"] = float(np.mean(f1s))
        result[f"{scheme}_logloss"] = float(
            log_loss(y, np.clip(mean_p, 1e-9, 1), labels=list(range(N_CLASSES)))
        )
        arrays[f"oof_{scheme}"] = mean_p

    if with_test:
        # Bagged over several seeds because a single small transformer trained
        # on 1,285 rows is noticeably seed-sensitive.
        preds = []
        for s in range(3):
            preds.append(train_one({**cfg, "seed": cfg["seed"] + s}, L, T, y, L_te, T_te, pretrained))
        arrays["test"] = np.mean(preds, axis=0).astype(np.float32)

    result["fit_seconds"] = round(time.time() - t0, 2)
    C.OOF_DIR.mkdir(parents=True, exist_ok=True)
    path = C.OOF_DIR / f"{tag}_{result['hash']}.npz"
    np.savez_compressed(path, meta=json.dumps(result), **arrays)
    result["oof_path"] = str(path)
    return result
