"""Modal app for Packaging Material Difference Mining.

One-time data upload (2.5 GB, from the repository root):

    modal volume create pmdm-data
    modal volume put pmdm-data \
        Task1/PackagingMaterialDifferenceMiningDataset/train /raw/train
    modal volume put pmdm-data \
        Task1/PackagingMaterialDifferenceMiningDataset/test /raw/test

Pipeline:

    modal run modal_app.py::prep                       # stage 0, CPU
    modal run modal_app.py::smoke                      # cheap shape/loss check
    modal run modal_app.py::synth --shards 16 --per-shard 400
    modal run modal_app.py::train --fold 0 --epochs 40
    modal run modal_app.py::oof --fold 0
    modal run modal_app.py::verify
    modal run modal_app.py::predict --folds 0,1,2,3,4
"""
from __future__ import annotations

import json

import modal

APP_NAME = "pmdm"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm==1.0.11",
        "opencv-python-headless==4.10.0.84",
        "numpy<2",
        "pandas",
    )
    .env(
        {
            "PMDM_DATA": "/data/raw",
            "PMDM_WORK": "/data/work",
            "PMDM_CKPT": "/ckpt",
            "PYTHONPATH": "/root/src",
            "HF_HOME": "/ckpt/hf",
        }
    )
    .add_local_dir("src", "/root/src")
)

data_vol = modal.Volume.from_name("pmdm-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("pmdm-ckpt", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/ckpt": ckpt_vol}

app = modal.App(APP_NAME, image=image)

HOUR = 3600


# --------------------------------------------------------------------------- #
# Stage 0: preprocessing
# --------------------------------------------------------------------------- #
@app.function(volumes=VOLUMES, timeout=2 * HOUR, cpu=4, memory=8192)
def preprocess_one(args: tuple[str, int]) -> dict:
    from pmdm.preprocess import preprocess_pair

    split, idx = args
    info = preprocess_pair(split, idx)
    data_vol.commit()
    return info


@app.function(volumes=VOLUMES, timeout=4 * HOUR)
def preprocess_all() -> dict:
    from pmdm.config import N_TEST, N_TRAIN, WORK

    jobs = [("train", i) for i in range(N_TRAIN)] + [("test", i) for i in range(N_TEST)]
    infos = list(preprocess_one.map(jobs, order_outputs=False))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "prep_info.json").write_text(json.dumps(infos, indent=2))
    data_vol.commit()
    modes: dict[str, int] = {}
    for i in infos:
        modes[i["mode"]] = modes.get(i["mode"], 0) + 1
    return {"pairs": len(infos), "align_modes": modes}


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
@app.function(volumes=VOLUMES, timeout=6 * HOUR, cpu=4, memory=8192)
def synth_shard(args: tuple) -> int:
    from pathlib import Path

    from pmdm.config import SYNTH
    from pmdm.synth import generate

    seed, n = args[0], args[1]
    small_bias = args[2] if len(args) > 2 else 0.0
    splits = tuple(args[3]) if len(args) > 3 else ("train", "test")
    boxes = generate(n=n, seed=seed, source_splits=splits, small_bias=small_bias)
    out = Path(SYNTH) / "synth"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"boxes_shard_{seed:03d}.json").write_text(json.dumps(boxes))
    data_vol.commit()
    return len(boxes)


@app.function(volumes=VOLUMES, timeout=1 * HOUR, cpu=2)
def synth_merge() -> dict:
    """Merge shard box files that are already on the volume, without regenerating anything."""
    from pathlib import Path

    from pmdm.config import SYNTH
    from pmdm.synth import merge_box_index

    data_vol.reload()
    shards = sorted((Path(SYNTH) / "synth").glob("boxes_shard_*.json"))
    total = merge_box_index()
    data_vol.commit()
    return {"shard_files": len(shards), "indexed": total}


@app.function(volumes=VOLUMES, timeout=8 * HOUR)
def synth_all(shards: int, per_shard: int, small_bias: float = 0.6,
              splits: tuple = ("train", "test")) -> dict:
    from pmdm.synth import merge_box_index

    counts = list(synth_shard.map(
        [(s, per_shard, small_bias, splits) for s in range(shards)], order_outputs=False))
    # The shards ran in their own containers; without this reload their commits are invisible
    # here and the merge sees an empty directory.
    data_vol.reload()
    total = merge_box_index()
    data_vol.commit()
    return {"shards": shards, "generated": sum(counts), "indexed": total}


# --------------------------------------------------------------------------- #
# Stage 1: detector
# --------------------------------------------------------------------------- #
@app.function(volumes=VOLUMES, gpu="A100-40GB", timeout=24 * HOUR, memory=32768)
def train_stage1(fold: int, epochs: int, backbone: str, n_synth: int,
                 samples_per_pair: int, batch: int, tag: str = "",
                 use_ema: bool = True, eval_every: int = 5, real_repeat: int = 1,
                 wh_relative: bool = True) -> dict:
    import torch

    from pmdm.train import train_fold

    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    result = train_fold(
        fold=fold,
        epochs=epochs,
        backbone=backbone,
        n_synth=n_synth,
        samples_per_pair=samples_per_pair,
        batch=batch,
        use_synth=n_synth != 0,
        use_ema=use_ema,
        eval_every=eval_every,
        tag=tag,
        real_repeat=real_repeat,
        wh_relative=wh_relative,
        on_epoch_end=ckpt_vol.commit,
    )
    ckpt_vol.commit()
    return result


@app.function(volumes=VOLUMES, gpu="L4", timeout=2 * HOUR, memory=16384)
def smoke_test() -> dict:
    """Cheap end-to-end check: one batch forward/backward plus one full-image decode."""
    import numpy as np
    import torch

    from pmdm.dataset import PairTileDataset, load_gt
    from pmdm.infer import predict_pair
    from pmdm.losses import total_loss
    from pmdm.model import SiamCenterNet

    gt_raw = load_gt()
    items = [("train", i) for i in range(4)]
    gt = {("train", i): gt_raw.get(i, np.zeros((0, 4), np.float32)) for i in range(4)}
    ds = PairTileDataset(items, gt, samples_per_pair=2)
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in ds[0]}

    model = SiamCenterNet(pretrained=True).cuda()
    out = model(batch["a"].cuda(), batch["b"].cuda())
    losses = total_loss(out, {k: v.cuda() for k, v in batch.items() if k not in ("a", "b")})
    losses["total"].backward()

    model.eval()
    boxes, scores = predict_pair(model, "train", 0, device="cuda", score_thr=0.2)
    return {
        "hm_shape": list(out["hm"].shape),
        "losses": {k: float(v) for k, v in losses.items()},
        "untrained_boxes_on_pair_0": int(len(boxes)),
        "gt_boxes_on_pair_0": int(len(gt[("train", 0)])),
    }


# --------------------------------------------------------------------------- #
# Out-of-fold candidates, verifier, submission
# --------------------------------------------------------------------------- #
@app.function(volumes=VOLUMES, gpu="L4", timeout=6 * HOUR, memory=16384)
def predict_oof(fold: int, backbone: str, score_thr: float, tag: str = "",
                flips: bool = False, offsets: tuple = (0,), max_boxes: int = 40,
                raw: bool = False, suffix: str = "") -> dict:
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.folds import split as fold_split
    from pmdm.infer import load_model, predict_pair
    from pmdm.metric import sweep_threshold

    ckpt = Path(CKPT) / f"stage1_fold{fold}_{backbone}{tag}" / "best.pt"
    model = load_model(ckpt, backbone=backbone)
    _, val_idx = fold_split(fold)
    gt_raw = load_gt()

    store, gt_eval = {}, {}
    for idx in val_idx:
        boxes, scores = predict_pair(model, "train", idx, device="cuda", score_thr=score_thr,
                                     flips=flips, offsets=tuple(offsets),
                                     max_boxes=max_boxes,
                                     use_snap=not raw, use_polarity=not raw)
        key = f"train_{idx:03d}"
        store[key] = (boxes, scores)
        gt_eval[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))

    out = Path(CKPT) / "oof"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"fold{fold}_{backbone}{suffix}.npz",
        **{f"{k}__boxes": v[0] for k, v in store.items()},
        **{f"{k}__scores": v[1] for k, v in store.items()},
    )
    thr, best = sweep_threshold(store, gt_eval)
    best["threshold"] = thr
    (out / f"fold{fold}_{backbone}{suffix}.json").write_text(json.dumps(best, indent=2))
    ckpt_vol.commit()
    return best


@app.function(volumes=VOLUMES, gpu="L4", timeout=8 * HOUR, memory=16384)
def train_verifier_fn(epochs: int) -> dict:
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.verifier import build_records, train_verifier

    gt_raw = load_gt()
    candidates, gt = {}, {}
    for path in sorted((Path(CKPT) / "oof").glob("*.npz")):
        z = np.load(path)
        keys = {k.split("__")[0] for k in z.files}
        for key in keys:
            idx = int(key.split("_")[-1])
            candidates[key] = ("train", idx, z[f"{key}__boxes"], z[f"{key}__scores"])
            gt[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))

    records = build_records(candidates, gt)
    result = train_verifier(records, epochs=epochs)
    ckpt_vol.commit()
    return result


@app.function(volumes=VOLUMES, gpu="L4", timeout=8 * HOUR, memory=16384)
def verifier_ab(fold: int, backbone: str, epochs: int) -> dict:
    """Honest verifier measurement on one fold.

    The validation pairs are split in half by index parity: the verifier trains on
    half A's candidates and is measured on half B, with half B's pre-verifier sweep
    as the control. Training and scoring on the same candidates would leak.
    """
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.metric import sweep_threshold
    from pmdm.verifier import apply_verifier, build_records, load_verifier, train_verifier

    z = np.load(Path(CKPT) / "oof" / f"fold{fold}_{backbone}.npz")
    keys = sorted({k.split("__")[0] for k in z.files})
    gt_raw = load_gt()

    half_a = [k for k in keys if int(k.split("_")[-1]) % 2 == 0]
    half_b = [k for k in keys if int(k.split("_")[-1]) % 2 == 1]

    def bundle(subset):
        cand, gt = {}, {}
        for key in subset:
            idx = int(key.split("_")[-1])
            cand[key] = ("train", idx, z[f"{key}__boxes"], z[f"{key}__scores"])
            gt[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))
        return cand, gt

    cand_a, gt_a = bundle(half_a)
    cand_b, gt_b = bundle(half_b)

    records = build_records(cand_a, gt_a)
    stats = train_verifier(records, epochs=epochs, out_dir=Path(CKPT) / "stage2_ab")
    model = load_verifier(Path(CKPT) / "stage2_ab" / "last.pt")

    before = {k: (v[2], v[3]) for k, v in cand_b.items()}
    thr_before, score_before = sweep_threshold(before, gt_b)

    after = {}
    for key, (split, idx, boxes, scores) in cand_b.items():
        after[key] = apply_verifier(model, split, idx, boxes, scores, device="cuda")
    thr_after, score_after = sweep_threshold(after, gt_b)

    ckpt_vol.commit()
    return {
        "train_pairs": len(half_a), "eval_pairs": len(half_b),
        "records": stats,
        "before": {**score_before, "threshold": thr_before},
        "after": {**score_after, "threshold": thr_after},
        "delta_f1": round(score_after["f1"] - score_before["f1"], 4),
    }


@app.function(volumes=VOLUMES, gpu="L4", timeout=8 * HOUR, memory=16384)
def predict_oof_ensemble(fold: int, backbone: str, tags: tuple, score_thr: float,
                         flips: bool, offsets: tuple, max_boxes: int, raw: bool,
                         suffix: str) -> dict:
    """Out-of-fold prediction fusing several checkpoints of the same fold.

    Every member trained on the same fold split, so none of them has seen this fold's
    validation pages — ensembling them changes nothing about leakage. Candidates are pooled
    before fusion so the coordinate averaging sees each model's raw opinion.
    """
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.folds import split as fold_split
    from pmdm.infer import load_model, predict_pair_ensemble
    from pmdm.metric import sweep_threshold

    models = [load_model(Path(CKPT) / f"stage1_fold{fold}_{backbone}{t}" / "best.pt",
                         backbone=backbone) for t in tags]
    print(f"[fold {fold}] ensemble of {len(models)}: {list(tags)}", flush=True)
    _, val_idx = fold_split(fold)
    gt_raw = load_gt()

    store, gt_eval = {}, {}
    for idx in val_idx:
        boxes, scores = predict_pair_ensemble(
            models, "train", idx, device="cuda", score_thr=score_thr, flips=flips,
            offsets=tuple(offsets), max_boxes=max_boxes,
            use_snap=not raw, use_polarity=not raw)
        key = f"train_{idx:03d}"
        store[key] = (boxes, scores)
        gt_eval[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))

    out = Path(CKPT) / "oof"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"fold{fold}_{backbone}{suffix}.npz",
        **{f"{k}__boxes": v[0] for k, v in store.items()},
        **{f"{k}__scores": v[1] for k, v in store.items()},
    )
    thr, best = sweep_threshold(store, gt_eval)
    best["threshold"] = thr
    (out / f"fold{fold}_{backbone}{suffix}.json").write_text(json.dumps(best, indent=2))
    ckpt_vol.commit()
    return best


@app.function(volumes=VOLUMES, gpu="L4", timeout=8 * HOUR, memory=16384)
def verifier_cv_fold(target_fold: int, backbone: str, epochs: int, folds: tuple,
                     suffix: str, blend: float, refine: bool) -> dict:
    """Train a verifier on every fold except `target_fold`, then apply it to that fold.

    This is the leakage-free arrangement. Fold j's candidates were produced by a detector that
    never trained on fold j, so they are honest predictions; the verifier trained on folds
    != k has seen neither the target fold's pages nor its detector. The previous attempt at
    stage 2 failed for a different reason — it trained on post-processed candidates, which are
    capped at 40 per image and already polarity-filtered, so it almost never saw a negative.
    These records come from the raw candidate dump instead.
    """
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.evaluate_cv import load_fold_candidates
    from pmdm.metric import sweep_threshold
    from pmdm.verifier import (apply_verifier, build_records, load_verifier,
                               train_verifier)

    root = Path(CKPT) / "oof"
    gt_raw = load_gt()

    def as_candidates(fold: int) -> dict:
        part = load_fold_candidates(root / f"fold{fold}_{backbone}{suffix}.npz")
        return {k: ("train", int(k.split("_")[-1]), v[0], v[1]) for k, v in part.items()}

    train_records = []
    for f in folds:
        if f == target_fold:
            continue
        cands = as_candidates(f)
        gt = {k: gt_raw.get(v[1], np.zeros((0, 4), np.float32)) for k, v in cands.items()}
        train_records += build_records(cands, gt)

    out_dir = Path(CKPT) / f"stage2_cv/fold{target_fold}"
    train_verifier(train_records, epochs=epochs, out_dir=out_dir)
    ckpt_vol.commit()

    model = load_verifier(out_dir / "last.pt")
    target = as_candidates(target_fold)
    refined, gt_eval = {}, {}
    for key, (split, idx, boxes, scores) in target.items():
        nb, ns = apply_verifier(model, split, idx, boxes, scores, refine=refine, blend=blend)
        refined[key] = (nb, ns)
        gt_eval[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))

    np.savez_compressed(
        root / f"fold{target_fold}_{backbone}{suffix}_verified.npz",
        **{f"{k}__boxes": v[0] for k, v in refined.items()},
        **{f"{k}__scores": v[1] for k, v in refined.items()},
    )
    detector_only = {k: (v[2], v[3]) for k, v in target.items()}
    thr_before, before = sweep_threshold(detector_only, gt_eval)
    thr_after, after = sweep_threshold(refined, gt_eval)
    ckpt_vol.commit()
    return {"fold": target_fold, "records": len(train_records),
            "before": {**before, "threshold": thr_before},
            "after": {**after, "threshold": thr_after},
            "delta_f1": after["f1"] - before["f1"]}


@app.function(volumes=VOLUMES, timeout=12 * HOUR, memory=8192)
def verify_cv_all(backbone: str, epochs: int, suffix: str, blend: float, refine: bool,
                  folds: tuple) -> list:
    """Server-side fan-out over folds.

    `starmap` from a local entrypoint is driven by the client, so a laptop network blip kills
    the whole map even under `--detach` — that is exactly how the first attempt at this died.
    Running the fan-out inside a remote function means the client only has to survive one call,
    and with `--detach` it does not have to survive at all.
    """
    args = [(f, backbone, epochs, tuple(folds), suffix, blend, refine) for f in folds]
    return list(verifier_cv_fold.starmap(args))


@app.function(volumes=VOLUMES, timeout=12 * HOUR, memory=8192)
def oof_ens_all_fn(backbone: str, tags: tuple, score_thr: float, flips: bool, offsets: tuple,
                   max_boxes: int, raw: bool, suffix: str, folds: tuple) -> list:
    """Server-side fan-out for the ensemble out-of-fold pass, same reasoning as above."""
    args = [(f, backbone, tuple(tags), score_thr, flips, tuple(offsets), max_boxes, raw, suffix)
            for f in folds]
    return list(predict_oof_ensemble.starmap(args))


@app.function(volumes=VOLUMES, gpu="L4", timeout=4 * HOUR, memory=16384)
def score_pooled(backbone: str, suffix: str, folds: tuple) -> dict:
    """Global F1 over the union of every fold's out-of-fold candidates."""
    from pmdm.evaluate_cv import cross_validated_score

    return cross_validated_score(tuple(folds), backbone, suffix)


@app.function(volumes=VOLUMES, gpu="L4", timeout=8 * HOUR, memory=16384)
def predict_test_shard(shard: int, n_shards: int, folds: tuple, tags: tuple, backbone: str,
                       score_thr: float, flips: bool, offsets: tuple, max_boxes: int,
                       use_verifier: bool, blend: float) -> dict:
    """Predict one slice of the test set with every fold model ensembled.

    Out-of-fold scoring can only use the one model that did not train on a given pair; the test
    set was seen by none of them, so all of them are fair game here. That is why the submission
    should score slightly above the reported out-of-fold number rather than matching it.
    """
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT, N_TEST
    from pmdm.infer import load_model, predict_pair_ensemble
    from pmdm.verifier import apply_verifier, load_verifier

    models = [load_model(Path(CKPT) / f"stage1_fold{f}_{backbone}{t}" / "best.pt",
                         backbone=backbone)
              for f in folds for t in tags]
    verifiers = []
    if use_verifier:
        for f in folds:
            vp = Path(CKPT) / f"stage2_cv/fold{f}" / "last.pt"
            if vp.exists():
                verifiers.append(load_verifier(vp))
    print(f"[shard {shard}] {len(models)} detectors, {len(verifiers)} verifiers", flush=True)

    idxs = [i for i in range(N_TEST) if i % n_shards == shard]
    store = {}
    for idx in idxs:
        boxes, scores = predict_pair_ensemble(
            models, "test", idx, device="cuda", score_thr=score_thr, flips=flips,
            offsets=tuple(offsets), max_boxes=max_boxes)
        if verifiers and len(boxes):
            # Average the fold verifiers' opinions, then blend with the detector score once.
            probs = []
            for v in verifiers:
                _, vs = apply_verifier(v, "test", idx, boxes, scores, device="cuda",
                                       refine=False, blend=1.0)
                probs.append(vs)
            vmean = np.mean(probs, axis=0)
            scores = (vmean ** blend) * (scores ** (1 - blend))
        store[idx] = (boxes, scores)

    out = Path(CKPT) / "test_shards"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"shard{shard:02d}.npz",
        **{f"{i}__boxes": v[0] for i, v in store.items()},
        **{f"{i}__scores": v[1] for i, v in store.items()},
    )
    ckpt_vol.commit()
    return {"shard": shard, "images": len(store),
            "candidates": int(sum(len(v[0]) for v in store.values()))}


@app.function(volumes=VOLUMES, timeout=12 * HOUR, memory=8192)
def predict_test_all(folds: tuple, tags: tuple, backbone: str, score_thr: float, flips: bool,
                     offsets: tuple, max_boxes: int, use_verifier: bool, blend: float,
                     threshold: float, n_shards: int) -> dict:
    """Server-side fan-out over test shards, then one submission file."""
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT, N_TEST
    from pmdm.metric import write_submission

    args = [(sh, n_shards, tuple(folds), tuple(tags), backbone, score_thr, flips,
             tuple(offsets), max_boxes, use_verifier, blend) for sh in range(n_shards)]
    parts = list(predict_test_shard.starmap(args))
    ckpt_vol.reload()

    candidates = {}
    for sh in range(n_shards):
        z = np.load(Path(CKPT) / "test_shards" / f"shard{sh:02d}.npz")
        for key in sorted({k.rsplit("__", 1)[0] for k in z.files}):
            candidates[f"template/test_template_{int(key):03d}.png"] = (
                z[f"{key}__boxes"], z[f"{key}__scores"])

    missing = [i for i in range(N_TEST)
               if f"template/test_template_{i:03d}.png" not in candidates]
    out = Path(CKPT) / "submission.csv"
    df = write_submission(candidates, threshold, out)
    ckpt_vol.commit()
    return {"rows": len(df), "images": len(candidates), "missing_images": missing,
            "boxes_per_image": round(len(df) / max(1, len(candidates)), 2),
            "shards": parts, "threshold": threshold, "path": str(out)}


# --------------------------------------------------------------------------- #
# Local entrypoints
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def prep():
    print(preprocess_all.remote())


@app.local_entrypoint()
def smoke():
    print(smoke_test.remote())


@app.local_entrypoint()
def synth(shards: int = 16, per_shard: int = 400, small_bias: float = 0.6):
    print(synth_all.remote(shards, per_shard, small_bias))


@app.local_entrypoint()
def merge_synth():
    print(synth_merge.remote())


@app.local_entrypoint()
def train(fold: int = 0, epochs: int = 40, backbone: str = "convnext_tiny",
          n_synth: int = 0, samples_per_pair: int = 8, batch: int = 8, tag: str = "",
          use_ema: bool = True, eval_every: int = 5, real_repeat: int = 1,
          wh_relative: bool = True):
    print(train_stage1.remote(fold, epochs, backbone, n_synth, samples_per_pair, batch,
                              tag, use_ema, eval_every, real_repeat, wh_relative))


@app.local_entrypoint()
def train_all(epochs: int = 40, backbone: str = "convnext_tiny", n_synth: int = 0,
              samples_per_pair: int = 8, batch: int = 8, tag: str = "",
              use_ema: bool = True, eval_every: int = 5, folds: str = "0,1,2,3,4",
              real_repeat: int = 1, wh_relative: bool = True):
    fold_list = [int(f) for f in folds.split(",") if f.strip() != ""]
    args = [(f, epochs, backbone, n_synth, samples_per_pair, batch, tag, use_ema, eval_every,
             real_repeat, wh_relative) for f in fold_list]
    for result in train_stage1.starmap(args):
        print(result)


@app.local_entrypoint()
def ab(epochs: int = 40, backbone: str = "convnext_tiny", n_synth: int = 6000,
       samples_per_pair: int = 8, batch: int = 8, fold: int = 0, eval_every: int = 5):
    """Two fold-0 runs side by side: the new recipe without and with synthetic data.

    Both carry the geometric augmentation, relative wh loss and EMA, so the only difference
    between them is the synthetic corpus. Whichever wins by more than the +-0.02 noise band
    decides what trains on all five folds.
    """
    args = [
        (fold, epochs, backbone, 0, samples_per_pair, batch, "_aug", True, eval_every, 1),
        (fold, 15, backbone, n_synth, 2, batch, "_augsyn", True, 2, 4),
    ]
    for result in train_stage1.starmap(args):
        print(result)


@app.local_entrypoint()
def oof(fold: int = 0, backbone: str = "convnext_tiny", score_thr: float = 0.05,
        tag: str = "", flips: bool = False, offsets: str = "0", max_boxes: int = 40,
        raw: bool = False, suffix: str = ""):
    off = tuple(int(o) for o in offsets.split(",") if o.strip() != "")
    print(predict_oof.remote(fold, backbone, score_thr, tag, flips, off, max_boxes,
                             raw, suffix))


@app.local_entrypoint()
def oof_all(backbone: str = "convnext_tiny", score_thr: float = 0.05, tag: str = "",
            flips: bool = False, offsets: str = "0", max_boxes: int = 40,
            raw: bool = False, suffix: str = "", folds: str = "0,1,2,3,4"):
    off = tuple(int(o) for o in offsets.split(",") if o.strip() != "")
    fold_list = [int(f) for f in folds.split(",") if f.strip() != ""]
    args = [(f, backbone, score_thr, tag, flips, off, max_boxes, raw, suffix)
            for f in fold_list]
    for result in predict_oof.starmap(args):
        print(result)


@app.local_entrypoint()
def verify(epochs: int = 8):
    print(train_verifier_fn.remote(epochs))


@app.local_entrypoint()
def oof_ens(fold: int = 0, backbone: str = "convnext_tiny", tags: str = "_aug,_augsyn",
            score_thr: float = 0.02, flips: bool = True, offsets: str = "0",
            max_boxes: int = 300, raw: bool = False, suffix: str = "_ens"):
    tag_list = tuple(t for t in tags.split(",") if t.strip() != "")
    off = tuple(int(o) for o in offsets.split(",") if o.strip() != "")
    print(predict_oof_ensemble.remote(fold, backbone, tag_list, score_thr, flips, off,
                                      max_boxes, raw, suffix))


@app.local_entrypoint()
def oof_ens_all(backbone: str = "convnext_tiny", tags: str = "_aug,_augsyn",
                score_thr: float = 0.02, flips: bool = True, offsets: str = "0",
                max_boxes: int = 300, raw: bool = False, suffix: str = "_ens",
                folds: str = "0,1,2,3,4"):
    tag_list = tuple(t for t in tags.split(",") if t.strip() != "")
    off = tuple(int(o) for o in offsets.split(",") if o.strip() != "")
    fold_list = tuple(int(f) for f in folds.split(",") if f.strip() != "")
    for r in oof_ens_all_fn.remote(backbone, tag_list, score_thr, flips, off,
                                   max_boxes, raw, suffix, fold_list):
        print(r)


@app.local_entrypoint()
def verify_cv(backbone: str = "convnext_tiny", epochs: int = 8, suffix: str = "_raw",
              blend: float = 0.5, refine: bool = True, folds: str = "0,1,2,3,4"):
    fold_list = tuple(int(f) for f in folds.split(",") if f.strip() != "")
    for result in verify_cv_all.remote(backbone, epochs, suffix, blend, refine, fold_list):
        print(result)


@app.local_entrypoint()
def pooled(backbone: str = "convnext_tiny", suffix: str = "", folds: str = "0,1,2,3,4"):
    fold_list = tuple(int(f) for f in folds.split(",") if f.strip() != "")
    print(score_pooled.remote(backbone, suffix, fold_list))


@app.local_entrypoint()
def verify_ab(fold: int = 0, backbone: str = "convnext_tiny", epochs: int = 8):
    print(verifier_ab.remote(fold, backbone, epochs))


@app.local_entrypoint()
def predict(folds: str = "0,1,2,3,4", tags: str = "_aug,_augsyn",
            backbone: str = "convnext_tiny", score_thr: float = 0.02, flips: bool = True,
            offsets: str = "0", max_boxes: int = 300, use_verifier: bool = False,
            blend: float = 0.5, threshold: float = 0.30, n_shards: int = 5):
    fold_list = tuple(int(f) for f in folds.split(",") if f.strip() != "")
    tag_list = tuple(t for t in tags.split(",") if t.strip() != "")
    off = tuple(int(o) for o in offsets.split(",") if o.strip() != "")
    print(predict_test_all.remote(fold_list, tag_list, backbone, score_thr, flips, off,
                                  max_boxes, use_verifier, blend, threshold, n_shards))
