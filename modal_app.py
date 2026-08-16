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
def synth_shard(args: tuple[int, int]) -> int:
    from pathlib import Path

    from pmdm.config import SYNTH
    from pmdm.synth import generate

    seed, n = args
    boxes = generate(n=n, seed=seed)
    out = Path(SYNTH) / "synth"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"boxes_shard_{seed:03d}.json").write_text(json.dumps(boxes))
    data_vol.commit()
    return len(boxes)


@app.function(volumes=VOLUMES, timeout=8 * HOUR)
def synth_all(shards: int, per_shard: int) -> dict:
    from pmdm.synth import merge_box_index

    counts = list(synth_shard.map([(s, per_shard) for s in range(shards)], order_outputs=False))
    total = merge_box_index()
    data_vol.commit()
    return {"shards": shards, "generated": sum(counts), "indexed": total}


# --------------------------------------------------------------------------- #
# Stage 1: detector
# --------------------------------------------------------------------------- #
@app.function(volumes=VOLUMES, gpu="A100-40GB", timeout=24 * HOUR, memory=32768)
def train_stage1(fold: int, epochs: int, backbone: str, n_synth: int,
                 samples_per_pair: int, batch: int) -> dict:
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
def predict_oof(fold: int, backbone: str, score_thr: float) -> dict:
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT
    from pmdm.dataset import load_gt
    from pmdm.folds import split as fold_split
    from pmdm.infer import load_model, predict_pair
    from pmdm.metric import sweep_threshold

    ckpt = Path(CKPT) / f"stage1_fold{fold}_{backbone}" / "best.pt"
    model = load_model(ckpt, backbone=backbone)
    _, val_idx = fold_split(fold)
    gt_raw = load_gt()

    store, gt_eval = {}, {}
    for idx in val_idx:
        boxes, scores = predict_pair(model, "train", idx, device="cuda", score_thr=score_thr)
        key = f"train_{idx:03d}"
        store[key] = (boxes, scores)
        gt_eval[key] = gt_raw.get(idx, np.zeros((0, 4), np.float32))

    out = Path(CKPT) / "oof"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"fold{fold}_{backbone}.npz",
        **{f"{k}__boxes": v[0] for k, v in store.items()},
        **{f"{k}__scores": v[1] for k, v in store.items()},
    )
    thr, best = sweep_threshold(store, gt_eval)
    best["threshold"] = thr
    (out / f"fold{fold}_{backbone}.json").write_text(json.dumps(best, indent=2))
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
def predict_test(folds: list[int], backbone: str, score_thr: float, threshold: float,
                 use_verifier: bool) -> dict:
    from pathlib import Path

    import numpy as np

    from pmdm.config import CKPT, N_TEST
    from pmdm.decode import postprocess, wbf
    from pmdm.infer import load_model, predict_pair
    from pmdm.metric import write_submission
    from pmdm.preprocess import load_prepared
    from pmdm.verifier import apply_verifier, load_verifier

    models = [load_model(Path(CKPT) / f"stage1_fold{f}_{backbone}" / "best.pt", backbone=backbone)
              for f in folds]
    verifier = None
    v_path = Path(CKPT) / "stage2" / "last.pt"
    if use_verifier and v_path.exists():
        verifier = load_verifier(v_path)

    candidates = {}
    for idx in range(N_TEST):
        boxes_all, scores_all = [], []
        for model in models:
            b, s = predict_pair(model, "test", idx, device="cuda", score_thr=score_thr,
                                use_snap=False, use_polarity=False)
            if len(b):
                boxes_all.append(b)
                scores_all.append(s)
        if boxes_all:
            boxes, scores = wbf(np.concatenate(boxes_all), np.concatenate(scores_all))
        else:
            boxes, scores = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        if verifier is not None and len(boxes):
            boxes, scores = apply_verifier(verifier, "test", idx, boxes, scores, device="cuda")
        _, _, tn, pn = load_prepared("test", idx)
        boxes, scores = postprocess(boxes, scores, tn, pn)
        candidates[f"template/test_template_{idx:03d}.png"] = (boxes, scores)

    out = Path(CKPT) / "submission.csv"
    df = write_submission(candidates, threshold, out)
    ckpt_vol.commit()
    return {"rows": len(df), "images": len(candidates), "path": str(out),
            "boxes_per_image": round(len(df) / max(1, len(candidates)), 2)}


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
def synth(shards: int = 16, per_shard: int = 400):
    print(synth_all.remote(shards, per_shard))


@app.local_entrypoint()
def train(fold: int = 0, epochs: int = 40, backbone: str = "convnext_tiny",
          n_synth: int = 0, samples_per_pair: int = 8, batch: int = 8):
    print(train_stage1.remote(fold, epochs, backbone, n_synth, samples_per_pair, batch))


@app.local_entrypoint()
def train_all(epochs: int = 40, backbone: str = "convnext_tiny", n_synth: int = 0,
              samples_per_pair: int = 8, batch: int = 8):
    args = [(f, epochs, backbone, n_synth, samples_per_pair, batch) for f in range(5)]
    for result in train_stage1.starmap(args):
        print(result)


@app.local_entrypoint()
def oof(fold: int = 0, backbone: str = "convnext_tiny", score_thr: float = 0.05):
    print(predict_oof.remote(fold, backbone, score_thr))


@app.local_entrypoint()
def verify(epochs: int = 8):
    print(train_verifier_fn.remote(epochs))


@app.local_entrypoint()
def verify_ab(fold: int = 0, backbone: str = "convnext_tiny", epochs: int = 8):
    print(verifier_ab.remote(fold, backbone, epochs))


@app.local_entrypoint()
def predict(folds: str = "0", backbone: str = "convnext_tiny", score_thr: float = 0.05,
            threshold: float = 0.35, use_verifier: bool = True):
    fold_list = [int(f) for f in folds.split(",") if f.strip() != ""]
    print(predict_test.remote(fold_list, backbone, score_thr, threshold, use_verifier))
