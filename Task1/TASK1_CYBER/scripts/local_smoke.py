"""CPU smoke test: preprocess two pairs, build a batch, run the model, decode, score.

Run before spending anything on Modal:

    PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
    PMDM_WORK=/tmp/pmdm_work PMDM_CKPT=/tmp/pmdm_ckpt \
    .venv/bin/python scripts/local_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pmdm import config  # noqa: E402
from pmdm.baseline import baseline_pair  # noqa: E402
from pmdm.dataset import PairTileDataset, load_gt  # noqa: E402
from pmdm.decode import postprocess  # noqa: E402
from pmdm.losses import total_loss  # noqa: E402
from pmdm.metric import sweep_threshold  # noqa: E402
from pmdm.model import SiamCenterNet  # noqa: E402
from pmdm.preprocess import load_prepared, preprocess_pair  # noqa: E402
from pmdm.synth import apply_edits, degrade  # noqa: E402

N_PAIRS = 2
TILE = 256  # smaller than production so this finishes on a laptop CPU


def main() -> None:
    print("data:", config.DATA, "work:", config.WORK)

    print("\n[1] preprocess")
    for i in range(N_PAIRS):
        info = preprocess_pair("train", i)
        print("   ", {k: info[k] for k in ("idx", "mode", "response", "blur_sigma", "h", "w")})

    print("\n[2] dataset")
    gt_raw = load_gt()
    items = [("train", i) for i in range(N_PAIRS)]
    gt = {("train", i): gt_raw.get(i, np.zeros((0, 4), np.float32)) for i in range(N_PAIRS)}
    ds = PairTileDataset(items, gt, samples_per_pair=2, tile=TILE, augment=True)
    sample = ds[0]
    print("   ", {k: tuple(v.shape) for k, v in sample.items()})
    print("    positives in tile:", float(sample["reg"].sum()))

    print("\n[3] model forward/backward")
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in sample}
    model = SiamCenterNet(pretrained=False)
    out = model(batch["a"], batch["b"])
    losses = total_loss(out, {k: v for k, v in batch.items() if k not in ("a", "b")})
    losses["total"].backward()
    print("   ", {k: tuple(v.shape) for k, v in out.items()})
    print("   ", {k: round(float(v), 4) for k, v in losses.items()})

    print("\n[4] classical baseline + scorer on the preprocessed pairs")
    candidates, gt_eval = {}, {}
    for i in range(N_PAIRS):
        boxes, scores = baseline_pair("train", i)
        _, _, tn, pn = load_prepared("train", i)
        boxes, scores = postprocess(boxes, scores, tn, pn)
        candidates[f"train_{i:03d}"] = (boxes, scores)
        gt_eval[f"train_{i:03d}"] = gt[("train", i)]
    thr, best = sweep_threshold(candidates, gt_eval)
    print("    threshold", round(thr, 3), {k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in best.items()})

    print("\n[5] synthetic generator")
    import cv2

    src = cv2.imread(str(config.TRAIN_DIR / "template" / "train_template_000.png"))
    rng = np.random.RandomState(0)
    edited, boxes = apply_edits(src, rng)
    photo = degrade(edited, rng)
    print("    edits:", len(boxes), "photo shape:", photo.shape,
          "mean shift:", round(float(photo.mean() - src.mean()), 2))

    print("\nOK")


if __name__ == "__main__":
    main()
