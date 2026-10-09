"""Run stage-1 training without Modal, on any machine with a GPU.

This is the same `train_fold` loop that `modal_app.py::train_stage1` calls; only the
execution environment differs. Preprocessing must have been run first (see below).

    # one-time: preprocess pairs into PMDM_WORK
    PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset PMDM_WORK=./work \
    python scripts/train_local.py --preprocess-only

    # train fold 0
    PMDM_DATA=Task1/PackagingMaterialDifferenceMiningDataset \
    PMDM_WORK=./work PMDM_CKPT=./ckpt \
    python scripts/train_local.py --fold 0 --epochs 40

Checkpoints land in $PMDM_CKPT/stage1_fold<N>_<backbone>/ and resume automatically,
exactly as on Modal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-synth", type=int, default=0,
                    help="synthetic pairs to mix in (0 disables; requires synth output in PMDM_WORK)")
    ap.add_argument("--samples-per-pair", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default=None, help="defaults to cuda if available, else cpu")
    ap.add_argument("--preprocess-only", action="store_true",
                    help="run stage 0 over all 300 pairs and exit")
    args = ap.parse_args()

    import torch

    from pmdm.config import N_TEST, N_TRAIN

    if args.preprocess_only:
        from pmdm.preprocess import preprocess_pair

        for split, n in (("train", N_TRAIN), ("test", N_TEST)):
            for idx in range(n):
                info = preprocess_pair(split, idx)
                if idx % 25 == 0:
                    print(f"{split} {idx}/{n} mode={info['mode']} sigma={info['blur_sigma']}",
                          flush=True)
        print("preprocessing complete")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: training on CPU will be impractically slow; this is for smoke tests only",
              flush=True)

    from pmdm.train import train_fold

    result = train_fold(
        fold=args.fold,
        epochs=args.epochs,
        backbone=args.backbone,
        batch=args.batch,
        n_synth=args.n_synth,
        use_synth=args.n_synth != 0,
        samples_per_pair=args.samples_per_pair,
        num_workers=args.num_workers,
        eval_every=args.eval_every,
        device=device,
    )
    print(result)


if __name__ == "__main__":
    main()
