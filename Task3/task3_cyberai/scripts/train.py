"""Train the shipped model and store its out-of-fold and test probabilities.

This is the whole training path in one file, independent of Modal. It fits the
two views of the shipped design - the flat ten-class model and the hierarchical
application x mode composition - under the stored grouped folds, then refits
both on all 1,285 training rows to produce test probabilities.

    python scripts/train.py --model tabicl          # the shipped configuration
    python scripts/train.py --model lgbm --repeats 2

Output: <work>/oof/train_<model>.npz holding oof_flat, oof_hier, test_flat,
test_hier and a JSON metadata blob. `scripts/predict.py` turns that into a
submission; nothing else is needed between the two.

The grouped scheme is the decisive one: near-duplicate flows very likely come
from the same source call, and a random split would score a memorised call
rather than a generalising model. `models.cv` builds the folds once so every
run and every container sees the same ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import config as C
import io_utils as IO
from features import base as fbase
from models import cv as cvmod
from models import hier

DEFAULT_PARAMS = {
    "tabicl": {"n_estimators": 16},
    "lgbm": {"n_estimators": 600, "num_leaves": 31, "learning_rate": 0.05,
             "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1},
    "xgb": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.06,
            "colsample_bytree": 0.5, "subsample": 0.9},
    "rf": {"n_estimators": 1200, "max_features": "sqrt"},
    "et": {"n_estimators": 1200, "max_features": "sqrt"},
    "cat": {"iterations": 1000, "depth": 6, "learning_rate": 0.03},
}


def load_or_build_folds(y: np.ndarray, groups: np.ndarray) -> list:
    try:
        return cvmod.load_folds()["group"]
    except (FileNotFoundError, OSError):
        folds = cvmod.build_folds(y, groups)
        C.ensure_dirs()
        cvmod.save_folds(folds)
        return folds["group"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tabicl", choices=sorted(DEFAULT_PARAMS))
    ap.add_argument("--repeats", type=int, default=2,
                    help="repeats of the grouped 5-fold scheme (2 are stored)")
    ap.add_argument("--params", default=None, help="JSON overriding the defaults")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the full-data refit and test prediction")
    args = ap.parse_args(argv)

    params = DEFAULT_PARAMS[args.model] | json.loads(args.params or "{}")
    train, test = IO.load_train(), IO.load_test()
    y = IO.labels_to_ids(train["label"])
    groups = IO.group_keys(train)
    X = fbase.build(train[C.FEATURE_COLS])
    X_test = fbase.build(test[C.FEATURE_COLS])
    audio = hier.audio_only(train)

    splits = load_or_build_folds(y, groups)
    n_rep = min(args.repeats, len(splits) // C.N_SPLITS)
    splits = splits[: n_rep * C.N_SPLITS]

    t0 = time.time()
    flat = np.zeros((n_rep, len(y), len(C.LABELS)), dtype=np.float32)
    hi = np.zeros_like(flat)
    for i, (tr_idx, va_idx) in enumerate(splits):
        rep = i // C.N_SPLITS
        f, h = hier.heads_predict(args.model, params, X.iloc[tr_idx], y[tr_idx],
                                  X.iloc[va_idx])
        flat[rep, va_idx], hi[rep, va_idx] = f, h
        print(f"fold {i + 1}/{len(splits)} done", flush=True)

    oof_flat, oof_hier = flat.mean(axis=0), hi.mean(axis=0)
    blend = _normalise(_normalise(oof_flat) + _normalise(oof_hier))
    scores = {
        "flat": hier.score(y, oof_flat.argmax(axis=1)),
        "hier": hier.score(y, oof_hier.argmax(axis=1)),
        "blend": hier.score(y, blend.argmax(axis=1)),
        "blend+zoom_rule": hier.score(y, hier.apply_zoom_rule(blend, audio)),
    }

    arrays = {"oof_flat": oof_flat, "oof_hier": oof_hier}
    if not args.no_test:
        f_te, h_te = hier.heads_predict(args.model, params, X, y, X_test)
        arrays |= {"test_flat": f_te.astype(np.float32),
                   "test_hier": h_te.astype(np.float32)}

    C.ensure_dirs()
    path = C.OOF_DIR / f"train_{args.model}.npz"
    meta = {"model": args.model, "params": params, "repeats": n_rep,
            "scores": scores, "fit_seconds": round(time.time() - t0, 2)}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    print(json.dumps(meta | {"path": str(path)}, indent=2))
    return 0


def _normalise(p: np.ndarray) -> np.ndarray:
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)


if __name__ == "__main__":
    raise SystemExit(main())
