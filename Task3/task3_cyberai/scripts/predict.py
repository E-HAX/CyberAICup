"""Turn stored probability matrices into the competition submission.

    python scripts/predict.py --runs <work>/oof/train_tabicl.npz
    python scripts/predict.py --runs a.npz b.npz --weights 2 1   # a blend

Each run file comes from `scripts/train.py` (or the Modal `hier_point`
function) and holds the flat and hierarchical test matrices. The two views are
normalised, averaged, blended across runs if more than one is given, and then
passed through the Zoom rule before the labels are written.

The rule is applied to labels rather than probabilities on purpose: it is a
decision about an inseparable subset, not a change of belief, and folding it
into the probability matrix would misreport the model's confidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import config as C
import io_utils as IO
from models import hier
from predict import write_submission


def _normalise(p: np.ndarray) -> np.ndarray:
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)


def combine(paths: list[str], weights: list[float] | None, split: str) -> np.ndarray:
    weights = weights or [1.0] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("one weight per run file, or none at all")
    total = None
    for path, w in zip(paths, weights):
        d = np.load(path, allow_pickle=True)
        views = [_normalise(d[k]) for k in (f"{split}_flat", f"{split}_hier") if k in d.files]
        if not views:
            raise ValueError(f"{path} holds no '{split}' matrices")
        stacked = w * _normalise(np.mean(views, axis=0))
        total = stacked if total is None else total + stacked
    return _normalise(total)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--no-zoom-rule", action="store_true")
    ap.add_argument("--score-oof", action="store_true",
                    help="also score the same blend on the out-of-fold matrices")
    args = ap.parse_args(argv)

    if args.score_oof:
        train = IO.load_train()
        oof = combine(args.runs, args.weights, "oof")
        y = IO.labels_to_ids(train["label"])
        audio = hier.audio_only(train)
        print(json.dumps({
            "blend": hier.score(y, oof.argmax(axis=1)),
            "blend+zoom_rule": hier.score(y, hier.apply_zoom_rule(oof, audio)),
        }, indent=2))

    test = IO.load_test()
    proba = combine(args.runs, args.weights, "test")
    if args.no_zoom_rule:
        labels = IO.ids_to_labels(proba.argmax(axis=1))
    else:
        labels = IO.ids_to_labels(hier.apply_zoom_rule(proba, hier.audio_only(test)))
    print(json.dumps(write_submission(proba, labels=labels), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
