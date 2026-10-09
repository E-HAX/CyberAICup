"""Paired comparison of two stored models on identical folds.

The ablation protocol requires McNemar's test for classifier-versus-classifier
claims: with both models scored on the same rows, only the discordant pairs
carry information about which is better, and an exact binomial test on those
pairs is the right instrument. A point estimate without it is not a result.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import binomtest

import config as C
import io_utils as IO


def _load(path: str) -> tuple[np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    return z["oof_group"], json.loads(str(z["meta"]))


def compare(path_a: str, path_b: str) -> dict:
    y = IO.labels_to_ids(IO.load_train()["label"])
    pa, ma = _load(path_a)
    pb, mb = _load(path_b)
    ca = pa.argmax(axis=1) == y
    cb = pb.argmax(axis=1) == y

    b_only = int((~ca & cb).sum())   # B right, A wrong
    a_only = int((ca & ~cb).sum())   # A right, B wrong
    n_disc = a_only + b_only
    test = binomtest(b_only, n_disc, 0.5) if n_disc else None

    zoom = np.isin(y, [C.LABEL_TO_ID["Zoom_voice"], C.LABEL_TO_ID["Zoom_video"]])
    return {
        "a": {"model": ma["model"], "hash": ma["hash"], "acc": float(ca.mean()),
              "zoom_acc": float(ca[zoom].mean())},
        "b": {"model": mb["model"], "hash": mb["hash"], "acc": float(cb.mean()),
              "zoom_acc": float(cb[zoom].mean())},
        "delta_b_minus_a": float(cb.mean() - ca.mean()),
        "b_right_a_wrong": b_only,
        "a_right_b_wrong": a_only,
        "discordant_pairs": n_disc,
        "mcnemar_p": float(test.pvalue) if test else None,
        "verdict": (
            "B significantly better" if test and test.pvalue < 0.05 and b_only > a_only
            else "A significantly better" if test and test.pvalue < 0.05 and a_only > b_only
            else "not established"
        ),
    }


def compare_best(model_a: str, model_b: str) -> dict:
    """Compare the strongest stored point of two model families."""
    best: dict[str, tuple[float, str]] = {}
    for path in sorted(C.OOF_DIR.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if "oof_group" not in z:
            continue
        meta = json.loads(str(z["meta"]))
        name, acc = meta["model"], meta.get("group_acc", 0.0)
        if name in (model_a, model_b) and acc > best.get(name, (0.0, ""))[0]:
            best[name] = (acc, str(path))
    if model_a not in best or model_b not in best:
        raise RuntimeError(f"missing stored points: have {sorted(best)}")
    result = compare(best[model_a][1], best[model_b][1])
    (C.REPORTS_DIR / f"compare_{model_a}_vs_{model_b}.json").write_text(
        json.dumps(result, indent=2, default=float)
    )
    return result
