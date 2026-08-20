"""Experiment A1: how much of the test set is answerable by direct lookup?

Only 2 of 1,136 distinct training length tuples map to more than one class, so
the mapping from packet-length tuple to label is almost injective on the data we
have. The open question is whether test flows reuse those tuples. If they do,
those flows can be answered without a model; if they do not, a whole branch of
the phase-2 plan (lookup features, tuple-based overrides) is dead and should be
dropped before any effort goes into it.

The cross-validated arm answers the same question honestly inside the training
set, where labels are known: hold out a fold, look its tuples up in the other
four, and measure how often the lookup is available and correct.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
import io_utils as IO
from models import cv as cvmod

OUT = C.REPORTS_DIR / "analysis"


def _tuples(df: pd.DataFrame, tol: int = 0) -> list[tuple]:
    L = df[C.LEN_COLS].to_numpy(dtype=np.int64)
    if tol:
        L = (np.round(L / (tol * 2.0)) * (tol * 2.0)).astype(np.int64)
    return [tuple(r) for r in L]


def run(tolerances: tuple[int, ...] = (0, 1, 2, 4)) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = IO.load_train(), IO.load_test()
    y = IO.labels_to_ids(train["label"])
    report: dict = {"n_train": len(train), "n_test": len(test), "tolerances": {}}

    for tol in tolerances:
        tr_keys = _tuples(train, tol)
        te_keys = _tuples(test, tol)

        table: dict[tuple, list[int]] = {}
        for k, lab in zip(tr_keys, y):
            table.setdefault(k, []).append(int(lab))

        pure = {k: v[0] for k, v in table.items() if len(set(v)) == 1}
        matched = [k in table for k in te_keys]
        matched_pure = [k in pure for k in te_keys]

        # Honest in-training estimate: look each fold's tuples up in the rest.
        folds = cvmod.load_folds()["group"][: C.N_SPLITS]
        hits, correct, covered = 0, 0, 0
        for tr_idx, va_idx in folds:
            sub: dict[tuple, list[int]] = {}
            for i in tr_idx:
                sub.setdefault(tr_keys[i], []).append(int(y[i]))
            sub_pure = {k: v[0] for k, v in sub.items() if len(set(v)) == 1}
            for i in va_idx:
                covered += 1
                k = tr_keys[i]
                if k in sub_pure:
                    hits += 1
                    correct += int(sub_pure[k] == y[i])

        report["tolerances"][f"pm{tol}"] = {
            "distinct_train_tuples": len(table),
            "pure_train_tuples": len(pure),
            "impure_train_tuples": len(table) - len(pure),
            "test_rows_matched": int(sum(matched)),
            "test_rows_matched_share": float(np.mean(matched)),
            "test_rows_matched_pure": int(sum(matched_pure)),
            "test_rows_matched_pure_share": float(np.mean(matched_pure)),
            "cv_lookup_coverage": hits / max(1, covered),
            "cv_lookup_accuracy_when_available": correct / max(1, hits),
            "cv_rows_with_lookup": hits,
        }

    # Verdict against the gate written into the ablation plan.
    exact = report["tolerances"]["pm0"]
    report["verdict"] = {
        "gate": "exact matches > 5% of test AND lookup accuracy > 0.95",
        "exact_match_share": exact["test_rows_matched_pure_share"],
        "lookup_accuracy": exact["cv_lookup_accuracy_when_available"],
        "passes": bool(
            exact["test_rows_matched_pure_share"] > 0.05
            and exact["cv_lookup_accuracy_when_available"] > 0.95
        ),
    }

    pd.DataFrame(report["tolerances"]).T.to_csv(OUT / "a1_tuple_lookup.csv")
    (C.REPORTS_DIR / "a1_tuple_lookup.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def override_gain(tolerances: tuple[int, ...] = (0, 1, 2, 4)) -> dict:
    """Experiment A3: the differential that decides whether lookup is worth it.

    Coverage and lookup accuracy are only half the question. What matters is
    whether the lookup is *better than the model* on the rows it covers - if the
    model already answers those easy rows correctly, the override buys nothing.
    """
    import json

    ens = json.loads((C.MODELS_DIR / "ensemble.json").read_text())
    best = max(ens["members"], key=lambda m: m["group_acc"])
    prob = np.load(best["path"], allow_pickle=False)["oof_group"]
    model_pred = prob.argmax(axis=1)

    train = IO.load_train()
    y = IO.labels_to_ids(train["label"])
    folds = cvmod.load_folds()["group"][: C.N_SPLITS]
    out = {"member": best["name"], "model_accuracy": float((model_pred == y).mean()), "tolerances": {}}

    for tol in tolerances:
        keys = _tuples(train, tol)
        pred = model_pred.copy()
        covered = np.zeros(len(y), dtype=bool)
        for tr_idx, va_idx in folds:
            sub: dict[tuple, list[int]] = {}
            for i in tr_idx:
                sub.setdefault(keys[i], []).append(int(y[i]))
            pure = {k: v[0] for k, v in sub.items() if len(set(v)) == 1}
            for i in va_idx:
                if keys[i] in pure:
                    covered[i] = True
                    pred[i] = pure[keys[i]]

        n = int(covered.sum())
        out["tolerances"][f"pm{tol}"] = {
            "rows_covered": n,
            "coverage": float(covered.mean()),
            "model_accuracy_on_covered": float((model_pred[covered] == y[covered]).mean()) if n else None,
            "lookup_accuracy_on_covered": float((pred[covered] == y[covered]).mean()) if n else None,
            "overall_accuracy_with_override": float((pred == y).mean()),
            "delta_vs_model": float((pred == y).mean() - (model_pred == y).mean()),
            "rows_changed": int((pred != model_pred).sum()),
        }

    (C.REPORTS_DIR / "a3_lookup_override.json").write_text(json.dumps(out, indent=2, default=float))
    return out
