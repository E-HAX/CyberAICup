"""Ensembling over stored out-of-fold probability matrices.

Every scored point writes its out-of-fold probabilities to the work Volume, so
ensembling refits nothing: it is a search over how to combine matrices that
already exist. Three combiners are compared and the winner is chosen on an
honest, nested estimate rather than on the score the weights were fitted to.

    hill-climb   greedy forward selection with replacement over soft votes,
                 the standard robust choice when the number of rows is small
    stacking     multinomial logistic regression on the concatenated member
                 probabilities
    rank-average a low-variance fallback that ignores calibration entirely
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

import config as C

N_CLASSES = len(C.LABELS)


@dataclass
class Member:
    name: str
    path: str
    oof: np.ndarray  # (n_train, n_classes) from the grouped scheme
    test: np.ndarray | None
    meta: dict

    @property
    def score(self) -> float:
        return float(self.meta.get("group_acc", 0.0))


def load_members(min_score: float = 0.0, require_test: bool = False) -> list[Member]:
    members: list[Member] = []
    for path in sorted(C.OOF_DIR.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        if "oof_group" not in z:
            continue
        meta = json.loads(str(z["meta"]))
        if meta.get("group_acc", 0.0) < min_score:
            continue
        test = z["test"] if "test" in z else None
        if require_test and test is None:
            continue
        members.append(
            Member(
                name=f"{meta['model']}_{meta['hash'][:8]}",
                path=str(path),
                oof=z["oof_group"].astype(np.float64),
                test=None if test is None else test.astype(np.float64),
                meta=meta,
            )
        )
    return members


def dedupe(members: list[Member], per_model: int = 6) -> list[Member]:
    """Keep the strongest few points per model family to limit redundancy."""
    by_model: dict[str, list[Member]] = {}
    for m in members:
        by_model.setdefault(m.meta["model"], []).append(m)
    kept: list[Member] = []
    for model, group in by_model.items():
        group.sort(key=lambda m: m.score, reverse=True)
        kept.extend(group[:per_model])
    return kept


# ---------------------------------------------------------- combiners ----


def hill_climb(
    probs: list[np.ndarray], y: np.ndarray, n_iter: int = 80, seed: int = C.SEED
) -> np.ndarray:
    """Greedy forward selection with replacement; returns member weights."""
    rng = np.random.RandomState(seed)
    n = len(probs)
    counts = np.zeros(n)
    current = np.zeros_like(probs[0])
    best_acc = 0.0
    for step in range(n_iter):
        best_i, best_step_acc = -1, -1.0
        order = rng.permutation(n)
        for i in order:
            cand = (current * step + probs[i]) / (step + 1)
            acc = accuracy_score(y, cand.argmax(axis=1))
            if acc > best_step_acc:
                best_i, best_step_acc = i, acc
        if best_step_acc < best_acc - 1e-9 and step > 0:
            break
        current = (current * step + probs[best_i]) / (step + 1)
        counts[best_i] += 1
        best_acc = best_step_acc
    return counts / max(counts.sum(), 1)


def blend(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs[0])
    for p, w in zip(probs, weights):
        if w:
            out += w * p
    return out


def rank_average(probs: list[np.ndarray]) -> np.ndarray:
    stacked = np.zeros_like(probs[0])
    for p in probs:
        stacked += np.apply_along_axis(rankdata, 0, p) / len(p)
    return stacked / len(probs)


def stack_fit_predict(
    oof: list[np.ndarray], y: np.ndarray, test: list[np.ndarray] | None
) -> tuple[np.ndarray, float, np.ndarray | None]:
    X = np.hstack(oof)
    meta_model = LogisticRegression(max_iter=3000, C=1.0, random_state=C.SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    oof_pred = np.zeros((len(y), N_CLASSES))
    for tr, va in cv.split(X, y):
        meta_model.fit(X[tr], y[tr])
        oof_pred[va] = meta_model.predict_proba(X[va])
    acc = accuracy_score(y, oof_pred.argmax(axis=1))
    meta_model.fit(X, y)
    test_pred = meta_model.predict_proba(np.hstack(test)) if test else None
    return oof_pred, float(acc), test_pred


def nested_hill_climb_score(
    probs: list[np.ndarray], y: np.ndarray, n_splits: int = 5
) -> float:
    """Honest estimate: fit the weights on 4/5 of the rows, score the rest."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=C.SEED)
    correct = 0
    for tr, va in cv.split(probs[0], y):
        w = hill_climb([p[tr] for p in probs], y[tr], n_iter=40)
        pred = blend([p[va] for p in probs], w).argmax(axis=1)
        correct += int((pred == y[va]).sum())
    return correct / len(y)


# -------------------------------------------------------------- driver ----


def build_ensemble(
    min_score: float = 0.0, per_model: int = 6, top_n: int = 25, require_test: bool = True
) -> dict:
    import io_utils as IO

    y = IO.labels_to_ids(IO.load_train()["label"])
    # Only members carrying a full-data refit can contribute to a submission,
    # so the blend is searched over exactly the set it will be applied to.
    members = dedupe(
        load_members(min_score=min_score, require_test=require_test), per_model=per_model
    )
    members.sort(key=lambda m: m.score, reverse=True)
    members = members[:top_n]
    if not members:
        raise RuntimeError("no scored members with out-of-fold matrices found")

    probs = [m.oof for m in members]
    have_test = all(m.test is not None for m in members)
    test_probs = [m.test for m in members] if have_test else None

    results = {}
    single_best = max(members, key=lambda m: m.score)
    results["single_best"] = {
        "name": single_best.name,
        "acc": accuracy_score(y, single_best.oof.argmax(axis=1)),
        "macro_f1": f1_score(y, single_best.oof.argmax(axis=1), average="macro"),
    }

    w = hill_climb(probs, y)
    hc = blend(probs, w)
    results["hill_climb"] = {
        "acc_fitted": accuracy_score(y, hc.argmax(axis=1)),
        "acc_nested": nested_hill_climb_score(probs, y),
        "macro_f1": f1_score(y, hc.argmax(axis=1), average="macro"),
        "weights": {m.name: float(wi) for m, wi in zip(members, w) if wi > 0},
    }

    ra = rank_average(probs)
    results["rank_average"] = {
        "acc": accuracy_score(y, ra.argmax(axis=1)),
        "macro_f1": f1_score(y, ra.argmax(axis=1), average="macro"),
    }

    _, stack_acc, stack_test = stack_fit_predict(probs, y, test_probs)
    results["stacking"] = {"acc_nested": stack_acc}

    # Selection uses the honest numbers only: the fitted hill-climb accuracy is
    # reported for reference but never used to choose.
    candidates = {
        "hill_climb": results["hill_climb"]["acc_nested"],
        "rank_average": results["rank_average"]["acc"],
        "stacking": results["stacking"]["acc_nested"],
        "single_best": results["single_best"]["acc"],
    }
    winner = max(candidates, key=candidates.get)
    results["winner"] = winner
    results["winner_acc"] = candidates[winner]
    results["members"] = [
        {"name": m.name, "path": m.path, "group_acc": m.score, "model": m.meta["model"],
         "params": m.meta["spec"]["params"], "features": m.meta["features"],
         "target_feats": m.meta["target_feats"]}
        for m in members
    ]
    results["has_test_probs"] = bool(have_test)

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODELS_DIR / "ensemble.json").write_text(json.dumps(results, indent=2, default=float))

    if have_test:
        if winner == "hill_climb":
            test_blend = blend(test_probs, w)
        elif winner == "rank_average":
            test_blend = rank_average(test_probs)
        elif winner == "stacking":
            test_blend = stack_test
        else:
            test_blend = single_best.test
        np.save(C.MODELS_DIR / "test_proba.npy", test_blend.astype(np.float32))

    return results
