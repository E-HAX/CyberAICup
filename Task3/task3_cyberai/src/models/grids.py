"""Hyperparameter grids, materialised as explicit lists of specifications.

Rather than letting scikit-learn iterate a grid inside one process, each grid
point becomes a standalone specification that Modal ships to its own container.
The grid is therefore a distributed GridSearchCV whose parallelism lives at the
infrastructure layer, which is why every model is configured with a single
thread internally.

Screening runs use one repeat per scheme; the finalists are re-scored with the
full repeat budget in a second pass, so the wide sweep stays cheap and the
decisive comparison stays low-variance.
"""

from __future__ import annotations

from itertools import product

# Screening scores only the decisive (grouped) scheme; the finalists add the
# optimistic scheme and extra repeats once the field has been narrowed.
SCREEN_REPEATS = {"group": 1}
FINAL_REPEATS = {"group": 2, "plain": 3}

# The ablation showed the fitted label-aware blocks F9-F11 costing roughly two
# accuracy points on the grouped scheme even with an inner out-of-fold loop, so
# they are off by default and remain a searchable option rather than a given.
DEFAULT_TARGET_FEATS = False


def _expand(model: str, grid: dict, **common) -> list[dict]:
    keys = list(grid)
    specs = []
    for values in product(*(grid[k] for k in keys)):
        specs.append(
            {
                "model": model,
                "params": dict(zip(keys, values)),
                "features": common.get("features", "full"),
                "target_feats": common.get("target_feats", DEFAULT_TARGET_FEATS),
                "probes": common.get("probes", False),
                "repeats": common.get("repeats", SCREEN_REPEATS),
            }
        )
    return specs


def xgb_grid(**common) -> list[dict]:
    return _expand(
        "xgb",
        {
            "n_estimators": [400, 800],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.03, 0.06, 0.12],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.5, 0.8],
            "min_child_weight": [1, 3],
            "reg_lambda": [1.0, 5.0],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )


def lgbm_grid(**common) -> list[dict]:
    return _expand(
        "lgbm",
        {
            "n_estimators": [400, 800],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.06, 0.12],
            "min_child_samples": [5, 10, 20],
            "subsample": [0.8],
            "subsample_freq": [1],
            "colsample_bytree": [0.5, 0.8],
            "reg_lambda": [0.0, 5.0],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )


def cat_grid(**common) -> list[dict]:
    return _expand(
        "cat",
        {
            "iterations": [500, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.08],
            "l2_leaf_reg": [1.0, 5.0],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )


def rf_grid(**common) -> list[dict]:
    return _expand(
        "rf",
        {
            "n_estimators": [600, 1200],
            "max_depth": [None, 12, 24],
            "max_features": ["sqrt", "log2", 0.3],
            "min_samples_leaf": [1, 2, 4],
            "class_weight_mode": ["none", "balanced", "balanced_subsample"],
        },
        **common,
    )


def et_grid(**common) -> list[dict]:
    return _expand(
        "et",
        {
            "n_estimators": [600, 1200],
            "max_depth": [None, 16],
            "max_features": ["sqrt", 0.3],
            "min_samples_leaf": [1, 2],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )


def hgb_grid(**common) -> list[dict]:
    return _expand(
        "hgb",
        {
            "max_iter": [400],
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [5, 20],
            "l2_regularization": [0.0, 1.0],
        },
        **common,
    )


def classic_grid(**common) -> list[dict]:
    specs = []
    specs += _expand(
        "logreg",
        {
            "C": [0.03, 0.1, 0.3, 1.0, 3.0],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )
    specs += _expand(
        "svm",
        {
            "C": [1.0, 3.0, 10.0, 30.0],
            "gamma": ["scale", 0.01, 0.05],
            "class_weight_mode": ["none", "balanced"],
        },
        **common,
    )
    specs += _expand(
        "knn",
        {
            "n_neighbors": [1, 3, 5, 9, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        **common,
    )
    specs += _expand(
        "mlp",
        {
            "hidden_layer_sizes": [(128,), (256, 128), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 3e-3],
        },
        **common,
    )
    return specs


def ablation_specs() -> list[dict]:
    """Feature-block and feature-set ablations on two representative models."""
    specs = []
    for model, params in (
        ("lgbm", {"n_estimators": 600, "num_leaves": 31, "learning_rate": 0.05}),
        ("xgb", {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05}),
    ):
        for features in ("full", "top120", "top60"):
            for target_feats in (True, False):
                specs.append(
                    {
                        "model": model,
                        "params": dict(params),
                        "features": features,
                        "target_feats": target_feats,
                        "repeats": FINAL_REPEATS,
                    }
                )
    return specs


def probe_ablation_specs() -> list[dict]:
    """Does feature block F12 (auxiliary-corpus probes) earn its place?"""
    specs = []
    for model, params in (
        ("lgbm", {"n_estimators": 800, "num_leaves": 31, "learning_rate": 0.03,
                  "colsample_bytree": 0.5, "subsample": 0.8, "subsample_freq": 1}),
        ("xgb", {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.06,
                 "colsample_bytree": 0.5, "subsample": 0.9}),
        ("rf", {"n_estimators": 1200, "max_features": "sqrt", "min_samples_leaf": 1}),
    ):
        for probes in (True, False):
            specs.append(
                {
                    "model": model,
                    "params": dict(params),
                    "features": "full",
                    "target_feats": False,
                    "probes": probes,
                    "repeats": FINAL_REPEATS,
                }
            )
    return specs


STAGES = {
    "smoke": lambda: [
        {"model": m, "params": p, "features": "full", "target_feats": True,
         "repeats": {"group": 1, "plain": 1}}
        for m, p in (
            ("lgbm", {"n_estimators": 300, "learning_rate": 0.05}),
            ("xgb", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05}),
            ("rf", {"n_estimators": 400}),
            ("knn", {"n_neighbors": 5, "weights": "distance"}),
        )
    ],
    "xgb": xgb_grid,
    "lgbm": lgbm_grid,
    "cat": cat_grid,
    "rf": rf_grid,
    "et": et_grid,
    "hgb": hgb_grid,
    "classic": classic_grid,
    "ablation": ablation_specs,
    "probe_ablation": probe_ablation_specs,
}


def stage_specs(stage: str) -> list[dict]:
    if stage == "all_trees":
        return xgb_grid() + lgbm_grid() + cat_grid() + rf_grid() + et_grid() + hgb_grid()
    if stage not in STAGES:
        raise ValueError(f"unknown stage '{stage}'; known: {sorted(STAGES)}")
    return STAGES[stage]()
