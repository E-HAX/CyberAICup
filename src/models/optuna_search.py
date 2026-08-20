"""Optuna search spaces and the ask/tell batching used to distribute them.

The grid stage covers the parameter space coarsely and in parallel; Optuna then
refines the continuous neighbourhood of the winner, which a grid wastes budget
on. Trials are distributed by asking the study for a batch, evaluating that
batch across containers, and telling the results back - no shared database and
no concurrent-writer semantics on a Volume are required. The study itself is
journalled so a killed driver resumes instead of restarting.
"""

from __future__ import annotations

import config as C

JOURNAL = C.CACHE / "optuna"


def suggest(model: str, trial) -> dict:
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 2.0, log=True),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced"]
            ),
        }
    if model == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 7, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced"]
            ),
        }
    if model == "cat":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced"]
            ),
        }
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=200),
            "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24, 32]),
            "max_features": trial.suggest_float("max_features", 0.05, 0.8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced", "balanced_subsample"]
            ),
        }
    if model == "et":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=200),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 32]),
            "max_features": trial.suggest_float("max_features", 0.05, 0.8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced"]
            ),
        }
    if model == "svm":
        return {
            "C": trial.suggest_float("C", 0.3, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 0.5, log=True),
            "class_weight_mode": trial.suggest_categorical(
                "class_weight_mode", ["none", "balanced"]
            ),
        }
    raise ValueError(f"no Optuna space defined for '{model}'")


def spec_from_trial(model: str, trial, features_choices=("full", "top120"), probes: bool = False) -> dict:
    from models.grids import SCREEN_REPEATS

    return {
        "model": model,
        "params": suggest(model, trial),
        "features": trial.suggest_categorical("features", list(features_choices)),
        "target_feats": trial.suggest_categorical("target_feats", [True, False]),
        "probes": probes,
        "repeats": SCREEN_REPEATS,
    }
