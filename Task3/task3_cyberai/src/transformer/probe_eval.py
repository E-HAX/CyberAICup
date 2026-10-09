"""Held-out evaluation of the frozen-encoder linear probes.

`probe.py` fits the probes and emits their outputs as feature block F12; this
module answers the separate question of how good those probes actually are:

1. probe quality on a held-out split of the auxiliary corpus, so the numbers
   are not the fit-to-itself accuracies reported during feature construction;
2. cross-corpus transfer - does an application probe trained only on MIRAGE
   mobile captures recognise the same application in the competition's lab
   Wi-Fi captures, which is the entire premise of the auxiliary dataset;
3. how much competition-relevant signal the 46 probe columns carry on their
   own, measured both by mutual information and by a model trained on nothing
   else.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    r2_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as C
import io_utils as IO
from transformer.mirage_stream import load_shards
from transformer.probe import (
    MAX_AUX,
    N_ALIGN_PACKETS,
    TOP_APPS,
    _aux_targets,
    _embed,
    _load_encoder,
)

# MIRAGE writes Google Meet as "Meet"; the competition writes "GoogleMeet".
APP_ALIAS = {"Meet": "GoogleMeet"}
SHARED_APPS = ["Discord", "GoogleMeet", "Messenger", "WhatsApp", "Zoom"]


def evaluate(pretrained: str, max_aux: int = MAX_AUX, test_size: float = 0.2) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_encoder(pretrained)
    model.to(device)

    aux = load_shards()
    rng = np.random.RandomState(C.SEED)
    sel = rng.permutation(len(aux["lengths"]))[: min(len(aux["lengths"]), max_aux)]
    keep = aux["mask"][sel][:, :N_ALIGN_PACKETS].all(axis=1)
    sel = sel[keep]
    aux_L = aux["lengths"][sel][:, :N_ALIGN_PACKETS].astype(np.float64)
    aux_G = aux["gaps"][sel][:, :N_ALIGN_PACKETS].astype(np.float64)
    aux_G[:, 0] = 0.0
    aux_labels = aux["label_code"][sel]
    aux_names = np.asarray([APP_ALIAS.get(str(n), str(n)) for n in aux["label_names"]])
    is_udp = aux["is_udp"][sel] if "is_udp" in aux else np.ones(len(sel), dtype=bool)

    E_aux = _embed(model, aux_L, aux_G, device)
    targets = _aux_targets(aux_L, aux_G)
    scaler = StandardScaler().fit(E_aux)
    A = scaler.transform(E_aux)

    report: dict = {
        "aux_flows_used": int(len(A)),
        "aux_udp_share": float(is_udp.mean()),
        "aux_apps": int(len(set(aux_labels))),
        "embedding_dim": int(A.shape[1]),
    }

    # -- P1: application family, held out ----------------------------------
    counts = np.bincount(aux_labels, minlength=len(aux_names))
    top = np.argsort(counts)[::-1][:TOP_APPS]
    mask = np.isin(aux_labels, top)
    remap = {v: i for i, v in enumerate(top)}
    y_app = np.array([remap[v] for v in aux_labels[mask]])
    Xtr, Xte, ytr, yte = train_test_split(
        A[mask], y_app, test_size=test_size, random_state=C.SEED, stratify=y_app
    )
    p1 = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    proba = p1.predict_proba(Xte)
    names_in_order = [str(aux_names[top[c]]) for c in range(len(top))]
    report["p1"] = {
        "n_classes": int(len(top)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "majority_baseline": float(np.bincount(yte).max() / len(yte)),
        "test_top1": float(accuracy_score(yte, proba.argmax(axis=1))),
        "test_top3": float(top_k_accuracy_score(yte, proba, k=3, labels=np.arange(len(top)))),
        "train_top1": float(p1.score(Xtr, ytr)),
        "per_class": classification_report(
            yte, proba.argmax(axis=1), labels=np.arange(len(top)),
            target_names=names_in_order, output_dict=True, zero_division=0,
        ),
        "class_names": names_in_order,
    }

    # -- P2: media modality, held out --------------------------------------
    y_vid = targets["video_like"]
    Xtr, Xte, ytr, yte = train_test_split(
        A, y_vid, test_size=test_size, random_state=C.SEED, stratify=y_vid
    )
    p2 = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    pv = p2.predict_proba(Xte)[:, 1]
    report["p2"] = {
        "positive_rate": float(y_vid.mean()),
        "test_acc": float(accuracy_score(yte, pv > 0.5)),
        "test_auc": float(roc_auc_score(yte, pv)),
    }

    # -- P3: bitrate and pacing regressions, held out -----------------------
    report["p3"] = {}
    for name in ("log_bitrate", "log_mean_gap", "frac_mtu"):
        Xtr, Xte, ytr, yte = train_test_split(
            A, targets[name], test_size=test_size, random_state=C.SEED
        )
        reg = Ridge(alpha=1.0).fit(Xtr, ytr)
        report["p3"][name] = {
            "test_r2": float(r2_score(yte, reg.predict(Xte))),
            "train_r2": float(reg.score(Xtr, ytr)),
        }

    # -- P4: padding quantum, held out --------------------------------------
    # As originally defined the target was the modulus with the largest share
    # of divisible packets, which is always 2 - every byte count is divisible
    # by 2 at least as often as by 4. The target had a single class, the probe
    # was skipped, and P4 contributed no columns to the feature block. The
    # corrected target below asks the sharper question: what is the strongest
    # power-of-two alignment the payload sizes actually obey?
    y_pad_original = targets["padding_class"]
    report["p4_original"] = {
        "n_classes_present": int(len(np.unique(y_pad_original))),
        "degenerate": bool(len(np.unique(y_pad_original)) < 2),
        "note": "single-class target; probe skipped, contributed 0 features",
    }

    aligned = np.stack(
        [(aux_L % m == 0).mean(axis=1) >= 0.8 for m in (2, 4, 8, 16)], axis=1
    )
    y_pad = aligned.sum(axis=1).astype(int)  # 0 = unaligned, 4 = 16-byte aligned
    if len(np.unique(y_pad)) > 1:
        Xtr, Xte, ytr, yte = train_test_split(
            A, y_pad, test_size=test_size, random_state=C.SEED, stratify=y_pad
        )
        p4 = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
        report["p4_corrected"] = {
            "target": "strongest power-of-two alignment obeyed by >=80% of packets",
            "n_classes": int(len(np.unique(y_pad))),
            "majority_baseline": float(np.bincount(yte).max() / len(yte)),
            "test_acc": float(p4.score(Xte, yte)),
            "class_shares": {
                f"level_{c}": float((y_pad == c).mean()) for c in np.unique(y_pad)
            },
        }
    else:
        report["p4_corrected"] = {"degenerate": True}

    # -- cross-corpus transfer of the application probe ---------------------
    train = IO.load_train()
    rtc_L = train[C.LEN_COLS].to_numpy(float)
    rtc_T = train[C.TIME_COLS].to_numpy(float)
    rtc_G = np.zeros_like(rtc_T)
    rtc_G[:, 1:] = np.diff(rtc_T, axis=1)
    R = scaler.transform(_embed(model, rtc_L, rtc_G, device))
    rtc_app, rtc_mode = IO.split_app_mode(train["label"])

    p1_full = LogisticRegression(max_iter=3000, C=1.0).fit(A[mask], y_app)
    rtc_proba = p1_full.predict_proba(R)
    pred_app = np.asarray(names_in_order)[rtc_proba.argmax(axis=1)]
    cross = pd.crosstab(rtc_app.to_numpy(), pred_app)
    cross.to_csv(C.REPORTS_DIR / "probe_transfer_crosstab.csv")

    shared_cols = [c for c in SHARED_APPS if c in cross.columns]
    restricted = np.asarray(
        [names_in_order.index(a) for a in SHARED_APPS if a in names_in_order]
    )
    restricted_names = [names_in_order[i] for i in restricted]
    pred_restricted = np.asarray(restricted_names)[rtc_proba[:, restricted].argmax(axis=1)]
    report["transfer"] = {
        "shared_applications": SHARED_APPS,
        "share_predicted_into_shared_apps": float(np.isin(pred_app, SHARED_APPS).mean()),
        "agreement_20way": float((pred_app == rtc_app.to_numpy()).mean()),
        "agreement_restricted_to_shared": float(
            (pred_restricted == rtc_app.to_numpy()).mean()
        ),
        "chance_restricted": 1.0 / len(restricted_names),
        "crosstab_shared": cross[shared_cols].to_dict() if shared_cols else {},
    }

    # -- how much of the competition label do the probe columns carry? ------
    probes_train = pd.read_parquet(C.FEATURES_DIR / "probes_train.parquet")
    y10 = IO.labels_to_ids(train["label"])
    mi = mutual_info_classif(probes_train.to_numpy(), y10, random_state=C.SEED)
    mi_series = pd.Series(mi, index=probes_train.columns).sort_values(ascending=False)
    mi_series.to_frame("mutual_info").to_csv(C.REPORTS_DIR / "probe_mutual_info.csv")
    report["mutual_info_top15"] = {k: float(v) for k, v in mi_series.head(15).items()}

    base = pd.read_parquet(C.FEATURES_DIR / "base_train.parquet")
    base_mi = mutual_info_classif(base.to_numpy(), y10, random_state=C.SEED)
    report["mutual_info_context"] = {
        "best_probe_feature": float(mi_series.max()),
        "best_base_feature": float(np.max(base_mi)),
        "median_probe_feature": float(mi_series.median()),
        "median_base_feature": float(np.median(base_mi)),
    }

    # A model trained on the probe columns alone, scored on the same grouped
    # folds every other model in the search used.
    import lightgbm as lgb

    from models import cv as cvmod

    folds = cvmod.load_folds()["group"][: C.N_SPLITS]
    preds = np.zeros(len(y10), dtype=int)
    for tr_idx, va_idx in folds:
        clf = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            random_state=C.SEED, n_jobs=1, verbose=-1,
        )
        clf.fit(probes_train.iloc[tr_idx].to_numpy(), y10[tr_idx])
        preds[va_idx] = clf.predict(probes_train.iloc[va_idx].to_numpy())
    report["probes_only_model"] = {
        "grouped_cv_accuracy": float(accuracy_score(y10, preds)),
        "majority_baseline": float(np.bincount(y10).max() / len(y10)),
        "full_feature_reference": 0.8249,
    }

    (C.REPORTS_DIR / "probe_evaluation.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    return report
