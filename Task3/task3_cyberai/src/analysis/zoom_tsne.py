"""Prediction-space visualisation and a focused study of the Zoom voice/video pair.

Two questions drive this module:

1. What does the model's decision surface look like in the engineered feature
   space - where do the out-of-fold errors sit relative to the class manifolds?
2. Why is Zoom_voice versus Zoom_video responsible for roughly a quarter of all
   errors, and is the pair separable at all from five packets?

Everything here reads the artifacts the search already produced; nothing is
re-searched or re-tuned.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import config as C
import io_utils as IO
from features.pipeline import Cache

OUT = C.REPORTS_DIR / "analysis"
PALETTE = sns.color_palette("tab10", 10)
CLASS_COLOR = dict(zip(C.LABELS, PALETTE))


def _best_oof() -> tuple[np.ndarray, str]:
    """The out-of-fold matrix of the strongest member with stored test probs."""
    ens = json.loads((C.MODELS_DIR / "ensemble.json").read_text())
    best = max(ens["members"], key=lambda m: m["group_acc"])
    z = np.load(best["path"], allow_pickle=False)
    return z["oof_group"], best["name"]


def _tsne(X: np.ndarray, seed: int = C.SEED) -> np.ndarray:
    return TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        random_state=seed, max_iter=1500,
    ).fit_transform(X)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Cache.get()
    train = cache.train
    y = cache.y
    base = cache.base_train
    oof, source = _best_oof()
    pred = oof.argmax(axis=1)
    conf = oof.max(axis=1)
    correct = pred == y

    Xs = StandardScaler().fit_transform(base.to_numpy())
    Z = _tsne(Xs)
    labels = np.asarray(C.LABELS)

    report: dict = {"oof_source": source, "accuracy": float(correct.mean())}

    # ---- panel 1: the whole prediction space --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))

    ax = axes[0][0]
    for k, lab in enumerate(C.LABELS):
        m = y == k
        ax.scatter(Z[m, 0], Z[m, 1], s=11, alpha=.8, color=CLASS_COLOR[lab], label=lab)
    ax.set_title("True class", fontsize=11)
    ax.legend(fontsize=7, ncol=2, markerscale=1.6)

    ax = axes[0][1]
    for k, lab in enumerate(C.LABELS):
        m = pred == k
        ax.scatter(Z[m, 0], Z[m, 1], s=11, alpha=.8, color=CLASS_COLOR[lab], label=lab)
    ax.set_title(f"Predicted class (grouped out-of-fold, {source})", fontsize=11)

    ax = axes[1][0]
    ax.scatter(Z[correct, 0], Z[correct, 1], s=9, alpha=.35, color="#B7C4C2", label="correct")
    sc = ax.scatter(
        Z[~correct, 0], Z[~correct, 1], s=26, c=conf[~correct], cmap="magma_r",
        vmin=0, vmax=1, edgecolors="k", linewidths=.3, label="error",
    )
    plt.colorbar(sc, ax=ax, label="model confidence in the wrong answer")
    ax.set_title(f"Errors ({int((~correct).sum())} of {len(y)}) shaded by confidence", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    zoom_mask = np.isin(y, [C.LABEL_TO_ID["Zoom_voice"], C.LABEL_TO_ID["Zoom_video"]])
    ax.scatter(Z[~zoom_mask, 0], Z[~zoom_mask, 1], s=8, alpha=.18, color="#C4CFCD", label="other classes")
    for lab, colour in (("Zoom_voice", "#0E8F82"), ("Zoom_video", "#9A5B00")):
        k = C.LABEL_TO_ID[lab]
        m_ok = (y == k) & correct
        m_bad = (y == k) & ~correct
        ax.scatter(Z[m_ok, 0], Z[m_ok, 1], s=16, color=colour, alpha=.85, label=f"{lab} correct")
        ax.scatter(Z[m_bad, 0], Z[m_bad, 1], s=42, facecolors="none", edgecolors=colour,
                   linewidths=1.4, label=f"{lab} error")
    ax.set_title("The Zoom pair: filled = correct, ringed = misclassified", fontsize=11)
    ax.legend(fontsize=8)

    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        "t-SNE of the 241-dimensional engineered feature space, coloured four ways", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(OUT / "tsne_predictions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- panel 2: inside the Zoom pair --------------------------------------
    zi = np.where(zoom_mask)[0]
    y_bin = (y[zi] == C.LABEL_TO_ID["Zoom_video"]).astype(int)
    Xz = StandardScaler().fit_transform(base.iloc[zi].to_numpy())
    Zz = _tsne(Xz, seed=C.SEED + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    ax = axes[0]
    for v, lab, colour in ((0, "Zoom_voice", "#0E8F82"), (1, "Zoom_video", "#9A5B00")):
        m = y_bin == v
        ax.scatter(Zz[m, 0], Zz[m, 1], s=26, color=colour, alpha=.85, label=lab)
    ax.set_title("Zoom flows only, true label", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    ok = correct[zi]
    ax.scatter(Zz[ok, 0], Zz[ok, 1], s=22, color="#B7C4C2", label="correct")
    ax.scatter(Zz[~ok, 0], Zz[~ok, 1], s=40, color="#A03225", label="misclassified")
    ax.set_title(f"Zoom flows only, errors ({int((~ok).sum())} of {len(zi)})", fontsize=11)
    ax.legend(fontsize=9)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Is the Zoom voice/video boundary visible at all in feature space?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "tsne_zoom.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Zoom-specific diagnostics ------------------------------------------
    mi = mutual_info_classif(base.iloc[zi].to_numpy(), y_bin, random_state=C.SEED)
    mi_rank = pd.Series(mi, index=base.columns).sort_values(ascending=False)
    mi_rank.to_frame("mutual_info").to_csv(OUT / "zoom_feature_ranking.csv")

    # A binary specialist on the Zoom subset, scored on the same grouped folds
    # restricted to those rows, tells us whether the pair is separable at all.
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    pred_bin = np.zeros(len(zi), dtype=int)
    prob_bin = np.zeros(len(zi))
    for tr, va in cv.split(Xz, y_bin):
        clf = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=10,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.7,
            random_state=C.SEED, n_jobs=1, verbose=-1,
        )
        clf.fit(base.iloc[zi[tr]].to_numpy(), y_bin[tr])
        p = clf.predict_proba(base.iloc[zi[va]].to_numpy())[:, 1]
        prob_bin[va] = p
        pred_bin[va] = (p > .5).astype(int)

    flat_zoom_acc = float(correct[zi].mean())
    report["zoom"] = {
        "n_flows": int(len(zi)),
        "n_voice": int((y_bin == 0).sum()),
        "n_video": int((y_bin == 1).sum()),
        "flat_10class_accuracy_on_zoom": flat_zoom_acc,
        "binary_specialist_accuracy": float(accuracy_score(y_bin, pred_bin)),
        "binary_specialist_macro_f1": float(f1_score(y_bin, pred_bin, average="macro")),
        "majority_baseline": float(max(y_bin.mean(), 1 - y_bin.mean())),
        "top_features": {k: float(v) for k, v in mi_rank.head(12).items()},
    }

    # How much of the Zoom confusion is the constant-size keepalive population?
    const = base["len_all_equal"].to_numpy()[zi] > 0.5
    report["zoom"]["constant_size_flows"] = int(const.sum())
    report["zoom"]["error_rate_constant_size"] = (
        float((~correct[zi][const]).mean()) if const.any() else None
    )
    report["zoom"]["error_rate_variable_size"] = float((~correct[zi][~const]).mean())

    # Where do the two Zoom classes actually overlap in packet size?
    lens = base.loc[base.index[zi], ["len_mean", "len_max", "len_min", "frac_mtu"]]
    summary = lens.assign(cls=np.where(y_bin == 1, "Zoom_video", "Zoom_voice")).groupby("cls").describe(
        percentiles=[.1, .5, .9]
    )
    summary.to_csv(OUT / "zoom_length_summary.csv")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, feat in zip(axes.ravel(), list(mi_rank.head(8).index)):
        v = base.iloc[zi][feat].to_numpy()
        for cls, colour in ((0, "#0E8F82"), (1, "#9A5B00")):
            sns.kdeplot(x=v[y_bin == cls], ax=ax, fill=True, alpha=.35, color=colour,
                        label="Zoom_video" if cls else "Zoom_voice", bw_adjust=.7)
        ax.set_title(feat, fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("The eight most discriminative features inside the Zoom pair", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "zoom_feature_densities.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- per-class error budget ---------------------------------------------
    budget = []
    for k, lab in enumerate(C.LABELS):
        m = y == k
        budget.append({
            "class": lab,
            "support": int(m.sum()),
            "errors": int((~correct[m]).sum()),
            "error_share_of_total": float((~correct[m]).sum() / max(1, (~correct).sum())),
            "mean_confidence_when_wrong": float(conf[m & ~correct].mean()) if (m & ~correct).any() else 0.0,
        })
    pd.DataFrame(budget).to_csv(OUT / "error_budget.csv", index=False)
    report["error_budget"] = budget
    report["total_errors"] = int((~correct).sum())

    (C.REPORTS_DIR / "zoom_analysis.json").write_text(json.dumps(report, indent=2, default=float))
    return report
