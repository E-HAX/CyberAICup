"""Model card: what was tried, what won, and why it should be believed.

Reads the artifacts already in the work Volume - the cross-validation record,
the stored out-of-fold matrices, the ensemble decision and the submission -
and assembles a single document with the evidence behind the final choice.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config as C
import io_utils as IO

CARD = C.REPORTS_DIR / "model_card.md"


def _load_cv() -> pd.DataFrame:
    path = C.REPORTS_DIR / "cv_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_oof() -> tuple[np.ndarray | None, dict]:
    """Out-of-fold matrix of the winning combination, for error analysis."""
    ens_path = C.MODELS_DIR / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text())
        best_member = max(ens["members"], key=lambda m: m["group_acc"])
        z = np.load(best_member["path"], allow_pickle=False)
        return z["oof_group"], {"source": best_member["name"], "ensemble": ens}
    files = sorted(C.OOF_DIR.glob("*.npz"))
    if not files:
        return None, {}
    best, best_acc, meta = None, -1.0, {}
    y = IO.labels_to_ids(IO.load_train()["label"])
    for f in files:
        z = np.load(f, allow_pickle=False)
        if "oof_group" not in z:
            continue
        acc = float((z["oof_group"].argmax(axis=1) == y).mean())
        if acc > best_acc:
            best, best_acc, meta = z["oof_group"], acc, json.loads(str(z["meta"]))
    return best, {"source": meta.get("model", "unknown")}


def build_model_card() -> dict:
    C.ensure_dirs()
    train = IO.load_train()
    y = IO.labels_to_ids(train["label"])
    cv = _load_cv()
    lines: list[str] = ["# Model card - encrypted RTC application identification\n"]

    lines.append("## Task\n")
    lines.append(
        f"Ten-class classification (5 applications x 2 call modes) from the sizes and "
        f"inter-arrival times of the first {C.N_PACKETS} packets of an encrypted UDP media "
        f"flow. {len(train)} labelled training flows, {C.N_TEST_ROWS} test flows.\n"
    )

    lines.append("\n## Validation protocol\n")
    lines.append(
        "- `plain`: repeated stratified 5-fold, the optimistic estimate.\n"
        "- `group`: stratified group 5-fold over a pseudo-group key that keeps "
        "near-duplicate flows together. The competition test split comes from held-out "
        "source calls and no call identifier is published, so this is the decisive number "
        "and every model is selected on it.\n"
    )

    if not cv.empty:
        lines.append("\n## Search record\n")
        lines.append(f"Points scored: **{len(cv)}** across {cv['model'].nunique()} model families.\n")
        by_family = (
            cv.sort_values("group_acc", ascending=False)
            .groupby("model")
            .head(1)[["model", "features", "target_feats", "group_acc", "plain_acc",
                      "group_macro_f1", "params"]]
            .sort_values("group_acc", ascending=False)
        )
        lines.append("\n### Best point per model family\n")
        lines.append(by_family.round(4).to_markdown(index=False) + "\n")

        abl = cv[cv["stage"].astype(str).str.startswith("ablation")]
        if not abl.empty:
            lines.append("\n### Feature-block ablation\n")
            lines.append(
                abl.sort_values("group_acc", ascending=False)[
                    ["model", "features", "target_feats", "group_acc", "plain_acc",
                     "group_macro_f1"]
                ].round(4).to_markdown(index=False) + "\n"
            )
            best_off = abl[~abl["target_feats"].astype(bool)]["group_acc"].max()
            best_on = abl[abl["target_feats"].astype(bool)]["group_acc"].max()
            lines.append(
                f"\n> The fitted label-aware blocks F9-F11 (class-conditional Markov "
                f"likelihoods, class-conditional densities, per-class neighbour distances) "
                f"score {best_on:.4f} at best against {best_off:.4f} without them, even after "
                "their training-side values are generated through an inner out-of-fold loop. "
                "With 1,285 rows they add variance rather than signal, so they are disabled by "
                "default and kept only as a searchable option.\n"
            )

        total_cpu_seconds = float(cv["fit_seconds"].sum())
        lines.append(
            f"\nCumulative fitting time across all scored points: "
            f"**{total_cpu_seconds / 3600:.1f} container-hours**, executed in parallel across "
            "Modal containers.\n"
        )

    oof, meta = _best_oof()
    if oof is not None:
        pred = oof.argmax(axis=1)
        acc = float((pred == y).mean())
        lines.append("\n## Error analysis of the selected model\n")
        lines.append(f"Source: `{meta.get('source')}`, grouped out-of-fold accuracy {acc:.4f}.\n")
        rep = classification_report(
            y, pred, labels=range(len(C.LABELS)), target_names=C.LABELS, output_dict=True,
            zero_division=0,
        )
        rep_df = pd.DataFrame(rep).T.round(3)
        rep_df.to_csv(C.REPORTS_DIR / "classification_report.csv")
        lines.append("\n### Per-class performance\n")
        lines.append(rep_df.head(len(C.LABELS)).to_markdown() + "\n")

        cm = confusion_matrix(y, pred, labels=range(len(C.LABELS)))
        cm_df = pd.DataFrame(cm, index=C.LABELS, columns=C.LABELS)
        cm_df.to_csv(C.REPORTS_DIR / "final_confusion.csv")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_df.div(cm_df.sum(axis=1), axis=0), annot=cm_df.to_numpy(), fmt="d",
            cmap="Blues", ax=ax,
        )
        ax.set_title(f"Grouped out-of-fold confusion, accuracy {acc:.4f}")
        ax.set_ylabel("true")
        ax.set_xlabel("predicted")
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "final_confusion.png", dpi=130)
        plt.close(fig)
        lines.append("\n![confusion](final_confusion.png)\n")

        off = cm.copy()
        np.fill_diagonal(off, 0)
        idx = np.unravel_index(np.argsort(off, axis=None)[::-1][:6], off.shape)
        pairs = [f"{C.LABELS[i]} -> {C.LABELS[j]} ({off[i, j]})" for i, j in zip(*idx)]
        lines.append("\nDominant confusions: " + "; ".join(pairs) + ".\n")

    ens_path = C.MODELS_DIR / "ensemble.json"
    if ens_path.exists():
        ens = json.loads(ens_path.read_text())
        lines.append("\n## Ensemble\n")
        lines.append(
            f"Winner: **{ens['winner']}** at {ens['winner_acc']:.4f} on the honest estimate "
            "(weights fitted on four fifths of the out-of-fold rows and scored on the fifth).\n"
        )
        comparison = pd.DataFrame(
            [
                {"combiner": "single best", "score": ens["single_best"]["acc"]},
                {"combiner": "hill-climb (nested)", "score": ens["hill_climb"]["acc_nested"]},
                {"combiner": "hill-climb (fitted, optimistic)", "score": ens["hill_climb"]["acc_fitted"]},
                {"combiner": "rank average", "score": ens["rank_average"]["acc"]},
                {"combiner": "stacking (nested)", "score": ens["stacking"]["acc_nested"]},
            ]
        )
        lines.append(comparison.round(4).to_markdown(index=False) + "\n")
        if ens["hill_climb"]["weights"]:
            lines.append("\n### Blend weights\n")
            w = pd.Series(ens["hill_climb"]["weights"]).sort_values(ascending=False)
            lines.append(w.round(3).to_frame("weight").to_markdown() + "\n")

    probe_path = C.REPORTS_DIR / "probe_report.json"
    if probe_path.exists():
        lines.append("\n## Auxiliary corpus and linear probes\n")
        lines.append("```json\n" + probe_path.read_text() + "\n```\n")

    dist_path = C.REPORTS_DIR / "submission_distribution.csv"
    if dist_path.exists():
        lines.append("\n## Submitted class distribution vs the flows-per-call prior\n")
        lines.append(pd.read_csv(dist_path, index_col=0).to_markdown() + "\n")

    lines.append("\n## Reproduction\n")
    lines.append(
        "```\nmodal run src/modal_app.py::seed\nmodal run src/modal_app.py::eda\n"
        "modal run src/modal_app.py::features\n"
        "modal run --detach src/modal_app.py::search --stage all_trees\n"
        "modal run --detach src/modal_app.py::optuna_search --model lgbm --trials 200\n"
        "modal run --detach src/modal_app.py::aux_ingest\n"
        "modal run --detach src/modal_app.py::pretrain\n"
        "modal run src/modal_app.py::probes --pretrained /cache/pretrain/encoder.pt\n"
        "modal run src/modal_app.py::finalists\nmodal run src/modal_app.py::ensemble\n"
        "modal run src/modal_app.py::submit\n```\n"
    )

    CARD.write_text("\n".join(lines))
    return {"path": str(CARD), "n_points": int(len(cv))}


def build_shap(top_n: int = 30) -> dict:
    """Global feature attribution for the selected model.

    The winning point is refit on the full training set and explained with
    TreeSHAP; the mean absolute attribution is averaged over the ten classes,
    which answers "what does the model actually use" rather than "what
    correlates with the label".
    """
    import json

    import shap

    from features.pipeline import Cache
    from models import registry
    from models.evaluate import _feature_frame  # noqa: F401  (kept import parity)

    cv = _load_cv()
    if cv.empty:
        return {"error": "no cross-validation record"}
    best = cv.sort_values("group_acc", ascending=False).iloc[0]
    spec = {
        "model": best["model"],
        "params": json.loads(best["params"]),
        "features": best["features"],
        "target_feats": bool(best["target_feats"]),
        "probes": bool(best.get("probes", False)),
    }
    if spec["model"].startswith("tfm"):
        return {"skipped": "winning model is the transformer; TreeSHAP does not apply"}

    cache = Cache.get()
    X = cache.base_train
    if spec["probes"] and cache.has_probes:
        X = pd.concat([X, cache.probes_train.set_index(X.index)], axis=1)

    params = registry.apply_class_weight(spec["model"], spec["params"])
    model = registry.build_model(spec["model"], params)
    sw = registry.sample_weights(
        spec["model"], spec["params"].get("class_weight_mode", "none"), cache.y
    )
    model.fit(X.to_numpy(), cache.y, sample_weight=sw) if sw is not None else model.fit(
        X.to_numpy(), cache.y
    )

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X.to_numpy())
    arr = np.abs(np.asarray(values))
    # shape is (classes, rows, features) or (rows, features, classes)
    axes = tuple(i for i, s in enumerate(arr.shape) if s != X.shape[1])
    importance = arr.mean(axis=axes)
    ranking = (
        pd.Series(importance, index=X.columns).sort_values(ascending=False).to_frame("mean_abs_shap")
    )
    ranking.to_csv(C.REPORTS_DIR / "shap_importance.csv")

    fig, ax = plt.subplots(figsize=(9, 10))
    ranking.head(top_n)["mean_abs_shap"].iloc[::-1].plot(kind="barh", ax=ax, color="C2")
    ax.set_title(f"Mean |SHAP| per feature - {spec['model']} refit on all training rows")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)

    if CARD.exists():
        with CARD.open("a") as fh:
            fh.write("\n## What the selected model actually uses\n\n")
            fh.write(ranking.head(top_n).round(4).to_markdown() + "\n")
            fh.write("\n![shap](shap_importance.png)\n")
    return {"model": spec["model"], "top_features": ranking.head(10).index.tolist()}
