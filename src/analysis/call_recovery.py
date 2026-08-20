"""Item 3 / D branch — can flows be regrouped into the calls that produced them?

The remaining errors sit on flows that are individually unlabelable: a call opens
several UDP flows, some carrying media and separable, others carrying signalling
and looking the same whichever mode the call is in. Their label is a property of
the call, not of the flow. If flows can be regrouped by call at prediction time,
a confident sibling can answer for an ambiguous one - the classic correlated-flow
argument in traffic classification.

No call identifier is published, so the pseudo-group key stands in for call
membership throughout. That makes this an optimistic proxy, and the gates below
are set accordingly:

    D1  same-call scorer, evaluated on held-out groups          gate: AUC >= 0.75
    D2  clustering quality against the pseudo-group partition   gate: application purity >= 0.90
    D3  correlated-flow aggregation of the stored probabilities
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config as C
import io_utils as IO
from features.pipeline import Cache

OUT = C.REPORTS_DIR / "analysis"

# A compact subspace keeps the pair representation small enough to enumerate.
PAIR_COLS = [
    "len_mean", "len_std", "len_max", "len_min", "len_median", "len_entropy",
    "log_duration", "logiat_mean", "frac_mtu", "frac_tiny", "frac_audio",
    "log_bytes_per_sec", "n_groups_1000us", "len_0", "len_4",
]


def _pair_features(Z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Symmetric pair representation: absolute difference and mean."""
    return np.hstack([np.abs(Z[i] - Z[j]), (Z[i] + Z[j]) / 2.0])


def _sample_pairs(groups: np.ndarray, idx: np.ndarray, rng, n_neg_per_pos: int = 3):
    """All positive pairs inside the given rows, plus sampled negatives."""
    by_group: dict[str, list[int]] = {}
    for i in idx:
        by_group.setdefault(groups[i], []).append(int(i))
    pos = [
        (a, b)
        for members in by_group.values()
        if len(members) > 1
        for k, a in enumerate(members)
        for b in members[k + 1 :]
    ]
    if not pos:
        return np.empty((0, 2), int), np.empty(0, int)
    n_neg = len(pos) * n_neg_per_pos
    neg = []
    while len(neg) < n_neg:
        a, b = rng.choice(idx, 2, replace=False)
        if groups[a] != groups[b]:
            neg.append((int(a), int(b)))
    pairs = np.array(pos + neg)
    labels = np.r_[np.ones(len(pos), int), np.zeros(len(neg), int)]
    return pairs, labels


def run(min_auc: float = 0.75, min_purity: float = 0.90) -> dict:
    import lightgbm as lgb

    OUT.mkdir(parents=True, exist_ok=True)
    cache = Cache.get()
    train = cache.train
    y = cache.y
    groups = IO.group_keys(train)
    cols = [c for c in PAIR_COLS if c in cache.base_train.columns]
    Z = StandardScaler().fit_transform(cache.base_train[cols].to_numpy())
    rng = np.random.RandomState(C.SEED)
    report: dict = {"n_flows": len(train), "n_pseudo_groups": int(len(set(groups)))}

    # ---- D1: same-call scorer, trained and tested on disjoint groups --------
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    cut = int(len(uniq) * 0.7)
    tr_groups, te_groups = set(uniq[:cut]), set(uniq[cut:])
    tr_idx = np.array([i for i, g in enumerate(groups) if g in tr_groups])
    te_idx = np.array([i for i, g in enumerate(groups) if g in te_groups])

    pairs_tr, lab_tr = _sample_pairs(groups, tr_idx, rng)
    pairs_te, lab_te = _sample_pairs(groups, te_idx, rng)
    report["D1"] = {"train_pairs": int(len(lab_tr)), "test_pairs": int(len(lab_te))}

    if len(lab_tr) < 50 or len(lab_te) < 20:
        report["D1"]["auc"] = None
        report["D1"]["gate_passed"] = False
        report["verdict"] = "D1 skipped: too few same-call pairs to learn from"
        (C.REPORTS_DIR / "d_branch.json").write_text(json.dumps(report, indent=2, default=float))
        return report

    scorer = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        random_state=C.SEED, n_jobs=4, verbose=-1,
    )
    scorer.fit(_pair_features(Z, pairs_tr[:, 0], pairs_tr[:, 1]), lab_tr)
    p_te = scorer.predict_proba(_pair_features(Z, pairs_te[:, 0], pairs_te[:, 1]))[:, 1]
    auc = float(roc_auc_score(lab_te, p_te))
    report["D1"]["auc"] = auc
    report["D1"]["gate"] = f"AUC >= {min_auc}"
    report["D1"]["gate_passed"] = bool(auc >= min_auc)

    if auc < min_auc:
        report["verdict"] = (
            f"D branch stopped at D1: same-call AUC {auc:.3f} below the {min_auc} gate"
        )
        (C.REPORTS_DIR / "d_branch.json").write_text(json.dumps(report, indent=2, default=float))
        return report

    # ---- D2: cluster the whole training set under the learned metric --------
    n = len(train)
    ii, jj = np.triu_indices(n, k=1)
    scores = np.zeros(len(ii), dtype=np.float32)
    step = 200_000
    for s in range(0, len(ii), step):
        sl = slice(s, s + step)
        scores[sl] = scorer.predict_proba(_pair_features(Z, ii[sl], jj[sl]))[:, 1]
    dist = np.ones((n, n), dtype=np.float32)
    dist[ii, jj] = 1.0 - scores
    dist[jj, ii] = 1.0 - scores
    np.fill_diagonal(dist, 0.0)

    n_clusters = int(len(set(groups)))
    cl = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist)

    true_codes = pd.factorize(groups)[0]
    app = pd.Series([lab.rsplit("_", 1)[0] for lab in train["label"]])
    purity = float(
        pd.DataFrame({"c": cl, "a": app}).groupby("c")["a"]
        .apply(lambda s: s.value_counts().iloc[0] / len(s))
        .mean()
    )
    label_purity = float(
        pd.DataFrame({"c": cl, "y": y}).groupby("c")["y"]
        .apply(lambda s: s.value_counts().iloc[0] / len(s))
        .mean()
    )
    sizes = pd.Series(cl).value_counts()
    report["D2"] = {
        "n_clusters": n_clusters,
        "adjusted_rand_vs_pseudo_groups": float(adjusted_rand_score(true_codes, cl)),
        "mean_application_purity": purity,
        "mean_label_purity": label_purity,
        "singleton_clusters": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "gate": f"application purity >= {min_purity}",
        "gate_passed": bool(purity >= min_purity),
    }
    if purity < min_purity:
        report["verdict"] = (
            f"D branch stopped at D2: cluster application purity {purity:.3f} "
            f"below the {min_purity} gate"
        )
        (C.REPORTS_DIR / "d_branch.json").write_text(json.dumps(report, indent=2, default=float))
        return report

    # ---- D3: aggregate the stored probabilities within clusters -------------
    ens = json.loads((C.MODELS_DIR / "ensemble.json").read_text())
    best = max(ens["members"], key=lambda m: m["group_acc"])
    prob = np.load(best["path"], allow_pickle=False)["oof_group"].astype(np.float64)
    base_pred = prob.argmax(axis=1)
    base_acc = float((base_pred == y).mean())

    arms = {}
    for name in ("logsum", "confidence_vote"):
        pred = base_pred.copy()
        for c in np.unique(cl):
            m = np.where(cl == c)[0]
            if len(m) < 2:
                continue
            if name == "logsum":
                agg = np.log(np.clip(prob[m], 1e-9, 1)).sum(axis=0)
            else:
                w = prob[m].max(axis=1) ** 4
                agg = (prob[m] * w[:, None]).sum(axis=0)
            pred[m] = int(agg.argmax())
        arms[name] = {
            "accuracy": float((pred == y).mean()),
            "delta": float((pred == y).mean() - base_acc),
            "rows_changed": int((pred != base_pred).sum()),
        }

    # The ceiling: what aggregation would buy with perfect call knowledge.
    oracle = base_pred.copy()
    for g in np.unique(true_codes):
        m = np.where(true_codes == g)[0]
        if len(m) < 2:
            continue
        agg = np.log(np.clip(prob[m], 1e-9, 1)).sum(axis=0)
        oracle[m] = int(agg.argmax())
    report["D3"] = {
        "baseline_accuracy": base_acc,
        "arms": arms,
        "oracle_grouping_accuracy": float((oracle == y).mean()),
        "oracle_delta": float((oracle == y).mean() - base_acc),
    }
    best_arm = max(arms, key=lambda k: arms[k]["accuracy"])
    report["verdict"] = (
        f"D3 best arm '{best_arm}' at {arms[best_arm]['accuracy']:.4f} "
        f"({arms[best_arm]['delta']:+.4f} vs {base_acc:.4f})"
    )
    (C.REPORTS_DIR / "d_branch.json").write_text(json.dumps(report, indent=2, default=float))
    return report
