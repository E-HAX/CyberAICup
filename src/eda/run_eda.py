"""Domain-grounded exploratory analysis of the RTC media-flow dataset.

Every figure answers a question a network-traffic analyst would ask about
SRTP-over-DTLS media flows: what do the encrypted packet sizes reveal about the
codec and transport, what does the packet pacing reveal about the media type,
and which of the ten application/mode classes are actually confusable.

Each figure is written alongside the numeric table that backs it, so the
findings can be audited without re-running the analysis.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import config as C
import io_utils as IO
from features import base as fbase

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("tab10", n_colors=10)
CLASS_COLOR = dict(zip(C.LABELS, PALETTE))

FIGSIZE_GRID = (16, 9)
DPI = 130


class Report:
    """Accumulates markdown sections and the tables they reference."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.lines: list[str] = []
        self.findings: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.lines.append(f"\n{'#' * level} {text}\n")

    def p(self, text: str) -> None:
        self.lines.append(text + "\n")

    def fig(self, name: str, caption: str) -> None:
        self.lines.append(f"\n![{caption}](eda/{name}.png)\n\n*{caption}*\n")

    def table(self, df: pd.DataFrame, name: str, caption: str, max_rows: int = 25) -> None:
        df.to_csv(self.out_dir / f"{name}.csv", index=True)
        shown = df.head(max_rows)
        self.lines.append(f"\n*{caption}* (`eda/{name}.csv`)\n\n")
        self.lines.append(shown.to_markdown() + "\n")

    def finding(self, text: str) -> None:
        self.findings.append(text)
        self.lines.append(f"\n> **Finding.** {text}\n")

    def write(self, path: Path) -> None:
        head = ["# EDA report - encrypted RTC application identification\n"]
        head.append(
            "Generated on Modal. Figures live in `reports/eda/`, and every figure "
            "has a `.csv` next to it holding the numbers it plots.\n"
        )
        head.append("\n## Findings at a glance\n")
        for i, fnd in enumerate(self.findings, 1):
            head.append(f"{i}. {fnd}")
        path.write_text("\n".join(head) + "\n" + "\n".join(self.lines))


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================ section A ====


def section_a(rep: Report, train: pd.DataFrame, test: pd.DataFrame) -> None:
    rep.h(2, "A. Integrity, duplication and train/test exchangeability")

    L = IO.lengths(train)
    T = IO.times(train)
    integrity = pd.DataFrame(
        {
            "train_rows": [len(train)],
            "test_rows": [len(test)],
            "missing_cells_train": [int(train[C.FEATURE_COLS].isna().sum().sum())],
            "missing_cells_test": [int(test[C.FEATURE_COLS].isna().sum().sum())],
            "t0_all_zero": [bool(np.allclose(T[:, 0], 0.0))],
            "non_monotonic_time_rows": [int((np.diff(T, axis=1) < 0).any(axis=1).sum())],
            "zero_iat_rows": [int((np.diff(T, axis=1) == 0).any(axis=1).sum())],
            "min_length": [float(L.min())],
            "max_length": [float(L.max())],
            "min_duration_s": [float(T[:, -1].min())],
            "max_duration_s": [float(T[:, -1].max())],
        }
    ).T.rename(columns={0: "value"})
    rep.table(integrity, "a1_integrity", "Dataset integrity checks")

    # Duplicate structure. A length tuple that maps to more than one class puts
    # a floor on achievable accuracy, so it is worth measuring explicitly.
    len_key = train[C.LEN_COLS].astype(int).astype(str).agg("-".join, axis=1)
    full_key = train[C.FEATURE_COLS].round(9).astype(str).agg("|".join, axis=1)
    per_key_classes = train.groupby(len_key)["label"].nunique()
    dup = pd.DataFrame(
        {
            "unique_length_tuples": [int(len_key.nunique())],
            "exact_duplicate_rows": [int(len(train) - full_key.nunique())],
            "length_tuples_multiclass": [int((per_key_classes > 1).sum())],
            "rows_in_multiclass_tuples": [
                int(len_key.isin(per_key_classes[per_key_classes > 1].index).sum())
            ],
        }
    ).T.rename(columns={0: "value"})
    rep.table(dup, "a2_duplication", "Duplicate and collision structure")
    ambiguous_rows = int(len_key.isin(per_key_classes[per_key_classes > 1].index).sum())
    amb_detail = (
        train.loc[len_key.isin(per_key_classes[per_key_classes > 1].index)]
        .groupby([len_key[len_key.isin(per_key_classes[per_key_classes > 1].index)], "label"])
        .size()
        .unstack(fill_value=0)
    )
    rep.table(amb_detail, "a2b_ambiguous_tuples", "Class mix inside label-ambiguous length tuples")
    rep.finding(
        f"Only {int((per_key_classes > 1).sum())} of {int(len_key.nunique())} distinct "
        "packet-length tuples map to more than one class, but those few tuples cover "
        f"{ambiguous_rows} rows ({ambiguous_rows / len(train):.1%} of the training set) - they "
        "are the repeated constant-size keepalive patterns. Outside that pocket the label is "
        "essentially determined by the length sequence, so the ceiling is set by those "
        "ambiguous rows plus generalisation from 1,285 samples, not by broad label noise."
    )

    # Train vs test comparison: per-feature KS test plus a discriminator.
    ks_rows = []
    for col in C.FEATURE_COLS:
        stat, pval = ks_2samp(train[col].to_numpy(), test[col].to_numpy())
        ks_rows.append({"feature": col, "ks_stat": stat, "p_value": pval})
    ks = pd.DataFrame(ks_rows).set_index("feature").sort_values("ks_stat", ascending=False)
    rep.table(ks, "a3_ks_train_test", "Kolmogorov-Smirnov train vs test, per raw feature")

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_GRID)
    for ax, col in zip(axes.ravel(), C.FEATURE_COLS):
        for name, frame, color in (("train", train, "C0"), ("test", test, "C1")):
            v = np.sort(frame[col].to_numpy())
            ax.plot(v, np.linspace(0, 1, len(v)), label=name, color=color, lw=1.5)
        ax.set_title(col, fontsize=9)
        if "time" in col:
            ax.set_xscale("symlog", linthresh=1e-5)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Train vs test ECDF for every raw column")
    savefig(fig, "a4_ecdf_train_test")
    rep.fig("a4_ecdf_train_test", "Train vs test ECDF overlays for the ten raw columns")

    # Adversarial validation: can a model tell train rows from test rows?
    Xa = pd.concat(
        [fbase.build(train[C.FEATURE_COLS]), fbase.build(test[C.FEATURE_COLS])], axis=0
    ).to_numpy()
    ya = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = RandomForestClassifier(n_estimators=400, random_state=C.SEED, n_jobs=-1)
    prob = cross_val_predict(
        clf, Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
        method="predict_proba",
    )[:, 1]
    auc = roc_auc_score(ya, prob)
    rep.p(f"Adversarial-validation AUC (train vs test discriminator): **{auc:.3f}**.")
    if auc < 0.6:
        rep.finding(
            f"Adversarial validation AUC is {auc:.3f}, close to chance, so train and test "
            "are drawn from the same distribution. No covariate-shift correction is needed "
            "and cross-validation on the training set is a meaningful proxy for the "
            "leaderboard - subject to the call-grouping caveat below."
        )
    else:
        rep.finding(
            f"Adversarial validation AUC is {auc:.3f}, well above chance: train and test "
            "differ measurably. Shift-aware validation and importance weighting should be "
            "considered before trusting plain cross-validation."
        )

    # Group structure. There is no call identifier, so near-duplicate flows are
    # the visible trace of multiple flows coming from one source call.
    groups = IO.group_keys(train)
    gs = pd.Series(groups).value_counts()
    grp = pd.DataFrame(
        {
            "n_pseudo_groups": [int(gs.size)],
            "largest_group": [int(gs.max())],
            "rows_in_groups_gt1": [int(gs[gs > 1].sum())],
            "share_rows_in_groups_gt1": [float(gs[gs > 1].sum() / len(train))],
        }
    ).T.rename(columns={0: "value"})
    rep.table(grp, "a5_pseudo_groups", "Pseudo-group structure used for leakage-aware CV")
    rep.finding(
        f"{int(gs[gs > 1].sum())} of {len(train)} training flows "
        f"({gs[gs > 1].sum() / len(train):.0%}) share a near-duplicate signature with at "
        "least one other flow. Since the test split comes from held-out calls, plain "
        "stratified CV will be optimistic; model selection uses the grouped variant."
    )


# ============================================================ section B ====


def section_b(rep: Report, train: pd.DataFrame) -> None:
    rep.h(2, "B. Packet-length structure: the codec and transport fingerprint")

    long = _melt_lengths(train)

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_GRID, sharex=True)
    for ax, lab in zip(axes.ravel(), C.LABELS):
        v = long.loc[long["label"] == lab, "length"]
        ax.hist(v, bins=60, range=(0, 1400), color=CLASS_COLOR[lab])
        ax.set_title(f"{lab}  (n={len(v)})", fontsize=9)
        ax.set_xlabel("UDP payload bytes", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Packet-length distribution per class (all five packets pooled)")
    savefig(fig, "b1_length_hist")
    rep.fig("b1_length_hist", "Packet-length histograms per class")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=long, x="label", y="length", order=C.LABELS, palette=PALETTE,
        cut=0, ax=ax,
    )
    ax.set_yscale("log")
    ax.set_ylabel("UDP payload bytes (log)")
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Packet-length spread per class")
    savefig(fig, "b2_length_violin")
    rep.fig("b2_length_violin", "Per-class packet-length violins on a log axis")

    # Quantisation: encrypted media still exposes discrete frame sizes created
    # by the codec, the RTP/SRTP overhead and per-application proprietary
    # headers, so the repeated values themselves are a fingerprint.
    top_n = 20
    top_lengths = long["length"].value_counts().head(60).index
    pivot = (
        long[long["length"].isin(top_lengths)]
        .pivot_table(index="label", columns="length", values="packet_index", aggfunc="count")
        .reindex(C.LABELS)
        .fillna(0)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "share of packets"})
    ax.set_title("Most frequent discrete packet lengths, share within each class")
    savefig(fig, "b3_length_quantisation")
    rep.fig("b3_length_quantisation", "Discrete packet-length modes per class")
    rep.table(
        pivot.round(4), "b3_length_quantisation", "Share of each frequent length per class"
    )

    per_class_top = (
        long.groupby("label")["length"]
        .apply(lambda s: ", ".join(f"{int(v)}({c})" for v, c in s.value_counts().head(top_n).items()))
        .to_frame("top_lengths(count)")
    )
    rep.table(per_class_top, "b4_top_lengths", "Top 20 discrete lengths per class", max_rows=10)

    # Band composition: tiny (STUN/RTCP/DTX), audio-frame, mid, MTU-regime.
    bands = _band_table(long)
    rep.table(bands.round(4), "b5_band_composition", "Share of packets per length band")
    fig, ax = plt.subplots(figsize=(12, 6))
    bands.loc[C.LABELS].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("share of packets")
    ax.set_title("Length-band composition per class (MTU band = fragmented video)")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, "b5_band_composition")
    rep.fig("b5_band_composition", "Length-band composition per class")
    video_mtu = bands.loc[[l for l in C.LABELS if l.endswith("video")], "mtu"].mean()
    voice_mtu = bands.loc[[l for l in C.LABELS if l.endswith("voice")], "mtu"].mean()
    rep.finding(
        f"Packets in the 1100-1600 B path-MTU band make up {video_mtu:.1%} of video-class "
        f"packets against {voice_mtu:.1%} for voice classes: MTU-regime fragmentation is a "
        "direct video marker and motivates the frac_mtu / n_mtu features."
    )

    # Sequence position: packet 0 is frequently a binding or handshake packet.
    pos = long.pivot_table(index="label", columns="packet_index", values="length", aggfunc="mean")
    rep.table(pos.round(1), "b6_mean_length_by_index", "Mean packet length by position in flow")
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab in C.LABELS:
        ax.plot(pos.columns, pos.loc[lab], marker="o", label=lab, color=CLASS_COLOR[lab])
    ax.set_xlabel("packet index within flow")
    ax.set_ylabel("mean UDP payload bytes")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Length profile across the first five packets")
    savefig(fig, "b6_length_profile")
    rep.fig("b6_length_profile", "Mean packet length by position, per class")

    # Constant-size flows are a distinct flow type rather than a distinct app.
    L = IO.lengths(train)
    const = pd.DataFrame(
        {
            "constant_length_flows": train.loc[
                [len(set(r)) == 1 for r in L], "label"
            ].value_counts(),
            "total_flows": train["label"].value_counts(),
        }
    ).reindex(C.LABELS).fillna(0)
    const["share"] = const["constant_length_flows"] / const["total_flows"]
    rep.table(const.round(3), "b7_constant_flows", "Flows whose five packets share one length")
    rep.finding(
        "Constant-size flows (keepalive/STUN-like, frequently 47 B) are concentrated in the "
        f"Discord classes ({int(const.loc['Discord_voice', 'constant_length_flows'])} voice, "
        f"{int(const.loc['Discord_video', 'constant_length_flows'])} video). They are a flow "
        "type rather than an application signature, so they are flagged explicitly "
        "(is_constant_keepalive) instead of being learned implicitly."
    )


# ============================================================ section C ====


def section_c(rep: Report, train: pd.DataFrame) -> None:
    rep.h(2, "C. Timing structure: the pacing fingerprint")

    D = IO.iats(train)
    iat_long = pd.DataFrame(
        {
            "label": np.repeat(train["label"].to_numpy(), D.shape[1]),
            "gap_index": np.tile(np.arange(1, D.shape[1] + 1), len(train)),
            "iat": D.ravel(),
        }
    )
    iat_long["log_iat"] = np.log10(np.maximum(iat_long["iat"], 1e-7))

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_GRID, sharex=True)
    for ax, lab in zip(axes.ravel(), C.LABELS):
        v = iat_long.loc[iat_long["label"] == lab, "log_iat"]
        sns.kdeplot(x=v, ax=ax, fill=True, color=CLASS_COLOR[lab], bw_adjust=0.6)
        for ref, txt in ((-3, "1 ms"), (np.log10(0.02), "20 ms")):
            ax.axvline(ref, color="k", ls=":", lw=1)
            ax.text(ref, ax.get_ylim()[1] * 0.9, txt, fontsize=6, rotation=90)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("log10 inter-arrival (s)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Inter-arrival distribution per class: sub-millisecond fragment bursts vs "
        "packetisation-interval pacing"
    )
    savefig(fig, "c1_iat_kde")
    rep.fig("c1_iat_kde", "Per-class inter-arrival KDEs on a log scale")

    burst = (
        iat_long.assign(sub_ms=iat_long["iat"] < 1e-3, over_20ms=iat_long["iat"] > 0.02)
        .groupby("label")[["sub_ms", "over_20ms"]]
        .mean()
        .reindex(C.LABELS)
    )
    burst["median_iat_s"] = iat_long.groupby("label")["iat"].median().reindex(C.LABELS)
    rep.table(burst.round(4), "c2_burstiness", "Share of sub-millisecond and >20 ms gaps")
    fig, ax = plt.subplots(figsize=(11, 5))
    burst[["sub_ms", "over_20ms"]].plot(kind="bar", ax=ax, color=["C0", "C3"])
    ax.set_ylabel("share of inter-arrival gaps")
    ax.set_title("Burstiness: sub-millisecond fragmentation vs idle gaps")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, "c2_burstiness")
    rep.fig("c2_burstiness", "Sub-millisecond and long-gap shares per class")

    dur = train.assign(duration=IO.times(train)[:, -1])
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab in C.LABELS:
        v = np.sort(dur.loc[dur["label"] == lab, "duration"].to_numpy())
        ax.plot(np.maximum(v, 1e-6), np.linspace(0, 1, len(v)), label=lab, color=CLASS_COLOR[lab])
    ax.set_xscale("log")
    ax.set_xlabel("time span of the first five packets (s, log)")
    ax.set_ylabel("ECDF")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Five-packet time span per class")
    savefig(fig, "c3_duration_ecdf")
    rep.fig("c3_duration_ecdf", "Flow-duration ECDF per class")
    dur_tab = dur.groupby("label")["duration"].describe(percentiles=[0.1, 0.5, 0.9, 0.99])
    rep.table(dur_tab.reindex(C.LABELS).round(6), "c3_duration_stats", "Five-packet time span")

    fig, ax = plt.subplots(figsize=(8, 7))
    for lab in C.LABELS:
        m = train["label"].to_numpy() == lab
        ax.scatter(
            np.maximum(D[m, 0], 1e-7), np.maximum(D[m, 1], 1e-7),
            s=8, alpha=0.6, label=lab, color=CLASS_COLOR[lab],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap 1 (s)")
    ax.set_ylabel("gap 2 (s)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Consecutive-gap regularity")
    savefig(fig, "c4_gap_scatter")
    rep.fig("c4_gap_scatter", "First versus second inter-arrival gap, coloured by class")

    rep.finding(
        "Video classes concentrate their gaps below one millisecond (packets of a single "
        "frame fragmented across the MTU) while voice classes carry a heavier mass near the "
        "packetisation interval and in the idle tail, which is what the burst/frame-group "
        "feature block (F6) is built to capture."
    )


# ============================================================ section D ====


def section_d(rep: Report, train: pd.DataFrame) -> None:
    rep.h(2, "D. Joint size-timing structure")

    L = IO.lengths(train)[:, 1:]
    D = np.maximum(IO.iats(train), 1e-7)
    lab_rep = np.repeat(train["label"].to_numpy(), L.shape[1])
    x = np.log10(D.ravel())
    y = L.ravel()

    fig, axes = plt.subplots(2, 5, figsize=FIGSIZE_GRID, sharex=True, sharey=True)
    for ax, lab in zip(axes.ravel(), C.LABELS):
        m = lab_rep == lab
        ax.hexbin(x[m], y[m], gridsize=32, cmap="magma_r", bins="log", extent=(-7, 1, 0, 1400))
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("log10 IAT (s)", fontsize=8)
        ax.set_ylabel("bytes", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Size-versus-timing manifold per class (five-packet FlowPic view)")
    savefig(fig, "d1_size_time_hexbin")
    rep.fig("d1_size_time_hexbin", "Joint packet-size / inter-arrival density per class")

    rate = pd.DataFrame({"label": lab_rep, "rate": y / D.ravel()})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=rate, x="label", y="rate", order=C.LABELS, palette=PALETTE, cut=0, ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("instantaneous rate (bytes/s, log)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Per-gap instantaneous rate: a proxy for stream bitrate")
    savefig(fig, "d2_instantaneous_rate")
    rep.fig("d2_instantaneous_rate", "Instantaneous per-gap rate per class")
    rep.table(
        rate.groupby("label")["rate"].describe(percentiles=[0.5]).reindex(C.LABELS).round(1),
        "d2_instantaneous_rate",
        "Instantaneous rate summary",
    )


# ============================================================ section E ====


def section_e(rep: Report, train: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> None:
    rep.h(2, "E. Class separability and the actual hard pairs")

    mi = mutual_info_classif(X.to_numpy(), y, random_state=C.SEED)
    fstat, _ = f_classif(X.to_numpy(), y)
    rank = (
        pd.DataFrame({"feature": X.columns, "mutual_info": mi, "anova_f": np.nan_to_num(fstat)})
        .set_index("feature")
        .sort_values("mutual_info", ascending=False)
    )
    rep.table(rank.round(4), "e1_feature_ranking", "Feature ranking by mutual information", 30)
    fig, ax = plt.subplots(figsize=(10, 10))
    rank.head(40)["mutual_info"].iloc[::-1].plot(kind="barh", ax=ax, color="C0")
    ax.set_title("Top 40 engineered features by mutual information with the class")
    savefig(fig, "e1_mutual_info")
    rep.fig("e1_mutual_info", "Mutual information of engineered features with the label")

    Xs = StandardScaler().fit_transform(X.to_numpy())
    app, mode = IO.split_app_mode(train["label"])
    emb = {
        "PCA": PCA(n_components=2, random_state=C.SEED).fit_transform(Xs),
        "t-SNE": TSNE(
            n_components=2, perplexity=30, init="pca", random_state=C.SEED, max_iter=1000
        ).fit_transform(Xs),
    }
    try:
        import umap

        emb["UMAP"] = umap.UMAP(n_components=2, random_state=C.SEED).fit_transform(Xs)
    except Exception as exc:  # pragma: no cover - optional dependency
        rep.p(f"UMAP unavailable: {exc}")

    fig, axes = plt.subplots(2, len(emb), figsize=(6 * len(emb), 11), squeeze=False)
    for j, (name, Z) in enumerate(emb.items()):
        for i, (col, title) in enumerate(((app, "application"), (mode, "call mode"))):
            ax = axes[i][j]
            for k, val in enumerate(sorted(col.unique())):
                m = col.to_numpy() == val
                ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=val, color=PALETTE[k])
            ax.set_title(f"{name} coloured by {title}", fontsize=10)
            ax.legend(fontsize=7)
    fig.suptitle("Embedding of the engineered feature space")
    savefig(fig, "e2_embeddings")
    rep.fig("e2_embeddings", "PCA / t-SNE / UMAP coloured by application and by call mode")

    # A quick baseline names the confusable pairs before feature work continues.
    import lightgbm as lgb

    base = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.9,
        colsample_bytree=0.8, random_state=C.SEED, n_jobs=-1, verbose=-1,
    )
    cv = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    pred = cross_val_predict(base, X.to_numpy(), y, cv=cv, n_jobs=1)
    acc = float((pred == y).mean())
    cm = confusion_matrix(y, pred, labels=range(len(C.LABELS)))
    cm_df = pd.DataFrame(cm, index=C.LABELS, columns=C.LABELS)
    rep.table(cm_df, "e3_baseline_confusion", f"Baseline LightGBM confusion matrix (acc={acc:.3f})")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_df.div(cm_df.sum(axis=1), axis=0), annot=cm_df.to_numpy(), fmt="d",
        cmap="Blues", ax=ax, cbar_kws={"label": "row-normalised"},
    )
    ax.set_title(f"Baseline LightGBM, 5-fold CV accuracy = {acc:.3f}")
    ax.set_ylabel("true")
    ax.set_xlabel("predicted")
    savefig(fig, "e3_baseline_confusion")
    rep.fig("e3_baseline_confusion", "Baseline confusion matrix on engineered features")

    off = cm_df.to_numpy().copy()
    np.fill_diagonal(off, 0)
    pairs = [
        (C.LABELS[i], C.LABELS[j], int(off[i, j]))
        for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:10], off.shape))
    ]
    hard = pd.DataFrame(pairs, columns=["true", "predicted", "count"])
    rep.table(hard, "e4_hard_pairs", "Most frequent confusions")
    rep.finding(
        f"A plain LightGBM on the engineered features already reaches {acc:.3f} accuracy in "
        f"5-fold CV. The dominant confusion is {pairs[0][0]} -> {pairs[0][1]} "
        f"({pairs[0][2]} flows), so feature work and ensembling should be judged on whether "
        "they move that pair rather than on overall accuracy alone."
    )

    # Is the label better attacked as application x mode than as ten flat classes?
    app_id = pd.Categorical(app, categories=C.APPS).codes
    mode_id = pd.Categorical(mode, categories=C.MODES).codes
    acc_app = float((cross_val_predict(base, X.to_numpy(), app_id, cv=cv) == app_id).mean())
    acc_mode = float((cross_val_predict(base, X.to_numpy(), mode_id, cv=cv) == mode_id).mean())
    hier = pd.DataFrame(
        {
            "target": ["flat 10-class", "application (5)", "call mode (2)", "product bound"],
            "cv_accuracy": [acc, acc_app, acc_mode, acc_app * acc_mode],
        }
    ).set_index("target")
    rep.table(hier.round(4), "e5_hierarchical", "Flat versus factorised targets")
    harder, easier = (
        ("call mode", "application") if acc_mode < acc_app else ("application", "call mode")
    )
    rep.finding(
        f"Split into its two factors, the application is predicted at {acc_app:.3f} and the "
        f"call mode at {acc_mode:.3f}, so **{harder} is the harder half** and {easier} the "
        "easier one - the opposite of the usual assumption that voice-versus-video is the "
        f"trivial part. The independent-product bound ({acc_app * acc_mode:.3f}) sits "
        f"{'above' if acc_app * acc_mode > acc else 'at or below'} the flat 10-class model "
        f"({acc:.3f}), so the hierarchical decomposition earns no free win and is kept only "
        "as a diversity member in the ensemble, judged on grouped CV."
    )
    return {"baseline_cv_acc": acc, "app_cv_acc": acc_app, "mode_cv_acc": acc_mode}


# ============================================================ section F ====


def section_f(rep: Report, train: pd.DataFrame) -> None:
    rep.h(2, "F. Class imbalance and the flows-per-call prior")

    counts = train["label"].value_counts().reindex(C.LABELS)
    tab = pd.DataFrame(
        {
            "train_flows": counts,
            "source_calls": C.CALLS_PER_CLASS_TRAIN,
            "flows_per_call": counts / C.CALLS_PER_CLASS_TRAIN,
        }
    )
    tab["implied_test_flows"] = tab["flows_per_call"] * C.CALLS_PER_CLASS_TEST
    tab["implied_test_share"] = tab["implied_test_flows"] / tab["implied_test_flows"].sum()
    rep.table(tab.round(3), "f1_class_prior", "Flows per source call and the implied test prior")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts.plot(kind="bar", ax=axes[0], color=[CLASS_COLOR[l] for l in C.LABELS])
    axes[0].set_title("Training flows per class")
    axes[0].tick_params(axis="x", rotation=30)
    tab["flows_per_call"].plot(kind="bar", ax=axes[1], color=[CLASS_COLOR[l] for l in C.LABELS])
    axes[1].set_title("Flows produced per source call")
    axes[1].tick_params(axis="x", rotation=30)
    savefig(fig, "f1_class_prior")
    rep.fig("f1_class_prior", "Class counts and the flows-per-call ratio")

    implied_total = float(tab["implied_test_flows"].sum())
    rep.finding(
        f"Per-class flow counts vary {counts.max() / counts.min():.1f}x because applications "
        f"differ in how many UDP flows one call opens ({tab['flows_per_call'].min():.1f} to "
        f"{tab['flows_per_call'].max():.1f} flows per call). Extrapolating that ratio to the "
        f"100 held-out test calls predicts about {implied_total:.0f} test flows against the "
        f"{C.N_TEST_ROWS} actually supplied, so the prior is a useful sanity check on the "
        "submitted class distribution but not a hard constraint."
    )


# =============================================================== driver ====


def _melt_lengths(train: pd.DataFrame) -> pd.DataFrame:
    L = IO.lengths(train)
    return pd.DataFrame(
        {
            "label": np.repeat(train["label"].to_numpy(), L.shape[1]),
            "packet_index": np.tile(np.arange(L.shape[1]), len(train)),
            "length": L.ravel(),
        }
    )


def _band_table(long: pd.DataFrame) -> pd.DataFrame:
    bands = {
        "tiny(<100B)": C.BAND_TINY,
        "audio(100-300B)": C.BAND_AUDIO,
        "mid(300-1100B)": C.BAND_MID,
        "mtu": C.BAND_MTU,
    }
    out = {}
    for name, (lo, hi) in bands.items():
        key = "mtu" if name == "mtu" else name
        out[key] = long.assign(inb=(long["length"] >= lo) & (long["length"] < hi)).groupby(
            "label"
        )["inb"].mean()
    return pd.DataFrame(out).reindex(C.LABELS)


def main() -> dict:
    C.ensure_dirs()
    train, test = IO.load_train(), IO.load_test()
    y = IO.labels_to_ids(train["label"])
    X = fbase.build(train[C.FEATURE_COLS])

    rep = Report(C.EDA_DIR)
    rep.h(2, "Dataset at a glance")
    rep.p(
        f"- Training flows: **{len(train)}**, test flows: **{len(test)}**\n"
        f"- Features: five (relative_time, packet_length) pairs per flow\n"
        f"- Classes: **{len(C.LABELS)}** = 5 applications x 2 call modes\n"
        f"- Engineered feature columns available to the models here: **{X.shape[1]}**\n"
    )

    section_a(rep, train, test)
    section_b(rep, train)
    section_c(rep, train)
    section_d(rep, train)
    summary = section_e(rep, train, X, y)
    section_f(rep, train)

    out_md = C.REPORTS_DIR / "eda_report.md"
    rep.write(out_md)
    (C.REPORTS_DIR / "eda_summary.json").write_text(
        json.dumps({"n_features_base": int(X.shape[1]), **summary}, indent=2)
    )
    return {"report": str(out_md), "n_features": int(X.shape[1]), **summary}


if __name__ == "__main__":
    print(main())
