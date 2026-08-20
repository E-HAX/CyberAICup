"""Fitted, label-aware feature blocks F9-F11.

Unlike the row-wise blocks in features/base.py these estimators are fitted on
labelled data, so they can only ever be fitted on the training part of a fold
and applied to the held-out part. They are deliberately implemented as a single
transformer with an explicit ``fit(X, y)`` / ``transform(X)`` split so the
cross-validation loop cannot accidentally fit them on validation rows.

    F9   class-conditional Markov chains over quantised packet-length and
         inter-arrival state sequences (Korczynski-Duda style fingerprinting)
    F10  class-conditional Gaussian mixtures on a compact behavioural subspace
    F11  per-class nearest-neighbour distances and neighbourhood votes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config as C

EPS = 1e-12

# Compact behavioural subspace used by the density and neighbourhood blocks.
DENSITY_COLS = [
    "len_mean",
    "len_std",
    "len_max",
    "len_min",
    "len_entropy",
    "log_duration",
    "frac_iat_below_1000us",
    "frac_mtu",
    "frac_tiny",
    "log_bytes_per_sec",
    "logiat_mean",
    "n_groups_1000us",
]


class TargetAwareFeatures:
    """Fits F9-F11 on a fold's training rows and emits them for any rows."""

    def __init__(
        self,
        n_len_bins: int = 16,
        n_iat_bins: int = 12,
        gmm_components: int = 2,
        knn_ks: tuple[int, ...] = (1, 3, 5),
        pca_dim: int = 16,
        smoothing: float = 0.5,
        seed: int = C.SEED,
    ):
        self.n_len_bins = n_len_bins
        self.n_iat_bins = n_iat_bins
        self.gmm_components = gmm_components
        self.knn_ks = knn_ks
        self.pca_dim = pca_dim
        self.smoothing = smoothing
        self.seed = seed
        self.n_classes = len(C.LABELS)

    # ------------------------------------------------------------- fit ----

    def fit(self, raw: pd.DataFrame, base: pd.DataFrame, y: np.ndarray) -> "TargetAwareFeatures":
        L = raw[C.LEN_COLS].to_numpy(dtype=np.float64)
        D = np.diff(raw[C.TIME_COLS].to_numpy(dtype=np.float64), axis=1)

        # F9: quantile bin edges are part of the fitted state.
        self.len_edges_ = self._edges(L.ravel(), self.n_len_bins)
        self.iat_edges_ = self._edges(np.log10(np.maximum(D.ravel(), 1e-7)), self.n_iat_bins)
        len_states = self._digitise(L, self.len_edges_)
        iat_states = self._digitise(np.log10(np.maximum(D, 1e-7)), self.iat_edges_)
        self.len_chain_ = self._fit_chains(len_states, y, self.n_len_bins)
        self.iat_chain_ = self._fit_chains(iat_states, y, self.n_iat_bins)

        # F10: class-conditional densities on a standardised compact subspace.
        cols = [c for c in DENSITY_COLS if c in base.columns]
        self.density_cols_ = cols
        Z = base[cols].to_numpy(dtype=np.float64)
        self.density_scaler_ = StandardScaler().fit(Z)
        Zs = self.density_scaler_.transform(Z)
        self.gmms_ = []
        for k in range(self.n_classes):
            rows = Zs[y == k]
            n_comp = max(1, min(self.gmm_components, max(1, len(rows) // 25)))
            gm = GaussianMixture(
                n_components=n_comp,
                covariance_type="diag",
                reg_covar=1e-4,
                random_state=self.seed,
                max_iter=200,
            )
            gm.fit(rows if len(rows) >= n_comp else Zs)
            self.gmms_.append(gm)

        # F11: neighbourhood structure in a PCA-compressed full feature space.
        self.knn_scaler_ = StandardScaler().fit(base.to_numpy(dtype=np.float64))
        Bs = self.knn_scaler_.transform(base.to_numpy(dtype=np.float64))
        dim = min(self.pca_dim, Bs.shape[1], max(2, len(Bs) - 1))
        self.pca_ = PCA(n_components=dim, random_state=self.seed).fit(Bs)
        P = self.pca_.transform(Bs)
        self.knn_all_ = NearestNeighbors(n_neighbors=min(11, len(P))).fit(P)
        self.knn_all_y_ = y.copy()
        self.knn_per_class_ = []
        for k in range(self.n_classes):
            rows = P[y == k]
            if len(rows) == 0:
                self.knn_per_class_.append(None)
                continue
            self.knn_per_class_.append(
                NearestNeighbors(n_neighbors=min(max(self.knn_ks), len(rows))).fit(rows)
            )
        self.fitted_ = True
        return self

    # ------------------------------------------------------- transform ----

    def transform(self, raw: pd.DataFrame, base: pd.DataFrame, self_exclude: bool = False) -> pd.DataFrame:
        """Emit F9-F11 for the given rows.

        ``self_exclude`` drops the first neighbour when the rows being
        transformed are the very rows the neighbour index was fitted on, which
        keeps the training-side features from trivially describing themselves.
        """
        L = raw[C.LEN_COLS].to_numpy(dtype=np.float64)
        D = np.diff(raw[C.TIME_COLS].to_numpy(dtype=np.float64), axis=1)
        out: dict[str, np.ndarray] = {}

        # -- F9 -------------------------------------------------------------
        for tag, states, chain, n_bins in (
            ("len", self._digitise(L, self.len_edges_), self.len_chain_, self.n_len_bins),
            ("iat", self._digitise(np.log10(np.maximum(D, 1e-7)), self.iat_edges_), self.iat_chain_, self.n_iat_bins),
        ):
            ll = self._loglik(states, chain)
            post = self._softmax(ll)
            for k in range(self.n_classes):
                out[f"mk_{tag}_ll_{k}"] = ll[:, k]
                out[f"mk_{tag}_post_{k}"] = post[:, k]
            srt = np.sort(ll, axis=1)
            out[f"mk_{tag}_argmax"] = ll.argmax(axis=1).astype(np.float64)
            out[f"mk_{tag}_margin"] = srt[:, -1] - srt[:, -2]
            out[f"mk_{tag}_max"] = srt[:, -1]

        # -- F10 ------------------------------------------------------------
        Zs = self.density_scaler_.transform(base[self.density_cols_].to_numpy(dtype=np.float64))
        dens = np.column_stack([gm.score_samples(Zs) for gm in self.gmms_])
        dpost = self._softmax(dens)
        for k in range(self.n_classes):
            out[f"gmm_ll_{k}"] = dens[:, k]
            out[f"gmm_post_{k}"] = dpost[:, k]
        srt = np.sort(dens, axis=1)
        out["gmm_argmax"] = dens.argmax(axis=1).astype(np.float64)
        out["gmm_margin"] = srt[:, -1] - srt[:, -2]

        # -- F11 ------------------------------------------------------------
        P = self.pca_.transform(self.knn_scaler_.transform(base.to_numpy(dtype=np.float64)))
        skip = 1 if self_exclude else 0
        for k, nn in enumerate(self.knn_per_class_):
            if nn is None:
                for kk in self.knn_ks:
                    out[f"knn{kk}_dist_c{k}"] = np.full(len(P), 1e6)
                continue
            n_avail = nn.n_samples_fit_
            dist, _ = nn.kneighbors(P, n_neighbors=min(max(self.knn_ks) + skip, n_avail))
            dist = dist[:, skip:] if skip and dist.shape[1] > 1 else dist
            for kk in self.knn_ks:
                take = min(kk, dist.shape[1])
                out[f"knn{kk}_dist_c{k}"] = dist[:, :take].mean(axis=1)
        d1 = np.column_stack([out[f"knn1_dist_c{k}"] for k in range(self.n_classes)])
        out["knn1_argmin"] = d1.argmin(axis=1).astype(np.float64)
        srt = np.sort(d1, axis=1)
        out["knn1_margin"] = srt[:, 1] - srt[:, 0]
        n_neighbors = min(10 + skip, self.knn_all_.n_samples_fit_)
        _, idx = self.knn_all_.kneighbors(P, n_neighbors=n_neighbors)
        idx = idx[:, skip:] if skip and idx.shape[1] > 1 else idx
        votes = self.knn_all_y_[idx]
        for k in range(self.n_classes):
            out[f"knn_vote_{k}"] = (votes == k).mean(axis=1)

        return pd.DataFrame(out, index=raw.index).astype(np.float32).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)

    # ---------------------------------------------------------- helpers ----

    @staticmethod
    def _edges(values: np.ndarray, n_bins: int) -> np.ndarray:
        qs = np.linspace(0, 100, n_bins + 1)[1:-1]
        return np.unique(np.percentile(values, qs))

    @staticmethod
    def _digitise(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(x, edges).astype(np.int64)

    def _fit_chains(self, states: np.ndarray, y: np.ndarray, n_bins: int) -> dict:
        """First-order Markov chain per class with additive smoothing."""
        n_states = n_bins + 1
        chains = {}
        for k in range(self.n_classes):
            s = states[y == k]
            init = np.full(n_states, self.smoothing)
            trans = np.full((n_states, n_states), self.smoothing)
            if len(s):
                np.add.at(init, s[:, 0], 1.0)
                for i in range(s.shape[1] - 1):
                    np.add.at(trans, (s[:, i], s[:, i + 1]), 1.0)
            chains[k] = (
                np.log(init / init.sum()),
                np.log(trans / trans.sum(axis=1, keepdims=True)),
            )
        return chains

    def _loglik(self, states: np.ndarray, chains: dict) -> np.ndarray:
        n_rows = states.shape[0]
        ll = np.zeros((n_rows, self.n_classes))
        for k, (log_init, log_trans) in chains.items():
            v = log_init[states[:, 0]]
            for i in range(states.shape[1] - 1):
                v = v + log_trans[states[:, i], states[:, i + 1]]
            ll[:, k] = v
        return ll

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - x.max(axis=1, keepdims=True)
        e = np.exp(np.clip(z, -60, 0))
        return e / (e.sum(axis=1, keepdims=True) + EPS)
