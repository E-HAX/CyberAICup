# Encrypted RTC Application Identification — Methodology and Results

**Task:** CyberAI Cup 2026 (ICSDS), Task 3 — given the sizes and inter-arrival times of the
first five packets of an encrypted UDP media flow, predict one of ten classes formed by
{Discord, GoogleMeet, Messenger, WhatsApp, Zoom} × {voice, video}.

**Deliverable:** `submission.csv` — 327 rows, no header, `index(1-based),label`, in test-file order.

**Execution substrate:** every training and search stage ran on Modal (app `rtc-cyberai`). The
local machine edited source and issued launch commands; the round-3 diagnostics of section 24 also
ran locally, on the 1,285-row training table, where a container round trip costs more than the
computation.

---

## Update — round 3 (see `ABLATION3.md`)

**Shipped model: TabICL, flat and hierarchical application × mode views averaged, with the derived
Zoom rule — grouped-CV accuracy 0.8342, macro-F1 0.8292.** This supersedes the flat TabICL model
reported below (0.8381 / 0.8262); the difference is +0.0030 macro-F1 with a 95% interval of
[−0.0130, +0.0201], so it is a preference, not an established gain.

The durable result of round 3 is a diagnosis rather than a model. Every record is the first five
packets of a flow, so a video call whose opening packets carry no video fragment is indistinguishable
from a voice call. For Zoom this is exact: 160 training flows have audio-only windows, split 80
voice against 80 video, at grouped-CV AUC 0.551. **That fixes an accuracy ceiling of 0.9377 and a
macro-F1 ceiling of 0.9364 for the whole task**, and makes a macro-F1 of 0.90 unreachable — it
would require the other eight classes to average F1 ≥ 0.954 against a current 0.868.

The one thing the geometry does license is a parameter-free rule: a flow predicted Zoom with an
audio-only window is labelled Zoom_voice. It raises macro-F1 for all six model families tested.

---

# Part I — Methodology

## 1. Data

| Property | Value |
|---|---|
| Training flows | 1,285 |
| Test flows | 327 |
| Columns | `relative_time_0..4`, `packet_length_0..4`, `label` (train only) |
| `relative_time_0` | identically 0.0 (verified) |
| Packet-length range | 26 – 1,242 bytes |
| Five-packet time span | 48 µs – 23.21 s (median 571 µs) |
| Missing values | none |
| Non-monotonic timestamp rows | none |
| Class imbalance | Discord_video 256 → GoogleMeet_voice 40 (6.4×) |

The provided CSVs were copied into a Modal volume once (`seed_data`) so no remote stage depends
on the laptop being reachable.

### 1.1 Exploratory analysis performed

Before any modelling, the flows were examined with questions a traffic analyst would ask of
SRTP-over-DTLS media, not with generic summary statistics. Each analysis wrote both a figure and
the numeric table behind it, so findings are auditable without re-running anything.

- **Integrity and duplication** — schema and dtype checks, missingness, `relative_time_0 == 0`,
  timestamp monotonicity (capture reordering), exact duplicate rows, and length tuples that map to
  more than one class (an empirical floor on achievable accuracy).
- **Train/test exchangeability** — per-column Kolmogorov–Smirnov tests with ECDF overlays, plus
  adversarial validation (a RandomForest discriminating train rows from test rows under 5-fold CV).
- **Packet-length structure** — per-class histograms and log-axis violins; a quantisation heatmap
  of the most frequent discrete sizes (codec frames, RTP/SRTP overhead and per-app proprietary
  headers such as WhatsApp's WASP survive encryption as repeated exact sizes); band composition
  across the four protocol-meaningful bands; the mean length profile across packet positions 0–4;
  and separate treatment of constant-size flows.
- **Timing structure** — per-class log-scale kernel densities of inter-arrival gaps with the 1 ms
  and 20 ms reference lines drawn in (sub-millisecond gaps indicate one video frame fragmented
  across the MTU; ~20 ms indicates audio packetisation); sub-millisecond and >20 ms gap shares;
  five-packet span ECDFs; and a gap₁-versus-gap₂ scatter for pacing regularity.
- **Joint size–timing** — per-class hexbin of (length, log gap), a five-packet FlowPic-style view
  of each class manifold, and per-gap instantaneous rate distributions as a bitrate proxy.
- **Separability** — mutual information and ANOVA F per engineered feature; PCA, t-SNE and UMAP
  embeddings coloured by application and by call mode **separately**; a quick 5-fold LightGBM
  confusion matrix to name the hard pairs before investing in features; and a flat-versus-factorised
  comparison (10-class model against a 5-class application model and a 2-class mode model).
- **Imbalance structure** — per-class flow counts against the stated 40 source calls per class,
  giving a flows-per-call rate and an implied test-set class prior.

## 2. Validation protocol

This is the single most consequential methodological choice in the project.

The 1,285 training flows are drawn from 400 source calls and the 327 test flows from 100
**held-out** calls, but **no call identifier is published**. Multiple flows from one call are
near-identical, so a plain stratified K-fold places siblings on both sides of the split and
reports an optimistic score.

Two schemes therefore run side by side for every model:

- **`plain`** — `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`. The
  optimistic estimate, retained for comparability with published baselines.
- **`group`** — `StratifiedGroupKFold(n_splits=5)` over a **pseudo-group key**, repeated twice
  with different shuffles. The key is the flow's packet-length tuple rounded to 8-byte
  granularity, joined with `round(log10(duration), 1)`. Flows sharing a key cannot straddle a
  fold boundary. **This is the decisive number; every selection decision on this page uses it.**

The key produces 1,107 pseudo-groups over 1,285 flows; 253 flows (20%) share a group with at
least one other flow.

Fold indices are computed **once** and stored in the volume, so scores produced in hundreds of
different containers are directly comparable. Primary metric is accuracy (implied by the
submission format); macro-F1 is tracked as a secondary guard because the test distribution is
explicitly not guaranteed uniform.

**Screening vs final budget.** Grid screening scores one grouped repeat (5 fits) to stay cheap.
Finalists are re-scored with 2 grouped repeats + 3 plain repeats (25 fits) and additionally
refit on all training rows so their test probabilities are stored alongside their out-of-fold
matrices.

## 3. Feature engineering

Twelve blocks, grouped by whether they are label-free.

### Deterministic, label-free (F1–F8) — 241 columns, computed once and cached

| Block | Content |
|---|---|
| **F1** Raw & log | the ten raw columns plus `log1p` of each length and timestamp |
| **F2** Inter-arrival | gaps `d₁..d₄`, their logs, each gap as a fraction of the span, cumulative time fractions, and min/max/mean/std/median/IQR/CV/skew/kurtosis of the gap vector |
| **F3** Length distribution | the same nine-statistic summary over packet lengths, plus distinct-value count, all-equal flag, argmax/argmin index, and normalised Shannon entropy of the length multiset |
| **F4** Sequence shape | all 10 pairwise differences and 10 pairwise ratios, first-order steps and their signs, increase/decrease counts, longest monotone run, each length over the first and over the max, least-squares slope and R² of length against index and of cumulative bytes against time |
| **F5** Protocol semantics | counts and fractions in four bands — <100 B (STUN/RTCP/DTX), 100–300 B (Opus + RTP/SRTP overhead), 300–1100 B, 1100–1600 B (path-MTU video fragment); the 47-byte keepalive count; a constant-size-keepalive flag; `length mod 4/8/16` residues with zero-counts and distinct-counts; RTP-header-adjusted (`−12 B`) and SRTP-auth-tag-adjusted (`−22 B`) mean payloads; first-packet-versus-rest difference and ratio |
| **F6** Burst structure | flows segmented at gap thresholds {0.5 ms, 1 ms, 5 ms, 20 ms}: group count, packets and bytes in the largest group, mean inter-group gap, packets per group, sub-threshold counts and longest run |
| **F7** Rate | total bytes, bytes/s, packets/s, per-gap instantaneous rate `lᵢ/dᵢ` with full statistics, and log variants |
| **F8** Spectral | 5-point DCT of the length sequence and of the log-gap sequence, plus rFFT magnitudes of the length sequence |

Design rationale is drawn from the encrypted-traffic literature: statistical packet-size
summaries (Taylor et al., *AppScanner*), time-based flow features (Draper-Gil et al., *ISCXVPN*),
size–time joint structure (Shapira & Shavitt, *FlowPic*), per-packet size/IAT sequence models
(Lopez-Martin et al.), and the MIRAGE-lineage multimodal work (Aceto et al., *MIMETIC*/*DISTILLER*).

### Fitted and label-aware (F9–F11) — computed inside the CV loop

| Block | Content |
|---|---|
| **F9** Markov fingerprints | lengths quantised into 16 quantile bins and log-gaps into 12; one first-order Markov chain per class with additive smoothing; emits 10 log-likelihoods, 10 posteriors, argmax, top-1/top-2 margin, and max — for both the length chain and the gap chain (Korczyński–Duda style fingerprinting adapted to a 5-packet sequence) |
| **F10** Class-conditional densities | per-class diagonal Gaussian mixtures on a 12-dimensional behavioural subspace (mean/std/max/min length, length entropy, log duration, sub-millisecond fraction, MTU fraction, tiny fraction, log bytes/s, mean log-gap, group count); emits log-densities, posteriors, argmax and margin |
| **F11** Neighbourhood | PCA-16 of the standardised full feature space; distance to the 1/3/5 nearest fold-train neighbours **of each class**, nearest-class argmin and margin, and class-vote fractions among the 10 nearest overall |

These are fitted on labelled data, so they are refit per fold. Critically, their **training-side
values are generated through an inner 5-fold loop** — fitting on the fold-train rows and scoring
those same rows in-sample makes the features far sharper on training data than on validation
data, and a model trained on that mismatch over-trusts them. (The first implementation did
exactly this and scored measurably worse; see Results §17.)

### Auxiliary-derived (F12) — 46 columns from the linear probes

Described in §7. Probe targets come from the auxiliary corpus only, so these columns carry no
competition-label information and are computed once outside the fold loop.

## 4. Model families

Thirteen configurations across twelve families, deliberately spanning different inductive biases
so an ensemble has something to combine:

- **Boosted trees:** XGBoost (`multi:softprob`), LightGBM (`multiclass`), CatBoost (`MultiClass`),
  scikit-learn `HistGradientBoosting`
- **Bagged trees:** RandomForest, ExtraTrees
- **Kernel / distance:** RBF-SVM (probability-calibrated), k-NN with searched metric and weighting
- **Linear / neural:** multinomial logistic regression, MLP
- **Sequence:** PacketFormer, a small transformer over the 5-packet sequence (§6)

Class imbalance is handled as a **searched hyperparameter** — `{none, balanced, balanced_subsample}` —
translated per family into `class_weight` or per-sample weights, rather than assumed to help.

## 5. Hyperparameter search

**Stage 1 — distributed grid.** The grid is materialised as an explicit list of parameter dicts
and executed as one container per point (`fit_eval.map`), i.e. a `GridSearchCV` whose parallelism
lives at the infrastructure layer. Every model runs single-threaded inside its container so the
fan-out is not fighting itself for cores.

**Stage 2 — Optuna refinement.** TPE with a multivariate sampler over the continuous neighbourhood
of the grid winners, distributed by **ask/tell batching**: a driver container asks for 30 trials,
evaluates them across containers, tells the results back, and repeats. The study is journalled to
a volume so a killed driver resumes rather than restarting. Ask/tell was chosen over shared-RDB
storage because it needs no database and no concurrent-writer semantics on a network volume.

**Artifact contract.** Every scored point writes its own `.npz` containing its out-of-fold
probability matrices, a JSON metadata blob, and (for finalists) test probabilities. This makes the
search record reconstructible from storage alone — which mattered when a ~1,000-point sweep was
cancelled mid-run and was recovered in full without re-running anything.

## 6. Auxiliary corpus: MIRAGE-AppAct-2024

**Source.** `https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-AppAct-2024.zip`, 6,973,763,076
bytes, 2,266 members (2,245 JSON), 20 application directories — including all five competition
applications plus Skype, Teams, Webex, JitsiMeet, Trueconf, GotoMeeting and others.

**Reconnaissance without downloading.** The server advertises `Accept-Ranges: bytes`, so a small
`HttpZip` class gives Python's `zipfile` a seekable file object backed by HTTP range requests.
The archive's member list and one sample record's schema were read over the network before
committing to a 7 GB transfer.

**Schema.** Each JSON holds biflows keyed by 4-tuple, each with `packet_data`
(`timestamp`, `packet_dir`, `L4_payload_bytes`, `L4_header_bytes`, `iat`, `TCP_flags`,
`L4_raw_payload`, …), `flow_features`, and `flow_metadata`. The raw payload bytes are what make
the archive large and are discarded during reduction.

**Reduction.** Members are sharded across 32 containers; each parses only its shard and reduces
every flow to payload lengths, gaps and direction, truncated to the first 20 packets, written as
`float32` shards. Filters: at least 5 packets, at least 5 with non-zero payload, and a transport
flag derived from `L4_header_bytes == 8` (UDP).

Two passes were run: **UDP-only → 14,975 flows**, and **UDP+TCP → 90,610 flows** (16.5% UDP). The
second was adopted for pretraining volume, with the transport flag retained per flow.

## 7. Transformer and linear probes

**PacketFormer.** Each packet becomes one token: a linear projection of eight continuous
descriptors — `log1p(length)/7.5`, `length/1500`, `log1p(gap·10³)`, cumulative time share, gap
share, sub-millisecond flag, MTU-regime flag, small-packet flag — summed with a learned embedding
of the quantised size bin (24 geometric bins from 20 B to 1400 B) and a positional embedding, with
a `[CLS]` token. Encoder: 3 pre-norm layers, 4 heads, `d_model` 96, FFN 192, dropout 0.15.

**Pretraining** (8,000 steps, batch 256, AdamW, cosine schedule, one L4 GPU, 347 s):

1. **Masked packet modelling** — 15% of tokens replaced by a learned mask token; reconstruct the
   quantised size bin (cross-entropy) and the log-gap (Huber). ET-BERT/YaTC adapted to per-packet tokens.
2. **Contrastive** — NT-Xent between two traffic-plausible augmentations of the same flow
   (±10% multiplicative pacing jitter, ±2 B padding jitter).
3. **Domain-adversarial** — a gradient-reversal head separating auxiliary flows from unlabelled
   competition flows, with both sides presented as **five-packet views** so the discriminator
   cannot win on sequence length alone.
4. **Random truncation** — each batch truncated to 5 / 8 / 12 / 20 packets, so the encoder is
   equally at home on the five packets the competition provides.

No competition label is touched anywhere in pretraining.

**Linear probes.** The encoder is frozen and small heads are trained on its 96-dimensional
embedding. Every probe target comes from the auxiliary corpus, never from the competition label —
that is what makes probe outputs new information rather than leakage. All auxiliary flows are
truncated to the same five packets the competition provides, so each probe is applied to exactly
the input shape it was trained on.

| Probe | Head | Target | Features emitted |
|---|---|---|---|
| **P1** Application family | multinomial logistic | which of the 20 MIRAGE applications | 20 class probabilities + top-1, entropy, margin = 23 |
| **P2** Media modality | binary logistic | auxiliary proxy for interactive video: MTU fraction > 0.2 **and** sub-millisecond gap fraction > 0.4 | 1 video-likeness score |
| **P3** Bitrate & pacing | ridge ×3 | log bitrate, log mean gap, MTU fraction | 3 predictions + 3 **residuals** against the observed values | 
| **P4** Transport fingerprint | multinomial logistic | padding quantum | 0 — target was degenerate, see Results §19 |
| **P5** Embedding compression | PCA | none (label-free) | 16 principal components |

P3's residuals are the interesting half: a residual says *how unlike the auxiliary corpus this
flow paces*, which is itself discriminative in a way the raw prediction is not.

## 8. Ensembling and final selection

Ensembling reads the stored out-of-fold matrices and refits nothing. Members are restricted to
points carrying a full-data refit, deduplicated to the best few per family, and combined three
ways: hill-climbed soft-vote weights (greedy forward selection with replacement), stacking with a
multinomial logistic meta-learner, and rank averaging.

Selection uses only **honest estimates**: hill-climb weights are also scored *nested* (fitted on
four fifths of the out-of-fold rows, evaluated on the fifth), and stacking is scored under its own
inner CV. The fitted hill-climb number is reported for reference and never used to choose.

## 9. Submission construction

The winning configuration is refit on all 1,285 training rows and applied to the 327 test rows in
file order. Validation runs **before** the file is written and aborts on failure: exactly 327 rows,
two columns, no header, indices strictly `1..327`, every label in the ten-string case-sensitive
vocabulary. A distribution check compares the predicted class mix against a flows-per-call prior
(training flows ÷ 40 source calls per class, scaled to the 10 held-out test calls per class) — a
sanity check, not a constraint, since the task states the test split is not guaranteed uniform.

## 10. Infrastructure

Three images (numeric / +boosted-trees / +CUDA-torch), three volumes split by lifecycle
(`rtc-work` for inputs, features, out-of-fold matrices and reports; `rtc-mirage` for the archive
and reduced shards; `rtc-cache` for checkpoints and the Optuna journal), and ~20 functions sized
per stage: 2-CPU fan-out workers capped at 80 concurrent, 8-CPU analysis containers, L4 GPUs for
the transformer.

**Resilience.** Orchestration lives server-side (`search_stage`, `finalists_stage`) rather than in
the local process, and jobs are launched with `.spawn()` against the **deployed** app — a
disconnected terminal cannot cancel a sweep. Fan-out units are pure and idempotent with
`retries=2`; long stages checkpoint (download offsets, pretraining steps, Optuna journal).

---

# Part II — Results

## 11. Exploratory findings

| Finding | Evidence |
|---|---|
| Train and test are exchangeable | adversarial-validation AUC **0.529** (chance = 0.5), so cross-validation is a meaningful proxy for the leaderboard |
| The label is nearly determined by length | only **2 of 1,136** distinct length tuples map to more than one class — but those tuples cover **95 rows (7.4%)**, the repeated constant-size keepalive patterns |
| MTU fragmentation is a clean video marker | packets in the 1100–1600 B band are **10.7%** of video-class packets and **0.0%** of voice-class packets |
| Constant-size flows are a *flow type*, not an app | 125 flows have five identical lengths, concentrated in Discord (58 voice, 37 video); flagged explicitly rather than learned implicitly |
| Call mode is the harder factor | factorised, the application is recoverable at **0.912** and the call mode only at **0.888** — the reverse of the usual assumption |
| Group leakage is real but moderate | 253 of 1,285 flows (20%) share a pseudo-group |
| The flows-per-call prior is consistent | extrapolating training flows-per-call to 100 held-out test calls predicts ~321 test flows against the 327 supplied |

Baseline LightGBM on the engineered features, before any search: **0.810** plain 5-fold.

## 12. Search record

**2,377 points scored**, 83.6 container-hours of fitting executed in parallel.

| Family | Points |
|---|---|
| XGBoost | 962 |
| LightGBM | 626 |
| RandomForest | 588 |
| CatBoost | 61 |
| ExtraTrees | 36 |
| SVM | 27 |
| MLP | 23 |
| k-NN | 22 |
| HistGB | 20 |
| Logistic regression | 10 |
| PacketFormer | 2 |

Optuna outcomes: LightGBM 180 trials → 0.8249; XGBoost 180 → 0.8202; RandomForest 150 → 0.8023.
The CatBoost study aborted when a single trial exceeded the 3,600 s per-point timeout; its 61 grid
points remain in the record.

## 13. Leaderboard — best point per family (grouped CV)

| Rank | Model | Grouped accuracy | Macro-F1 | Feature set |
|---|---|---|---|---|
| 1 | **LightGBM** | **0.8249** | 0.8069 | full, no label-aware blocks |
| 2 | XGBoost | 0.8218 | 0.8067 | full |
| 3 | HistGradientBoosting | 0.8156 | 0.7959 | full |
| 4 | ExtraTrees | 0.8125 | 0.8009 | full |
| 5 | CatBoost | 0.8070 | 0.7895 | full |
| 6 | RandomForest | 0.8023 | 0.7866 | full + label-aware |
| 7 | PacketFormer (pretrained) | 0.7346 | 0.7262 | raw sequence |
| 8 | PacketFormer (scratch) | 0.7331 | 0.7239 | raw sequence |
| 9 | MLP | 0.7292 | 0.6792 | full |
| 10 | SVM (RBF) | 0.7268 | 0.6595 | full |
| 11 | Logistic regression | 0.7144 | 0.6666 | full |
| 12 | k-NN | 0.7132 | 0.6747 | full + label-aware |
| — | Majority class | 0.1992 | 0.0332 | — |

**Winning hyperparameters:** `n_estimators=800, num_leaves=31, learning_rate=0.12,
subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=10, reg_lambda=0.0,
class_weight_mode=none`.

## 14. Selected model — per-class performance

The submitted model is the same family re-scored under the full repeat budget: **0.8210 grouped,
0.8062 plain, 0.8028 macro-F1**.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| WhatsApp_voice | 0.976 | 1.000 | **0.988** | 80 |
| WhatsApp_video | 0.901 | 0.890 | 0.896 | 82 |
| Messenger_voice | 0.883 | 0.883 | 0.883 | 94 |
| Discord_video | 0.905 | 0.859 | 0.882 | 256 |
| Discord_voice | 0.829 | 0.916 | 0.870 | 238 |
| Messenger_video | 0.835 | 0.847 | 0.841 | 131 |
| Zoom_video | 0.736 | 0.744 | 0.740 | 172 |
| GoogleMeet_voice | 0.705 | 0.775 | 0.738 | 40 |
| GoogleMeet_video | 0.804 | 0.661 | 0.725 | 112 |
| Zoom_voice | 0.468 | 0.462 | **0.465** | 80 |

## 15. Confusion matrix (grouped out-of-fold, rows = truth)

| true \ pred | Dis_vo | Dis_vi | GM_vo | GM_vi | Msg_vo | Msg_vi | WA_vo | WA_vi | Zm_vo | Zm_vi |
|---|---|---|---|---|---|---|---|---|---|---|
| Discord_voice | **218** | 8 | 2 | 0 | 3 | 0 | 0 | 0 | 4 | 3 |
| Discord_video | 23 | **220** | 0 | 3 | 1 | 4 | 0 | 3 | 0 | 2 |
| GoogleMeet_voice | 1 | 0 | **31** | 3 | 1 | 0 | 0 | 0 | 2 | 2 |
| GoogleMeet_video | 2 | 6 | 7 | **74** | 1 | 9 | 0 | 1 | 4 | 8 |
| Messenger_voice | 8 | 0 | 0 | 0 | **83** | 1 | 2 | 0 | 0 | 0 |
| Messenger_video | 0 | 6 | 0 | 6 | 5 | **111** | 0 | 3 | 0 | 0 |
| WhatsApp_voice | 0 | 0 | 0 | 0 | 0 | 0 | **80** | 0 | 0 | 0 |
| WhatsApp_video | 0 | 3 | 0 | 1 | 0 | 5 | 0 | **73** | 0 | 0 |
| Zoom_voice | 6 | 0 | 3 | 2 | 0 | 1 | 0 | 0 | **37** | 31 |
| Zoom_video | 5 | 0 | 1 | 3 | 0 | 2 | 0 | 1 | 32 | **128** |

**The Zoom voice/video boundary is 27% of all errors.** Zoom_voice → Zoom_video (31) plus
Zoom_video → Zoom_voice (32) is 63 of the model's ~230 misclassifications, in a pair that is 20%
of the data. Every other confusion is single- or low-double-digit. Exploration predicted this
before any tuning.

## 16. Feature attribution (TreeSHAP, winner refit on all rows)

| Feature | Mean \|SHAP\| |
|---|---|
| `len_max` | 0.570 |
| `len_min` | 0.508 |
| `len_mean` | 0.209 |
| `len_median` | 0.190 |
| `len_0` | 0.183 |
| `len_fftmag_2` | 0.156 |
| `len_trend_r2` | 0.147 |
| `len_4` | 0.145 |
| `len_3` | 0.139 |
| `len_skew` | 0.136 |
| `len_kurt` | 0.132 |
| `iat_frac_3` | 0.131 |

Packet-size statistics dominate; the first timing feature appears eleventh. This matches the
physics — codec frame sizes and MTU fragmentation survive encryption, while five packets is too
short a window to measure pacing precisely.

## 17. Ablation — label-aware blocks F9–F11

Matched configurations, full repeat budget:

| Model | Feature set | Label-aware blocks | Grouped | Plain | Macro-F1 |
|---|---|---|---|---|---|
| LightGBM | full | **off** | **0.8152** | 0.8122 | 0.7974 |
| LightGBM | top-120 | off | 0.8097 | 0.8057 | 0.7918 |
| XGBoost | full | off | 0.8016 | 0.8057 | 0.7812 |
| LightGBM | full | on | 0.7969 | 0.7925 | 0.7785 |
| LightGBM | top-120 | on | 0.7930 | 0.7855 | 0.7741 |
| XGBoost | top-120 | on | 0.7914 | 0.7816 | 0.7764 |
| XGBoost | top-60 | off | 0.7911 | 0.7834 | 0.7732 |
| XGBoost | top-120 | off | 0.7907 | 0.7979 | 0.7712 |
| LightGBM | top-60 | off | 0.7895 | 0.7953 | 0.7706 |
| XGBoost | full | on | 0.7883 | 0.7829 | 0.7726 |
| LightGBM | top-60 | on | 0.7825 | 0.7782 | 0.7674 |
| XGBoost | top-60 | on | 0.7732 | 0.7746 | 0.7563 |

**Two conclusions.** The class-conditional Markov, density and neighbourhood blocks cost about
**1.8 accuracy points** on LightGBM even after the inner out-of-fold fix; at 1,285 rows they
contribute variance rather than signal, and they are disabled by default. And the **full 241-column
set beats both compact subsets** in every matched pair — mutual-information pruning discards
usable signal here.

## 18. Ablation — probe block F12

| Model | Probes | Grouped | Plain | Macro-F1 | Δ grouped |
|---|---|---|---|---|---|
| LightGBM | off | 0.8144 | 0.8096 | 0.7957 | — |
| LightGBM | **on** | 0.8144 | 0.8031 | 0.7965 | **0.0000** |
| XGBoost | off | 0.8082 | 0.8091 | 0.7861 | — |
| XGBoost | **on** | 0.7996 | 0.7987 | 0.7810 | **−0.0086** |
| RandomForest | off | 0.7778 | 0.7803 | 0.7578 | — |
| RandomForest | **on** | 0.7782 | 0.7787 | 0.7563 | **+0.0004** |

## 19. Linear probes — held-out results

All metrics on a held-out 20% split of the auxiliary corpus (18,122 test flows for P1). Metrics
recorded during feature construction were fit-to-itself; these are not.

| Probe | Metric | Result | Baseline | Features shipped |
|---|---|---|---|---|
| **P1** application family | top-1 over 20 classes | **0.5029** | 0.1015 (majority) | 23 |
| **P1** application family | top-3 | **0.7318** | ~0.264 | — |
| **P2** media modality | ROC-AUC | **0.9977** | 0.5 | 1 |
| **P2** media modality | accuracy | 0.9955 | 0.991 | — |
| **P3** log bitrate | held-out R² | **0.9473** (train 0.9475) | — | 2 (pred + residual) |
| **P3** log mean gap | held-out R² | **0.9529** (train 0.9536) | — | 2 |
| **P3** MTU fraction | held-out R² | **0.9480** (train 0.9504) | — | 2 |
| **P4** transport fingerprint | — | **degenerate** | — | **0** |
| **P4** corrected | accuracy | **0.9403** | 0.7222 (majority) | not shipped |
| **P5** embedding PCA | variance retained | 96.4% in 16 components | — | 16 |

**P2's accuracy is misleading and its AUC is not.** Only 0.9% of auxiliary flows are positive
under the video-likeness proxy, so 0.9955 accuracy is nearly free; the 0.9977 AUC is the number
that says the embedding genuinely separates the classes.

**P3 shows no overfit** — held-out R² matches train R² to three decimals across all three targets.

### The P4 defect

P4's target was "which modulus in {2, 4, 8, 16} has the largest share of divisible packet sizes".
Every byte count divisible by 4 is also divisible by 2, so modulus 2 always wins and the target
has **exactly one class**. The builder's single-class guard skipped the probe silently, and P4
contributed **zero of the 46 shipped probe features**.

Re-specified as "the strongest power-of-two alignment obeyed by at least 80% of packets" (5 levels,
72.2% of flows at 16-byte alignment), the same frozen embedding predicts it at **0.9403 against a
0.7222 majority baseline**. The signal is real; the original definition was wrong. The corrected
probe was evaluated after the fact and is **not** in the feature set any model trained on.

### P1 per-class (20-way, held out)

| Application | Precision | Recall | F1 | Test flows |
|---|---|---|---|---|
| Webex | 0.761 | 0.689 | 0.723 | 1,162 |
| Telegram | 0.554 | 0.683 | 0.612 | 688 |
| Signal | 0.598 | 0.582 | 0.590 | 478 |
| **GoogleMeet** ● | 0.729 | 0.470 | 0.572 | 1,180 |
| **WhatsApp** ● | 0.918 | 0.410 | 0.567 | 217 |
| **Zoom** ● | 0.551 | 0.573 | 0.562 | 1,621 |
| **Messenger** ● | 0.533 | 0.526 | 0.529 | 1,280 |
| Trueconf | 0.455 | 0.620 | 0.525 | 540 |
| Twitch | 0.419 | 0.627 | 0.503 | 1,250 |
| Skype | 0.595 | 0.431 | 0.500 | 1,806 |
| **Discord** ● | 0.434 | 0.550 | 0.485 | 1,840 |
| Crunchyroll | 0.423 | 0.546 | 0.477 | 978 |
| Omlet | 0.587 | 0.345 | 0.434 | 528 |
| Teams | 0.350 | 0.555 | 0.429 | 1,589 |
| JitsiMeet | 0.537 | 0.330 | 0.408 | 355 |
| Slack | 0.500 | 0.288 | 0.365 | 1,247 |
| ClashRoyale | 0.714 | 0.225 | 0.342 | 89 |
| GotoMeeting | 0.450 | 0.266 | 0.334 | 958 |
| Line | 0.766 | 0.206 | 0.325 | 286 |
| KakaoTalk | 0.400 | 0.067 | 0.114 | 30 |

Macro average: precision 0.564, recall 0.449, F1 0.470. ● marks the five applications that are
also competition classes.

### Cross-corpus transfer

The premise of the auxiliary phase: MIRAGE contains all five competition applications, but
captured on mobile devices in Naples while the competition data was captured over lab Wi-Fi at
Inha University. Does an application's packet signature survive the change of capture environment?

| Measure | Value |
|---|---|
| Agreement, full 20-way head | **0.1525** |
| Agreement, restricted to the 5 shared apps | **0.2926** (chance 0.20) |
| Competition flows assigned to one of the 5 shared apps | 0.4996 |

Predicted application (columns) for each true competition application (rows), top columns:

| true \ probe says | Discord | GoogleMeet | Zoom | JitsiMeet | Trueconf | Teams | Twitch | Skype |
|---|---|---|---|---|---|---|---|---|
| Discord | **133** | 15 | 68 | 36 | 48 | 36 | 59 | 37 |
| GoogleMeet | 11 | **42** | 41 | 15 | 7 | 0 | 20 | 2 |
| Messenger | 62 | 23 | 26 | 2 | 24 | 38 | 9 | 13 |
| WhatsApp | 33 | 19 | 61 | 0 | 3 | 11 | 7 | 1 |
| Zoom | 2 | 64 | 14 | 68 | 44 | 11 | 5 | 6 |

**Transfer is weak and its failure mode is coherent.** Discord is the one clear success. Elsewhere
the probe drifts to a plausible neighbour: Zoom → JitsiMeet/GoogleMeet, WhatsApp → Zoom,
Messenger → Discord. The embedding learned *"this is video-conferencing traffic"* rather than
*"this is Zoom"* — exactly the granularity a mobile-versus-Wi-Fi domain gap would leave intact.

### How much competition signal do the probe columns carry?

| Probe feature | Source | Mutual information with the 10-class label |
|---|---|---|
| `probe_emb_pc0` | P5 | 0.749 |
| `probe_emb_pc1` | P5 | 0.590 |
| `probe_pred_log_bitrate` | P3 | 0.553 |
| `probe_pred_frac_mtu` | P3 | 0.542 |
| `probe_app_Crunchyroll` | P1 | 0.517 |
| `probe_resid_frac_mtu` | P3 | 0.510 |
| `probe_video_like` | P2 | 0.482 |
| `probe_emb_pc2` | P5 | 0.481 |
| `probe_app_JitsiMeet` | P1 | 0.429 |
| `probe_emb_pc9` | P5 | 0.410 |

For scale: strongest raw engineered feature **1.271**; median probe column **0.326**; median raw
feature **0.221**. A LightGBM trained on the 46 probe columns **and nothing else** scores
**0.7019** on the same grouped folds (majority class 0.1992).

**The probes are informative and redundant at the same time.** They carry real structure —
bitrate, fragmentation, media type — but nothing the direct measurements of those same five
packets do not already expose, which is why adding them to the 241-column set changed nothing
(§18).

## 20. Transformer results

| Variant | Grouped | Plain | Macro-F1 | Wall clock |
|---|---|---|---|---|
| PacketFormer, from scratch | 0.7331 | 0.7276 | 0.7239 | 775 s (L4) |
| PacketFormer, MIRAGE-pretrained | **0.7346** | 0.7401 | 0.7262 | 1,004 s (L4) |

Pretraining bought **+0.0015**. Final pretraining losses: total 2.664, masked-bin 1.146,
masked-gap 0.959, contrastive 2.074, domain 0.016. Both variants sit ~9 points below the boosted
trees: five tokens is very little sequence for attention, and 1,285 labelled rows is very little
to fine-tune on. Kept in the ensemble pool as a diversity member.

## 21. Ensemble comparison

| Combiner | Score | Estimate type |
|---|---|---|
| **Single best LightGBM** | **0.8210** | grouped out-of-fold — **selected** |
| Hill-climbed blend | 0.8195 | nested (weights fitted on 4/5, scored on 1/5) |
| Stacking, logistic meta-learner | 0.8187 | nested |
| Hill-climbed blend | 0.8319 | fitted to its own evaluation rows — optimistic, not used |
| Rank averaging | 0.7735 | grouped out-of-fold |

The blend selected four members at equal weight (two LightGBM, one XGBoost, one HistGB), and its
fitted score is 1.1 points above its nested score — a clean illustration of why the fitted number
cannot be used to choose. On honest estimates the single model wins.

## 22. Submission

327 rows, no header, indices 1–327, all ten labels present, mean predicted confidence **0.958**.

| Class | Predicted flows | Predicted share | Prior share | Ratio |
|---|---|---|---|---|
| Discord_voice | 70 | 21.41% | 18.52% | 1.156 |
| Discord_video | 59 | 18.04% | 19.92% | 0.906 |
| GoogleMeet_voice | 9 | 2.75% | 3.11% | 0.884 |
| GoogleMeet_video | 22 | 6.73% | 8.72% | 0.772 |
| Messenger_voice | 23 | 7.03% | 7.32% | 0.962 |
| Messenger_video | 35 | 10.70% | 10.19% | 1.050 |
| WhatsApp_voice | 19 | 5.81% | 6.23% | 0.933 |
| WhatsApp_video | 22 | 6.73% | 6.38% | 1.054 |
| Zoom_voice | 19 | 5.81% | 6.23% | 0.933 |
| Zoom_video | 49 | 14.98% | 13.39% | 1.119 |

Every class lands within 23% of its flows-per-call prior share, most within 10%.

## 23. Limitations and next steps

1. **The Zoom voice/video boundary is the whole remaining headroom.** 27% of errors sit in one
   pair. A dedicated binary model over Zoom-looking flows, with features built around the Zoom
   audio frame sizes visible in the exploration histograms, is worth more than further global
   hyperparameter search.
2. **P4 should be fixed and shipped.** The corrected padding-alignment probe reaches 0.940 against
   a 0.722 baseline and encodes per-application crypto and header overhead — orthogonal to the size
   statistics that dominate SHAP, and the one probe with a real chance of not being redundant.
3. **The auxiliary encoder should be retrained on UDP media only.** Only 16.5% of the reduced
   corpus is UDP; TCP pacing is governed by congestion control rather than a codec. The corpus was
   widened to TCP for volume, and that is a plausible cause of the weak transfer.
4. **Grouped CV is a proxy, not the truth.** Without call identifiers the pseudo-group key is a
   heuristic; the real held-out-call generalisation gap could be larger than 0.8210 suggests.
5. **CatBoost is under-searched** — its Optuna study aborted on a per-point timeout, so only 61
   grid points inform its 0.8070.

---

## Appendix A — reproduction

```
modal deploy src/modal_app.py
modal run src/modal_app.py::seed
modal run src/modal_app.py::eda
modal run src/modal_app.py::features
python scripts/spawn.py search_stage    '{"stage": "all_trees"}'
python scripts/spawn.py search_stage    '{"stage": "classic"}'
python scripts/spawn.py search_driver   '{"model": "lgbm", "n_trials": 180, "batch_size": 30}'
python scripts/spawn.py mirage_fetch    '{}'
python scripts/spawn.py pretrain_encoder '{}'
python scripts/spawn.py build_probe_features '{"pretrained": "/cache/pretrain/encoder.pt"}'
python scripts/spawn.py evaluate_probes '{}'
python scripts/spawn.py finalists_stage '{"top_k": 28, "per_family": 4}'
python scripts/spawn.py run_ensemble    '{"per_model": 5, "top_n": 22}'
python scripts/spawn.py make_submission '{}'
python scripts/spawn.py write_model_card '{}'
python scripts/spawn.py write_shap      '{}'
modal volume get rtc-work submission.csv ./submission.csv
modal volume get rtc-work reports ./reports_dl
```

`rebuild_cv_results` reconstructs the full search record from the stored out-of-fold files if a
run is interrupted.

## Appendix B — source map

| Path | Role |
|---|---|
| `src/config.py` | paths, class vocabulary, CV constants, protocol bands |
| `src/io_utils.py` | loading, validation, label encoding, pseudo-group key |
| `src/features/base.py` | blocks F1–F8 |
| `src/features/likelihood.py` | blocks F9–F11 (fitted, label-aware) |
| `src/features/pipeline.py` | cached artifacts and the per-container handle |
| `src/models/cv.py` | fold generation and storage |
| `src/models/registry.py` | model factory and class-weight translation |
| `src/models/evaluate.py` | the fan-out unit: score one point, store its matrices |
| `src/models/grids.py` | grid definitions and stage registry |
| `src/models/optuna_search.py` | Optuna spaces and ask/tell spec construction |
| `src/models/ensemble.py` | hill-climb, stacking, rank averaging, nested scoring |
| `src/transformer/model.py` | PacketFormer |
| `src/transformer/dataset_rtc.py` | tensor encoding and traffic-plausible augmentation |
| `src/transformer/mirage_stream.py` | ranged-HTTP zip access, download, reduction |
| `src/transformer/pretrain.py` | masked modelling + contrastive + domain-adversarial |
| `src/transformer/probe.py` | probes P1–P5 → feature block F12 |
| `src/transformer/probe_eval.py` | held-out probe evaluation and transfer analysis |
| `src/eda/run_eda.py` | exploratory analysis and report generation |
| `src/reporting.py` | model card and SHAP attribution |
| `src/predict.py` | submission construction and validation |
| `src/modal_app.py` | Modal app: images, volumes, functions, entrypoints |
| `scripts/spawn.py` | fire-and-forget launcher against the deployed app |
