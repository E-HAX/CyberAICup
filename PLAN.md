# Implementation Plan — RTC: Encrypted RTC Application Identification from Media Packet Sequences

**Competition:** CyberAI Cup 2026 (ICSDS), Task 3
**Goal:** Given the sizes and inter-arrival times of the first 5 packets of an encrypted UDP media flow, predict one of 10 classes = {Discord, GoogleMeet, Messenger, WhatsApp, Zoom} x {voice, video}.
**Data:** `Task3/RTC_CyberAICup2026/Training_set.csv` (1,285 labelled flows), `Testing_set.csv` (327 unlabelled flows).
**Deliverable:** `submission.csv`, 327 rows, no header, `index(1-based),label`, same order as the test file.
**Execution substrate:** **everything runs on [Modal](https://modal.com)**. The laptop only edits code and issues `modal run`. No EDA, training, search, download or inference happens locally.

---

## 0. Facts already verified about the data

Confirmed by direct inspection before writing this plan:

| Fact | Value |
|---|---|
| Train rows / Test rows | 1,285 / 327 |
| Columns | `relative_time_{0..4}`, `packet_length_{0..4}`, `label` (train only) |
| `relative_time_0` | always 0.0 |
| Packet length range (`packet_length_0`) | 26 – 1217 bytes |
| Flow duration (`relative_time_4`) | 48 µs – 23.2 s, median 571 µs |
| Class balance | Discord_video 256 … GoogleMeet_voice 40 (6.4x imbalance) |
| Unique 5-length tuples | 1,136 of 1,285 (149 exact duplicates) |
| Length tuples mapping to >1 class | 2 → intrinsic Bayes error is essentially zero on lengths alone; the task is learnable, not degenerate |
| Flows where all 5 packets have identical length | 125, dominated by Discord_voice (58) and Discord_video (37) — signature of STUN/keepalive-style constant-size flows (many are the 47-byte pattern) |
| Median mean-packet-length per class | voice classes 33–148 B, video classes 391–914 B → length alone separates voice/video strongly; app-ID is the hard half |

**Local machine:** macOS arm64, 10 cores, 16 GB RAM, 35 GB free disk, `~/.modal.toml` already present with `token_id` + `token_secret` (Modal is authenticated; the CLI itself is not installed yet). The local machine's disk and cores are now irrelevant to the plan — they were the binding constraint in the pre-Modal version and Modal removes them.

**Aux dataset:** `https://traffic.comics.unina.it/mirage/MIRAGE/MIRAGE-AppAct-2024.zip` is live, `Content-Length: 6,973,763,076` (6.97 GB), `Accept-Ranges: bytes`. It is downloaded **by a Modal container straight into a Modal Volume**, never onto the laptop.

---

## 1. Guiding constraints (things that decide the whole design)

1. **Tiny dataset, wide feature space.** 1,285 rows. Any feature engineering that produces hundreds of columns must be paired with strong regularisation and honest CV, or it will overfit silently.
2. **Grouping leakage risk.** Flows are drawn from 400 source calls, but **no call ID is provided**. Several flows from the same call sit in the training set and are near-identical. Plain stratified K-fold will therefore be *optimistic* relative to the test set, which comes from 100 *held-out* calls. Mitigation: report both plain stratified CV and a **pseudo-group CV** where near-duplicate flows (clustered by rounded length tuple + duration decade) are forced into the same fold. Model selection uses the pessimistic (grouped) score.
3. **Unknown test distribution.** The brief explicitly says do not assume the test split is uniform. Optimise **accuracy** (the submission format implies plain accuracy), track **macro-F1** as a secondary guard, and treat class weighting as a searched hyperparameter rather than an assumption.
4. **Any target-derived feature must be out-of-fold.** Class-conditional likelihoods, kNN-distance features and target encodings are computed *inside* the CV loop with fold-fitted estimators, and refit on full train for the test set.
5. **"Random forest regressor" → classifier here.** This is a 10-class classification problem, so the ensemble uses `RandomForestClassifier`. A `RandomForestRegressor` still appears, but as a *feature builder*: probe heads predicting continuous auxiliary targets (log bitrate, log mean IAT), whose predictions and residuals become features.
6. **Modal-specific constraint: containers are ephemeral and parallel.** Nothing may rely on local filesystem state surviving a call. Every artifact goes to a Modal **Volume**, with explicit `volume.commit()` after writes and `volume.reload()` before cross-container reads. Every function is idempotent and re-runnable, because preemption and retries are normal.

---

## 2. Modal architecture

### 2.1 App, images, volumes

Single Modal App, `rtc-cyberai`, defined in `src/modal_app.py`, with three images so a CPU job never pays to build the CUDA layer:

| Image | Base | Contents | Used by |
|---|---|---|---|
| `IMG_BASE` | `debian_slim(python_version="3.12")` | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn | EDA, feature builds, submission |
| `IMG_ML` | `IMG_BASE` + | xgboost, lightgbm, catboost, optuna, umap-learn, shap, imbalanced-learn | model fitting, search, ensembling |
| `IMG_GPU` | `debian_slim` + CUDA wheels | torch, plus `IMG_ML` contents | transformer pretrain/fine-tune/probes |

Images are built with `.uv_pip_install(...)` and **fully version-pinned** (a `requirements.lock` baked in with `add_local_file`), so a run six weeks from now reproduces exactly. Project source is attached with `image.add_local_python_source("src")` so code edits do not trigger an image rebuild — only dependency changes do.

Three Volumes, split by lifecycle rather than by convenience:

| Volume | Holds | Why separate |
|---|---|---|
| `rtc-work` | input CSVs, feature matrices, OOF probability matrices, fitted models, `reports/`, `submission.csv` | small, rewritten constantly |
| `rtc-mirage` | the 6.97 GB zip + the reduced `.npz` shards | large, written once, expensive to refetch |
| `rtc-cache` | Optuna journal, HF/torch caches, checkpoints | churn-heavy, safe to wipe |

The two provided CSVs (185 KB total) are uploaded into `rtc-work/data/` by a one-shot `seed_data` entrypoint using `add_local_dir`, so remote functions never depend on the laptop being online.

### 2.2 Function inventory

Every stage is one Modal function. Resource requests are per-stage rather than one fat container:

| Function | Image | Resources | Role |
|---|---|---|---|
| `seed_data` | BASE | 0.5 CPU | copy the provided CSVs into `rtc-work` |
| `run_eda` | BASE | 4 CPU / 8 GB | all of §3; writes figures + `eda_report.md` |
| `build_features` | ML | 4 CPU / 16 GB | deterministic feature blocks F1–F8 |
| `fit_eval` | ML | 2 CPU / 4 GB | **fan-out unit**: fit + CV-score one (model, hyperparameter) point, return metrics + OOF probs |
| `optuna_batch` | ML | 2 CPU / 4 GB | evaluate one asked batch of Optuna trials |
| `search_driver` | ML | 2 CPU / 4 GB | orchestrates grid + Optuna ask/tell over `fit_eval.map()` |
| `mirage_fetch` | BASE | 2 CPU / 8 GB, 4 h timeout | stream the 6.97 GB zip into `rtc-mirage` |
| `mirage_reduce` | BASE | 2 CPU / 8 GB | **fan-out unit**: parse one zip member shard → compact arrays |
| `pretrain` | GPU | 1× L4 (A10G if needed) / 16 GB | masked-modelling + contrastive pretraining on MIRAGE |
| `finetune` | GPU | 1× L4 | supervised transformer on RTC, per CV fold (fanned out) |
| `probe_features` | GPU | 1× L4 | frozen-encoder linear probes P1–P5 → feature block F12 |
| `ensemble` | ML | 4 CPU / 8 GB | hill-climb weights + stacking over stored OOF matrices |
| `predict_submit` | ML | 4 CPU / 8 GB | refit on full train, write + validate `submission.csv` |

Cross-cutting settings: `timeout=` sized per stage, `retries=2` on the fan-out units (they are pure functions of their inputs), `max_containers` capped per stage so a runaway grid cannot open 500 containers at once, and long stages launched with `modal run --detach` so closing the laptop does not kill them.

### 2.3 Why Modal actually helps here (not just "cloud")

- **Grid search is embarrassingly parallel and this plan has thousands of points.** `fit_eval.map(param_points)` turns a multi-hour sequential sweep into a wide fan-out across containers, each with its own 2 CPUs. This is the single largest wall-clock win.
- **Per-fold transformer training** fans out the same way (5 folds × 3 repeats = 15 independent GPU jobs).
- **The 6.97 GB aux download stops being a local-disk problem.** A Modal container on a fast link writes it into a Volume once; every later stage reads from the Volume. The plan can now use a much larger slice of MIRAGE than the 1–3 GB subset the 35 GB laptop disk forced.
- **GPU on demand.** An L4 for the pretraining hours, nothing the rest of the time.
- **The artifact store is the Volume**, so results survive across sessions and machines; the laptop pulls only `reports/` and `submission.csv` via `modal volume get`.

### 2.4 Orchestration and the local surface

`@app.local_entrypoint()` functions are the only things a human invokes:

```
modal run src/modal_app.py::seed_data
modal run src/modal_app.py::eda
modal run src/modal_app.py::features
modal run --detach src/modal_app.py::search      --stage gbdt
modal run --detach src/modal_app.py::aux_pipeline               # fetch → reduce → pretrain
modal run --detach src/modal_app.py::probes
modal run src/modal_app.py::ensemble
modal run src/modal_app.py::submit
modal volume get rtc-work reports ./reports
modal volume get rtc-work submission.csv ./submission.csv
```

Local setup is exactly: `uv tool install modal` (the token in `~/.modal.toml` is already valid). `modal shell` is used for interactive debugging against the real Volumes; `modal app logs rtc-cyberai` for detached runs.

---

## 3. Phase 1 — Domain-grounded EDA  *(Modal fn: `run_eda`)*

Not generic `df.describe()` dumps. Every plot answers a question a network-traffic analyst would actually ask about SRTP-over-DTLS media flows. Figures are written as PNG **and** the numeric backing tables as CSV into `rtc-work/reports/eda/`, so conclusions are auditable without re-running anything.

**A. Sanity & integrity**
- Schema/dtype validation, missingness, `relative_time_0 == 0` check, monotonicity of `relative_time_i` (non-monotonic rows = capture reordering, flagged).
- Exact-duplicate rows overall and per class; duplicate length-tuples spanning multiple classes (Bayes-error floor).
- Train vs test comparison: per-feature KS test + ECDF overlays, plus **adversarial validation** (train-vs-test discriminator AUC). AUC ≈ 0.5 means the splits are exchangeable; a high AUC forces a covariate-shift-aware strategy.

**B. Packet-length structure (the codec/transport fingerprint)**
- Per-class histograms and violins of all 5 lengths, on log and linear (0–1400) axes.
- **Quantisation analysis:** value-count spikes per class. Opus/AAC frames, SRTP header + auth-tag overhead and per-app proprietary headers (e.g. WhatsApp WASP) create discrete repeated sizes. Top-20 discrete lengths per class as a heatmap.
- **MTU-regime analysis:** fraction of packets in 1100–1250 B (WebRTC video fragments hitting path MTU) per class — expected to be a near-clean video/voice discriminator and an app discriminator.
- **Small-packet analysis:** the 26–100 B band (STUN binding, DTLS/RTCP, comfort-noise/DTX audio). The 47-byte constant-length flows are examined separately — a distinct *flow type*, not a distinct app, and models must not overfit to them.
- Length *sequence shape*: mean length by packet index per class, and first-packet-vs-rest deltas — packet 0 is frequently a handshake/binding packet, structurally unlike packets 1–4.

**C. Timing structure (the pacing fingerprint)**
- Inter-arrival times `d_i = t_i - t_{i-1}`: per-class log-scale KDEs. Expected multimodality: sub-millisecond (fragments of one video frame), ~10–30 ms (audio ptime pacing, typically 20 ms), and long idle/keepalive tails (up to 23 s here).
- Fraction of IATs `< 1 ms` per class → burst/fragmentation index, a strong video marker.
- Flow-duration ECDF per class on a log axis, with the heavy tail called out explicitly.
- `d_1` vs `d_2` scatter coloured by class, revealing per-app pacing regularity.

**D. Joint size–time structure**
- 2-D hexbin of (packet length, log IAT) per class — a 5-packet "FlowPic"-style view of the class manifold.
- Per-class instantaneous rate `length_i / d_i` distributions (proxy for stream bitrate).

**E. Class-separability diagnostics**
- Mutual information and ANOVA F of each raw and engineered feature vs label.
- PCA / t-SNE / UMAP scatter of the standardised feature space, coloured by app and by mode **separately** — expectation: mode separates cleanly, app is the residual difficulty.
- A quick 5-fold LightGBM baseline confusion matrix to name the actual hard pairs before investing in features.
- Hierarchical check: accuracy of a 5-class app-only model and a 2-class mode-only model vs the flat 10-class model — decides whether the hierarchical classifier in §5 is worth building.

**F. Class imbalance & flow-per-call structure**
- Flow count per class vs the brief's stated call counts → implied flows-per-call per app (Discord fans out ~6× more flows per call than GoogleMeet_voice). A real prior about the test set, documented rather than assumed.

**Output:** `rtc-work/reports/eda_report.md` with every figure embedded and a written findings list naming the specific features each finding motivates.

---

## 4. Phase 2 — Feature engineering  *(Modal fns: `build_features`, then fold-internal code inside `fit_eval`)*

Grounded in the encrypted-traffic-classification literature: Taylor et al. *AppScanner* (statistical packet-size summaries), Draper-Gil et al. *ISCXVPN* (time-based flow features), Korczyński & Duda (Markov-chain fingerprinting of encrypted sessions), Shapira & Shavitt *FlowPic* (2-D size–time histogram), Lopez-Martin et al. (CNN+RNN over per-packet size/IAT sequences), Aceto et al. *MIMETIC/DISTILLER* and the Nascita explainable-DL work (all built on MIRAGE — the same lineage as the aux dataset), Lin et al. *ET-BERT* and Zhao et al. *YaTC* (pretrained traffic transformers, which motivate §6).

**F1 — Raw & normalised.** 10 raw columns; `log1p` of lengths and of times.

**F2 — Inter-arrival.** `d_1..d_4`, `log1p(d_i)`, `d_i/duration`, cumulative time fractions, and min/max/mean/std/median/IQR/CV/skew/kurtosis of the IAT vector.

**F3 — Length statistics.** min/max/mean/std/median/IQR/range/sum/CV/skew/kurtosis, number of distinct lengths, normalised Shannon entropy of the length multiset, all-equal flag, `argmax`/`argmin` index.

**F4 — Sequence shape.** All 10 pairwise differences and 10 pairwise ratios of lengths; first-order diffs and their signs; count of increases/decreases; longest monotone run; `l_i/l_0` and `l_i/max(l)`; linear-trend slope and R² of length vs index; the same for cumulative bytes vs time.

**F5 — Protocol-semantics flags (domain features).** Indicators/counts for `length == 47` and the other observed constant-size keepalive values; `length < 100` (STUN/RTCP/DTX band); `100 ≤ length < 300` (Opus audio + SRTP overhead); `length > 1100` (MTU-regime video fragment); `length mod 4`, `mod 8`, `mod 16` (padding / block-cipher alignment residues); `length − 12` and `− 12 − 10` (RTP header and SRTP auth-tag adjusted payload); is-first-packet-anomalous flag.

**F6 — Burst / frame-group structure.** Segment the 5 packets at IAT thresholds {0.5 ms, 1 ms, 5 ms, 20 ms}: number of groups, packets per group, bytes per group, max group bytes, inter-group gap stats. A video frame fragmented across the MTU gives one tight group; audio at 20 ms ptime gives singletons. Plus counts of IATs below/above each threshold and the longest sub-millisecond run.

**F7 — Rate & throughput.** Total bytes, bytes/second, packets/second, per-gap instantaneous rate `l_i/d_i` (stats over the 4 gaps), and log versions.

**F8 — Spectral / transform.** 5-point DCT and FFT-magnitude coefficients of the length sequence and of the log-IAT sequence (captures alternation such as large/small/large fragment interleaving).

**F9 — Class-conditional Markov likelihoods (OOF).** Quantise lengths into K bins (K ∈ {8,16,32}, quantile edges fit on fold-train only) and IATs into log-spaced bins. Fit one first-order Markov chain per class on fold-train; emit the 10 log-likelihoods, softmax posteriors, argmax, and top1–top2 margin. Korczyński–Duda fingerprinting adapted to a 5-packet sequence.

**F10 — Class-conditional density likelihoods (OOF).** Per-class GMM (2–4 diagonal components) and KDE on a compact standardised subspace (mean length, log duration, burst index, MTU fraction); emit per-class log-densities + posteriors.

**F11 — Neighbourhood features (OOF).** Distance to the k nearest fold-train neighbours *of each class* (k ∈ {1,3,5}) under a scaled metric, plus class-vote fractions among the 10 nearest overall.

**F12 — Transformer / linear-probe features.** Delivered by §6; slots into the same fold-safe pipeline.

**Split of labour on Modal.** F1–F8 are pure row-wise transforms with no fitted state → computed once by `build_features` and cached to `rtc-work/features/base.parquet`, so the hundreds of `fit_eval` containers each load a ready matrix instead of recomputing. F9–F11 are *fitted* and therefore live **inside** the CV loop in `fit_eval`, refit per fold. This split is what keeps a wide fan-out cheap without leaking.

**Selection & hygiene.** After assembly (~250–400 columns): drop zero-variance and >0.995-correlated duplicates; rank by permutation importance and SHAP on a fold-averaged LightGBM; keep a "compact" set (top ~60) and a "full" set, and let CV choose per model family. All fitted transformers (scalers, bin edges, GMMs, kNN indices, encoders) live inside an sklearn `Pipeline` so they refit per fold — no leakage by construction.

---

## 5. Phase 3 — Models, hyperparameter search, ensembling  *(Modal fns: `fit_eval`, `optuna_batch`, `search_driver`, `ensemble`)*

**CV protocol (fixed once, used everywhere):** `RepeatedStratifiedKFold(n_splits=5, n_repeats=3, seed=42)` for the primary score, plus the pseudo-grouped 5-fold from §1.2 as the pessimistic score. The fold assignment is computed **once** by `build_features` and stored in the Volume as an explicit index array, so every one of the hundreds of parallel containers scores on byte-identical folds — otherwise cross-container comparisons are meaningless.

**Individual models**
- Baselines: majority class, multinomial logistic regression (elastic-net), linear SVM.
- Distance/kernel: kNN (distance weighting, searched metric), RBF-SVM (probability-calibrated).
- Bagged trees: **RandomForestClassifier**, ExtraTreesClassifier.
- Boosted trees: **XGBoost** (`multi:softprob`), **LightGBM** (`multiclass` + a `dart` variant), **CatBoost** (`MultiClass`), sklearn `HistGradientBoosting`.
- Neural: sklearn MLP and a small PyTorch MLP with label smoothing + mixup (a useful diversity source).
- Hierarchical variant: mode (voice/video) × per-mode 5-way app classifier, and 5-way app × 2-way mode, combined by the product rule. §3E decides whether it beats the flat model; it is kept as an ensemble member regardless because its error profile differs.

**Hyperparameter search — the part Modal changes most**
- **Grid stage (explicitly requested).** Rather than sklearn's in-process `GridSearchCV`, the grid is materialised as an explicit list of parameter dicts and executed as `fit_eval.map(points)` — a *distributed* `GridSearchCV` where each point is its own container. Example grids: XGBoost `max_depth {3,4,6,8} × lr {0.02,0.05,0.1} × subsample {0.7,0.9} × colsample {0.6,0.8,1.0} × min_child_weight {1,3,5} × reg_lambda {0.5,1,5}` (1,296 points); RF `n_estimators {500,1000} × max_depth {None,8,16} × max_features {sqrt,log2,0.3} × min_samples_leaf {1,2,4} × class_weight {None,balanced,balanced_subsample}`; comparable grids for LGBM/CatBoost/SVM/kNN/LogReg. Sequentially these are hours; fanned out they are minutes. `sklearn`'s own `n_jobs` stays at 1 inside each container — parallelism lives at the Modal layer, not nested underneath it.
- **Optuna stage.** TPE + median pruner over the *continuous* neighbourhood of each grid winner, 200–400 trials per GBDT. Distributed via **ask/tell batching**: `search_driver` holds the study, asks for a batch of ~32 trials, evaluates them with `optuna_batch.map()`, tells the results back, repeats. The study is persisted to a journal file in `rtc-cache` so a killed driver resumes instead of restarting. (Ask/tell batching is chosen over a shared RDB storage because it needs no database and no concurrent-writer semantics on a Volume.)
- Both stages report their best; whichever wins on grouped CV is used. Every evaluation writes `{model, params, cv_acc, cv_macro_f1, grouped_acc, fit_seconds}` plus its OOF probability matrix to `rtc-work/oof/`, giving a complete searchable record in `reports/cv_results.csv`.
- Class imbalance is a searched hyperparameter, not an assumption: `{no weighting, balanced weights, SMOTE-variant oversampling}`.

**Ensembling (three layers; best on grouped CV wins)** — runs in one container because it reads stored OOF matrices and refits nothing expensive:
1. **Soft voting** over calibrated per-model probabilities, weights found by hill-climbing on OOF (greedy forward selection with replacement — robust on small data).
2. **Stacking**: OOF probability matrices (10 columns per base model) + a few strong raw features → multinomial logistic regression / shallow LGBM meta-learner, with nested CV so the meta-score stays honest.
3. **Rank/logit averaging** as a low-variance fallback.
Diversity is deliberate: tree, kernel, neighbour, neural, hierarchical, and transformer members all contribute.

**Interpretation:** SHAP summary + per-class SHAP for the winner, and a confusion matrix naming the hard pairs, written to `reports/model_card.md`.

---

## 6. Phase 4 — Transformer + MIRAGE pretraining + customized linear probes  *(Modal fns: `mirage_fetch`, `mirage_reduce`, `pretrain`, `finetune`, `probe_features`)*

**6.1 The RTC transformer (`PacketFormer`).** Each flow is a 5-token sequence. Token *i* embeds a small per-packet vector — `[log1p(length), length/1500, log1p(IAT), IAT/duration, quantised-length one-hot, burst-flag]` — through a linear projection to `d_model` (64–128), plus a learned positional embedding and a `[CLS]` token. 2–4 encoder layers, 4 heads, GELU, dropout 0.1–0.3, pre-norm, classification head on `[CLS]`. Label smoothing 0.05, mixup on token embeddings, AdamW + cosine schedule, early stopping on the shared folds. Augmentations are traffic-plausible: multiplicative time jitter (±10%), small length jitter within the padding quantum, packet dropout with re-indexing. The 15 fold-jobs (5 folds × 3 repeats) run as a `finetune.map()` fan-out on L4s.

**6.2 Auxiliary corpus: MIRAGE-AppAct-2024 (6.97 GB) — Modal-native ingestion.**
The laptop's 35 GB free disk no longer constrains anything; the archive lands in the `rtc-mirage` Volume:
1. `mirage_fetch` streams the zip to `/mirage/raw/MIRAGE-AppAct-2024.zip` in chunks with a resumable HTTP-range loop (the server advertises `Accept-Ranges: bytes`), checkpointing byte offsets so a preempted container resumes instead of restarting. `volume.commit()` at each checkpoint. Runs `--detach`.
2. `mirage_reduce` is a **fan-out** unit: the driver reads the zip's central directory, partitions members into shards, and `.map()`s them across containers. Each container decompresses only its own members, parses the JSON, and reduces every flow to what the model needs — per-packet L4 payload length, IAT and direction — truncated to the first N packets (N = 5 for probe alignment with the RTC data, N = 20 for richer pretraining). Output: `float32` `.npz`/memmap shards in `/mirage/reduced/`.
3. Because storage is now cheap, the corpus target rises from the 1–3 GB subset the old plan was forced into, to **the full archive if reduction throughput allows, else a stratified subset covering all apps/activities biased toward UDP/real-time apps**. Actual coverage, subset criteria and flow counts are logged to `reports/mirage_manifest.json` so the result is reproducible.
4. After reduction succeeds, the raw 6.97 GB zip can be deleted from the Volume to stop paying for it — gated behind an explicit flag, never automatic.

**6.3 Pretraining objectives (self-supervised, no RTC labels involved)** — `pretrain` on one L4:
- **Masked packet modelling** (ET-BERT / YaTC style): mask 15% of tokens, reconstruct the quantised length bin (cross-entropy) and the log-IAT (Huber).
- **Contrastive** (SimCLR-style): two augmented views of one flow attract, other flows repel — this is what makes the embedding robust to the gap between a lab Wi-Fi capture and MIRAGE's mobile captures.
- **Domain-adaptation guard:** a small gradient-reversal domain-adversarial head, plus an unconditional A/B of the pretrained encoder against the from-scratch encoder on grouped CV. Pretraining is kept only if it wins.
- Checkpoints every N steps into `rtc-cache/pretrain/`, so a preempted GPU container resumes from the last checkpoint.

**6.4 Customized linear probes → new features (the point of this phase)** — `probe_features`:
Freeze the pretrained encoder; train *small, purpose-built* heads on its embeddings. Each probe's target is **not** the competition label, so its output is a legitimate new feature rather than a leak:
- **P1 — App-family probe:** trained on MIRAGE app/activity labels; softmax over MIRAGE app groups. Applied to RTC flows it reports "this looks like a video-conferencing app / a messenger / a streaming app".
- **P2 — Media-modality probe:** trained on MIRAGE-derived proxies for interactive media (high packet rate, MTU-regime fragmentation); emits a "video-likeness" scalar.
- **P3 — Bitrate/pacing regression probes:** `RandomForestRegressor` / ridge heads on the embedding predicting log bitrate, log mean IAT and fragmentation ratio. Both the predictions **and the residuals versus observed values** become features — a residual captures "this flow paces unlike anything in the aux corpus", which is itself app-discriminative.
- **P4 — Transport-fingerprint probe:** predicts padding-quantum / length-alignment class, exposing per-app crypto and header overhead.
- **P5 — Embedding compression:** top 8–16 PCA components of the frozen embedding, plus the fine-tuned model's own OOF class posteriors.
All probe outputs are written as feature block **F12** to `rtc-work/features/probes.parquet` and consumed by the GBDT/ensemble stack of §5 — i.e. the GPU phase feeds the CPU phase, and `fit_eval` simply sees more columns. P5 (RTC-target-derived) is produced **out-of-fold**; P1–P4 carry no RTC label information and apply directly.

**6.5 Final integration.** The fine-tuned transformer joins the ensemble as its own member, *and* its probe features enrich the tree models. Both contributions are ablated separately (`no-probes`, `probes-only`, `full`) so the report states what the transformer actually bought.

---

## 7. Phase 5 — Final model, submission, reporting  *(Modal fn: `predict_submit`)*

1. Pick the winner by grouped-CV accuracy, with plain-CV accuracy and macro-F1 reported alongside; prefer the simpler model when scores sit within noise (±1 SD across repeats).
2. Refit the entire pipeline — every fitted transformer, encoder and probe — on all 1,285 training rows with the chosen hyperparameters, in one container, from artifacts already in the Volume.
3. Predict the 327 test rows **in file order**; write `submission.csv` with no header, 1-based index, exact case-sensitive label strings.
4. Automated validation before the file is committed to the Volume: 327 rows, 2 columns, no header, indices 1..327 strictly increasing, every label in the 10-string vocabulary, and a sanity check that the predicted class distribution is not wildly implausible against the flows-per-call prior from §3F. Validation failure aborts the write.
5. `reports/model_card.md`: data summary, CV table for every model and ensemble, ablations (feature blocks F1–F12 added cumulatively), SHAP importances, confusion matrix, failure analysis, seeds, package versions, Modal image digests, per-stage wall-clock and GPU-hours.
6. Pull to the laptop: `modal volume get rtc-work submission.csv ./submission.csv` and `modal volume get rtc-work reports ./reports`.

---

## 8. Sequencing, effort and risk

| Phase | Content | Modal shape | Rough effort |
|---|---|---|---|
| 0 | `uv tool install modal`, app scaffold, images, volumes, `seed_data` | 1 container | short |
| 1 | EDA (§3) | 1 container, 4 CPU | medium |
| 2 | Features F1–F8 (§4) | 1 container, cached to Volume | medium |
| 3 | Grid + Optuna + ensemble (§5) | **wide fan-out**, hundreds of 2-CPU containers | long sequentially, minutes-to-hours fanned out |
| 4a | RTC transformer from scratch (§6.1) | 15 L4 jobs via `.map()` | medium |
| 4b | MIRAGE fetch → reduce → pretrain → probes (§6.2–6.4) | 1 network container + reduce fan-out + 1 L4 | long (network + GPU-bound) |
| 5 | Final fit, submission, report (§7) | 1 container | short |

Phases 1–3 already produce a complete, submittable result; phase 4 is additive and gated on beating phase 3 on grouped CV. That ordering guarantees a valid `submission.csv` exists early rather than being hostage to a 6.97 GB download.

**Risks and their mitigations**
- *Optimistic CV from same-call flows* → pseudo-group CV, and model selection on the pessimistic number.
- *Overfitting from ~400 features on 1,285 rows* → fold-internal fitting, correlation pruning, compact-vs-full selection by CV, preference for regularised/bagged models.
- *Fan-out cost blowup* → `max_containers` caps per stage, cheap `fit_eval` containers (2 CPU), `reports/cv_results.csv` logging `fit_seconds` so cost per point is visible, and a dry-run mode that prices a grid before launching it.
- *Volume write races across hundreds of containers* → every fan-out unit writes to a **unique** path keyed by its parameter hash; no shared mutable file, and aggregation happens in a single driver container after the map completes.
- *Preemption / retries* → all fan-out units are pure and idempotent with `retries=2`; long stages checkpoint (download offsets, pretraining steps, Optuna journal) and resume.
- *Domain gap MIRAGE (mobile) vs RTC (lab Wi-Fi)* → contrastive + domain-adversarial pretraining, and an explicit A/B against the from-scratch encoder; discard if it does not help.
- *Silent leakage through target-derived features* → every such feature has an out-of-fold implementation plus a unit test asserting the fold-train reference set never contains fold-validation rows.
- *Non-reproducibility across containers* → pinned image digests, one shared precomputed fold-index array in the Volume, and seeds recorded per run.

---

## 9. Execution log and results

Everything below was produced by running the plan on Modal. Numbers are grouped-CV
(the pessimistic, decisive scheme) unless stated otherwise.

**What ran**

| Stage | Outcome |
|---|---|
| `seed_data`, `run_eda` | EDA report with 20 figures and their backing tables in `reports/eda/` |
| `build_features` | 241 deterministic feature columns (F1–F8), shared fold indices, per-fold mutual-information ranking |
| Grid + Optuna search | **2,377 scored points** across 12 model families, 83.6 container-hours of fitting executed in parallel |
| `mirage_fetch` + `mirage_reduce` | 6.97 GB archive downloaded into the Volume; 2,245 members reduced to **90,610 flows** across 20 applications |
| `pretrain_encoder` | PacketFormer pretrained (masked packet modelling + contrastive + domain-adversarial), 8,000 steps in 347 s on one L4 |
| `build_probe_features` | 46 probe columns (F12); P1 app-family probe 0.50 accuracy over 20 auxiliary apps, P3 regressions R² ≈ 0.95 |
| `train_transformer` | from-scratch and pretrained variants, both cross-validated |
| `finalists_stage`, `run_ensemble`, `make_submission`, `write_model_card`, `write_shap` | final selection, `submission.csv`, model card, SHAP attribution |

**Leaderboard (best point per family, grouped CV)**

LightGBM 0.8249 · XGBoost 0.8218 · HistGradientBoosting 0.8156 · ExtraTrees 0.8125 ·
CatBoost 0.8070 · RandomForest 0.8023 · PacketFormer (pretrained) 0.7346 ·
PacketFormer (scratch) 0.7331 · MLP 0.7292 · SVM 0.7268 · LogReg 0.7144 · kNN 0.7132

**Three results that changed the plan**

1. *The fitted label-aware blocks F9–F11 hurt.* Class-conditional Markov likelihoods,
   class-conditional densities and per-class neighbour distances cost about two accuracy
   points (LightGBM 0.7969 with, 0.8152 without) even after their training-side values are
   generated through an inner out-of-fold loop. With 1,285 rows they add variance rather
   than signal. They are off by default and remain a searchable option.
2. *The auxiliary corpus is close to neutral.* MIRAGE-AppAct-2024 covers all five
   competition applications, and pretraining plus probing was implemented in full, but the
   pretrained transformer beat the from-scratch one by only 0.0015, and the probe features
   left LightGBM unchanged (0.8144 both ways), helped RandomForest by 0.0004 and hurt
   XGBoost by 0.009. Reported as measured rather than assumed to help.
3. *Ensembling did not beat the single best model on an honest estimate.* Hill-climb
   weights score 0.8319 on the rows they were fitted to but 0.8195 nested; stacking 0.8187;
   the single best LightGBM 0.8210. The submission therefore uses the single model, which
   is the outcome the nested comparison supports.

**Infrastructure lesson worth recording.** `modal run --detach` still dies with the local
client in this setup: a sweep of roughly a thousand points was cancelled when its terminal
was reaped. The fix has two parts, both now in the repository — orchestration moved
server-side into `search_stage`/`finalists_stage` so no local process holds the sweep, and
`scripts/spawn.py` fires calls against the **deployed** app so nothing depends on the
laptop. Because every scored point writes its own `.npz`, the cancelled work was fully
recovered with `rebuild_cv_results`; that crash-tolerance is now a permanent property of
the pipeline rather than a lucky accident.

**Final submission.** Single LightGBM (800 trees, 31 leaves, learning rate 0.12,
subsample 0.8, colsample 0.8), refit on all 1,285 training rows over the 241 deterministic
features. `submission.csv`: 327 rows, no header, 1-based index, all ten labels present,
mean predicted confidence 0.958. Weakest class is Zoom_voice (F1 0.465), confused with
Zoom_video — the same pair the EDA baseline flagged before any modelling was done.
