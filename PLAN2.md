# Improvement Plan — Phase 2

**Baseline to beat:** 0.8249 grouped-CV accuracy (LightGBM, 241 features); 0.8210 for the model
actually shipped. 230 errors on 1,285 training flows.

This plan is written after a dedicated error study (`src/analysis/zoom_tsne.py`, figures in
`reports/analysis/`). Its first job is to correct a recommendation I made in the previous report.

---

## 1. What the error study found

### 1.1 The error budget

| Class | Support | Errors | Share of all errors | Mean confidence when wrong |
|---|---|---|---|---|
| Zoom_video | 172 | 44 | 19.1% | 0.766 |
| Zoom_voice | 80 | 43 | 18.7% | 0.815 |
| GoogleMeet_video | 112 | 38 | 16.5% | 0.725 |
| Discord_video | 256 | 36 | 15.7% | 0.853 |
| Discord_voice | 238 | 20 | 8.7% | 0.702 |
| Messenger_video | 131 | 20 | 8.7% | 0.823 |
| Messenger_voice | 94 | 11 | 4.8% | 0.854 |
| GoogleMeet_voice | 40 | 9 | 3.9% | 0.614 |
| WhatsApp_video | 82 | 9 | 3.9% | 0.860 |
| WhatsApp_voice | 80 | 0 | 0.0% | — |

**Zoom flows carry 87 of 230 errors (37.8%)** — 63 of them are Zoom↔Zoom, the rest leak to other
applications. GoogleMeet_video is the second concentration at 16.5%.

Every error is made **confidently**: mean probability assigned to the wrong class is 0.61–0.86.
This is not a model sitting on the fence; it is a model that is sure and wrong, which rules out
"tune the threshold" as a general fix and points at calibration and at missing information.

### 1.2 The Zoom pair is not separable per-flow — my earlier recommendation was wrong

The previous report recommended a dedicated binary Zoom voice/video model. **I built it and it
does not work:**

| Approach on the 252 Zoom flows | Accuracy |
|---|---|
| Always predict Zoom_video (majority) | **0.6825** |
| Dedicated binary LightGBM specialist, 5-fold | 0.6746 |
| Current flat 10-class model | 0.6548 |

A specialist trained on nothing but Zoom flows, with the full feature set and a proper CV, lands
**below the constant predictor**. The information needed to separate Zoom voice from Zoom video is
not present in these five packets.

The t-SNE confirms the structure (`reports/analysis/tsne_zoom.png`). Zoom splits into two
populations:

- **A clean Zoom_video island** (~85 flows on the right of the plot) that is almost entirely
  classified correctly — these are the flows actually carrying fragmented video frames.
- **A large interleaved region** where Zoom_voice and Zoom_video sit on top of each other, holding
  essentially all 87 errors.

The same shape appears in the global t-SNE (`reports/analysis/tsne_predictions.png`): most classes
form tight islands, while one central blob mixes Zoom_voice, Zoom_video, GoogleMeet and
Messenger_voice — and that blob is where nearly every error lives.

Supporting numbers: constant-size Zoom flows (18 of them) have a **5.6%** error rate, while
variable-size Zoom flows have **36.8%**. The keepalive-shaped flows are easy; the media-shaped
ones are the problem.

### 1.3 The interpretation that reorganises the plan

A "call" opens several UDP flows. Some carry media and are separable; others carry signalling,
probing or a secondary stream and look nearly identical whichever mode the call is in. Per-flow,
those auxiliary flows are close to unlabelable — **their label is a property of the call they
belong to, not of the flow itself**.

That reframes the remaining headroom. It is not mostly a feature-engineering problem. It is:

1. a **decision-rule** problem (what to answer when the evidence genuinely does not discriminate), and
2. a **grouping** problem (recover which test flows share a call and let confident siblings speak
   for ambiguous ones).

---

## 2. Strategies, ranked by expected value

Expected values are stated in points of overall accuracy and are deliberately conservative.

### S1 — Prior-corrected decision rule inside the ambiguous region
**Expected: +0.4 to +0.9 points. Cost: minutes. Risk: low-moderate.**

The flat model scores 0.6548 on Zoom flows where the constant predictor scores 0.6825. It is
*losing* 2.8 points against a rule that ignores the input. Rather than a hard override, fit a
**cost-sensitive decision layer**: on the probability vector, learn a per-class multiplicative
prior correction (equivalently, a per-class logit bias) by maximising grouped-CV accuracy over a
small simplex search. This subsumes threshold shifting, handles the other confusable pairs at the
same time, and is a single extra vector of ten numbers.

Guardrail: the task states the test class mix is not guaranteed uniform, so the correction is
fitted on grouped CV only and its magnitude is capped; if the improvement does not survive the
grouped scheme it is not shipped.

### S2 — Transductive call reconstruction and correlated-flow aggregation
**Expected: +1 to +3 points if call recovery works at all; 0 if it does not. Cost: a day. Risk: high variance, well bounded.**

The test set is 327 flows drawn from 100 calls — roughly 3.3 flows per call. If flows can be
grouped by call, then a confident prediction on the media-carrying flow of a call can be
propagated to that call's ambiguous siblings. This is the classic *correlated flows* result in
traffic classification (Zhang et al.), and it is exactly the structure the pseudo-group key
already exploits inside cross-validation.

Implementation:
1. Learn a same-call similarity metric on the **training** set, where call membership is
   approximated by the pseudo-group key: a small siamese/metric model or a learned Mahalanobis
   distance trained to score "same call" versus "different call".
2. Cluster the 327 test flows under that metric (agglomerative, with the cluster count anchored
   near 100 and a distance cap).
3. Aggregate class probabilities within a cluster — sum of log-probabilities, or a weighted vote
   dominated by the most confident member — and re-argmax every member.
4. Validate the entire chain on the training set under the grouped scheme, where true pseudo-call
   membership is known and can be scored directly.

The failure mode is a cluster that mixes two applications, which would spread one confident error
across several flows. Mitigate by only aggregating when within-cluster agreement on the
**application** is already high, and by leaving singleton clusters untouched.

### S3 — Tabular foundation models as a new family
**Expected: +0.5 to +1.5 points. Cost: hours. Risk: low.**

1,285 rows × 241 features sits precisely in the regime where prior-fitted tabular transformers
dominate gradient boosting. TabPFN v2 reports outperforming tuned tree ensembles on datasets under
10,000 rows, and TabPFN-2.5 reports a 100% win rate against default XGBoost at this scale. This is
the single cheapest new model family to add, it needs no hyperparameter search, and its errors
should decorrelate from the boosted trees — which also gives the ensemble something new to work
with (§S4).

Add as three members: TabPFN-2.5 on the full feature set, on the top-60 subset, and on the raw ten
columns. Also evaluate TabICLv2 as an alternative in-context learner.

### S4 — Calibration, then re-ensemble
**Expected: +0.2 to +0.6 points. Cost: hours. Risk: low.**

Ensembling previously lost to the single model (hill-climb 0.8195 nested vs 0.8210). One plausible
cause is visible in the error budget: members are confidently wrong in overlapping regions, so
averaging preserves the error. Fit per-model temperature scaling (or vector/Dirichlet calibration)
on out-of-fold probabilities *before* the weight search, and re-run hill-climb and stacking on
calibrated inputs. Calibration is also a prerequisite for S1 and S2 to behave sensibly.

### S5 — Exact-tuple and near-tuple lookup against the training set
**Expected: unknown until measured; potentially large. Cost: an hour. Risk: none.**

Only 2 of 1,136 distinct training length tuples map to more than one class — the mapping from
length tuple to label is almost injective. The unmeasured quantity is **how many test tuples
appear verbatim, or within a byte or two, in the training set**. If that fraction is material,
those test flows can be answered by lookup with near-certainty and removed from the model's
burden. This is a measurement first and a feature second; run it before anything else in the plan.

### S6 — Zoom-specific protocol features from the measurement literature
**Expected: +0 to +0.5 points. Cost: a day. Risk: moderate — may be a dead end, but it is cheap to falsify.**

Zoom does not send bare RTP: it wraps media in its own encapsulation header, and the field layout
is documented by Michel et al. (IMC 2022), whose open-source tooling infers media type (audio,
video, screen share) from packet structure. Two concrete features follow: payload sizes corrected
for Zoom's specific header stack (rather than the generic RTP −12 / SRTP −22 used now), and
membership tests against the size families that Zoom's audio versus video encoders produce. The
related Springer 2024 chapter on information leakage through packet lengths in RTC traffic is the
other source to mine for size-family structure.

Honest prior: the t-SNE says the ambiguous Zoom flows do not look like media flows at all, so
media-type features may simply not apply to them. This strategy is ranked below the decision-rule
and grouping work for that reason.

### S7 — Use MIRAGE as labelled auxiliary data, not only as a probe source
**Expected: +0 to +0.7 points. Cost: a day. Risk: moderate.**

The probes were neutral (§18 of the main report) and cross-corpus transfer was weak (0.293 on the
five shared applications against 0.20 chance). But the auxiliary corpus contains labelled Discord,
Meet, Messenger, WhatsApp and Zoom flows, and the *application* half of the task is the half the
probe transferred least badly on. Two things to try: retrain the encoder on **UDP media flows
only** (16.5% of the current corpus — the TCP majority is governed by congestion control, not by a
codec), and train a five-way application head jointly on auxiliary and competition data with a
domain-adaptation loss, rather than freezing and probing.

### S8 — Ship the corrected P4 probe
**Expected: +0 to +0.3 points. Cost: an hour. Risk: none.**

The padding-alignment probe reaches 0.940 against a 0.722 baseline under its corrected definition
and contributed nothing to the models because the original target was degenerate. It encodes
per-application crypto and header overhead, which is orthogonal to the size statistics that
dominate SHAP — the one probe with a real chance of not being redundant.

### S9 — Hierarchical model with per-application mode heads
**Expected: +0 to +0.5 points. Cost: hours. Risk: low.**

Predict the application first (0.912 achievable standalone), then the mode with a head conditioned
on the predicted application, so the Zoom mode head can carry its own prior and its own threshold
while the WhatsApp head — which is already perfect — is left alone. This is a cleaner way to
express S1's per-class correction and gives the ensemble another decorrelated member.

### S10 — Feature additions worth one round
**Expected: +0 to +0.4 points. Cost: hours. Risk: none.**

Order-invariant views of the length tuple (sorted lengths, sorted gaps) so flows that differ only
in packet ordering collapse together; quantile-rank encodings of each length against the training
marginal; explicit "distance to nearest training tuple" and "label of nearest training tuple"
features (fold-safe, from S5); and pairwise ratios binned to the padding quantum.

---

## 3. What is deliberately not being done

- **More hyperparameter search.** 2,377 points already spread 0.8249 to 0.80 across families; the
  gradient there is exhausted. Optuna added 0.0097 over the grid winner.
- **A bigger transformer.** Five tokens, 1,285 labelled rows, and pretraining bought +0.0015. More
  capacity will not change that arithmetic.
- **A dedicated Zoom binary specialist.** Measured at 0.6746 against a 0.6825 constant predictor.
  Retired.
- **Compact feature subsets.** The full 241 columns beat top-120 and top-60 in every matched pair.

---

## 4. Sequencing

| Order | Work | Gate to continue |
|---|---|---|
| 1 | S5 measurement (test-tuple overlap) | always run — it informs S2 and S10 |
| 2 | S3 TabPFN-2.5 / TabICLv2 members | ship if grouped CV beats 0.8249 |
| 3 | S4 calibration, then re-ensemble | ship if calibrated blend beats the single model on the nested estimate |
| 4 | S1 cost-sensitive decision layer | ship if grouped CV improves and the correction is small |
| 5 | S2 transductive grouping | ship only if the training-set simulation shows a gain and application purity holds |
| 6 | S8 corrected P4, S10 features | ship if the ablation is positive |
| 7 | S6 Zoom protocol features, S7 auxiliary rework, S9 hierarchy | opportunistic |

Every step is evaluated by the ablation protocol in `ABLATION2.md`, on the same stored folds, so
results are directly comparable to the 2,377 points already scored.

---

## 5. Implementation notes on Modal

The existing topology absorbs all of this without redesign:

- **New model families (S3)** slot into `src/models/registry.py` and are scored by the existing
  `fit_eval` fan-out. TabPFN needs the GPU image and a `max_containers` cap, since each container
  loads model weights.
- **Decision layer and calibration (S1, S4)** read stored out-of-fold matrices from `rtc-work/oof/`
  and refit nothing — they belong in `ensemble.py` alongside the existing combiners.
- **Transductive grouping (S2)** is a new module `src/analysis/call_recovery.py` plus one function;
  it needs both train and test features, which the `Cache` already exposes.
- **Feature work (S6, S8, S10)** extends `features/base.py` and the probe builder; the fold and
  selection caches must be rebuilt afterwards (`build_features`), and the ablation reruns against
  the new cache.

Cost control stays as it is: screening on one grouped repeat, finalists on the full budget, per-point
artifacts so an interrupted run is recoverable.

---

## 6. Research references

Directly relevant to the strategies above.

**Tabular foundation models (S3)**
- Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature, 2025 — https://www.nature.com/articles/s41586-024-08328-6
- *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models*, arXiv:2511.08667 — https://arxiv.org/abs/2511.08667
- *A Closer Look at TabPFN v2: Strength, Limitation, and Extension*, arXiv:2502.17361 — https://arxiv.org/html/2502.17361v1
- *TabICLv2: A better, faster, scalable, and open tabular foundation model*, arXiv:2602.11139 — https://arxiv.org/pdf/2602.11139
- *Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data*, arXiv:2507.03971 — https://arxiv.org/pdf/2507.03971

**Correlated flows and semi-supervised traffic classification (S2)**
- *A New Semi-Supervised Method for Network Traffic Classification Based on X-Means Clustering and Label Propagation*, IEEE, 2018 — https://ieeexplore.ieee.org/document/8566608/
- *SemTra: A semi-supervised approach to traffic flow labeling with minimal human effort*, Pattern Recognition, 2019 — https://www.sciencedirect.com/science/article/abs/pii/S0031320319300652
- *Network Traffic Classification Using Semi-Supervised Approach*, IEEE — https://ieeexplore.ieee.org/document/5460712

**RTC packet-length leakage and Zoom protocol structure (S6)**
- *Information Leakage Through Packet Lengths in RTC Traffic*, Springer, 2024 — https://link.springer.com/chapter/10.1007/978-981-97-7737-2_16
- Michel et al., *Enabling Passive Measurement of Zoom Performance in Production Networks*, ACM IMC 2022 — https://www.cs.princeton.edu/~ravian/publications/zoom_imc22.pdf
- Princeton-Cabernet, `zoom-analysis` open-source tooling — https://github.com/Princeton-Cabernet/zoom-analysis
- Karamollahi et al., *Packet-Level Analysis of Zoom Performance Anomalies*, ICPE 2023 — https://research.spec.org/icpe_proceedings/2023/proceedings/p221.pdf

**Encrypted-traffic pretraining, for S7 and for context on the transformer result**
- Zhou et al., *TrafficFormer: An Efficient Pre-trained Model for Traffic Data*, IEEE S&P 2025 — https://www.computer.org/csdl/proceedings-article/sp/2025/223600a102/22K50xTq93y
- *MM4flow: A Pre-trained Multi-modal Model for Versatile Network Traffic Analysis*, ACM CCS 2025 — https://dl.acm.org/doi/10.1145/3719027.3744804
- *FlowletFormer: Network Behavioral Semantic Aware Pre-training Model for Traffic Classification*, arXiv:2508.19924 — https://arxiv.org/pdf/2508.19924
- *UniAlign: A Model-Agnostic Framework for Robust Network Traffic Classification under Distribution Shifts*, arXiv:2605.17575 — https://arxiv.org/pdf/2605.17575
- *Mean Masked Autoencoder with Flow-Mixing for Encrypted Traffic Classification*, arXiv:2603.29537 — https://arxiv.org/pdf/2603.29537
- ETA-Resource, a maintained index of encrypted-traffic-analysis work — https://github.com/linwhitehat/ETA-Resource

**Audio/video separation by packet size, for S6's thresholds**
- *Video QoE Metrics from Encrypted Traffic: Application-agnostic Methodology*, arXiv:2504.14720 — https://arxiv.org/pdf/2504.14720
- *Learning QoE from Packet-Level Measurements in Encrypted Video Conferencing Traffic*, arXiv:2601.06862 — https://arxiv.org/pdf/2601.06862
