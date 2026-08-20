# Second Ablation Plan — Implementation Specification

Companion to `PLAN2.md`. The first ablation round asked *which feature blocks help*. This round
asks *which of the ten phase-2 strategies survive contact with an honest evaluation*, and it is
specified tightly enough that the answers cannot be argued with after the fact.

---

## 1. Protocol

**Folds.** The existing stored fold arrays in `rtc-work/features/folds.npz` — no regeneration, so
every number here is directly comparable to the 2,377 points already scored. Screening uses one
grouped repeat (5 fits); confirmation uses 2 grouped repeats + 3 plain repeats (25 fits).

**Metric.** Grouped-CV accuracy decides. Macro-F1, grouped-CV standard deviation across repeats,
and accuracy restricted to the Zoom pair are recorded alongside for every experiment.

**Baselines, fixed once and reused.**

| Reference | Value |
|---|---|
| Current best single model | 0.8249 grouped |
| Shipped model | 0.8210 grouped |
| Majority class | 0.1992 |
| Constant predictor within the Zoom pair | 0.6825 on 252 flows |
| Current model within the Zoom pair | 0.6548 |

**Decision rule.** A change ships only if it beats its own matched control by more than the noise
floor. The noise floor is one standard deviation of the control across grouped repeats — measured,
not assumed, and reported per experiment. For classifier-versus-classifier comparisons on identical
folds, report **McNemar's test on the paired error sets** (exact binomial, discordant pairs only);
a p-value above 0.05 with a positive point estimate is recorded as "not established", not as a win.

**Paired evaluation.** Every experiment is run as a matched pair (or matched set) differing in one
factor. Controls are re-run rather than quoted from the earlier round, so both arms see identical
folds, identical seeds and identical feature caches.

**Negative controls.** Two experiments (N1, N2) exist to detect a broken harness. If either
produces a "win", the round is invalid and stops.

**Artifacts.** Each arm writes its own `.npz` under `rtc-work/oof/` exactly as before, so the round
is reconstructible with `rebuild_cv_results` and nothing is lost to an interrupted run.

**Budget.** Screening arms are ~2 CPU-minutes each on 2-CPU containers except where noted;
confirmation arms are 5× that. The whole round is estimated at 25–40 container-hours, run at a cap
of 80 concurrent containers.

---

## 2. Experiment matrix

Each experiment states a hypothesis that can be falsified, the factor being varied, and what
happens next in either outcome.

### A. Measurement first

**A1 — Test/train tuple overlap (no model)**
*Hypothesis:* a material fraction of test length tuples appear verbatim, or within ±2 bytes per
packet, in the training set.
*Method:* exact-match and near-match join of the 327 test tuples against the 1,285 training tuples;
report match counts, and the label purity of matched training tuples. Cross-validated version on
training data to estimate accuracy on matched rows.
*Decision:* if exact matches exceed 5% of the test set with purity above 0.95, a lookup feature and
a lookup override both enter the round as A2/A3. If below 1%, S5 and part of S10 are dropped.
*Cost:* minutes, no GPU.

**A2 — Nearest-training-tuple features**
*Factor:* base features vs base + {distance to nearest training tuple, label of that tuple as
one-hot, margin between nearest two class distances}, computed fold-safely.
*Decision:* ships if it beats control; also feeds A3.

**A3 — Lookup override at prediction time**
*Factor:* model argmax vs "if an exact tuple match exists in training and is pure, use its label,
else model argmax".
*Decision:* ships only if grouped CV improves **and** the override fires on more than 20 training
rows, so the estimate is not built on a handful of flows.

### B. New model families

**B1 — TabPFN-2.5, full feature set**
*Hypothesis:* a tabular foundation model beats tuned gradient boosting at n=1,285.
*Factor:* LightGBM (control, re-run) vs TabPFN-2.5 vs TabICLv2.
*Notes:* GPU image, `max_containers` capped at 8 because each container loads weights; no
hyperparameter search — that is the point of the family.
*Decision:* ships as an ensemble member if it beats 0.8249 grouped; enters the ensemble pool
regardless if it is within one noise floor, because decorrelated errors are worth having.

**B2 — TabPFN feature-set sensitivity**
*Factor:* full 241 columns vs top-60 vs the raw 10 columns.
*Purpose:* prior-fitted transformers are known to degrade with many weakly informative columns;
this locates the best input width rather than assuming it.

**B3 — Hierarchical application → mode**
*Factor:* flat 10-class (control) vs {5-way application model × per-application binary mode head},
combined by the product rule.
*Decision:* ships if it beats control **or** if its Zoom-pair accuracy exceeds 0.6825 without
overall loss.

### C. Decision layer

**C1 — Probability calibration**
*Factor:* raw probabilities (control) vs temperature scaling vs vector scaling vs Dirichlet
calibration, each fitted out-of-fold.
*Metrics:* accuracy, plus expected calibration error and negative log-likelihood, because the
error study showed the model is confidently wrong (mean 0.61–0.86 confidence on errors).
*Decision:* the best calibrator is adopted as the input transform for C2, C3 and D1 regardless of
whether it changes accuracy on its own.

**C2 — Cost-sensitive per-class logit bias**
*Hypothesis:* the model loses to a constant predictor inside the Zoom pair, so a learned per-class
prior correction recovers accuracy.
*Factor:* argmax (control) vs argmax over `p · exp(b)` with a ten-dimensional bias vector fitted by
grouped-CV search, with an L2 cap on ‖b‖ to keep the correction small.
*Guardrail:* the fitted bias is reported explicitly; if it amounts to "always predict Zoom_video",
that is stated as such rather than hidden inside a score.
*Decision:* ships if grouped CV improves beyond the noise floor **and** the improvement survives
when the bias is refit inside each fold (nested), not once globally.

**C3 — Reject-and-defer within confusable pairs**
*Factor:* for flows where the top-two classes are the same application in different modes and the
margin is below a threshold, defer to the application-level prior instead of the argmax.
*Purpose:* a narrower, more interpretable version of C2; the two are compared directly.

### D. Transductive grouping

**D1 — Same-call metric learning**
*Hypothesis:* flows from the same call are identifiable from features alone.
*Method:* train a binary "same call" scorer on training pairs, using the pseudo-group key as the
call proxy; evaluate with ROC-AUC on held-out **groups**, never held-out pairs from the same group.
*Decision:* AUC below 0.75 kills the whole D branch — no downstream experiment is run.

**D2 — Cluster quality on held-out data**
*Method:* apply the D1 metric, cluster, and score against the pseudo-group partition with adjusted
Rand index and application purity.
*Decision:* application purity below 0.9 kills D3, since impure clusters would propagate errors.

**D3 — Correlated-flow aggregation**
*Factor:* per-flow argmax (control) vs cluster-aggregated log-probability sum vs
confidence-weighted vote, with singleton clusters untouched in all arms.
*Evaluation:* on training data under the grouped scheme, where the true pseudo-call partition is
known, so the ceiling and the realised gain are both visible.
*Decision:* ships only if the realised gain is positive **and** the gain does not come entirely
from the Zoom pair — a Zoom-only gain is more cheaply obtained by C2 and less risky.

### E. Features

**E1 — Corrected P4 probe**
*Factor:* current 46 probe columns vs 46 + corrected padding-alignment probe outputs.
*Note:* run on the three families that were in the first probe ablation (LightGBM, XGBoost,
RandomForest) so the comparison is matched to a published control.

**E2 — Order-invariant views**
*Factor:* base vs base + {sorted length tuple, sorted gap vector, and their pairwise differences}.

**E3 — Quantile-rank encodings**
*Factor:* base vs base + per-length empirical quantile rank against the training marginal,
computed fold-safely.

**E4 — Zoom protocol features**
*Factor:* base vs base + {payload sizes corrected for Zoom's media-encapsulation header stack,
membership indicators for the documented audio and video size families}.
*Note:* this is the arm most likely to fail, and it is scoped to one round so that it fails cheaply.

**E5 — Auxiliary encoder retrained on UDP media only**
*Factor:* current probes (from the 90,610-flow mixed corpus) vs probes from an encoder pretrained
on the 14,975 UDP-only flows.
*Secondary metric:* re-run the cross-corpus transfer measurement — restricted-agreement 0.293 is
the number to beat, and it is the honest test of whether the TCP majority was the problem.

### F. Ensembling, re-opened

**F1 — Calibrated ensemble**
*Factor:* the previous hill-climb and stacking (control, 0.8195 and 0.8187 nested) vs the same
combiners on C1-calibrated inputs, with the B1 and B3 members added to the pool.
*Decision:* ships if the nested estimate beats the best single model of this round.

### N. Negative controls

**N1 — Shuffled labels.** Re-run the control model with permuted training labels. Expected grouped
accuracy ≈ 0.10–0.20. Anything materially higher means the fold or feature cache leaks, and the
round is void.

**N2 — Prior-only predictor.** Predict the training class distribution, ignoring features.
Expected 0.1992. Any experiment that fails to beat this is a harness bug, not a result.

---

## 3. Analysis plan

For each experiment, report a single row:

`experiment · arm · grouped_acc ± sd · macro_F1 · zoom_pair_acc · Δ vs control · McNemar p · fit_seconds · decision`

Aggregate into `reports/ablation2_results.csv` and a short narrative section appended to
`REPORT.md`. Three rules govern the write-up:

1. **Negative results are reported at the same length as positive ones.** The first round's value
   was mostly in what it ruled out.
2. **A point estimate without its noise floor is not a result.** Every Δ is quoted with the
   control's across-repeat standard deviation.
3. **Anything that ships must be reproducible from a stored artifact** — a spec dict, a calibrator,
   a bias vector, or a cluster assignment — committed to the volume alongside the model.

---

## 4. Stop rules

- **Cost ceiling:** 60 container-hours for the round. Beyond that, remaining arms are dropped in
  reverse rank order.
- **D branch:** killed at D1 if same-call AUC < 0.75, or at D2 if application purity < 0.9.
- **E branch:** if E1–E3 are all within one noise floor of control, E4 and E5 are dropped — the
  feature space is saturated and further columns are not where the remaining accuracy lives.
- **Overall:** if no arm beats 0.8249 by more than its noise floor, the shipped model stays as it
  is and the round is written up as a negative result. That is an acceptable outcome; shipping a
  change that only looks better is not.

---

## 5. Execution order and dependencies

```
A1 ─┬─> A2 ─> A3
    └─> (informs D1 features, E-branch scope)
B1 ─┬─> B2
    └─> B3 ──┐
C1 ─┬─> C2   ├─> F1  (needs calibrated members + new families)
    └─> C3 ──┘
D1 ─> D2 ─> D3        (independent branch, gated twice)
E1..E3 ─> (gate) ─> E4, E5
N1, N2 run first and must pass before anything else is interpreted
```

Wall-clock estimate with an 80-container cap: N and A branches in under an hour; B, C and E
branches in an afternoon; the D branch is the long pole at roughly a day including metric-model
training.

---

## 6. New code required

| Module | Purpose | Experiments |
|---|---|---|
| `src/analysis/tuple_lookup.py` | exact and near tuple matching, purity statistics, lookup features | A1–A3 |
| `src/models/registry.py` (extend) | TabPFN-2.5 and TabICLv2 entries | B1, B2 |
| `src/models/hierarchy.py` | application → per-application mode composition | B3 |
| `src/models/calibrate.py` | temperature, vector and Dirichlet calibration on stored out-of-fold matrices | C1 |
| `src/models/decision.py` | per-class logit bias search, margin-based deferral | C2, C3 |
| `src/analysis/call_recovery.py` | same-call metric, clustering, correlated-flow aggregation | D1–D3 |
| `src/features/base.py` (extend) | order-invariant views, quantile ranks, Zoom header corrections | E2–E4 |
| `src/transformer/probe.py` (fix) | corrected padding-alignment target | E1 |
| `src/transformer/pretrain.py` (option) | UDP-only corpus filter | E5 |

All of these are additive; none changes the existing evaluation path, so the 2,377 stored points
remain valid comparisons throughout.

---

# Round 1 results — A1/A3, B1, C1/C2, N1/N2

Executed on the stored grouped folds; controls first, as specified.

## Negative controls — round is valid

| Control | Result | Expected | Verdict |
|---|---|---|---|
| N1 shuffled labels | 0.1549 | 0.08–0.20 | pass — no leakage in folds or feature cache |
| N2 prior-only | 0.1992 | 0.1992 | pass |

## A1 / A3 — tuple lookup: measured, then rejected

| Tolerance | Distinct train tuples | Pure | Test rows matched | CV coverage | Lookup accuracy when available |
|---|---|---|---|---|---|
| exact | 1,136 | 1,134 | 25 (7.6%) | 5.5% | 0.972 |
| ±1 B | 1,092 | 1,087 | 40 (12.2%) | 9.8% | 0.976 |
| ±2 B | 1,032 | 1,015 | 52 (15.9%) | 14.7% | 0.905 |
| ±4 B | 920 | 895 | 73 (22.3%) | 18.1% | 0.892 |

A1 passed its gate (>5% coverage at >0.95 accuracy), so A3 measured the differential that actually
decides — and killed it:

| Tolerance | Rows covered | Model accuracy on those rows | Lookup accuracy on those rows | Overall with override | Δ |
|---|---|---|---|---|---|
| exact | 71 | **1.0000** | 0.9718 | 0.8195 | **−0.0016** |
| ±1 B | 126 | 0.9921 | 0.9762 | 0.8195 | −0.0016 |
| ±2 B | 189 | 0.9365 | 0.9048 | 0.8163 | −0.0047 |
| ±4 B | 232 | 0.9181 | 0.8922 | 0.8163 | −0.0047 |

**The model is already perfect on every exactly-matched row.** Lookup can only take rows away from
a model that answers them correctly. S5 and the lookup half of S10 are dropped.

## B1 — TabICL: the round's only win

TabPFN could not be run: as of this release its weights sit behind a one-time licence acceptance
and a `TABPFN_TOKEN`, which needs an interactive browser login at ux.priorlabs.ai. The arm was
re-pointed at **TabICL**, an in-context tabular classifier with openly downloadable weights.

| Arm | Grouped | Plain | Macro-F1 | Fit seconds |
|---|---|---|---|---|
| TabICL, full 241 features | **0.8389** | 0.8195 | **0.8232** | 83 |
| TabICL, top-120 | 0.8280 | 0.8197 | 0.8117 | 64 |
| TabICL, top-60 | 0.8272 | 0.8130 | 0.8117 | 51 |
| LightGBM control (best of 2,377 points) | 0.8249 | — | 0.8069 | — |

All three widths beat the best of the entire previous search, with **no hyperparameter search at
all** — three arms against 2,377.

Paired McNemar against the LightGBM control: 66 rows TabICL wins, 49 rows LightGBM wins, 115
discordant, **p = 0.135**. By the protocol this is recorded as **"not established"**, not as a
significant win, despite +1.3 points. It enters the ensemble pool on the strength of the point
estimate and the consistency across three widths.

## C1 — calibration: fixes confidence, not accuracy

| Calibrator | Accuracy | NLL | ECE | Mean confidence when wrong |
|---|---|---|---|---|
| raw | 0.8210 | 0.7931 | 0.1012 | 0.7833 |
| **temperature** | **0.8210** | **0.5628** | 0.0451 | 0.6095 |
| vector | 0.8031 | 0.5898 | 0.0359 | 0.6253 |
| dirichlet | 0.8101 | 0.5450 | **0.0349** | 0.6211 |

Temperature scaling cuts negative log-likelihood by 29% and expected calibration error by 55% at
**zero accuracy cost**; the richer calibrators buy a little more calibration and pay 1–2 accuracy
points for it. Selection rule was amended mid-round to "best NLL among calibrators within one point
of raw accuracy", because the first version picked Dirichlet and silently gave up accuracy.

## C2 / C3 — the decision layer does not deliver

| Arm | Accuracy | Zoom-pair accuracy | Rows changed |
|---|---|---|---|
| raw argmax (control) | 0.8210 | 0.6548 | — |
| C2 logit bias, nested, on raw probabilities | 0.8226 | 0.6508 | 5 |
| C2 logit bias, nested, on calibrated probabilities | 0.8187 | 0.6468 | 18 |
| C3 defer, margin 0.10 | 0.8202 | 0.6587 | 19 |
| C3 defer, margin 0.35 | 0.8117 | 0.6349 | 50 |

The best arm moves 5 rows for +0.0016 — inside the noise floor, and it does **not** ship. The
fitted bias vector is tiny (max |b| = 0.10) and, notably, **does not push toward Zoom_video**: the
gap between the model and a constant predictor inside the Zoom pair is not recoverable by a global
per-class prior shift, because buying Zoom rows costs rows elsewhere. C3's deferral rule is
uniformly neutral-to-harmful.

## Ensemble, re-opened with the new member

| Combiner | Score | Estimate |
|---|---|---|
| **Single best TabICL** | **0.8381** | grouped out-of-fold — selected |
| Stacking | 0.8327 | nested |
| Hill-climb blend (TabICL ×0.75, LightGBM ×0.25) | 0.8311 | nested |
| Hill-climb blend | 0.8451 | fitted — optimistic, not used |
| Rank averaging | 0.7837 | grouped |

Same verdict as the first round: the single model wins on honest estimates.

## Round summary

| Experiment | Outcome | Shipped |
|---|---|---|
| N1, N2 | controls pass, round valid | — |
| A1 | 7.6% exact test coverage at 0.972 accuracy | — |
| A3 | model already perfect on covered rows | **no** |
| B1 | TabICL 0.8389 vs 0.8249; McNemar p=0.135 | **yes** — new submission model |
| C1 | temperature scaling, −29% NLL at no accuracy cost | yes, as ensemble input |
| C2, C3 | +0.0016 at best, inside noise | **no** |
| F1 (partial) | blend loses to single model, again | no |

**Net: 0.8249 → 0.8389 grouped (+1.40 points), macro-F1 0.8069 → 0.8232 (+1.63 points).**

---

# Round 2 results — items 1–3 (TabICL tuning, probes, call recovery)

## Item 1 — TabICL bagging and its real knobs

Eighteen arms over the parameters that actually exist in the constructor, full repeat budget.

| bag | n_estimators | softmax_temperature | Grouped | sd | Plain | Macro-F1 | Seconds |
|---|---|---|---|---|---|---|---|
| 1 | **16** | 0.90 | **0.8405** | 0.0031 | 0.8236 | 0.8268 | 119 |
| 1 | 16 | 1.05 | 0.8405 | 0.0031 | 0.8236 | 0.8268 | 119 |
| 1 | 16 | 0.75 | 0.8405 | 0.0031 | 0.8236 | 0.8268 | 123 |
| 1 | 8 | any | 0.8389 | 0.0031 | 0.8195 | 0.8232 | 83 |
| 1 | 32 | any | 0.8358 | 0.0023 | 0.8236 | 0.8221 | 196 |
| 5 | 8 | 0.90 | 0.8350 | 0.0016 | 0.8223 | 0.8192 | 372 |
| 5 | 8 | 0.75 | 0.8346 | 0.0019 | 0.8226 | 0.8193 | 373 |

Three findings, none of which ships:

- **`n_estimators=16` gives +0.0016 over the default 8** — inside the 0.0031 noise floor, so recorded
  as not established. `n_estimators=32` is *worse* (−0.0031), so the internal ensemble has an
  interior optimum rather than a monotone one.
- **`softmax_temperature` has no effect on accuracy at all** — identical to four decimals across
  0.75/0.90/1.05. With `average_logits=True` the temperature cancels out of the argmax. It is a
  calibration knob here, not an accuracy knob.
- **Seed-bagging hurts: 0.8350 against 0.8389 unbagged.** It does what bagging is supposed to do to
  variance (sd 0.0031 → 0.0016) and loses accuracy doing it. TabICL already ensembles internally
  over feature shuffles and class shifts, so seed-averaging on top smooths toward a slightly worse
  consensus. The prediction in the plan — "bagging is the cheapest remaining point" — was wrong.

## Item 2 — probe block against the new baseline: still redundant

| n_estimators | Probes | Grouped | Macro-F1 |
|---|---|---|---|
| 32 | on | 0.8300 | 0.8158 |
| 16 | on | 0.8288 | 0.8160 |
| 8 | on | 0.8257 | 0.8108 |
| **16** | **off** | **0.8405** | **0.8268** |
| 8 | off | 0.8389 | 0.8232 |

Adding the 46 probe columns costs **−0.0105** on the best matched pair. The hypothesis was that an
in-context model would have different redundancy structure from gradient boosting and might find
something in the probes that LightGBM could not. It does not: the probe block is now measured as
neutral-to-harmful for three families (LightGBM, XGBoost, TabICL) and marginal for one
(RandomForest, +0.0004). Feature block F12 is retired.

## Item 3 — the D branch: both gates passed, and the idea still failed

| Stage | Measure | Gate | Result |
|---|---|---|---|
| D1 same-call scorer | ROC-AUC on held-out groups | ≥ 0.75 | **0.998** — pass |
| D2 clustering | mean application purity | ≥ 0.90 | **0.997** — pass |
| D2 clustering | adjusted Rand vs pseudo-groups | — | 0.775 |
| D2 clustering | mean label purity | — | 0.989 |
| D3 aggregation, log-probability sum | accuracy | > baseline | 0.8327 (**−0.0054**) |
| D3 aggregation, confidence-weighted vote | accuracy | > baseline | 0.8304 (**−0.0078**) |
| D3 **oracle grouping** (perfect call knowledge) | accuracy | — | 0.8311 (**−0.0070**) |

The oracle row is the one that settles it. **Even with perfect knowledge of which flows share a
call, aggregating within those groups makes accuracy worse.** The reason is visible in D2: 1,024 of
1,107 clusters are singletons and the largest holds 13 flows, so aggregation almost never has a
sibling to consult — and where it does, the siblings already agree with the model. All aggregation
can do is occasionally overrule a correct minority.

Recovering call structure is *easy* here (AUC 0.998, purity 0.997) and *worthless*, because the
structure carries no information the per-flow model has not already extracted. The D branch is
closed — not on a failed gate, but on a measured ceiling.

## Round 2 summary

| Item | Hypothesis | Outcome | Shipped |
|---|---|---|---|
| 1 | bagging and tuning lift TabICL | +0.0016 at best, inside noise; bagging harmful | **no** |
| 2 | probes help an in-context model | −0.0105 | **no** |
| 3 | sibling flows resolve ambiguous ones | aggregation negative even with an oracle grouping | **no** |

**Standing model unchanged: TabICL, defaults, full feature set — 0.8389 grouped / 0.8381 on the
averaged out-of-fold matrix, macro-F1 0.8232.** `submission.csv` is unchanged.

TabPFN v2 was also run in this round on the published checkpoint (`Prior-Labs/TabPFN-v2-clf`,
licence `priorlabs-1-1`, cached with its LICENSE.txt on the volume): 0.8311 at top-120, 0.8230 at
full width, 0.8191 at top-60 — better than every tuned tree but below TabICL, and notably degrading
as feature count rises, which is the documented weakness of prior-fitted transformers on wide,
weakly-informative inputs.
