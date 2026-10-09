# Round 3 — the label geometry, and what it costs

Rounds 1 and 2 searched for a better *model*. Every avenue closed: tuning and
bagging the in-context model, the auxiliary-corpus probe block, tuple lookup
tables, and call-structure recovery all measured neutral or negative. Round 3
therefore stopped asking which estimator to use and asked instead what the
label geometry of this dataset actually is, and what the best attainable score
under that geometry would be.

## 1. Where the errors are

A three-repeat grouped cross-validation of the standing gradient-boosting
configuration gives 0.8156 accuracy and 0.7982 macro-F1. Decomposing its 237
errors along the two label dimensions:

| Dimension | Accuracy |
|---|---|
| application (5-way) | 0.9035 |
| mode (voice/video) | 0.8934 |
| mode, given the application is right | 0.9027 |

and by packet-size regime, splitting flows on whether every one of their five
packets falls inside the audio band (`config.BAND_AUDIO`, < 300 bytes):

| Regime | Flows | Accuracy | Errors |
|---|---|---|---|
| audio-only window | 726 (56.5%) | 0.7590 | 175 |
| contains a larger packet | 559 (43.5%) | 0.8891 | 62 |

A single pair dominates: **Zoom_voice against Zoom_video accounts for 68 of the
237 errors.** Zoom_voice recall is 0.34.

## 2. Why that pair is hard

Every flow record is the first five packets of a flow. A voice call produces
only audio-sized packets. A video call produces audio-sized packets too — and
if its opening five packets contain no video fragment, the record it produces
is a video-call record that looks exactly like a voice-call record.

Measured, per application, over flows whose window is audio-only:

| Application | Audio-only flows | of which video | Grouped-CV AUC, video vs voice |
|---|---|---|---|
| Discord | 290 | 52 | 0.880 |
| Google Meet | 85 | 45 | 0.826 |
| Messenger | 111 | 19 | 0.879 |
| **Zoom** | **160** | **80** | **0.551** |

For three applications the two remain separable inside the audio band, which is
consistent with per-application differences in codec and packetisation. For
Zoom they do not. The Zoom subset is 80 voice flows against 80 video flows —
exactly balanced — and a gradient boosting model over all 241 features scores
AUC 0.551 on it, which at n = 160 is not distinguishable from a coin flip.

The split is not an artefact of the 300-byte boundary. Every threshold from 200
to 500 bytes isolates the same subset (75/76 at 200 bytes, 80/80 from 250
bytes upward).

## 3. The ceiling this implies

Those 160 flows can never contribute more than chance. Taking every other flow
in the dataset as perfectly classified:

* **accuracy ceiling 0.9377** — 80 of 1,285 flows are unwinnable;
* **macro-F1 ceiling 0.9364**, attained by sending the whole ambiguous subset to
  Zoom_voice, which yields Zoom_voice F1 = 0.667 and Zoom_video F1 = 0.697.

Reaching macro-F1 0.90 therefore requires the other eight classes to average
F1 ≥ 0.954. They currently average 0.868, and a large part of their remaining
error is the same audio-only-window ambiguity in its milder, still partly
irreducible form (AUC 0.83–0.88, not 1.0). **Macro-F1 of 0.90 is not reachable
on this dataset**, and the statement is a measurement, not an estimate.

## 4. The one rule the geometry licenses

On an evenly split, inseparable subset every assignment scores the same
accuracy — so the choice is free in accuracy, and it is not free in macro-F1.
Sending the entire subset to the smaller class rescues that class's recall
without costing the larger class anything it was going to keep.

> **Zoom rule.** If a flow is predicted to be Zoom and its window is audio-only,
> label it Zoom_voice.

The rule has no fitted parameter: the direction follows from the 80/80 balance
and the threshold is the band boundary already declared in `config.BAND_AUDIO`.
Measured on out-of-fold predictions:

| Base model | macro-F1 | with the Zoom rule |
|---|---|---|
| flat | 0.7942 | 0.8080 |
| hierarchical | 0.7978 | 0.8092 |
| average of the two | 0.8041 | 0.8181 |

Bootstrap over 2,000 resamples of the averaged model: **+0.0140 macro-F1, 95%
interval [+0.000, +0.028], P(gain > 0) = 0.975**, with accuracy unchanged
(0.8233 to 0.8241). Zoom_voice F1 rises from 0.384 to 0.580.

## 5. Model-side changes that were tested

| Change | Result | Shipped |
|---|---|---|
| Hierarchical application × mode heads | +0.004 macro-F1 alone; adds a useful second view | as an ensemble member |
| Fifteen-class mixture view (video split by window type) | 0.8063 alone, above flat | as an ensemble member |
| Average of flat + hierarchical + mixture | 0.8085, 0.8185 with the Zoom rule | **yes** |
| Global mode head instead of per-application heads | AUC 0.9661 vs 0.9680 — no difference | no |
| Class-balanced sample weights in every head | flat +0.005, ensemble −0.003 | no |
| Per-class log-probability bias fitted for macro-F1 | in-sample 0.8124, nested 0.7956 vs 0.7982 raw | no |
| Single global mode-bias parameter | nested 0.8173 vs 0.8181 unbiased | no |
| MIRAGE-supervised application transfer | see below | no |

Two decision-layer parameterisations were fitted for macro-F1 and both lost
under nested evaluation while gaining in sample — at 1,285 rows a fitted
threshold is a memorised threshold. The Zoom rule survives precisely because it
is derived rather than fitted.

## 6. MIRAGE, closed

MIRAGE-AppAct-2024 contains directories for all five competition applications,
so a classifier trained on its flows should transfer if the two corpora are
commensurable. Restricted to UDP media flows of those five applications with at
least five payload-bearing packets (7,655 flows), a gradient boosting model
trained purely on MIRAGE and applied to the competition training set scores:

| Length convention | Application accuracy on competition data |
|---|---|
| L4 payload bytes as stored | 0.3183 |
| plus 28 bytes of IP+UDP header | 0.2988 |
| majority-class baseline | 0.3844 |

**Both settings fall below predicting the majority class.** The domain gap
between a 2023 Android capture campaign and this corpus is total at the level
of packet sizes. This also retrospectively explains round 2's probe result: a
representation pretrained on MIRAGE has nothing commensurable to transfer.

The activity dimension of MIRAGE-AppAct-2024 was also checked as a possible
source of voice/video supervision. It is not present in the published archive —
`flow_metadata` carries the Android package name and byte/packet counters only,
and the archive contains no separate activity manifest. The application name in
the directory layout is the whole label.

## 7. Results

Six base learners, each run in three views (flat ten-class, hierarchical
application × mode, fifteen-class mixture) over the stored grouped folds, with
and without the Zoom rule.

### The rule holds everywhere

| Base learner | best macro-F1 without the rule | best with the rule |
|---|---|---|
| TabICL | 0.8262 | 0.8292 |
| LightGBM | 0.8066 | 0.8169 |
| XGBoost | 0.7965 | 0.8044 |
| ExtraTrees | 0.7815 | 0.7885 |
| CatBoost | 0.7735 | 0.7804 |
| RandomForest | 0.7640 | 0.7725 |

**Six families out of six improve, in every view.** That is the strongest
evidence available here that the rule reflects the data rather than the
estimator: it was derived from a class-balance measurement, never fitted, and
it survives a complete change of hypothesis class six times.

### Candidate blends

| Candidate | accuracy | macro-F1 | accuracy + rule | macro-F1 + rule |
|---|---|---|---|---|
| TabICL ×2 + LightGBM, flat+hier | 0.8381 | 0.8216 | 0.8374 | **0.8319** |
| **TabICL, flat+hier average** | 0.8358 | 0.8201 | 0.8342 | **0.8292** |
| TabICL, flat only (round-2 standing model) | 0.8381 | 0.8262 | 0.8304 | 0.8274 |
| TabICL + LightGBM + XGBoost | 0.8342 | 0.8147 | 0.8311 | 0.8231 |
| all six families | 0.8272 | 0.8082 | 0.8249 | 0.8172 |

### What ships, and how sure we are

The shipped model is **TabICL, flat and hierarchical views averaged, with the
Zoom rule**: grouped-CV accuracy 0.8342, **macro-F1 0.8292**. It is chosen
structurally rather than by taking the maximum of the table - the family was
already established in round 1, the two views are the pre-specified pair, and
the rule has no parameter. The weighted TabICL ×2 + LightGBM blend scores
higher (0.8319) but is the maximum of seventeen candidates evaluated on the
same 1,285 rows, which is exactly the kind of choice that does not reproduce.

Against the round-2 standing model, on paired resamples of the same
out-of-fold predictions:

| Comparison | Δ macro-F1 | 95% interval | P(Δ > 0) | McNemar |
|---|---|---|---|---|
| shipped vs standing | +0.0030 | [−0.0130, +0.0201] | 0.630 | 46/51, p = 0.685 |
| observed maximum vs standing | +0.0058 | [−0.0114, +0.0231] | 0.744 | 53/54, p = 1.000 |

**Neither difference is established.** The honest reading is that the model
change is worth roughly three macro-F1 thousandths with wide error bars, while
the *diagnosis* - which flows are unwinnable and why - is the durable result of
this round.

### Per class, shipped model

| Class | precision | recall | F1 |
|---|---|---|---|
| Discord_voice | 0.824 | 0.908 | 0.864 |
| Discord_video | 0.961 | 0.855 | 0.905 |
| GoogleMeet_voice | 0.769 | 0.750 | 0.759 |
| GoogleMeet_video | 0.790 | 0.741 | 0.765 |
| Messenger_voice | 0.888 | 0.926 | 0.906 |
| Messenger_video | 0.936 | 0.893 | 0.914 |
| WhatsApp_voice | 0.976 | 1.000 | 0.988 |
| WhatsApp_video | 0.899 | 0.976 | 0.936 |
| Zoom_voice | 0.431 | 0.900 | 0.583 |
| Zoom_video | 0.978 | 0.512 | 0.672 |

Zoom now sits at F1 0.583 + 0.672 = 1.255 against the structural maximum of
0.667 + 0.697 = 1.364, so roughly nine tenths of what that pair can ever yield
has been taken. The remaining headroom is in Google Meet (0.759 / 0.765 on 152
flows) and Discord_voice.

## 8. Submission

`submission.csv`, 327 rows, `index,label`, no header. 43 of the 327 test flows
are predicted Zoom with an audio-only window and are therefore routed to
Zoom_voice by the rule - consistent with the training-set proportion, where the
ambiguous subset is an even 80/80 split.

## 9. Standing answer on the 0.90 target

Not attainable. The ceiling is 0.9364 macro-F1 with all eight non-Zoom classes
perfect, and reaching 0.90 requires those eight to average F1 ≥ 0.954 against a
current 0.868 whose residual is itself partly irreducible. The shipped model is
at 0.8292.
