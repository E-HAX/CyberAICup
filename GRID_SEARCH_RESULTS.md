# Grid search — hybrid fusion weight × dense model

_Retrieval-quality grid search over the fusion weight (alpha = dense weight, 1-alpha = BM25 weight) and two dense models (bge-small vs bge-base). Metric = retrieval hit-rate against gold's cited evidence, over all 10 dev cases (369 CPs with a track reference, 142 with a quoted snippet). No LLM calls — pure local retrieval evaluation._

## hit@5 (track-level: did we retrieve a track gold's evidence points to)

| dense | alpha=0.0 (pure BM25) | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 (pure dense) |
|---|---|---|---|---|---|---|
| **bge-small** | 0.515 | 0.558 | 0.645 | 0.759 | 0.794 | **0.808** |
| bge-base | 0.515 | 0.547 | 0.580 | 0.659 | 0.705 | 0.726 |

## hit@5 (phrase-level: did we retrieve the decisive quoted sentence's tokens)

| dense | alpha=0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| **bge-small** | 0.261 | 0.324 | 0.451 | 0.556 | **0.556** |
| bge-base | 0.261 | 0.324 | 0.359 | 0.430 | 0.423 |

## Findings

1. **Dense retrieval dominates — the opposite of the BM25-heavy hypothesis.** Track-hit@5 rises monotonically from 0.515 (pure BM25) to 0.808 (pure dense, bge-small). Every unit of weight moved from BM25 to dense *improves* retrieval. Pure dense (alpha=1.0) is the single best setting; there is no hybrid point that beats it.

2. **bge-base is WORSE than bge-small** (0.726 vs 0.808 at pure dense; 0.423 vs 0.556 phrase-hit). The "upgrade" to the larger model would have *hurt*. On this short-fragment, boilerplate-heavy corpus, bge-small's embeddings happen to discriminate better. No Modal needed — and it would have been counterproductive.

3. **Even the best config has a hard ceiling.** At bge-small/alpha=1.0, track-hit@5 is 0.808 but phrase-hit@5 is only 0.556 — i.e. ~44% of the time the *decisive sentence* (a status flag, a "Mandarin" note) still isn't in the top-5 chunks. That's the fundamental reason RAG underperforms full-context on this task.

4. **The earlier RAG-vs-full-context run used alpha=0.5** (track-hit@5 0.645), which this grid shows is suboptimal. Re-running RAG at alpha=1.0 (pure bge-small) should recover a meaningful chunk of the gap — worth doing before concluding RAG is strictly worse.

## Recommendation

Set `alpha=1.0` (pure dense, bge-small) for any further RAG experiments. Do **not** adopt bge-base. Re-run the RAG LLM comparison at alpha=1.0 on the same 4 cases to get the honest "best-case RAG" number, then re-compare against full-context.
