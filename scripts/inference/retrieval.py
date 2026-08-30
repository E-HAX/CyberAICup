"""Hybrid retrieval for FRECA: lexical (BM25) + semantic (dense) -> weighted
fusion -> ColBERT-style late-interaction rerank -> top-k evidence chunks.

Deterministic / reproducible (no randomness; pinned ONNX models):
  dense    : BAAI/bge-small-en-v1.5   (384-dim, 67MB)
  reranker : answerdotai/answerai-colbert-small-v1 (ColBERT late-interaction, 130MB)
  lexical  : BM25 (rank_bm25, pure Python, tokenizer-only)
All inference is local CPU (torch-free, ONNX runtime via fastembed).
"""
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "answerdotai/answerai-colbert-small-v1"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

WORD_RE = re.compile(r"[a-zA-Z0-9]+")
TRACK_RE = re.compile(r"^## Track (\d+)\s*—")


def tokenize(text):
    return [t.lower() for t in WORD_RE.findall(text or "")]


def chunk_evidence(case_md_text, case_id, target_words=280):
    """Split evidence markdown into track-tagged chunks.
    Returns list of {case_id, track, chunk_id, text}.
    Boundaries: track headers, `### Sheet:` headers, blank lines, and hard
    size caps (oversized blocks split at line boundaries so dense tabular
    registers don't collapse into one giant unretrievable chunk).
    """
    lines = case_md_text.split("\n")
    track = None
    records = []          # (track, record_text)
    buf = []

    def flush():
        if buf:
            records.append((track, "\n".join(buf).strip()))
            buf[:] = []

    for ln in lines:
        m = TRACK_RE.match(ln)
        if m:
            flush()
            track = int(m.group(1))
            continue
        if ln.strip() == "" or not WORD_RE.search(ln):
            flush()
            continue
        if ln.startswith("### "):
            flush()
            buf.append(ln)
            continue
        buf.append(ln)
    flush()

    chunks = []
    for tr, rec in records:
        if tr is None:
            continue
        if len(rec.split()) <= target_words:
            chunks.append((tr, rec))
            continue
        # oversized: split at line boundaries
        cur, curw = [], 0
        for ln in rec.split("\n"):
            w = len(ln.split())
            if cur and curw + w > target_words:
                chunks.append((tr, "\n".join(cur)))
                cur, curw = [], 0
            cur.append(ln)
            curw += w
        if cur:
            chunks.append((tr, "\n".join(cur)))

    out = []
    for i, (tr, text) in enumerate(chunks):
        out.append({"case_id": case_id, "track": tr, "chunk_id": i, "text": text})
    return out


class HybridRetriever:
    """Lexical BM25 + dense bge-small, weighted fusion, optional ColBERT rerank."""

    def __init__(self, chunks, alpha=0.5):
        self.chunks = chunks
        self.alpha = alpha
        self.corpus = [c["text"] for c in chunks]
        self.tokenized = [tokenize(t) for t in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized)

        from fastembed import TextEmbedding
        self.dense = TextEmbedding(DENSE_MODEL)
        self.doc_embs = np.asarray(list(self.dense.embed(self.corpus)), dtype="float32")

        self._reranker = None  # lazy

    def _dense_query(self, query):
        q = BGE_QUERY_PREFIX + query
        return np.asarray(list(self.dense.embed([q]))[0], dtype="float32")

    def _bm25_scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")

    def _dense_scores(self, query):
        q = self._dense_query(query)
        q = q / (np.linalg.norm(q) + 1e-9)
        d = self.doc_embs / (np.linalg.norm(self.doc_embs, axis=1, keepdims=True) + 1e-9)
        return (d @ q).astype("float32")

    @staticmethod
    def _norm(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def retrieve(self, query, top_k=6, fuse_k=24):
        b = self._norm(self._bm25_scores(query))
        d = self._norm(self._dense_scores(query))
        fused = self.alpha * d + (1 - self.alpha) * b

        cand_idx = np.argsort(-fused)[:fuse_k]
        cand_idx = self._rerank(query, cand_idx)

        return [(self.chunks[i]["track"], float(fused[i]), self.corpus[i]) for i in cand_idx[:top_k]]

    def _rerank(self, query, cand_idx):
        """ColBERT late-interaction MaxSim rerank of the candidate set."""
        if len(cand_idx) <= 1:
            return cand_idx
        if self._reranker is None:
            from fastembed import LateInteractionTextEmbedding
            self._reranker = LateInteractionTextEmbedding(RERANK_MODEL)
        q = np.asarray(list(self._reranker.query_embed(query))[0], dtype="float32")  # [Q, D]
        docs = [self.corpus[i] for i in cand_idx]
        ps = list(self._reranker.passage_embed(docs))  # list of [P_i, D]
        scores = []
        for p in ps:
            p = np.asarray(p, dtype="float32")
            sim = q @ p.T  # [Q, P]
            scores.append(float(sim.max(axis=1).sum()))
        order = np.argsort(-np.asarray(scores))
        return cand_idx[order]


def build_query(cp_text, governing_sections, max_words=50):
    """Query = CP official text + a SHORT, clean governing-policy expansion.
    Markdown headers are stripped and the expansion is capped so dense/BM25
    see a focused query (long legalese dilutes retrieval)."""
    g_parts = []
    for g in governing_sections:
        # strip markdown `# ...` header lines and the duplicate raw 'X-Y Title' line
        lines = [ln for ln in g.split("\n") if not ln.startswith("#")]
        g_parts.append(" ".join(lines))
    g = " ".join(g_parts)
    g = " ".join(g.split()[:max_words])
    return f"{cp_text} {g}".strip()
