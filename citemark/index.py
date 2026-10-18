import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunk import Chunk

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = os.environ.get("CITEMARK_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Index:
    """Chunks plus two ways to find them: BM25 over words and cosine similarity over embeddings.
    Optionally reranks the merged candidates with a cross-encoder."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embed_model: str = EMBED_MODEL):
        self.chunks = chunks
        self.vectors = vectors
        self.embed_model = embed_model
        self.bm25 = BM25([f"{c.title} {c.text}" for c in chunks])
        self._embedder = None
        self._reranker = None

    # --- models are loaded lazily so tests and BM25-only use stay light ---------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    @classmethod
    def build(cls, chunks: list[Chunk], embed_model: str = EMBED_MODEL, batch_size: int = 64):
        idx = cls(chunks, np.zeros((len(chunks), 0), dtype=np.float32), embed_model)
        idx.vectors = idx.embedder.encode([f"{c.title}\n{c.text}" for c in chunks], batch_size=batch_size,
                                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return idx

    def save(self, folder: str):
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(p / "vectors.npy", self.vectors.astype(np.float32))
        (p / "meta.json").write_text(json.dumps({"embed_model": self.embed_model}))

    @classmethod
    def load(cls, folder: str):
        p = Path(folder)
        chunks = [Chunk(**json.loads(line)) for line in (p / "chunks.jsonl").open(encoding="utf-8")]
        meta = json.loads((p / "meta.json").read_text())
        return cls(chunks, np.load(p / "vectors.npy"), meta["embed_model"])

    # --- search ------------------------------------------------------------------------------------
    def dense(self, query: str, k: int) -> list[int]:
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        return list(np.argsort(-(self.vectors @ q))[:k])

    def keyword(self, query: str, k: int) -> list[int]:
        s = self.bm25.scores(query)
        top = np.argsort(-s)[:k]
        return [int(i) for i in top if s[i] > 0]

    def hybrid(self, query: str, k: int, pool: int = 50, rrf_k: int = 60) -> list[int]:
        """Reciprocal rank fusion: score = sum over the two rankings of 1 / (rrf_k + rank)."""
        score: dict[int, float] = {}
        for ranking in (self.dense(query, pool), self.keyword(query, pool)):
            for rank, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(score, key=score.get, reverse=True)[:k]

    def search(self, query: str, k: int = 5, mode: str = "hybrid+rerank", pool: int = 30) -> list[tuple[Chunk, float]]:
        if mode == "keyword":
            ids = self.keyword(query, k)
        elif mode == "dense":
            ids = self.dense(query, k)
        elif mode == "hybrid":
            ids = self.hybrid(query, k)
        elif mode == "hybrid+rerank":
            cand = self.hybrid(query, pool)
            scores = self.reranker.predict([(query, f"{self.chunks[i].title}\n{self.chunks[i].text}") for i in cand])
            order = np.argsort(-np.asarray(scores))[:k]
            return [(self.chunks[cand[j]], float(scores[j])) for j in order]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [(self.chunks[int(i)], 0.0) for i in ids]
