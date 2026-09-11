"""Retrieval — Week 5 최종 구성(R4: Hybrid + Cross-Encoder Rerank).

구성: Dense(e5) + BM25(Kiwi 형태소) → RRF 융합 → Cross-Encoder 재정렬 → top-k

모델과 인덱스는 최초 호출 시 1회만 로드한다(lazy singleton).
import 시점에 로드하지 않으므로 config 확인용으로 가볍게 import할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import config


@dataclass
class RetrievedDoc:
    text: str
    source: str          # 파일명
    page: object
    score: float         # rerank score
    lang: Optional[str] = None

    def citation(self) -> str:
        return f"{self.source} p.{self.page}"


class _Index:
    """코퍼스 + dense/BM25/reranker를 한 번만 로드해 보관."""

    _instance: Optional["_Index"] = None

    def __init__(self) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder

        self.emb = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectordb = Chroma(
            persist_directory=str(config.INDEX_DIR),
            embedding_function=self.emb,
            collection_name=config.COLLECTION_NAME,
        )

        raw = self.vectordb.get(include=["documents", "metadatas"])
        self.texts: list[str] = raw["documents"]
        self.metas: list[dict] = raw["metadatas"]
        if not self.texts:
            raise RuntimeError(
                f"인덱스가 비어 있습니다: {config.INDEX_DIR} / {config.COLLECTION_NAME}"
            )
        self.text2idx = {t: i for i, t in enumerate(self.texts)}

        self.kiwi = Kiwi()
        self.bm25 = BM25Okapi([self._tokenize(t) for t in self.texts])
        self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)

    def _tokenize(self, text: str) -> list[str]:
        return [t.form.lower() for t in self.kiwi.tokenize(text) if t.form.strip()]

    @classmethod
    def get(cls) -> "_Index":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- 개별 검색기 ---

    def dense_rank(self, query: str, n: int) -> list[int]:
        hits = self.vectordb.similarity_search(config.QUERY_PREFIX + query, k=n)
        return [self.text2idx[d.page_content] for d in hits if d.page_content in self.text2idx]

    def bm25_rank(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(self._tokenize(query))
        return list(np.argsort(scores)[::-1][:n])

    def hybrid_rrf(self, query: str, n_each: int) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self.dense_rank(query, n_each), self.bm25_rank(query, n_each)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """R4 검색. rerank score 내림차순으로 반환."""
    top_k = top_k or config.TOP_K
    idx = _Index.get()

    cand = idx.hybrid_rrf(query, config.CANDIDATE_EACH)[: config.RERANK_CANDIDATES]
    if not cand:
        return []

    scores = idx.reranker.predict([(query, idx.texts[i]) for i in cand])
    order = np.argsort(scores)[::-1][:top_k]

    docs = []
    for j in order:
        i = cand[j]
        meta = idx.metas[i]
        docs.append(
            RetrievedDoc(
                text=idx.texts[i],
                source=Path(str(meta.get("source", "?"))).name,
                page=meta.get("page", "?"),
                score=float(scores[j]),
                lang=meta.get("language"),
            )
        )
    return docs


def format_context(docs: list[RetrievedDoc]) -> str:
    """생성 프롬프트에 넣을 context 문자열. 출처를 함께 표기해 인용을 유도한다."""
    return "\n\n".join(
        f"[{i}] ({d.citation()})\n{d.text}" for i, d in enumerate(docs, 1)
    )
