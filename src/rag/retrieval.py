"""Retrieval — Week 5 최종 채택안(R4: Hybrid + Cross-Encoder Rerank).

이 파일은 Week 5 노트북의 함수를 그대로 옮긴 것이다.
각 함수 위 주석에 출처 노트북과 셀 번호를 적었다. 값을 바꿀 때는
반드시 해당 셀과 대조하고 decision_log에 기록한다.

원본
- week5_0_preprocess.ipynb  cell 16  청킹 · 메타데이터
- week5_0_preprocess.ipynb  cell 17  임베딩 · 인덱싱 (prefix 미사용)
- week5_2_reranking.ipynb   cell 2   상수 (TOP_K=5, FETCH_K=20, RRF_K=60)
- week5_2_reranking.ipynb   cell 6   BM25 토큰화 · hybrid_search
- week5_2_reranking.ipynb   cell 8   hybrid_rerank_search

주의
- 인덱싱 시 "passage: " prefix를 쓰지 않았으므로, 질의에도 prefix를 붙이지 않는다.
- BM25 토큰화는 내용어만 남긴다. 조사·어미를 포함하면 고빈도 토큰이 신호를
  희석시켜 표 형태 chunk가 밀려난다 (Week 6에서 실제로 발생).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config


@dataclass
class RetrievedDoc:
    text: str
    filename: str
    page: object
    score: float
    org: Optional[str] = None
    title: Optional[str] = None
    lang: Optional[str] = None

    @property
    def source(self) -> str:
        """하위 호환용 별칭."""
        return self.filename

    def citation(self) -> str:
        return f"{self.filename} p.{self.page}"


class _Index:
    """코퍼스 + dense/BM25/reranker를 한 번만 로드해 보관."""

    _instance: Optional["_Index"] = None

    def __init__(self) -> None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder

        # week5_0 cell 17 / week5_2 cell 5 와 동일
        self.emb = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
        self.vectordb = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.emb,
            persist_directory=str(config.INDEX_DIR),
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

        # week5_2 cell 8 와 동일
        self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device="cpu")

    # week5_2 cell 6 — 내용어(명사·동사·S계열·외국어·숫자)만 남긴다
    def _tokenize(self, text: str) -> list[str]:
        return [
            t.form.lower()
            for t in self.kiwi.tokenize(text)
            if t.tag[0] in ("N", "V", "S") or t.tag in ("SL", "SN", "XR")
        ]

    @classmethod
    def get(cls) -> "_Index":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # --- 개별 검색기 ---

    # week5_2 cell 5 — dense_search. 질의에 prefix를 붙이지 않는다.
    def dense_rank(self, query: str, n: int) -> list[int]:
        hits = self.vectordb.similarity_search(query, k=n)
        return [self.text2idx[d.page_content] for d in hits if d.page_content in self.text2idx]

    # week5_2 cell 6 — bm25_search
    def bm25_rank(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(self._tokenize(query))
        return list(np.argsort(scores)[::-1][:n])

    # week5_2 cell 6 — hybrid_search (RRF 융합)
    def hybrid_rrf(self, query: str, n_each: int) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self.dense_rank(query, n_each), self.bm25_rank(query, n_each)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)


# week5_2 cell 8 — hybrid_rerank_search
def retrieve(query: str, top_k: int | None = None) -> list[RetrievedDoc]:
    """Hybrid(dense FETCH_K + bm25 FETCH_K → RRF → top FETCH_K) → rerank → top_k."""
    top_k = top_k or config.TOP_K
    idx = _Index.get()

    cand = idx.hybrid_rrf(query, config.FETCH_K)[: config.FETCH_K]
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
                filename=meta.get("filename", "?"),
                page=meta.get("page", "?"),
                score=float(scores[j]),
                org=meta.get("org"),
                title=meta.get("title"),
                lang=meta.get("language"),
            )
        )
    return docs


# week5_0 cell 19 / week5_2 cell 13 — format_context
def format_context(docs: list[RetrievedDoc]) -> str:
    """생성 프롬프트에 넣을 context 문자열. Week 5와 동일한 형식."""
    return "\n\n---\n\n".join(
        f"[{i}] 출처: {d.org or '?'} / {d.title or '?'} / p.{d.page}\n{d.text}"
        for i, d in enumerate(docs, 1)
    )
