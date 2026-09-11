"""Baseline RAG 파이프라인 — retrieve → generate.

Week 5 최종 구성 그대로이며, Week 6 Agentic RAG의 비교 기준(baseline)이다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .generation import generate
from .retrieval import RetrievedDoc, retrieve


@dataclass
class RagResult:
    question: str
    answer: str
    docs: list[RetrievedDoc] = field(default_factory=list)
    latency: float = 0.0

    @property
    def contexts(self) -> list[str]:
        """RAGAS 입력용 context 리스트."""
        return [d.text for d in self.docs]

    @property
    def citations(self) -> list[str]:
        return [d.citation() for d in self.docs]

    @property
    def max_score(self) -> float:
        return max((d.score for d in self.docs), default=0.0)


def answer(question: str, top_k: int | None = None) -> RagResult:
    t0 = time.perf_counter()
    docs = retrieve(question, top_k=top_k or config.TOP_K)
    text = generate(question, docs)
    return RagResult(
        question=question,
        answer=text,
        docs=docs,
        latency=time.perf_counter() - t0,
    )
