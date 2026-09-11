"""LangGraph 조립 및 실행 진입점.

    질문
     ↓
    retrieve  ──────────────────────┐
     ↓                              │
    grade                           │
     ├─ score >= 0.5  → generate    │
     ├─ score <  0.1  → refuse      │
     └─ 그 사이       → rewrite ────┘  (최대 2회, 소진 시 generate)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from ..retrieval import RetrievedDoc
from .nodes import (
    detect_lang,
    generate_node,
    grade_node,
    refuse_node,
    retrieve_node,
    rewrite_node,
    route_after_grade,
)
from .state import AgentState

_app = None


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("refuse", refuse_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "refuse": "refuse", "rewrite": "rewrite"},
    )
    g.add_edge("rewrite", "retrieve")   # 재검색 루프
    g.add_edge("generate", END)
    g.add_edge("refuse", END)

    return g.compile()


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


@dataclass
class AgenticResult:
    question: str
    answer: str
    decision: str                      # "answer" | "refuse"
    docs: list[RetrievedDoc] = field(default_factory=list)
    max_score: float = 0.0
    retry_count: int = 0
    route_history: list[str] = field(default_factory=list)
    node_latencies: list[dict] = field(default_factory=list)
    latency: float = 0.0

    @property
    def contexts(self) -> list[str]:
        return [d.text for d in self.docs]

    @property
    def citations(self) -> list[str]:
        return [d.citation() for d in self.docs]

    @property
    def route(self) -> str:
        """route_history를 한 줄로 압축 (CSV 저장용)."""
        return " → ".join(self.route_history)


def run(question: str) -> AgenticResult:
    t0 = time.perf_counter()
    final = get_app().invoke({
        "question": question,
        "query": question,
        "lang": detect_lang(question),
        "retry_count": 0,
    })
    return AgenticResult(
        question=question,
        answer=final.get("answer", ""),
        decision=final.get("decision", ""),
        docs=final.get("docs", []),
        max_score=final.get("max_score", 0.0),
        retry_count=final.get("retry_count", 0),
        route_history=final.get("route_history", []),
        node_latencies=final.get("node_latencies", []),
        latency=time.perf_counter() - t0,
    )
