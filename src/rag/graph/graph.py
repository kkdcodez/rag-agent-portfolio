"""LangGraph 조립 및 실행 진입점 (v3).

    질문
     ↓
    retrieve  ────────────────────────────┐
     ↓                                    │
    filter (rerank score)                 │
     ├─ >= 0.5 ──────────→ generate       │
     └─ <  0.5 → judge (LLM)              │
                  ├─ answer  → generate   │
                  ├─ refuse  → refuse     │
                  └─ rewrite → rewrite ───┘  (최대 2회, 소진 시 answer로 대체)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph

from ..retrieval import RetrievedDoc
from .nodes import (
    detect_lang,
    filter_node,
    generate_node,
    judge_node,
    refuse_node,
    retrieve_node,
    rewrite_node,
    route_after_filter,
    route_after_judge,
)
from .state import AgentState

_app = None


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("filter", filter_node)
    g.add_node("judge", judge_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)
    g.add_node("refuse", refuse_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "filter")

    # score 필터: 충분하면 즉시 생성, 아니면 판정자에게
    g.add_conditional_edges(
        "filter",
        route_after_filter,
        {"generate": "generate", "judge": "judge"},
    )

    # 판정자의 결정을 따른다
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"},
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
    judge_calls: list[dict] = field(default_factory=list)
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

    @property
    def judged(self) -> bool:
        """판정자를 거쳤는지 여부."""
        return bool(self.judge_calls)

    def _last(self, key: str, default: str = "") -> str:
        return self.judge_calls[-1].get(key, default) if self.judge_calls else default

    @property
    def judge_decision(self) -> str:
        return self._last("decision")

    @property
    def judge_evidence(self) -> str:
        return self._last("evidence")

    @property
    def judge_category(self) -> str:
        return self._last("category")

    @property
    def judge_reason(self) -> str:
        return self._last("reason")

    @property
    def judge_trace(self) -> str:
        """판정 이력을 한 줄로 (CSV 저장용)."""
        return " | ".join(
            f"[{c.get('retry')}] {c.get('decision')}/{c.get('evidence')}/{c.get('category')}"
            for c in self.judge_calls
        )


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
        judge_calls=final.get("judge_calls", []),
        route_history=final.get("route_history", []),
        node_latencies=final.get("node_latencies", []),
        latency=time.perf_counter() - t0,
    )
