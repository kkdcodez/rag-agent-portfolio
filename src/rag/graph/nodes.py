"""Agentic RAG 노드 정의.

설계 원칙
- retrieve / generate 는 baseline(src/rag/pipeline.py)과 완전히 동일한 함수를 호출한다.
  Agentic과 baseline의 차이가 "검색 품질"이나 "프롬프트"가 아니라 "판단 구조"뿐이도록 하기 위함.
- grade 는 LLM을 호출하지 않고 rerank score만 사용한다.
  baseline의 실패 원인이 LLM의 판단 흔들림이었으므로, 해결책에 다시 LLM 판단을 넣지 않는다.
  임계값 근거: golden set 라벨 검증(week6_0) 및 baseline 실행에서 관측한 score 분포.
"""
from __future__ import annotations

import re
import time

from .. import config
from ..generation import generate, get_llm
from ..retrieval import retrieve
from .state import AgentState

# --- 거절 문구 (LLM 호출 없이 즉시 반환) ---
REFUSAL = {
    "ko": "제공된 문서에서는 해당 질문에 대한 근거를 확인할 수 없습니다.",
    "en": "The provided documents do not contain information to answer this question.",
}

REWRITE_PROMPT = """다음 질문으로 문서를 검색했으나 충분한 근거를 찾지 못했습니다.
같은 의도를 유지하면서, 문서에서 검색되기 쉬운 표현으로 질문을 다시 작성하세요.

- 문서에서 쓰일 법한 용어나 상위 개념을 사용하세요.
- 의미를 바꾸거나 새로운 조건을 추가하지 마세요.
- 다시 작성한 질문 한 문장만 출력하세요.

원래 질문: {question}
현재 질문: {query}"""


def detect_lang(text: str) -> str:
    """한글이 포함되어 있으면 ko, 아니면 en."""
    return "ko" if re.search(r"[가-힣]", text) else "en"


def _timed(name: str, fn):
    """노드 실행 시간을 측정해 (결과, latency 기록) 형태로 반환."""
    t0 = time.perf_counter()
    result = fn()
    return result, [{"node": name, "sec": round(time.perf_counter() - t0, 3)}]


# ─────────────────────────────────────────────────────────────
# 노드
# ─────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> dict:
    """검색. baseline과 동일한 R4(Hybrid + Rerank)를 그대로 사용한다."""
    query = state.get("query") or state["question"]
    docs, lat = _timed("retrieve", lambda: retrieve(query))
    max_score = max((d.score for d in docs), default=0.0)

    # 재검색 결과가 이전보다 나쁘면 이전 것을 유지한다.
    # (Week 5-3 Multi-Query가 멀쩡한 검색을 악화시켰던 사례를 반복하지 않기 위함)
    prev_docs = state.get("docs")
    prev_score = state.get("max_score", -1.0)
    if prev_docs and max_score < prev_score:
        return {
            "route_history": [f"retrieve(kept prev {prev_score:.3f} > new {max_score:.3f})"],
            "node_latencies": lat,
        }

    return {
        "docs": docs,
        "max_score": max_score,
        "route_history": [f"retrieve(score={max_score:.3f})"],
        "node_latencies": lat,
    }


def grade_node(state: AgentState) -> dict:
    """근거 충분성 판단. 계산은 분기 함수에서 하고, 여기서는 기록만 남긴다."""
    return {
        "route_history": [f"grade(score={state.get('max_score', 0.0):.3f}, "
                          f"retry={state.get('retry_count', 0)})"],
        "node_latencies": [{"node": "grade", "sec": 0.0}],
    }


def rewrite_node(state: AgentState) -> dict:
    """질문을 검색되기 쉬운 표현으로 다시 작성한다."""
    def _run():
        msg = REWRITE_PROMPT.format(
            question=state["question"],
            query=state.get("query") or state["question"],
        )
        return get_llm().invoke([("human", msg)]).content.strip()

    new_query, lat = _timed("rewrite", _run)
    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "route_history": [f"rewrite -> {new_query}"],
        "node_latencies": lat,
    }


def generate_node(state: AgentState) -> dict:
    """답변 생성. baseline과 동일한 프롬프트를 사용한다."""
    docs = state.get("docs", [])
    text, lat = _timed("generate", lambda: generate(state["question"], docs))
    return {
        "answer": text,
        "decision": "answer",
        "route_history": ["generate"],
        "node_latencies": lat,
    }


def refuse_node(state: AgentState) -> dict:
    """거절. LLM을 호출하지 않으므로 빠르며, 질문 언어에 맞춰 문구를 선택한다."""
    lang = state.get("lang") or detect_lang(state["question"])
    return {
        "answer": REFUSAL.get(lang, REFUSAL["ko"]),
        "decision": "refuse",
        "route_history": ["refuse"],
        "node_latencies": [{"node": "refuse", "sec": 0.0}],
    }


# ─────────────────────────────────────────────────────────────
# 분기 (조건부 엣지)
# ─────────────────────────────────────────────────────────────

def route_after_grade(state: AgentState) -> str:
    """rerank score와 재시도 횟수로 다음 노드를 결정한다.

    - score >= 0.5  : 근거 충분      -> generate
    - score <  0.1  : 근거 없음      -> refuse (재검색해도 없을 가능성이 높음)
    - 그 사이       : 애매           -> rewrite 후 재검색 (최대 MAX_RETRY회)
                      재시도 소진 시 -> generate (있는 근거로라도 답변)
    """
    score = state.get("max_score", 0.0)
    retry = state.get("retry_count", 0)

    if score >= config.RELEVANCE_THRESHOLD:
        return "generate"
    if score < config.NO_EVIDENCE_THRESHOLD:
        return "refuse"
    if retry < config.MAX_RETRY:
        return "rewrite"
    return "generate"
