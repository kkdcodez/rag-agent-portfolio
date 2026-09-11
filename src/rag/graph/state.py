"""Agentic RAG의 상태 정의.

LangGraph는 노드들이 하나의 dict(State)를 주고받으며 동작한다.
각 노드는 State를 읽고, 바뀐 부분만 반환한다.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from ..retrieval import RetrievedDoc


def append(a: list, b: list) -> list:
    """리스트 필드는 덮어쓰지 않고 이어붙인다 (경로 추적용)."""
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    # --- 입력 ---
    question: str          # 사용자의 원래 질문 (변하지 않음)
    lang: str              # "ko" | "en" — 거절 문구 언어 선택에 사용

    # --- 진행 상태 ---
    query: str             # 실제로 검색에 쓰는 질문 (rewrite로 바뀔 수 있음)
    docs: list[RetrievedDoc]
    max_score: float
    retry_count: int

    # --- 판정 기록 (v3에서 추가) ---
    judge_calls: Annotated[list[dict], append]   # 판정자 호출별 decision/reason

    # --- 출력 ---
    answer: str
    decision: Literal["answer", "refuse"]   # 최종 행동

    # --- 기록 (7주차 Routing Accuracy / latency 분석용) ---
    route_history: Annotated[list[str], append]
    node_latencies: Annotated[list[dict], append]
