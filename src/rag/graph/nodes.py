"""Agentic RAG 노드 정의 (v3).

v1 → v3 변경점
- rerank score의 역할을 "필터"로 한정한다. 0.5 이상이면 즉시 generate.
- 0.5 미만은 LLM 판정자(judge)가 조각 내용을 읽고 answer / rewrite / refuse를 결정한다.
- v1의 NO_EVIDENCE_THRESHOLD(0.1)를 제거했다. 그 선 아래로 떨어진 문항이
  판정 기회 없이 즉시 거절되어, 안내가 필요한 질문(qid 38)이 고정 문구만 받았다.
- 재작성 질의를 판정자가 직접 생성한다. 무엇이 부족했는지 판단한 주체가
  검색어도 만드는 편이 정확하고, LLM 호출도 1회 줄어든다.

판정 정책의 핵심
- refuse는 되돌릴 수 없다. 생성 단계에 "근거 없으면 확인할 수 없다고 답하라"는
  규칙이 있으므로, 잘못된 answer는 회복되지만 잘못된 refuse는 회복되지 않는다.
- 따라서 refuse에는 "범위 밖임을 적극적으로 입증하는 근거"를 요구한다.
  현재 조각에 답이 없다는 사실만으로는 refuse의 근거가 되지 않는다.

설계 원칙
- retrieve / generate 는 baseline(src/rag/pipeline.py)과 동일한 함수를 호출한다.
- 판정자는 골든셋 라벨(q_type, expected_behavior)을 절대 받지 않는다.
  실제 서비스에서 존재하지 않는 정보이기 때문이다.
"""
from __future__ import annotations

import json
import re
import time

from .. import config
from ..generation import generate, get_llm
from ..retrieval import RetrievedDoc, retrieve
from .state import AgentState

# --- 거절 문구 (LLM 호출 없이 즉시 반환) ---
REFUSAL = {
    "ko": "제공된 문서에서는 해당 질문에 대한 근거를 확인할 수 없습니다.",
    "en": "The provided documents do not contain information to answer this question.",
}

# 이 문서 모음이 다루는 정보의 종류. 인덱스 구성이 바뀌면 함께 갱신한다.
CORPUS_SCOPE = (
    "유방암의 검진, 진단, 병기, 치료, 예후, 치료 부작용, 환자 관리에 관한 "
    "진료지침 및 환자 안내 자료"
)

JUDGE_PROMPT = """당신은 검색 기반 RAG 시스템의 라우팅 판정자입니다.
질문에 직접 답하지 않습니다. 검색 결과를 보고 다음 행동 하나를 고릅니다.

- answer  : 지금 근거를 답변 생성기로 보낸다
- rewrite : 검색어를 다시 써서 재검색한다
- refuse  : 이 문서 모음으로 처리할 수 없는 요청으로 확정하고 거절한다

## 전제

answer는 "완전한 정답이 검색되었다"는 뜻이 아닙니다.
"생성기가 이 근거로 완전한 답변, 부분 답변, 일반 정보, 또는 근거 부족 안내 중
하나를 만들 수 있다"는 뜻입니다.

refuse는 되돌릴 수 없습니다. 생성 단계에는 근거가 없으면 확인할 수 없다고
답하는 규칙이 있으므로, 잘못된 answer는 회복되지만 잘못된 refuse는 회복되지
않습니다.

당신은 검색된 조각만 볼 수 있고, 검색되지 않은 나머지 조각은 볼 수 없습니다.
따라서 "지금 조각에 없다"에서 "문서 모음에 없다"를 추론하지 마십시오.

## 이 문서 모음이 다루는 것
{corpus_scope}

## 입력
- 원 질문: {question}
- 현재 검색어: {query}
- 질문 언어: {lang}
- 재검색 횟수: {retry_count} / 최대 {max_retry}
- 검색된 조각:
{docs_block}

## 판단 순서

**1. 질문이 요구하는 것 (asked)**
원 질문 기준으로 핵심 요구와 조건을 한 문장으로 적습니다.
시점, 대상군, 비교 대상 같은 조건이 있으면 함께 적습니다.
현재 검색어가 재작성되었더라도 원 질문의 조건을 기준으로 삼습니다.

**2. 근거 수준 (evidence)**
- full          : 핵심 요구에 직접 답할 근거가 있다
- partial       : 전부는 아니지만 의미 있는 부분 답변이나 일반 정보를 줄 수 있다
- related_only  : 주제나 용어만 관련되고, 묻는 내용에 대한 근거는 없다
- none          : 실질적으로 관련된 근거가 없다

여러 조각에 나뉘어 있어도 합쳐서 답이 되면 full 또는 partial입니다.
rerank 점수는 관련도 신호일 뿐입니다. 점수를 근거 수준의 판단에 쓰지 마십시오.

**3. 요구하는 정보의 종류 (category)**
- in_scope      : 위 "다루는 것"에 속한다. 이번 검색이 못 찾았을 뿐일 수 있다
- out_of_scope  : 속하지 않는다. 검색어를 바꿔도 나오지 않는다
                  (비용, 기관 평가·순위, 개인 기록, 실시간 외부 정보 등)

**4. 결론 (decision)**

refuse — 아래 중 하나가 명확할 때만 고릅니다.
- category가 out_of_scope이다
- 문서 모음이 구조적으로 보유할 수 없는 정보만 요구하며, 일반 정보로도
  의미 있는 응답을 줄 수 없다

다음은 refuse의 근거가 되지 않습니다.
- 지금 조각에 답이 없다
- rerank 점수가 낮다
- 검색 결과가 질문과 동떨어져 있다
- 관련 용어만 나오고 직접적인 답은 없다
- 질문이 요구한 시점의 자료를 찾지 못했다
- 개인의 상황에 대한 확정적인 판단을 내릴 수 없다

answer — 아래 중 하나면 고릅니다.
- evidence가 full 또는 partial이다
- 개인의 치료 시작·중단·변경·연기·감량 등을 묻는 질문에서, 그 결정 자체에
  대한 답은 없더라도 해당 주제의 일반 원칙·지침·주의사항이 조각에 있다.
  생성기가 일반 정보를 제시하고 개인 결정은 의료진 상담으로 안내한다
- 재검색 횟수를 모두 썼고 refuse 조건이 성립하지 않는다

rewrite — 아래를 모두 만족하면 고릅니다.
- evidence가 related_only 또는 none이다
- category가 in_scope이다
- 재검색 횟수가 최대에 도달하지 않았다

즉 "지금 조각에는 없지만 문서 모음에 없다고 확정할 수도 없다"면
refuse가 아니라 rewrite입니다.

## 재작성 검색어 (rewrite일 때만)
- 원 질문의 의미를 유지합니다
- 시점, 기간, 대상군, 비교 조건을 삭제하거나 완화하지 않습니다
- 현재 검색어와 실질적으로 다른 표현을 씁니다
- 지침 문서에서 쓰일 법한 용어나 상위 개념으로 바꿉니다
- 비교 질문이면 두 대상이 모두 검색되도록 씁니다
- 질문의 언어를 유지합니다

## 출력
아래 JSON 하나만 출력합니다. 키 순서를 지키고, 다른 텍스트를 덧붙이지 않습니다.

{{"asked": "1의 내용",
  "evidence": "full | partial | related_only | none",
  "category": "in_scope | out_of_scope",
  "reason": "결론의 근거 두 문장 이내",
  "decision": "answer | rewrite | refuse",
  "rewrite_query": "rewrite일 때만 검색어, 아니면 null"}}"""


def detect_lang(text: str) -> str:
    """한글이 포함되어 있으면 ko, 아니면 en."""
    return "ko" if re.search(r"[가-힣]", text) else "en"


def _timed(name: str, fn):
    """노드 실행 시간을 측정해 (결과, latency 기록) 형태로 반환."""
    t0 = time.perf_counter()
    result = fn()
    return result, [{"node": name, "sec": round(time.perf_counter() - t0, 3)}]


def format_docs_for_judge(docs: list[RetrievedDoc], max_chars: int = 800) -> str:
    """판정자에게 넘길 조각 블록. 점수와 출처를 함께 제시한다."""
    if not docs:
        return "(검색된 조각 없음)"
    blocks = []
    for i, d in enumerate(docs, 1):
        text = d.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + " …"
        blocks.append(f"[{i}] score {d.score:.3f} | {d.citation()}\n{text}")
    return "\n\n".join(blocks)


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


def filter_node(state: AgentState) -> dict:
    """score 필터. 계산은 분기 함수에서 하고, 여기서는 기록만 남긴다."""
    return {
        "route_history": [f"filter(score={state.get('max_score', 0.0):.3f}, "
                          f"retry={state.get('retry_count', 0)})"],
        "node_latencies": [{"node": "filter", "sec": 0.0}],
    }


def judge_node(state: AgentState) -> dict:
    """LLM 판정자. score 0.5 미만 문항에서 조각 내용을 읽고 다음 행동을 결정한다."""
    docs = state.get("docs", [])
    retry = state.get("retry_count", 0)

    def _run():
        msg = JUDGE_PROMPT.format(
            corpus_scope=CORPUS_SCOPE,
            question=state["question"],
            query=state.get("query") or state["question"],
            lang=state.get("lang", "ko"),
            retry_count=retry,
            max_retry=config.MAX_RETRY,
            docs_block=format_docs_for_judge(docs),
        )
        return get_judge_llm().invoke([("human", msg)]).content

    raw, lat = _timed("judge", _run)
    parsed = _parse_judge(raw)

    # 재시도를 모두 소진했는데 rewrite가 나오면 answer로 대체한다.
    if parsed["decision"] == "rewrite" and retry >= config.MAX_RETRY:
        parsed["reason"] = f"[retry 소진으로 answer 대체] {parsed['reason']}"
        parsed["decision"] = "answer"
        parsed["rewrite_query"] = None

    # rewrite인데 검색어를 만들지 못했으면 rewrite 노드가 생성하도록 둔다
    call = {"retry": retry, **parsed}
    return {
        "judge_calls": [call],
        "route_history": [f"judge -> {parsed['decision']} "
                          f"(evidence={parsed['evidence']}, category={parsed['category']})"],
        "node_latencies": lat,
    }


REWRITE_PROMPT = """다음 질문으로 문서를 검색했으나 충분한 근거를 찾지 못했습니다.
같은 의도를 유지하면서, 문서에서 검색되기 쉬운 표현으로 질문을 다시 작성하세요.

- 문서에서 쓰일 법한 용어나 상위 개념을 사용하세요.
- 의미를 바꾸거나 새로운 조건을 추가하지 마세요.
- 시점, 대상군, 비교 조건을 삭제하지 마세요.
- 다시 작성한 질문 한 문장만 출력하세요.

원래 질문: {question}
현재 질문: {query}"""


def rewrite_node(state: AgentState) -> dict:
    """검색어 재작성. 판정자가 만든 검색어를 우선 사용하고, 없을 때만 LLM을 호출한다."""
    calls = state.get("judge_calls") or []
    from_judge = calls[-1].get("rewrite_query") if calls else None

    if from_judge:
        return {
            "query": from_judge,
            "retry_count": state.get("retry_count", 0) + 1,
            "route_history": [f"rewrite(judge) -> {from_judge}"],
            "node_latencies": [{"node": "rewrite", "sec": 0.0}],
        }

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
        "route_history": [f"rewrite(llm) -> {new_query}"],
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
# 판정자 LLM
# ─────────────────────────────────────────────────────────────

_judge_llm = None


def get_judge_llm():
    """판정용 LLM. 생성 모델과 분리해 두어 필요 시 다른 모델로 교체할 수 있다."""
    global _judge_llm
    if _judge_llm is None:
        from langchain_openai import ChatOpenAI

        _judge_llm = ChatOpenAI(
            model=config.JUDGE_MODEL,
            temperature=config.JUDGE_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
        )
    return _judge_llm


VALID_DECISIONS = {"answer", "rewrite", "refuse"}
VALID_EVIDENCE = {"full", "partial", "related_only", "none"}
VALID_CATEGORY = {"in_scope", "out_of_scope"}


def _parse_judge(raw: str) -> dict:
    """판정 JSON을 파싱한다. 실패 시 answer로 처리해 오거절을 만들지 않는다."""
    fallback = {
        "asked": "", "evidence": "unknown", "category": "unknown",
        "reason": "", "decision": "answer", "rewrite_query": None,
    }

    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for d in VALID_DECISIONS:
            if re.search(rf'"decision"\s*:\s*"{d}"', text):
                fallback["decision"] = d
                fallback["reason"] = "(JSON 파싱 실패, 문자열에서 복구)"
                return fallback
        fallback["reason"] = f"[파싱 실패] {text[:80]}"
        return fallback

    decision = str(data.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        fallback["reason"] = f"[알 수 없는 decision={decision!r}] {data.get('reason', '')}"
        return fallback

    evidence = str(data.get("evidence", "")).strip().lower()
    category = str(data.get("category", "")).strip().lower()
    rq = data.get("rewrite_query")

    return {
        "asked": str(data.get("asked", "")).strip(),
        "evidence": evidence if evidence in VALID_EVIDENCE else "unknown",
        "category": category if category in VALID_CATEGORY else "unknown",
        "reason": str(data.get("reason", "")).strip() or "(사유 없음)",
        "decision": decision,
        "rewrite_query": str(rq).strip() if isinstance(rq, str) and rq.strip() else None,
    }


# ─────────────────────────────────────────────────────────────
# 분기 (조건부 엣지)
# ─────────────────────────────────────────────────────────────

def route_after_filter(state: AgentState) -> str:
    """rerank score를 필터로만 사용한다.

    - score >= 0.5 : 근거가 충분하다고 보고 즉시 generate (LLM 호출 없음)
    - score <  0.5 : 판정자에게 넘긴다
    """
    if state.get("max_score", 0.0) >= config.RELEVANCE_THRESHOLD:
        return "generate"
    return "judge"


def route_after_judge(state: AgentState) -> str:
    """판정자의 결정을 그대로 따른다."""
    calls = state.get("judge_calls") or []
    decision = calls[-1]["decision"] if calls else "answer"
    return {"answer": "generate", "rewrite": "rewrite", "refuse": "refuse"}[decision]
