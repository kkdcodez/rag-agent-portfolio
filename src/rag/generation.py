"""생성 — context 기반 답변.

Baseline과 Agentic RAG가 같은 프롬프트를 쓰도록 한 곳에서 관리한다.
(변수 격리: 6주차에 바뀌는 것은 routing 구조뿐이어야 한다.)
"""
from __future__ import annotations

from . import config
from .retrieval import RetrievedDoc, format_context

SYSTEM_PROMPT = """당신은 유방암 관련 문서를 근거로 답변하는 정보 검색 어시스턴트입니다.

규칙:
1. 제공된 context에 있는 내용만 근거로 답변합니다. context에 없는 내용을 추가하지 않습니다.
2. 답변에 사용한 근거의 출처를 [1], [2] 형식으로 표기합니다.
3. context에 답변의 근거가 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답하고, 추측하지 않습니다.
4. 개인의 진단·치료·투약 결정을 요구하는 질문에는 직접 결정하지 않습니다.
   문서에 있는 일반적인 정보는 제시하되, 실제 결정은 담당 의료진과 상담하도록 안내합니다.
5. 질문에 잘못된 전제가 포함되어 있으면 먼저 바로잡은 뒤 답변합니다.
6. 답변 언어는 반드시 질문의 언어를 따릅니다. context가 한국어 문서여도 질문이 영어면 영어로 답변합니다."""

USER_TEMPLATE = """다음 context를 근거로 질문에 답하세요.

<context>
{context}
</context>

질문: {question}

(답변은 위 질문과 같은 언어로 작성하세요.)"""

REFUSAL_MESSAGE = (
    "제공된 문서에서는 해당 질문에 대한 근거를 확인할 수 없습니다. "
    "추가 문서가 제공되면 더 정확히 답변할 수 있습니다."
)

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model=config.GEN_MODEL,
            temperature=config.GEN_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
        )
    return _llm


def generate(question: str, docs: list[RetrievedDoc]) -> str:
    """context 기반 답변 생성. docs가 비면 거절 문구를 반환한다."""
    if not docs:
        return REFUSAL_MESSAGE

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE.format(context=format_context(docs), question=question)),
    ]
    return get_llm().invoke(messages).content
