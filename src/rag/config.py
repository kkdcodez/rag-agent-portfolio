"""프로젝트 전역 설정.

Week 4에서 확정한 청킹(G2_ko4_en3)과 Week 5에서 확정한 retrieval(R4: Hybrid + Rerank)을
코드 한 곳에서 관리한다. 실험 노트북과 서비스 코드가 같은 값을 참조하도록 하기 위함.
"""
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# --- 경로 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Week 5 전처리(5-0, 참고문헌 제거)까지 적용된 최종 인덱스
INDEX_DIR = VECTOR_STORE_DIR / "week5pre_P1_ref_removed"
COLLECTION_NAME = "breast_rag_week5pre_P1_ref_removed"

GOLDEN_SET = EVAL_DIR / "golden_set_v2.csv"

# --- 모델 ---
EMBED_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "          # e5 계열은 질의에 prefix 필요
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GEN_MODEL = "gpt-4o-mini"
GEN_TEMPERATURE = 0.0

# --- Retrieval (Week 5 최종 R4) ---
RRF_K = 60                # Reciprocal Rank Fusion 상수
CANDIDATE_EACH = 30       # dense/BM25 각각의 1차 후보 수
RERANK_CANDIDATES = 20    # reranker 입력 수
TOP_K = 5                 # 최종 context 수

# --- Agentic RAG 라우팅 임계값 ---
# golden_set 라벨 검증(week6_0)에서 측정한 rerank score 분포에 근거:
#   근거 명확 0.92~0.99 / 부분 관련 0.36~0.43 / 근거 없음 0.03~0.05
RELEVANCE_THRESHOLD = 0.5   # 이상이면 충분한 근거로 판단
NO_EVIDENCE_THRESHOLD = 0.1 # 미만이면 재검색 없이 즉시 거절
MAX_RETRY = 2

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
