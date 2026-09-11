"""프로젝트 전역 설정.

모든 값은 3~5주차 실험에서 확정된 것이다. 임의로 바꾸지 않는다.
값을 바꿀 때는 출처 노트북/셀과 대조하고 decision_log에 기록한다.

출처
- Week 4 확정: 청킹 G2 (ko 540/80, en 620/90), 클렌징 미적용, PyMuPDFLoader
- Week 5-0: 참고문헌·색인 페이지 제거 인덱스 (2,240 chunk)
- Week 5-2 cell 2: TOP_K=5, FETCH_K=20, RRF_K=60
- Week 5-0 cell 17: 인덱싱 시 e5 prefix 미사용 → 질의에도 붙이지 않음
"""
from pathlib import Path
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv

load_dotenv()

# --- 경로 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

INDEX_DIR = VECTOR_STORE_DIR / "week5pre_P1_ref_removed"
COLLECTION_NAME = "breast_rag_week5pre_P1_ref_removed"

GOLDEN_SET = EVAL_DIR / "golden_set_v3.csv"

# --- 모델 (week5_2 cell 2) ---
EMBED_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GEN_MODEL = "gpt-4o-mini"
GEN_TEMPERATURE = 0.0

# 라우팅 판정용. 생성 모델과 분리해 두어 교체할 수 있게 한다.
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPERATURE = 0.0

# --- Retrieval: Week 5 최종 R4 (week5_2 cell 2, cell 6, cell 8) ---
TOP_K = 5        # 최종 context 수
FETCH_K = 20     # dense/BM25 각각의 1차 후보 수이자 rerank 입력 수
RRF_K = 60       # Reciprocal Rank Fusion 상수

# --- Agentic RAG 라우팅 (Week 6) ---
# v1은 rerank score만으로 3분기(0.5 / 0.1)했으나, 0.5 미만 구간에서
# "관련도"와 "답변 가능성"을 구분하지 못했다.
# v3에서는 score를 필터로만 쓰고(0.5 이상 즉시 통과), 그 미만은 LLM 판정자에게 넘긴다.
# 0.5는 bge-reranker의 시그모이드 중립점이다.
RELEVANCE_THRESHOLD = 0.5
MAX_RETRY = 2

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
