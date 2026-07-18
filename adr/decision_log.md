# 결정 로그 (Decision Log)

## Week 3
- PDF 로더 = PyMuPDFLoader (baseline) — 빠르고 안정적, pymupdf4llm는 미검증이라 4주차 실험으로 미룸
- 생성/채점 모델 분리 (생성 gpt-4o-mini / 채점 Claude) — self-preference 편향 회피
- 임베딩 = multilingual-e5-base (CPU) 
- 의존성 = Poetry — 버전 충돌로 재현성 깨져서 버전 고정

## Week 4
- 청킹 = 언어별 차등 크기 (한국어 540/80, 영어 620/90) — 한국어가 영어보다 2.3배 밀도 높아, 한국어는 잘게·영어는 크게 잘라야 두 언어 검색이 함께 최적
- 노이즈 제거(클렌징) 미적용 — 켰더니 검색·생성 다 하락. 반복 머리말이 노이즈가 아니라 문서 제목·섹션명 등 검색 앵커였음
- PDF 로더 = PyMuPDFLoader 유지 (pymupdf4llm 기각) — 표 markdown이 자연어 임베딩엔 노이즈이고 chunk를 파편화, 표 많은 한국어에서 가장 악화
- 평가셋 = 이중언어 golden_set (한국어 20 + 영어 10) — 영어 청킹 효과를 검증하려면 영어 질문 필요, NCCN 문서 근거로 출제
