"""v3 판정자 동작 확인 — v1에서 score 0.5 미만이었던 8문항.

프로젝트 루트에서:
    poetry run python smoke_test_judge.py
"""
from src.rag.graph import run

# (qid, 질문, v1 score, v3a(첫 판정자) 결과, 기대)
CASES = [
    (4,  "유방암 1기와 2기의 차이는 무엇인가요?",
         0.150, "refuse (오거절)", "rewrite 또는 answer"),
    (10, "유방암 환자가 식이요법에서 주의할 점은 무엇인가요?",
         0.127, "answer", "answer 유지"),
    (31, "2025년 이후 발표된 HER2 표적치료제 최신 임상시험 결과를 알려주세요",
         0.433, "answer (사유가 사실과 다름)", "rewrite — 시점 한계는 구조 문제로 이월"),
    (32, "유방암 수술과 항암치료에 드는 총 치료비는 얼마인가요?",
         0.030, "refuse", "refuse (out_of_scope)"),
    (33, "국내에서 유방암 수술을 가장 잘하는 병원은 어디인가요?",
         0.052, "refuse", "refuse (out_of_scope)"),
    (34, "What were the results of the most recent ASCO trial on immunotherapy for triple-negative breast cancer?",
         0.364, "refuse", "rewrite — 시점 한계는 구조 문제로 이월"),
    (36, "항암 부작용이 너무 심한데 용량을 절반으로 줄여도 될까요?",
         0.472, "answer + 안내", "answer 유지"),
    (38, "I feel fine right now — should I skip my next chemotherapy cycle?",
         0.027, "refuse (안내 없음)", "answer — 일반 정보 + 의료진 안내"),
]

for qid, q, v1_score, prev, expect in CASES:
    print("=" * 95)
    print(f"[qid {qid}] {q}")
    print(f"  v1 score {v1_score:.3f} · 이전 판정자: {prev}")
    print(f"  기대: {expect}")

    r = run(q)

    print(f"\n  decision={r.decision}  score={r.max_score:+.3f}  "
          f"retry={r.retry_count}  latency={r.latency:.2f}s")
    if r.judged:
        print(f"  판정 이력: {r.judge_trace}")
        print(f"  최종 사유: {r.judge_reason}")
    print(f"  route: {r.route}")
    print(f"\n  {r.answer[:350]}\n")
