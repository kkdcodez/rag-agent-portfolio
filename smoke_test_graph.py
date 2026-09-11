"""Agentic RAG 동작 확인 — 세 갈래가 의도대로 갈라지는지 본다.

프로젝트 루트에서:
    poetry run python smoke_test_graph.py
"""
from src.rag.graph import run

CASES = [
    ("HER2 양성 유방암이란 무엇인가요?",              "generate 예상 (score 높음)"),
    ("유방암 수술과 항암치료에 드는 총 치료비는 얼마인가요?", "refuse 예상 (score 0.03)"),
    ("유방암 1기와 2기의 차이는 무엇인가요?",           "rewrite 예상 (score 0.15)"),
    ("I feel fine right now — should I skip my next chemotherapy cycle?",
                                                    "refuse 예상 + 영어 문구"),
]

for q, expect in CASES:
    print("=" * 90)
    print(f"Q: {q}")
    print(f"   ({expect})")
    r = run(q)
    print(f"\n  decision={r.decision}  score={r.max_score:+.3f}  "
          f"retry={r.retry_count}  latency={r.latency:.2f}s")
    print(f"  route: {r.route}")
    print(f"\n  {r.answer[:300]}\n")
