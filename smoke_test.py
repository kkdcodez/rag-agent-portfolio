"""src/rag 모듈 동작 확인.

프로젝트 루트에서 실행:
    poetry run python smoke_test.py
"""
from src.rag import config
from src.rag.pipeline import answer
from src.rag.retrieval import retrieve

print("인덱스:", config.INDEX_DIR.name, "/", config.COLLECTION_NAME)
print("존재:", config.INDEX_DIR.exists())
print("골든셋:", config.GOLDEN_SET.exists(), config.GOLDEN_SET)
print()

Q = "HER2 양성 유방암이란 무엇인가요?"

docs = retrieve(Q, top_k=3)
print(f"[검색] {len(docs)}건")
for d in docs:
    print(f"  {d.score:+.3f} | {d.citation()} | {d.text[:60]}...")
print()

res = answer(Q)
print(f"[생성] latency={res.latency:.2f}s  max_score={res.max_score:+.3f}")
print(res.answer[:400])
