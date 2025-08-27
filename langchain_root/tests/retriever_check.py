# tests/retriever_check.py
"""
목적:
- Retriever(search.py)가 질문 → Top-k 근거 청크를 잘 찾는지 확인.
- contexts(생성기에 줄 재료)와 sources(출처 리스트)를 간단히 출력.

사전조건:
- 색인(indexer)이 한 번 이상 돌아 Chroma에 벡터가 저장돼 있어야 함.
  (예: python -m tests.indexer_check 로 먼저 색인을 확인)
"""

from src.utils.config import AppConfig
from src.llm.solar import SolarClient
from src.retriever.search import Retriever

def main():
    cfg = AppConfig()

    # 1) Solar 클라이언트 (질문 임베딩에 사용)
    solar = SolarClient(api_key=cfg.solar_api_key)

    # 2) 리트리버 생성 (Top-k와 MMR 파라미터는 필요시 조정)
    retriever = Retriever(
        chroma_dir=cfg.chroma_dir,
        solar_client=solar,
        collection_name="ai_news_rag",
        top_k=5,# 
        use_mmr=True,
        mmr_lambda=0.3,
    )

    # 3) 테스트 질문 (원하는 질문으로 바꿔도 됨)
    question = "최근 OpenAI 관련 중요한 발표를 요약해줘"

    result = retriever.search(question)

    # 4) 결과 요약 출력
    print("[RAW]", result["raw"])  # 몇 개 가져왔는지 숫자 확인
    print("\n[SOURCES]")
    for i, s in enumerate(result["sources"], 1):
        print(f"{i}. {s['title']} ({s['source']}) - {s['url']}  score={s['score']}")

    # 5) 컨텍스트 일부 확인 (너무 길면 앞부분만)
    ctx_preview = result["contexts"][:600].replace("\n", " ") + ("..." if len(result["contexts"]) > 600 else "")
    print("\n[CONTEXT PREVIEW]")
    print(ctx_preview)

if __name__ == "__main__":
    main()