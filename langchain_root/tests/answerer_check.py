# tests/answerer_check.py
"""
목적:
- retriever -> PromptBuilder -> SolarClient.generate 흐름이 실제로 동작하는지 확인
- 같은 컨텍스트로 solar-pro / solar-mini 두 모델 결과를 나란히 출력
사전조건:
- 색인 완료(Chroma에 벡터 있음): python -m tests.indexer_check
- SOLAR_API_KEY 설정
"""

from src.utils.config import AppConfig
from src.llm.solar import SolarClient
from src.retriever.search import Retriever
from src.llm.prompt import PromptBuilder, PromptOptions

def main():
    cfg = AppConfig()

    # 1) 검색: 질문 -> Top-k 근거
    solar = SolarClient(api_key=cfg.solar_api_key)
    retriever = Retriever(
        chroma_dir=cfg.chroma_dir,
        solar_client=solar,
        collection_name="ai_news_rag",
        top_k=5,
        use_mmr=True,
        mmr_lambda=0.3,
    )
    question = "최근 OpenAI 관련 중요한 발표를 요약해줘"
    ret = retriever.search(question)

    # 2) 프롬프트 만들기
    builder = PromptBuilder(PromptOptions(language="ko", style="bullets"))
    msgs = builder.build_messages(question, ret["sources"])

    system = msgs[0]["content"]
    user   = msgs[1]["content"]

    # 3) 모델 두 개로 생성
    for model in ["solar-pro", "solar-mini"]:
        try:
            ans = solar.generate(system_prompt=system, user_prompt=user, model=model, max_tokens=300)
            print(f"\n=== [{model}] ===")
            print(ans)
        except Exception as e:
            print(f"\n=== [{model}] ERROR ===")
            print(e)

    # 4) 근거 URL 요약 출력
    print("\n[SOURCES]")
    for i, s in enumerate(ret["sources"], 1):
        print(f"{i}. {s['title']} - {s['url']} (score={s.get('score')})")

if __name__ == "__main__":
    main()