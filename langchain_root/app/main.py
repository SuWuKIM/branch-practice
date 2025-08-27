# app/main.py

from src.utils.config import AppConfig
from src.sql.db import SqlStore
from src.crawler.rss_crawler import fetch_rss_docs
from src.llm.solar import SolarClient
from src.vector_store.indexer import Indexer
from src.qa.answerer import Answerer

class MainApp:
    def __init__(self):
        self.cfg = AppConfig()
        print("[INIT] 환경 로드 완료")
        print(f" - APP_ENV       : {self.cfg.env}")
        print(f" - CHROMA_DIR    : {self.cfg.chroma_dir}")
        print(f" - SQLITE_PATH   : {self.cfg.sqlite_path}")
        print(f" - RSS SOURCES   : {len(self.cfg.rss_list)}개 등록")

    # 1) 수집: RSS → 본문 추출 → SQLite 저장
    def run_ingest(self):
        """
        1) RSS에서 글 목록을 읽고
        2) 각 글의 본문을 추출한 뒤
        3) SQLite에 '중복 없이' 저장합니다.
        """
        store = SqlStore(self.cfg.sqlite_path)              # DB 연결(없으면 생성)
        docs = fetch_rss_docs(self.cfg.rss_list, per_feed_limit=50)  # RSS 2개 x 최대 20개
        inserted = 0
        for d in docs:
            doc_id = store.upsert_document(d)               # 중복이면 기존 id 반환
            if doc_id:
                inserted += 1
        print(f"[INGEST] new docs inserted: {inserted} / fetched: {len(docs)}")

    # 2) 인덱싱: 청킹/임베딩 → Chroma 업서트
    def run_index(self):
        """
        SQLite에서 문서를 불러와:
          - 청킹(조각내기)
          - 임베딩(숫자 벡터로 변환, embedding-passage)
          - Chroma(VectorDB)에 업서트(저장/갱신)
        를 수행합니다.
        """
        store = SqlStore(self.cfg.sqlite_path)
        solar = SolarClient(api_key=self.cfg.solar_api_key)

        indexer = Indexer(
            store=store,
            chroma_dir=self.cfg.chroma_dir,
            solar_client=solar,
            # 필요하면 configs로 빼서 조정 가능
            max_chars=1200,
            overlap=120,
            min_chunk_chars=200,
            batch_size=16,
        )

        result = indexer.index_recent(limit_docs=100)  # 최근 N개만 색인
        print("[INDEX RESULT]", result)

    # 3) 검색+생성: Top-k 검색 → LLM 답변 생성(+출처)
        # 3) 검색+생성: Top-k 검색 → LLM 답변 생성(+출처)
    def run_qa(self, question: str):
        """
        질문 한 번으로:
          - Retriever로 Top-k 근거 검색
          - PromptBuilder로 프롬프트 조립
          - Solar LLM(mini/pro)로 생성
        결과를 콘솔에 보기 좋게 출력합니다.
        """
        print(f"[QA    ] Q: {question}")

        # 오케스트레이터 준비(Top-k/MMR은 필요 시 조정)
        answerer = Answerer(
            cfg=self.cfg,
            top_k=5,
            use_mmr=True,
            mmr_lambda=0.3,
        )

        # 같은 컨텍스트로 두 모델 결과를 나란히 비교
        results = answerer.answer_multi(
            question=question,
            models=["solar-pro", "solar-mini"],
            max_tokens=320,
            extra_instructions=None,  # 필요하면 "불릿 3개 이내" 등 추가
        )

        # 콘솔 출력(모델별 답변 + 출처)
        for res in results:
            print(f"\n=== [{res['model']}] ===")
            print(res["answer"].strip())

            # Sources 요약
            if res.get("sources"):
                print("\n[SOURCES]")
                for i, s in enumerate(res["sources"], 1):
                    title = s.get("title", "(제목 없음)")
                    url = s.get("url", "")
                    score = s.get("score")
                    score_str = f" (score={round(float(score),4)})" if score is not None else ""
                    print(f"{i}. {title} - {url}{score_str}")
            else:
                print("\n[SOURCES] (없음)")

        # 필요 시 상위 레벨에서 활용할 수 있도록 반환
        return results
    
def main():
    app = MainApp()
    # 워킹 스켈레톤: 전체 흐름 자리만 호출
    app.run_ingest()  # 최신 뉴스 기사 수집
    app.run_index()   # 수집한 기사를 청킹/임베딩해 벡터DB에 색인
    app.run_qa("openai에 관한 뉴스를 요약해줘.")  # 검색+생성

if __name__ == "__main__":
    main()