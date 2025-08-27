# tests/indexer_check.py
"""
목적:
- SQLite에 저장된 문서들을 색인기(Indexer)로 청킹→임베딩→Chroma 업서트가 되는지 스모크 테스트.
- 완료 후 Chroma 컬렉션의 총 벡터 개수(count)도 확인.

사전조건:
- RSS 수집이 한 번이라도 돌았어야 함 (documents 테이블에 최소 1개 이상).
  예: python -m app.main 실행 후 [INGEST] 로그로 확인.
"""

# TODO: 개선 사항: 배치 크기, 청크 크기, 오버랩 비율 등을 파라미터화하여 실험해보기

from src.utils.config import AppConfig
from src.sql.db import SqlStore
from src.llm.solar import SolarClient
from src.vector_store.indexer import Indexer
import chromadb
from chromadb.config import Settings

def main():
    cfg = AppConfig()

    # 1) SQLite에 문서가 있는지 간단 확인
    store = SqlStore(cfg.sqlite_path)
    docs = store.fetch_all(limit=3)
    print(f"[CHECK] documents preview: {len(docs)}개 (색인 대상 미리보기)")

    if not docs:
        print("[WARN] documents가 비어있습니다. 먼저 수집(ingest)을 실행하세요: python -m app.main")
        return

    # 2) Solar 임베딩 클라이언트 (색인은 passage 임베딩 사용)
    solar = SolarClient(api_key=cfg.solar_api_key)

    # 3) 인덱서 생성 & 실행 (기본 파라미터: max_chars=1200, overlap=120)
    indexer = Indexer(
        store=store,
        chroma_dir=cfg.chroma_dir,
        solar_client=solar,
        max_chars=1200,
        overlap=120,
        min_chunk_chars=200,
        batch_size=16,
    )
    result = indexer.index_recent(limit_docs=50)
    print("[INDEX RESULT]", result)

    # 4) Chroma 컬렉션 카운트 확인 (총 벡터 수)
    client = chromadb.PersistentClient(
        path=cfg.chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection("ai_news_rag")
    cnt = col.count()
    print(f"[CHROMA] 현재 컬렉션 벡터 개수: {cnt}")

if __name__ == "__main__":
    main()

