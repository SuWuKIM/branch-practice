# src/vector_store/indexer.py
"""
- DB에 들어있는 문서(raw_text)를 불러와서 '청크'로 쪼갬.
- 각 청크를 Upstage 임베딩(embedding-passage)으로 벡터화함.
- 벡터 + 메타데이터를 Chroma(VectorDB)에 저장하는 파일!
"""

import os
import math
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

# 간단 길이 기반 청커(문단 경계 우선, 부족하면 길이로 잘라 오버랩 포함)
# 1200자면 대략 200 영단어라서 적당한 값으로 판단함.
def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    if not text:
        return []
    # 1) 문단 기준 1차 분할
    paras = [p.strip() for p in text.split("\n") if p.strip()] # 개행 기준으로 문단 단위로 묶기
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf}\n{p}"
        else:
            chunks.append(buf)
            # 오버랩: 끝부분 일정 길이만 남기고 새로 시작
            keep = buf[-overlap:] if overlap > 0 and len(buf) > overlap else ""
            buf = (keep + "\n" + p).strip()
    if buf:
        chunks.append(buf)

    # 2) 여전히 긴 덩어리는 강제로 쪼개기
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            step = max_chars - overlap
            start = 0
            while start < len(c):
                end = min(start + max_chars, len(c))
                piece = c[start:end]
                final_chunks.append(piece)
                start += step
    return final_chunks

class ChromaStore:
    """
    Chroma 컬렉션 래퍼.
    - persist_dir: data/chroma
    - collection: "ai_news_rag"
    """
    def __init__(self, persist_dir: str, collection_name: str = "ai_news_rag"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(collection_name)

    def upsert_chunks(
        self,
        doc_id: int,
        url: str,
        title: str,
        source: str,
        date_published: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> int:
        """
        청크+임베딩을 collection에 업서트.
        id 충돌을 피하려고 'doc_<id>_chunk_<i>' 규칙을 사용.
        """
        if not chunks:
            return 0
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "doc_id": doc_id,
            "url": url,
            "title": title,
            "source": source,
            "date_published": date_published,
            "chunk_index": i,
            "length": len(chunks[i]),
        } for i in range(len(chunks))]
        self.col.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

class Indexer:
    """
    색인 파이프라인:
    - DB에서 최근 문서 N개 읽기
    - 청킹
    - Upstage 임베딩(embedding-passage)
    - Chroma 업서트
    """
    def __init__(
        self,
        store,                  # SqlStore 인스턴스
        chroma_dir: str,        # data/chroma
        solar_client,           # SolarClient 인스턴스 (임베딩용)
        max_chars: int = 1200,
        overlap: int = 120,
        min_chunk_chars: int = 200,
        batch_size: int = 32,
    ):
        self.store = store
        self.vdb = ChromaStore(chroma_dir)
        self.solar = solar_client
        self.max_chars = max_chars
        self.overlap = overlap
        self.min_chunk_chars = min_chunk_chars
        self.batch_size = batch_size

    def _chunk_doc(self, text: str) -> List[str]:
        chunks = simple_chunk(text, self.max_chars, self.overlap)
        # 너무 짧은 청크 제거
        return [c for c in chunks if len(c) >= self.min_chunk_chars]

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        # 문서 색인에는 passage 임베딩 권장
        return self.solar.embed_passage(batch_texts)

    def index_recent(self, limit_docs: int = 200) -> Dict[str, Any]:
        docs = self.store.fetch_all(limit=limit_docs)
        total_chunks = 0
        total_embedded = 0
        total_upserted = 0

        for d in docs:
            doc_id = d["id"]
            url = d.get("url", "")
            title = d.get("title", "")
            source = d.get("source", "")
            date_published = d.get("date_published", "")
            text = d.get("raw_text", "")

            chunks = self._chunk_doc(text)
            if not chunks:
                continue

            # 배치 임베딩
            embeddings: List[List[float]] = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                embs = self._embed_batch(batch)
                embeddings.extend(embs)

            # 안전 체크
            if len(embeddings) != len(chunks):
                raise RuntimeError("임베딩 개수와 청크 개수가 일치하지 않습니다.")

            upserted = self.vdb.upsert_chunks(
                doc_id=doc_id,
                url=url,
                title=title,
                source=source,
                date_published=date_published,
                chunks=chunks,
                embeddings=embeddings,
            )

            total_chunks += len(chunks)
            total_embedded += len(embeddings)
            total_upserted += upserted

        return {
            "docs_processed": len(docs),
            "chunks_total": total_chunks,
            "embedded_total": total_embedded,
            "upserted_total": total_upserted,
        }