# src/retriever/search.py
"""
- 사용자의 질문을 임베딩(embedding-query)으로 바꾼 다음,
- Chroma(VectorDB)에서 가장 관련 있는 청크 Top-k를 찾아옵니다.
- (옵션) MMR로 비슷비슷한 청크를 덜어내고 다양하게 뽑습니다.
- 생성기에 넘길 '컨텍스트 문자열'과 '출처 메타데이터'를 함께 돌려줍니다.
"""

from typing import List, Dict, Any, Tuple
import math
import chromadb
from chromadb.config import Settings
from src.llm.solar import SolarClient

# 코사인 유사도 계산(벡터와 벡터 사이의 각도를 보고 얼마나 비슷한지 판단), MMR에서 두 벡터의 비슷함/겹침 계산할 때 사용
def _cosine(a: List[float], b: List[float]) -> float:
    # Upstage 임베딩은 정규화되어 있어 dot==cos지만, 안전하게 코사인 계산
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-12)

def _mmr_select(
    query_vec: List[float],
    cand_vecs: List[List[float]],
    cand_idxs: List[int],
    k: int,
    lambda_coef: float = 0.3,
) -> List[int]:
    """
    MMR(최대 한계 다양성) 간단 구현:
    - 관련성(relevance): query와의 유사도
    - 다양성(diversity): 이미 선택된 것들과의 '차이'
    점수 = λ * relevance - (1-λ) * max(similarity to selected)
    """
    selected: List[int] = []
    remaining = set(cand_idxs)
    rel_scores = {i: _cosine(query_vec, cand_vecs[i]) for i in cand_idxs}

    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in list(remaining):
            if not selected:
                score = rel_scores[i]
            else:
                max_sim = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
                score = lambda_coef * rel_scores[i] - (1 - lambda_coef) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected

class Retriever:
    def __init__(
        self,
        chroma_dir: str,
        solar_client: SolarClient,
        collection_name: str = "ai_news_rag",
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.3,
    ):
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

        self.solar = solar_client
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(collection_name)

    def search(self, question: str) -> Dict[str, Any]:
        """
        입력: 사용자 질문(문자열)
        출력: {
          "contexts": 컨텍스트 문자열(생성기용),
          "sources":  [{title,url,source,date_published,chunk_index,length,score}, ...],
          "raw":      Chroma 원본 결과(디버깅용 일부)
        }
        """
        # 1) 질문 임베딩 (query 전용)
        q_emb = self.solar.embed_query([question])[0]

        # 2) Chroma에서 후보 Top-(top_k*3) 먼저 가져오기 (MMR 위해 여유있게)
        n_initial = max(self.top_k * 3, self.top_k)
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=n_initial,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        embs  = res["embeddings"][0] if "embeddings" in res and res["embeddings"] else []

        if not docs:
            return {"contexts": "", "sources": [], "raw": {}}

        # 3) MMR 선택(옵션)
        idxs = list(range(len(docs)))
        if self.use_mmr and len(docs) > self.top_k:
            sel = _mmr_select(q_emb, embs, idxs, k=self.top_k, lambda_coef=self.mmr_lambda)
        else:
            sel = idxs[: self.top_k]

        # 4) 선택 결과 정리 (컨텍스트/소스)
        picked = [(docs[i], metas[i], 1 - dists[i]) for i in sel]  # 유사도 점수: 1 - distance
        # 점수로 정렬(높은 순)
        picked.sort(key=lambda x: x[2], reverse=True)

        # 컨텍스트 문자열 구성(짧게)
        context_blocks = []
        sources = []
        for text, meta, score in picked:
            title = meta.get("title", "")
            url = meta.get("url", "")
            source = meta.get("source", "")
            date = meta.get("date_published", "")
            idx = meta.get("chunk_index", -1)
            context_blocks.append(
                f"[{title}]({url})\n{(text[:800] + '...') if len(text) > 800 else text}\n---"
            )
            sources.append({
                "title": title, "url": url, "source": source,
                "date_published": date, "chunk_index": idx,
                "length": meta.get("length", len(text)),
                "score": round(float(score), 4)
            })

        contexts = "\n".join(context_blocks)
        return {
            "contexts": contexts,
            "sources": sources,
            "raw": {
                "n_initial": n_initial,
                "returned": len(docs),
                "selected": len(picked),
            }
        }