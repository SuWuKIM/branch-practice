# src/llm/solar.py
"""
- Solar API를 부르는 아주 얇은 어댑터(내 명령어를 솔라가 이해할 수 있는 형태로 변환해서 전달해줌)입니다.
- embed(texts): 문장/청크를 '숫자 벡터'로 바꿔서 벡터DB(Chroma)에 저장하거나 검색에 씁니다.
- generate(system_prompt, user_prompt): 리트리브된 근거로 최종 답변을 만듭니다.
"""

"""
페이로드는 API에 요청을 보낼 때 '실제로 전달하는 데이터 묶음', 즉 '화물'이나 '소포'라고 생각하면 돼.
- **URL:** 받는 사람 주소
- **Header:** 보내는 사람 정보, 내용물 종류
- **Payload:** **소포 상자 안에 담긴 실제 물건들**
"""
# src/llm/solar.py
import requests
from typing import List

class SolarClient:
    def __init__(self, api_key: str, base_url: str = "https://api.upstage.ai/v1"):
        if not api_key:
            raise ValueError("SOLAR_API_KEY가 비어있습니다. .env에 설정하세요.")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    # --- 임베딩 (색인/검색용) ---
    # 기본값을 업스테이지 권장 별칭으로 교체
    def embed(self, texts: List[str], model: str = "embedding-passage", timeout: int = 60) -> List[List[float]]:
        """
        입력: texts = ["문장1", "문장2", ...]
        출력: 각 문장을 고정 길이의 숫자 리스트(벡터)로 변환
        권장: 문서 색인용은 'embedding-passage', 질의용은 'embedding-query'
        """
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        payload = {"model": model, "input": texts}
        try:
            r = self.session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
        except requests.HTTPError as e:
            # 응답 바디를 그대로 보여줘서 원인을 빠르게 파악
            msg = getattr(e.response, "text", str(e))
            raise RuntimeError(f"[Solar Embeddings Error] {msg}") from e
        data = r.json().get("data", [])
        return [item["embedding"] for item in data]

    # 편의 함수: 역할 분리형 호출
    # embed_passage 함수: 문서를 번역할 때 사용
    def embed_passage(self, texts: List[str], timeout: int = 60) -> List[List[float]]:
        return self.embed(texts, model="embedding-passage", timeout=timeout)

    # embed_query 함수: 사용자의 질문을 번역할 때 사용
    def embed_query(self, texts: List[str], timeout: int = 60) -> List[List[float]]:
        return self.embed(texts, model="embedding-query", timeout=timeout)

    # --- 생성 (최종 QA용) ---
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "solar-pro",
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: int = 120,
    ) -> str:
        """
        입력: system_prompt(역할/규칙), user_prompt(실제 질문/참고 자료)
        출력: 모델이 생성한 답변 문자열
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            r = self.session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
        except requests.HTTPError as e:
            msg = getattr(e.response, "text", str(e))
            raise RuntimeError(f"[Solar Chat Error] {msg}") from e
        return r.json()["choices"][0]["message"]["content"]