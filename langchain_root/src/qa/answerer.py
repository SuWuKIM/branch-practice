# src/qa/answerer.py
"""
이 파일의 목적:
- 질문 한 번으로 '검색 → 프롬프트 조립 → LLM 생성'까지 수행하는 오케스트레이터.
- 내부적으로 Retriever, PromptBuilder, SolarClient를 호출한다.
- 결과는 '답변 텍스트 + Sources(출처)' 형태로 반환한다.

사용 흐름(예):
    from src.qa.answerer import Answerer
    ans = answerer.answer("최근 OpenAI 관련 중요한 발표를 요약해줘")
    print(ans["model"], ans["answer"])
    for s in ans["sources"]: print(s["title"], s["url"])

또는 두 모델 비교:
    answers = answerer.answer_multi("질문", models=["solar-pro","solar-mini"])
    for a in answers: print(a["model"], a["answer"][:120])
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from src.utils.config import AppConfig
from src.llm.solar import SolarClient
from src.retriever.search import Retriever
from src.llm.prompt import PromptBuilder, PromptOptions


class Answerer:
    """
    검색 → 프롬프트 → 생성까지 한 번에 수행하는 컨트롤러.
    - init 시점에 필요한 의존성(설정/클라이언트)들을 준비한다.
    - answer(): 단일 모델로 생성
    - answer_multi(): 여러 모델(mini/pro) 비교 생성
    """
    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.3,
        prompt_opt: Optional[PromptOptions] = None,
        collection_name: str = "ai_news_rag",
    ) -> None:
        self.cfg = cfg or AppConfig()

        # LLM/Solar 클라이언트 (임베딩·생성 모두 사용)
        self.solar = SolarClient(api_key=self.cfg.solar_api_key)

        # 리트리버(Chroma 연결)
        self.retriever = Retriever(
            chroma_dir=self.cfg.chroma_dir,
            solar_client=self.solar,
            collection_name=collection_name,
            top_k=top_k,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
        )

        # 프롬프트 옵션 (한국어·불릿·Sources 강제·CoT 조용히)
        self.prompt_builder = PromptBuilder(
            options=prompt_opt or PromptOptions(
                language="ko",
                style="bullets",
                include_sources=True,
                max_context_chars=3000,
                max_blocks=5,
                max_block_chars=800,
                cot_silent=True,
                react_hint=False,  # 필요시 True
            )
        )

    # ------------------- public API ------------------- #
    def answer(
        self,
        question: str,
        model: str = "solar-pro",
        max_tokens: int = 300,
        extra_instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        단일 모델로 답변 생성.
        반환: {
          "model": str,
          "answer": str,
          "sources": List[Dict[str, Any]],  # title/url/score 등
          "used_top_k": int,
          "raw": Dict[str, Any]            # 디버깅 숫자(n_initial/selected 등)
        }
        """
        # 1) 검색: Top-k 근거
        ret = self.retriever.search(question)
        sources = ret.get("sources", [])

        # 2) 프롬프트 조립
        msgs = self.prompt_builder.build_messages(
            question=question,
            sources=sources,
            extra_instructions=extra_instructions,
        )
        system = msgs[0]["content"]
        user = msgs[1]["content"]

        # 3) LLM 생성 호출
        try:
            answer_text = self.solar.generate(
                system_prompt=system,
                user_prompt=user,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            answer_text = f"(오류) 모델 호출 실패: {e}"

        # 4) 결과 패키징
        return {
            "model": model,
            "answer": answer_text,
            "sources": sources,        # UI에서 카드로 표기 가능
            "used_top_k": len(sources),
            "raw": ret.get("raw", {}),
        }

    def answer_multi(
        self,
        question: str,
        models: Optional[List[str]] = None,
        max_tokens: int = 300,
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        여러 모델을 같은 컨텍스트로 호출해 결과를 나란히 비교.
        기본: ["solar-pro", "solar-mini"]
        """
        models = models or ["solar-pro", "solar-mini"]

        # 한 번만 검색하고(같은 근거)
        ret = self.retriever.search(question)
        sources = ret.get("sources", [])
        msgs = self.prompt_builder.build_messages(
            question=question, sources=sources, extra_instructions=extra_instructions
        )
        system = msgs[0]["content"]
        user = msgs[1]["content"]

        results: List[Dict[str, Any]] = []
        for m in models:
            try:
                text = self.solar.generate(
                    system_prompt=system,
                    user_prompt=user,
                    model=m,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                text = f"(오류) 모델 호출 실패: {e}"

            results.append({
                "model": m,
                "answer": text,
                "sources": sources,
                "used_top_k": len(sources),
                "raw": ret.get("raw", {}),
            })
        return results