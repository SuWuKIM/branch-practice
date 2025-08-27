# src/llm/prompt.py
"""
목적:
- LLM에 보낼 '문장 묶음(프롬프트)'을 깔끔하고 일관되게 만들어 줍니다.
- 구성: System(역할/규칙) + User(질문+근거 Evidence 블록)
- 옵션: 한국어 출력, 불릿 위주, Sources(URL) 강제, CoT(조용한 단계적 사고) 지시 등

사용 흐름(예):
- retriever.search(question) -> contexts(텍스트), sources(메타) 를 받는다.
- PromptBuilder.build_messages(question, sources) 로 system/user 메시지를 만든다.
- SolarClient.generate(system, user, model=...) 로 답변을 생성한다.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PromptOptions:
    language: str = "ko"                 # 답변 언어
    style: str = "bullets"               # 출력 스타일 힌트
    include_sources: bool = True         # Sources 섹션 강제 여부
    max_context_chars: int = 3000        # Evidence 전체 길이 상한(너무 길면 잘라냄)
    max_blocks: int = 5                  # Evidence 블록 최대 개수(Top-k 추천 3~5)
    max_block_chars: int = 800           # 블록별 본문 최대 길이
    cot_silent: bool = True              # 조용한 단계적 사고(CoT) 유도 (사고 과정을 출력하지 말 것)
    react_hint: bool = False             # 근거 부족 시 재검색(ReAct) 힌트를 프롬프트에 넣을지 여부


class PromptBuilder:
    """
    프롬프트 빌더:
    - System Prompt(역할/규칙/형식)를 문자열로 생성
    - Evidence(리트리버 결과)를 깔끔한 블록으로 정리
    - 질문(User 메시지)와 합쳐 최종 메시지 2개(system, user)를 반환
    """
    def __init__(self, options: Optional[PromptOptions] = None):
        self.opt = options or PromptOptions()

    # --------- public API --------- #
    def build_messages(
        self,
        question: str,
        sources: List[Dict[str, Any]],
        extra_instructions: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        반환: [{"role":"system","content":...}, {"role":"user","content":...}]
        """
        system = self.build_system_prompt()
        context_block = self.build_context_block(sources)
        user = self.build_user_message(question, context_block, extra_instructions)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # --------- core builders --------- #
    def build_system_prompt(self) -> str:
        """
        System(역할/규칙/형식) 지시문을 생성합니다.
        - 근거(Evidence)만 사용
        - 근거 없으면 '모르겠습니다' 또는 '근거 없음'으로 답
        - 한국어, 불릿 중심, 간결히
        - 마지막에 Sources(URL) 표기
        - (선택) CoT: 조용히 단계적으로 생각하되, 사고 과정은 출력 금지
        - (선택) ReAct 힌트: 근거 부족 시 추가 검색이 필요하다고 스스로 판단하도록 유도(실제 도구 호출은 코드에서 처리)
        """
        lines = []
        lines.append("당신은 뉴스 RAG 어시스턴트입니다.")
        lines.append("다음 규칙을 반드시 지키세요:")
        lines.append("1) 아래 Evidence에 포함된 정보만 사용해 답변하세요. 추측/창작 금지.")
        lines.append("2) Evidence가 불충분하면 '모르겠습니다'라고 명확히 말하세요.")
        lines.append("3) 답변은 한국어로 간결한 불릿 위주로 작성하세요.")
        if self.opt.include_sources:
            lines.append("4) 답변 마지막에 'Sources' 섹션을 만들고, 참고한 URL을 나열하세요.")
        # CoT: 내부 사고 유도(출력 금지)
        if self.opt.cot_silent:
            lines.append("5) 답변을 작성하기 전, 조용히 단계적으로 생각하되 그 생각 과정을 출력하지 마세요.")
        # ReAct 힌트(도구 루프는 코드에서 처리하되, 모델에게 힌트를 줌)
        if self.opt.react_hint:
            lines.append("6) Evidence가 부족하다고 판단되면, 추가 검색이 필요하다는 점을 답변 내에서 간단히 알려주세요.")
        return "\n".join(lines)

    def build_context_block(self, sources: List[Dict[str, Any]]) -> str:
        """
        리트리버 결과(sources)를 Evidence 블록으로 정리합니다.
        sources 원소 예시:
        {
          "title": str, "url": str, "source": str, "date_published": str,
          "chunk_index": int, "length": int, "score": float,
          (선택적으로) "text": 청크 본문 (리트리버 쪽에서 넣어줄 수도 있음)
        }
        """
        # 상위 max_blocks까지만 사용
        items = sources[: self.opt.max_blocks]

        # 블록 문자열 만들기
        blocks = []
        used_chars = 0
        for i, meta in enumerate(items, 1):
            title = meta.get("title", "").strip() or "(제목 없음)"
            url = meta.get("url", "").strip()
            src = meta.get("source", "").strip()
            date = meta.get("date_published", "").strip()
            score = meta.get("score", None)
            # 본문 텍스트: retriever에서 넘겨주지 않았다면 생략 가능
            body = meta.get("text", "")
            if body and len(body) > self.opt.max_block_chars:
                body = body[: self.opt.max_block_chars] + "..."

            block = []
            head = f"[{i}] {title}"
            if url:
                head += f" ({url})"
            if src or date:
                head += f"  |  {src or ''}{' · ' if src and date else ''}{date or ''}"
            if score is not None:
                head += f"  |  score={round(float(score), 4)}"
            block.append(head)
            if body:
                block.append(body)
            block.append("---")
            block_str = "\n".join(block)

            # 전체 Evidence 길이 제한
            if used_chars + len(block_str) > self.opt.max_context_chars:
                break
            blocks.append(block_str)
            used_chars += len(block_str)

        if not blocks:
            return "(Evidence 없음)"

        return "Evidence:\n" + "\n".join(blocks)

    def build_user_message(
        self,
        question: str,
        context_block: str,
        extra_instructions: Optional[str] = None,
    ) -> str:
        """
        User 메시지: 질문 + Evidence + 출력 형식 힌트를 한 번에 전달
        """
        lines = []
        lines.append(f"질문: {question.strip()}")
        lines.append("")
        lines.append(context_block)
        lines.append("")
        # 출력 형식 힌트
        fmt_hints = []
        if self.opt.language.lower().startswith("ko"):
            fmt_hints.append("- 한국어로 답변")
        if self.opt.style == "bullets":
            fmt_hints.append("- 핵심 불릿 3~5개")
        if self.opt.include_sources:
            fmt_hints.append("- 마지막에 Sources 섹션 (URL 나열)")
        if extra_instructions:
            fmt_hints.append(f"- 추가 지시: {extra_instructions.strip()}")
        if fmt_hints:
            lines.append("요청 포맷:")
            lines.extend(fmt_hints)
        return "\n".join(lines)