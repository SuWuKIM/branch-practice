# tests/prompt_check.py
# 프롬프트 조립
from src.llm.prompt import PromptBuilder, PromptOptions

def main():
    # 1) 프롬프트 옵션 (필요시 바꿀 수 있음)
    opt = PromptOptions(language="ko", style="bullets")

    # 2) 빌더 생성
    builder = PromptBuilder(options=opt)

    # 3) 가짜 sources (retriever가 리턴했다고 가정)
    fake_sources = [
        {
            "title": "GPT-5 출시 소식",
            "url": "https://openai.com/gpt-5",
            "source": "OpenAI Blog",
            "date_published": "2025-08-01",
            "text": "OpenAI는 GPT-5 모델을 발표하며 안전성 향상과 새로운 기능을 강조했습니다.",
            "score": 0.92,
        },
        {
            "title": "AI 규제 관련 정책",
            "url": "https://example.com/ai-policy",
            "source": "Google AI Blog",
            "date_published": "2025-07-28",
            "text": "유럽연합은 AI 규제 법안을 통과시켜, 기업들이 투명성과 책임성을 강화해야 한다고 발표했습니다.",
            "score": 0.88,
        },
    ]

    # 4) 메시지 생성
    messages = builder.build_messages(
        "최근 AI 규제 동향을 요약해줘.",
        fake_sources,
        extra_instructions="답변은 3줄 이내"
    )

    # 5) 결과 출력
    print("[SYSTEM PROMPT]")
    print(messages[0]["content"])
    print("\n[USER PROMPT]")
    print(messages[1]["content"])

if __name__ == "__main__":
    main()