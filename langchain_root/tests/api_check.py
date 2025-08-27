from src.utils.config import AppConfig
from src.llm.solar import SolarClient

def main():
    # 1. 환경설정 불러오기 (.env → AppConfig)
    cfg = AppConfig()

    # 2. Solar 클라이언트 초기화
    cli = SolarClient(api_key=cfg.solar_api_key)

    # 3. 간단한 챗 생성 요청 (Chat 생성이 되는지 확인)
    system = "너는 사실만 말하는 간단한 도우미야."
    user = "인공지능을 한 문장으로 설명해줘."

    try:
        ans = cli.generate(
            system_prompt=system,
            user_prompt=user,
            model="solar-pro-2",   # 답변 품질 확인용
            max_tokens=50
        )
        print("=== Solar Chat API 응답 ===")
        print(ans)
    except Exception as e:
        print("에러 발생:", e)

if __name__ == "__main__":
    main()