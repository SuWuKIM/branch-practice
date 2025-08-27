import os, yaml
from dotenv import load_dotenv

load_dotenv()  # .env 로드

def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# .env와 configs/*를 읽어 하나의 런타임 설정 객체로 합침
# AppConfig 클래스 덕분에 파일들은 경로/키 걱정 없이 다음과 같이 쉽게 접근 가능
class AppConfig:
    def __init__(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.root = root
        self.app = _read_yaml(os.path.join(root, "configs", "app.yaml"))
        self.chunking = _read_yaml(os.path.join(root, "configs", "chunking.yaml"))

        # 환경변수 우선 적용
        self.env = os.getenv("APP_ENV", self.app["app"].get("env", "dev"))
        self.chroma_dir = os.getenv("CHROMA_DIR", self.app["paths"]["chroma_dir"])
        self.sqlite_path = os.getenv("SQLITE_PATH", self.app["paths"]["sqlite_path"])

        self.solar_api_key = os.getenv("SOLAR_API_KEY", "")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")

        # W&B
        self.wandb_project = os.getenv("WANDB_PROJECT", self.app["logging"]["wandb"]["project"])
        self.wandb_entity  = os.getenv("WANDB_ENTITY",  self.app["logging"]["wandb"].get("entity",""))

        # RSS 소스
        self.rss_list = self.app["sources"]["rss"]