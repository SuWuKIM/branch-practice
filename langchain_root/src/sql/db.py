# src/sql/db.py
import sqlite3, os, time

# 정형 데이터(메타 데이터, 원본) 등을 저장할 테이블
# url, title, source, date_published, date_crawled, content_hash, raw_text, lang 저장

# 스키마란? 데이터베이스의 구조를 만드는 sql 명령어.
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE,
  title TEXT,
  source TEXT,
  date_published TEXT,
  date_crawled TEXT,
  content_hash TEXT,
  raw_text TEXT,
  lang TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
"""

class SqlStore:
    """
    - 생성자에서 SQLite 파일을 만들고(없으면 생성), 우리에게 필요한 테이블(documents)을 만들어 둡니다.
    - upsert_document(): 같은 URL 또는 같은 내용(content_hash)이면 중복 저장을 막습니다.
    - fetch_all(): 최신 문서 몇 개를 읽어옵니다(색인 단계에서 사용할 예정).
    """
    def __init__(self, db_path: str):
        # 폴더가 없다면 먼저 만들어 둠
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # DB 연결 (파일이 없으면 새로 만들어짐)
        self.conn = sqlite3.connect(db_path)
        # 동시 접근 안정화(기본 성능 개선)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # 스키마 적용
        self.conn.executescript(SCHEMA)

    def upsert_document(self, doc: dict) -> int | None:
        """
        doc 딕셔너리 예:
        {
          "url": "...", "title": "...", "source": "...",
          "date_published": "...", "content_hash": "...",
          "raw_text": "...", "lang": "ko" 또는 "en"
        }
        이미 같은 URL 또는 같은 해시가 있으면 새로 넣지 않고 기존 id를 돌려줍니다.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM documents WHERE url=? OR content_hash=?",
                    (doc["url"], doc["content_hash"]))
        row = cur.fetchone()
        if row:
            return row[0]  # 기존 문서 id

        # 먼저 url, title, source, date_published, content_hash, raw_text, lang에 그 내용이 있는지 확인
        # 만일 없다면 추가
        cur.execute("""
          INSERT INTO documents(url,title,source,date_published,date_crawled,content_hash,raw_text,lang)
          VALUES(?,?,?,?,?,?,?,?)
        """, (
          doc["url"], doc.get("title",""), doc.get("source",""),
          doc.get("date_published",""),
          time.strftime("%Y-%m-%dT%H:%M:%S"),  # 지금 수집 시간
          doc["content_hash"], doc.get("raw_text",""), doc.get("lang","")
        ))
        self.conn.commit()
        return cur.lastrowid  # 새 문서 id

    def fetch_all(self, limit: int = 200):
        """
        최근 문서를 몇 개 읽어옵니다.
        나중에 색인(청킹/임베딩) 단계에서 사용합니다.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, url, title, source, date_published, raw_text, lang
            FROM documents
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]