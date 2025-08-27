# src/crawler/rss_crawler.py

# FIXME: 지금은 빨리 돌려보기용 코드라서 함수로 만들었습니다. 나중에 클래스로 수정할게요!

"""
- RSS 주소(여러 개)를 받아서, 각 글의 URL로 들어가 본문을 뽑아옵니다.
- 본문을 '정규화'해서 공백/줄바꿈을 정리하고, 같은 내용이면 같은 '해시'가 나오도록 만듭니다.
- 결과는 DB에 넣기 좋은 dict 리스트로 돌려줍니다.
"""


import feedparser          # RSS 파서(피드 읽기)
import trafilatura         # 웹페이지에서 '본문'만 추출
import hashlib
import re
from urllib.parse import urlparse, urlunparse

def _normalize_url(u: str) -> str:
    """URL에서 추적용 쿼리스트링(utm 등)을 제거해 같은 글을 같은 주소로 인식."""
    p = urlparse(u)
    clean = p._replace(query="")  # 쿼리 제거
    return urlunparse(clean)

def _normalize_text(text: str) -> str:
    """여러 공백/개행을 하나로 줄여 '같은 내용이면 같은 문자열'이 되게 정리."""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _content_hash(text: str) -> str:
    """정규화된 본문으로 지문(해시)을 생성. 중복 문서 방지에 사용."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fetch_rss_docs(rss_urls: list[str], per_feed_limit: int = 20) -> list[dict]:
    """
    입력: RSS 주소 리스트
    출력: 문서 dict 리스트 (DB upsert용)
      dict 예시:
      {
        "url": "...", "title": "...", "source": "...",
        "date_published": "...", "raw_text": "...",
        "content_hash": "...", "lang": "en" 또는 "ko"
      }
    """
    docs = []
    for rss in rss_urls:
        feed = feedparser.parse(rss)          # RSS 목록 읽기
        source_name = feed.feed.get("title", "").strip() if feed.feed else ""
        # entries: 파싱한 RSS의 글 목록. 메타 데이터만 있고 본문 내용은 보통 없음 -> 그래서 링크에 들어가서 본문을 추출해야 함!
        entries = feed.entries[:per_feed_limit]

        for e in entries:
            # 1) URL 정리
            url = _normalize_url(e.get("link", "")) or ""
            if not url:
                continue

            # 2) 웹페이지 다운로드 + 본문 추출
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                continue
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False
            )
            if not extracted:
                continue

            # 3) 텍스트 정규화
            text = _normalize_text(extracted)
            if len(text) < 400:  # 너무 짧은 본문은 노이즈일 확률↑
                continue

            # 4) 메타데이터 정리
            title = (e.get("title") or "").strip()
            date_published = (e.get("published") or e.get("updated") or "").strip()
            lang = (e.get("language") or "en").strip()  # RSS가 언어를 잘 안 줄 때가 많아 기본 en

            # 5) 해시 생성
            h = _content_hash(text)

            # 6) DB에 넣기 좋은 dict로 패키징
            docs.append({
                "url": url,
                "title": title,
                "source": source_name,
                "date_published": date_published,
                "raw_text": text,
                "content_hash": h,
                "lang": lang
            })

    return docs