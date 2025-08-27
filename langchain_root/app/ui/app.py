# app/ui/app.py
# 실행:  streamlit run app/ui/app.py

import time
import streamlit as st

from src.utils.config import AppConfig
from src.qa.answerer import Answerer
from src.sql.db import SqlStore
from src.crawler.rss_crawler import fetch_rss_docs
from src.llm.solar import SolarClient
from src.vector_store.indexer import Indexer
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -----------------------------
# 초기 설정 / 세션 준비
# -----------------------------
st.set_page_config(page_title="AI News RAG QA", page_icon="📰", layout="wide")
st.title("📰 AI News RAG — QA & Evidence Viewer")

if "cfg" not in st.session_state:
    st.session_state.cfg = AppConfig()
if "answerer" not in st.session_state:
    st.session_state.answerer = Answerer(cfg=st.session_state.cfg)
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None


# -----------------------------
# 사이드바: 옵션/액션
# -----------------------------
with st.sidebar:
    st.header("⚙️ Options")

    # 모델 선택
    model_mode = st.radio(
        "Model",
        options=("Both (pro & mini)", "solar-pro only", "solar-mini only"),
        index=0,
    )

    # Retrieval 파라미터 (실시간 반영을 위해 Answerer 재구성 버튼 제공)
    top_k = st.slider("Top-k", min_value=3, max_value=8, value=5, step=1)
    mmr_lambda = st.slider("MMR λ (관련성↔다양성)", 0.0, 1.0, 0.3, 0.05)
    use_mmr = st.checkbox("Use MMR", value=True)

    # 생성 길이
    max_tokens = st.slider("Max tokens (generation)", 150, 800, 320, 50)

    st.caption("※ Retrieval/Generation 파라미터 변경 후 아래 버튼으로 적용하세요.")
    if st.button("Apply Retrieval Settings", use_container_width=True):
        st.session_state.answerer = Answerer(
            cfg=st.session_state.cfg,
            top_k=top_k,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
        )
        st.success("Retrieval settings applied.")

    st.divider()
    st.header("🧹 Data Ops")

    # 수집(ingest)
    if st.button("Ingest: RSS → SQLite (Fetch latest)", use_container_width=True):
        cfg = st.session_state.cfg
        store = SqlStore(cfg.sqlite_path)
        with st.spinner("Fetching RSS and extracting main content..."):
            docs = fetch_rss_docs(cfg.rss_list, per_feed_limit=20)
            inserted = 0
            for d in docs:
                doc_id = store.upsert_document(d)
                if doc_id:
                    inserted += 1
            st.success(f"INGEST 완료: 새 문서 {inserted} / 총 가져온 문서 {len(docs)}")

    # 색인(index)
    if st.button("Index: Chunk → Embed → Chroma upsert", use_container_width=True):
        cfg = st.session_state.cfg
        store = SqlStore(cfg.sqlite_path)
        solar = SolarClient(api_key=cfg.solar_api_key)
        indexer = Indexer(
            store=store,
            chroma_dir=cfg.chroma_dir,
            solar_client=solar,
            max_chars=1200,
            overlap=120,
            min_chunk_chars=200,
            batch_size=16,
        )
        with st.spinner("Indexing documents... (chunking/embedding/upsert)"):
            result = indexer.index_recent(limit_docs=100)
            st.success(f"INDEX 결과: {result}")

    st.divider()
    st.caption(f"ENV: {st.session_state.cfg.env}")
    st.caption(f"Chroma: {st.session_state.cfg.chroma_dir}")
    st.caption(f"SQLite: {st.session_state.cfg.sqlite_path}")


# -----------------------------
# 메인 입력 / 실행
# -----------------------------
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("❓ Ask a question")
    question = st.text_area(
        "질문을 입력하세요",
        value="최근 생성형 AI 규제 동향을 요약해줘.",
        height=120,
        help="예) 오늘 기사 중 OpenAI 관련 정책 이슈만 요약해줘",
    )

    extra_ins = st.text_input(
        "추가 지시 (선택)",
        placeholder="예) 불릿 3개 이내, 반드시 Sources 표시",
    )

    do_search = st.button("Run QA", type="primary", use_container_width=True)

    st.markdown("—")
    st.caption("💡 팁: 먼저 Ingest/Index를 실행해 KB를 최신으로 만들어두면 정확도가 올라갑니다.")

with col_right:
    st.subheader("🧠 Answers & Evidence")

    if do_search:
        t0 = time.time()

        # 모델 선택 모드 설정
        if model_mode == "Both (pro & mini)":
            models = ["solar-pro", "solar-mini"]
        elif model_mode == "solar-pro only":
            models = ["solar-pro"]
        else:
            models = ["solar-mini"]

        # 오케스트레이션 실행
        with st.spinner("Retrieving evidence and generating answers..."):
            if len(models) == 1:
                res = st.session_state.answerer.answer(
                    question=question,
                    model=models[0],
                    max_tokens=max_tokens,
                    extra_instructions=extra_ins or None,
                )
                st.session_state.last_results = [res]
                st.session_state.last_sources = res["sources"]
            else:
                res_list = st.session_state.answerer.answer_multi(
                    question=question,
                    models=models,
                    max_tokens=max_tokens,
                    extra_instructions=extra_ins or None,
                )
                st.session_state.last_results = res_list
                st.session_state.last_sources = res_list[0]["sources"] if res_list else []

        t1 = time.time()
        st.success(f"완료: {(t1 - t0)*1000:.0f} ms")

    # 결과 표시
    if st.session_state.last_results:
        results = st.session_state.last_results
        sources = st.session_state.last_sources or []

        # 탭 구성: 모델별 + 근거뷰
        tab_labels = [f"🧩 {r['model']}" for r in results] + ["📚 Evidence"]
        tabs = st.tabs(tab_labels)

        # 모델별 답변 탭
        for i, r in enumerate(results):
            with tabs[i]:
                st.markdown(f"**Model:** `{r['model']}`  |  **Top-k used:** {r['used_top_k']}")
                st.markdown("---")
                st.markdown(r["answer"])

                # Sources 섹션 (답변 본문과 별도로 다시 한번 명확히)
                st.markdown("#### Sources")
                if r.get("sources"):
                    for idx, s in enumerate(r["sources"], 1):
                        title = s.get("title", "(제목 없음)")
                        url = s.get("url", "")
                        score = s.get("score", None)
                        meta_line = []
                        if s.get("source"):
                            meta_line.append(s["source"])
                        if s.get("date_published"):
                            meta_line.append(s["date_published"])
                        meta_txt = " · ".join(meta_line)

                        score_txt = f" · score={round(float(score), 4)}" if score is not None else ""
                        st.markdown(f"- **[{idx}] {title}**  \n  {meta_txt}{score_txt}  \n  <{url}>")
                else:
                    st.info("선택된 근거가 없습니다.")

                # Raw 디버깅 요약
                if r.get("raw"):
                    st.caption(f"raw: {r['raw']}")

        # Evidence 탭 (공통 근거 뷰)
        with tabs[-1]:
            st.markdown("**Retrieval 결과 (Top-k Evidence)**")
            if not sources:
                st.info("근거가 없습니다. 먼저 질문을 실행하세요.")
            else:
                for i, s in enumerate(sources, 1):
                    with st.expander(f"[{i}] {s.get('title','(제목 없음)')}", expanded=(i == 1)):
                        meta = []
                        if s.get("source"):
                            meta.append(s["source"])
                        if s.get("date_published"):
                            meta.append(s["date_published"])
                        meta_txt = " · ".join(meta)
                        st.caption(meta_txt if meta_txt else "—")

                        # 점수/길이
                        score = s.get("score", None)
                        length = s.get("length", None)
                        stat_line = []
                        if score is not None:
                            stat_line.append(f"score={round(float(score),4)}")
                        if length is not None:
                            stat_line.append(f"len={length}")
                        st.caption(" | ".join(stat_line) if stat_line else "—")

                        # 본문 미리보기(리트리버가 text를 포함시켰다면)
                        preview = s.get("text", None)
                        if preview:
                            st.write(preview[:600] + ("..." if len(preview) > 600 else ""))
                        st.markdown(f"[원문 링크]({s.get('url','')})")
    else:
        st.info("좌측에서 질문을 입력하고 **Run QA**를 눌러 실행하세요.")