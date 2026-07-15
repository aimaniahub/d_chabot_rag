"""Streamlit chat UI for PDF RAG."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when launched via streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config import DATA_DIR, MIN_SCORE, TOP_K, ensure_dirs
from modules.chat import ChatService
from modules.ingest import get_index_stats, ingest_paths

ensure_dirs()

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Chat")
st.caption("Ingest PDFs → Chroma embeddings → grounded Gemini answers with sources.")


@st.cache_resource
def get_chat_service(top_k: int, min_score: float) -> ChatService:
    return ChatService(top_k=top_k, min_score=min_score)


with st.sidebar:
    st.header("Index")
    stats = get_index_stats()
    st.metric("Chunks", stats["chunk_count"])
    st.metric("Documents", len(stats.get("documents") or []))
    st.write(f"**Model:** `{stats.get('embedding_model')}`")
    docs = stats.get("documents") or []
    if docs:
        st.markdown("**Indexed files**")
        for d in docs:
            st.write(f"- {d.get('doc_id')} ({d.get('chunk_count')} chunks)")

    st.divider()
    st.subheader("Ingest")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None:
        dest = DATA_DIR / uploaded.name
        if st.button("Save & index upload"):
            dest.write_bytes(uploaded.getvalue())
            with st.spinner("Indexing..."):
                report = ingest_paths(paths=[dest], force=True)
            st.success(
                f"Indexed={report.indexed} skipped={report.skipped} failed={report.failed}"
            )
            st.json(report.to_dict())
            st.cache_resource.clear()
            st.rerun()

    if st.button("Ingest all in data/"):
        with st.spinner("Indexing data/ ..."):
            report = ingest_paths()
        st.success(
            f"Indexed={report.indexed} skipped={report.skipped} failed={report.failed} "
            f"({report.total_ms:.0f} ms)"
        )
        st.cache_resource.clear()
        st.rerun()

    if st.button("Rebuild index"):
        with st.spinner("Full rebuild..."):
            report = ingest_paths(rebuild=True)
        st.warning(f"Rebuild done. Indexed={report.indexed} failed={report.failed}")
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Retrieval")
    top_k = st.slider("top_k", 1, 12, TOP_K)
    min_score = st.slider("min_score", 0.0, 1.0, float(MIN_SCORE), 0.01)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['n']}]** `{s['source']}` "
                        f"p.{s['page_start']}-{s['page_end']} "
                        f"(score={s['score']})"
                    )
                    st.caption(s.get("preview", ""))
        if msg.get("metrics"):
            m = msg["metrics"]
            t = m.get("timings", {})
            st.caption(
                f"total={t.get('total_ms', 0):.0f}ms · "
                f"retrieve={t.get('retrieve_ms', 0):.0f}ms · "
                f"llm={t.get('llm_ms', 0):.0f}ms · "
                f"tokens={m.get('tokens', {}).get('total_tokens')}"
            )

prompt = st.chat_input("Ask about your PDFs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    stats = get_index_stats()
    if stats["chunk_count"] == 0:
        answer = "Index is empty. Upload a PDF or click **Ingest all in data/** in the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if m["role"] in ("user", "assistant")
                    ]
                    service = get_chat_service(top_k, min_score)
                    resp = service.ask(prompt, history=history, top_k=top_k, min_score=min_score)
                    st.markdown(resp.answer)
                    if resp.sources:
                        with st.expander("Sources", expanded=True):
                            for s in resp.sources:
                                st.markdown(
                                    f"**[{s['n']}]** `{s['source']}` "
                                    f"p.{s['page_start']}-{s['page_end']} "
                                    f"(score={s['score']})"
                                )
                                st.caption(s.get("preview", ""))
                    m = resp.metrics.to_dict()
                    t = m["timings"]
                    st.caption(
                        f"total={t['total_ms']:.0f}ms · "
                        f"embed={t['embed_ms']:.0f}ms · "
                        f"retrieve={t['retrieve_ms']:.0f}ms · "
                        f"llm={t['llm_ms']:.0f}ms · "
                        f"tokens={m['tokens'].get('total_tokens')}"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": resp.answer,
                            "sources": resp.sources,
                            "metrics": m,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    err = f"Error: {exc}"
                    st.error(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
