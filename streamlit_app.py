"""
PDF Agent - Streamlit Frontend
Features: classify, summarize, metadata, tables, chat (single + multi-doc), compare
"""

import os
import time

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")
HEALTH_URL = API_BASE.replace("/api/v1", "") + "/health"

TIMEOUTS = {
    "upload": 600,
    "classify": 120,
    "summarize": 300,
    "metadata": 120,
    "chat": 180,
    "compare": 240,
    "default": 60,
}

st.set_page_config(page_title="PDF Agent", page_icon="D", layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --pa-primary: #4c6ef5;
    --pa-primary-soft: #eef1ff;
    --pa-ink: #1f2430;
    --pa-muted: #6b7280;
    --pa-border: #e6e8ee;
}
.block-container { padding-top: 2.2rem; }
.pa-hero {
    background: linear-gradient(135deg, #4c6ef5 0%, #7048e8 100%);
    color: #fff; border-radius: 18px; padding: 34px 38px; margin-bottom: 22px;
}
.pa-hero h1 { color:#fff; margin:0 0 6px 0; font-size:2.0rem; }
.pa-hero p  { color:#e8ebff; margin:0; font-size:1.02rem; }
.pa-tile { border:1px solid var(--pa-border); border-radius:14px; padding:18px 20px; height:100%; background:#fff; }
.pa-tile h4 { margin:0 0 6px 0; color:var(--pa-ink); font-size:1.02rem; }
.pa-tile p  { margin:0; color:var(--pa-muted); font-size:0.9rem; }
.pa-status { display:flex; align-items:center; gap:8px; font-size:0.9rem; font-weight:600; }
.pa-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.pa-dot.ok  { background:#37b24d; box-shadow:0 0 0 3px rgba(55,178,77,.18); }
.pa-dot.off { background:#e03131; box-shadow:0 0 0 3px rgba(224,49,49,.18); }
.chip { display:inline-block; padding:3px 11px; border-radius:14px; margin:3px 3px 3px 0; font-size:0.8rem; font-weight:600; }
.chip-topic   { background:#e7f5ff; color:#1971c2; }
.chip-keyword { background:#ebfbee; color:#2f9e44; }
.chip-entity  { background:#fff4e6; color:#e8590c; }
.chip-meta    { background:#f3f0ff; color:#7048e8; }
.cite {
    background:#f8f9fc; border:1px solid var(--pa-border); border-left:3px solid var(--pa-primary);
    border-radius:8px; padding:10px 13px; margin:6px 0; font-size:0.83rem; color:#3a3f4b;
}
.cite b { color:var(--pa-primary); }
.cite .score { float:right; font-size:0.78rem; background:#4c6ef520; color:#4c6ef5; padding:1px 7px; border-radius:8px; }
.cite .doc-badge { font-size:0.75rem; background:#7048e820; color:#7048e8; padding:1px 7px; border-radius:8px; margin-left:4px; }
.pa-label { font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase; color:var(--pa-muted); font-weight:700; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "doc_id": None,
    "doc_info": None,
    "_last_upload_key": None,   # prevents re-upload on st.rerun()
    "chat_history": [],
    "insights_loaded": False,
    "classify_data": None,
    "summary_data": None,
    "metadata_data": None,
    "tables_data": None,
    "questions_data": None,
    # Multi-doc
    "all_docs": [],
    "selected_doc_ids": [],
    "multi_chat_history": [],
    # Compare
    "compare_doc_a": None,
    "compare_doc_b": None,
    "compare_result": None,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ── API helpers ───────────────────────────────────────────────────────────────
def api_post(endpoint: str, **kwargs):
    key = endpoint.split("/")[0]
    timeout = TIMEOUTS.get(key, TIMEOUTS["default"])
    for attempt in range(2):
        try:
            r = requests.post(f"{API_BASE}/{endpoint}", timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(5)
                continue
            return None, {"message": "Cannot reach the backend. Wait ~30s and try again.", "retry_after": 0}
        except requests.exceptions.Timeout:
            return None, {"message": f"Request timed out after {timeout}s. Try again.", "retry_after": 0}
        except requests.exceptions.HTTPError as e:
            return None, _parse_http_error(e)


def api_get(endpoint: str, timeout: int = 60):
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, {"message": "Cannot connect to the backend.", "retry_after": 0}
    except requests.exceptions.HTTPError as e:
        return None, _parse_http_error(e)


def _parse_http_error(e: requests.exceptions.HTTPError) -> dict:
    try:
        body = e.response.json()
        detail = body.get("detail", {})
        if isinstance(detail, dict):
            return {"message": detail.get("message", str(e)), "retry_after": detail.get("retry_after", 0)}
        return {"message": str(detail), "retry_after": 0}
    except Exception:
        return {"message": str(e), "retry_after": 0}


def _show_error(error, context: str = ""):
    if not isinstance(error, dict):
        st.error(f"Error: {context}{error}")
        return
    msg = error.get("message", "An unexpected error occurred.")
    retry = error.get("retry_after", 0)
    if retry:
        st.warning(f"Rate limit. {msg}")
    elif any(k in msg.lower() for k in ("api key", "check your")):
        st.error(f"Config error. {msg}")
    elif any(k in msg.lower() for k in ("unavailable", "overloaded")):
        st.warning(msg)
    elif any(k in msg.lower() for k in ("cannot reach", "connect")):
        st.error(f"Connection error. {msg}")
    else:
        st.error(f"{context}{msg}")


def backend_online() -> bool:
    try:
        return requests.get(HEALTH_URL, timeout=10).status_code == 200
    except Exception:
        return False


def reset_document_state():
    keys = ["doc_id", "doc_info", "chat_history", "insights_loaded",
            "classify_data", "summary_data", "metadata_data", "tables_data",
            "questions_data", "_last_upload_key"]
    for k in keys:
        st.session_state[k] = _DEFAULTS[k]


def chips(items, css_class):
    if not items:
        return "<span style='color:#9aa0ac;font-size:0.85rem;'>None found</span>"
    return " ".join(f"<span class='chip {css_class}'>{i}</span>" for i in items)


def render_citation(c: dict, show_doc: bool = False):
    score = c.get("relevance_score", 0)
    score_pct = f"{score*100:.0f}%" if score else ""
    doc_badge = ""
    if show_doc and c.get("document_name"):
        doc_badge = f"<span class='doc-badge'>doc: {c['document_name']}</span>"
    score_html = f"<span class='score'>{score_pct}</span>" if score_pct else ""
    st.markdown(
        f"<div class='cite'>{score_html}{doc_badge}"
        f"<b>Page {c.get('page', '?')}</b><br>"
        f"<span style='white-space:pre-wrap;'>{c.get('text', '')}</span></div>",
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PDF Agent")
    st.caption("AI-powered document intelligence")

    online = backend_online()
    dot = "ok" if online else "off"
    label = "Backend connected" if online else "Backend offline"
    st.markdown(f"<div class='pa-status'><span class='pa-dot {dot}'></span>{label}</div>", unsafe_allow_html=True)

    if not online:
        st.error("Start the backend first:\n```\nuvicorn app.main:app --reload --port 8000\n```")

    st.divider()
    st.markdown("#### Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        # Deduplicate: Streamlit re-runs after every state change, which would
        # re-trigger the upload on each rerun. Use a (name, size) fingerprint
        # to ensure we only send each file to the backend once.
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if upload_key != st.session_state._last_upload_key:
            with st.spinner("Uploading & indexing..."):
                result, error = api_post(
                    "upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )
            if error:
                _show_error(error)
            elif result.get("status") == "error":
                st.error(result.get("message", "Processing failed."))
            else:
                reset_document_state()
                st.session_state.doc_id = result["document_id"]
                st.session_state.doc_info = result
                st.session_state._last_upload_key = upload_key
                st.success(f"Indexed {result.get('page_count', '?')} pages.")
                docs, _ = api_get("documents")
                if docs:
                    st.session_state.all_docs = docs
                st.rerun()

    if st.session_state.doc_id:
        st.divider()
        info = st.session_state.doc_info or {}
        st.markdown("#### Current document")
        st.markdown(f"**{info.get('filename', 'Unknown')}**")
        size_kb = (info.get("file_size", 0) or 0) // 1024
        st.caption(f"{info.get('page_count', '?')} pages · {size_kb} KB")
        if st.button("Clear document", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/document/{st.session_state.doc_id}", timeout=10)
            except Exception:
                pass
            reset_document_state()
            st.rerun()

# ── Landing screen ────────────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.markdown("""
        <div class="pa-hero">
            <h1>Understand any PDF in seconds</h1>
            <p>Upload a document to classify, summarize, extract metadata and tables, and chat with it.</p>
        </div>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "Insights + Metadata", "Classification, summaries, topics, and auto-extracted metadata."),
        (c2, "Table Extraction", "Pulls all tables from the PDF as clean markdown."),
        (c3, "Chat with citations", "Ask questions — answers cite the exact page and relevance score."),
        (c4, "Compare & Multi-PDF", "Compare two docs side-by-side or chat across a library."),
    ]
    for col, title, body in tiles:
        with col:
            st.markdown(f"<div class='pa-tile'><h4>{title}</h4><p>{body}</p></div>", unsafe_allow_html=True)
    st.markdown("")
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

# ── Main tabs ─────────────────────────────────────────────────────────────────
doc_id = st.session_state.doc_id
tab_insights, tab_tables, tab_chat, tab_multi, tab_compare = st.tabs(
    ["Insights", "Tables", "Chat", "Multi-PDF", "Compare"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB: Insights
# ════════════════════════════════════════════════════════════════════════════
with tab_insights:
    if not st.session_state.insights_loaded:
        st.markdown("#### Analyse this document")
        st.caption("Classify, summarize, extract topics and metadata in one click.")
        col_a, col_b = st.columns(2)
        run_classify = col_a.button("Classify only", use_container_width=True)
        run_full = col_b.button("Full analysis", use_container_width=True, type="primary")

        if run_classify or run_full:
            progress = st.progress(0, "Starting...")
            err = None

            progress.progress(15, "Classifying...")
            classify_data, err = api_post(f"classify/{doc_id}")
            if err:
                _show_error(err, "Classification: ")

            summary_data = None
            if run_full and not err:
                progress.progress(40, "Summarizing...")
                summary_data, err = api_post(f"summarize/{doc_id}")
                if err:
                    _show_error(err, "Summarization: ")

            metadata_data = None
            if run_full and not err:
                progress.progress(70, "Extracting metadata...")
                metadata_result, merr = api_post(f"metadata/{doc_id}")
                if not merr:
                    metadata_data = metadata_result.get("metadata", {})

            progress.progress(90, "Fetching suggested questions...")
            questions_data, _ = api_get(f"questions/{doc_id}")
            progress.progress(100, "Done")

            st.session_state.classify_data = classify_data
            st.session_state.summary_data = summary_data
            st.session_state.metadata_data = metadata_data
            st.session_state.questions_data = questions_data
            st.session_state.insights_loaded = True
            st.rerun()

    else:
        classify_data = st.session_state.classify_data
        summary_data = st.session_state.summary_data
        metadata_data = st.session_state.metadata_data
        questions_data = st.session_state.questions_data

        # Classification
        if classify_data:
            c1, c2 = st.columns([2, 1])
            c1.metric("Document type", classify_data.get("document_type", "Unknown"))
            conf = classify_data.get("confidence", 0) or 0
            c2.metric("Confidence", f"{conf:.0%}")
            c2.progress(min(max(conf, 0.0), 1.0))
            st.divider()

        # Summaries
        if summary_data:
            st.markdown("<div class='pa-label'>Quick summary</div>", unsafe_allow_html=True)
            st.write(summary_data.get("short_summary") or "_No summary._")
            with st.expander("Detailed analysis"):
                st.write(summary_data.get("detailed_summary") or "_No detailed summary._")
            st.markdown("")
            t, k, e = st.columns(3)
            with t:
                st.markdown("<div class='pa-label'>Topics</div>", unsafe_allow_html=True)
                st.markdown(chips(summary_data.get("topics", []), "chip-topic"), unsafe_allow_html=True)
            with k:
                st.markdown("<div class='pa-label'>Keywords</div>", unsafe_allow_html=True)
                st.markdown(chips(summary_data.get("keywords", []), "chip-keyword"), unsafe_allow_html=True)
            with e:
                st.markdown("<div class='pa-label'>Entities</div>", unsafe_allow_html=True)
                st.markdown(chips(summary_data.get("entities", []), "chip-entity"), unsafe_allow_html=True)
            st.divider()

        # Metadata
        if metadata_data:
            st.markdown("<div class='pa-label'>Document metadata</div>", unsafe_allow_html=True)
            LABELS = {
                "title": "Title", "author": "Author", "date": "Date",
                "language": "Language", "tone": "Tone",
                "target_audience": "Audience", "reading_time_minutes": "Reading time (min)",
                "word_count": "Word count", "document_length": "Length",
                "pdf_title": "PDF Title", "pdf_author": "PDF Author",
                "pdf_creation_date": "Creation date",
            }
            items = [(LABELS.get(k, k.replace("_", " ").title()), v)
                     for k, v in metadata_data.items() if v and v != "null"]
            if items:
                cols = st.columns(3)
                for i, (key, val) in enumerate(items):
                    with cols[i % 3]:
                        st.markdown(
                            f"<div style='background:#f8f9fc;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;'>"
                            f"<div style='color:#6b7280;font-size:0.75rem;text-transform:uppercase;'>{key}</div>"
                            f"<div style='color:#1f2430;font-weight:600;'>{val}</div></div>",
                            unsafe_allow_html=True,
                        )
            st.divider()

        # Suggested questions — clicking sends directly to chat
        if questions_data and questions_data.get("questions"):
            st.markdown("<div class='pa-label'>Suggested questions — click to ask</div>", unsafe_allow_html=True)
            for i, q in enumerate(questions_data["questions"]):
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    with st.spinner("Searching document..."):
                        result, error = api_post(
                            "chat",
                            json={"document_id": doc_id, "message": q, "include_history": True},
                        )
                    if error:
                        _show_error(error)
                    else:
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "citations": result.get("citations", []),
                        })
                        st.toast("Answer ready - open the Chat tab!")

        if not classify_data and not summary_data:
            st.info("No insights yet. Run an analysis above.")

        if st.button("Re-analyse", help="Clear cached results and run again"):
            st.session_state.insights_loaded = False
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB: Tables
# ════════════════════════════════════════════════════════════════════════════
with tab_tables:
    if st.session_state.tables_data is None:
        if st.button("Extract tables", type="primary", use_container_width=True):
            with st.spinner("Extracting tables from PDF..."):
                result, error = api_get(f"tables/{doc_id}")
            if error:
                _show_error(error)
            else:
                st.session_state.tables_data = result
                st.rerun()
    else:
        tables_data = st.session_state.tables_data
        count = tables_data.get("table_count", 0)
        if count == 0:
            st.info("No tables found in this document.")
        else:
            st.markdown(f"**{count} table(s) found**")
            for t in tables_data.get("tables", []):
                with st.expander(f"{t.get('caption', 'Table')} - page {t.get('page', '?')}"):
                    st.markdown(t.get("markdown", "_No content_"))
        if st.button("Re-extract", help="Re-run table extraction"):
            st.session_state.tables_data = None
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB: Chat (single document)
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.caption("Answers are grounded in this document only, with page-level citations and relevance scores.")

    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])} citation(s))"):
                    for c in msg["citations"]:
                        render_citation(c)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about this document...", label_visibility="collapsed")
        col_send, col_clear = st.columns([5, 1])
        send = col_send.form_submit_button("Send", use_container_width=True, type="primary")
        clear = col_clear.form_submit_button("Clear", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Searching document and generating answer..."):
            result, error = api_post(
                "chat",
                json={"document_id": doc_id, "message": user_input, "include_history": True},
            )
        if error:
            st.session_state.chat_history.pop()
            _show_error(error)
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result.get("citations", []),
            })
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB: Multi-PDF Chat
# ════════════════════════════════════════════════════════════════════════════
with tab_multi:
    st.caption("Query across multiple uploaded documents simultaneously.")

    col_r, _ = st.columns([1, 4])
    if col_r.button("Refresh list"):
        docs, _ = api_get("documents")
        if docs:
            st.session_state.all_docs = docs

    all_docs = st.session_state.all_docs
    if not all_docs:
        docs, _ = api_get("documents")
        if docs:
            st.session_state.all_docs = docs
            all_docs = docs

    if not all_docs:
        st.info("Upload at least one PDF to use multi-doc chat.")
    else:
        doc_options = {
            f"{d.get('original_filename') or d['filename']} ({d.get('page_count', 0)} pages)": d["id"]
            for d in all_docs
        }
        selected_labels = st.multiselect(
            "Select documents to query",
            options=list(doc_options.keys()),
            default=[list(doc_options.keys())[0]] if doc_options else [],
        )
        selected_ids = [doc_options[l] for l in selected_labels]
        st.session_state.selected_doc_ids = selected_ids

        if selected_ids:
            for msg in st.session_state.multi_chat_history:
                with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        with st.expander(f"Sources ({len(msg['citations'])} citation(s))"):
                            for c in msg["citations"]:
                                render_citation(c, show_doc=True)

            with st.form("multi_chat_form", clear_on_submit=True):
                multi_input = st.text_input("Ask across selected documents...", label_visibility="collapsed")
                col_ms, col_mc = st.columns([5, 1])
                msend = col_ms.form_submit_button("Send", use_container_width=True, type="primary")
                mclear = col_mc.form_submit_button("Clear", use_container_width=True)

            if mclear:
                st.session_state.multi_chat_history = []
                st.rerun()

            if msend and multi_input.strip():
                st.session_state.multi_chat_history.append({"role": "user", "content": multi_input})
                with st.spinner(f"Searching {len(selected_ids)} document(s)..."):
                    result, error = api_post(
                        "chat/multi",
                        json={"document_ids": selected_ids, "message": multi_input},
                    )
                if error:
                    st.session_state.multi_chat_history.pop()
                    _show_error(error)
                else:
                    st.session_state.multi_chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result.get("citations", []),
                    })
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB: Compare
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.caption("Compare two documents - similarities, differences, and a recommendation.")

    all_docs = st.session_state.all_docs
    if not all_docs:
        docs, _ = api_get("documents")
        if docs:
            st.session_state.all_docs = docs
            all_docs = docs

    if len(all_docs) < 2:
        st.info("Upload at least two PDFs to use document comparison.")
    else:
        doc_options = {
            f"{d.get('original_filename') or d['filename']}": d["id"]
            for d in all_docs
        }
        labels = list(doc_options.keys())

        col_a, col_b = st.columns(2)
        doc_a_label = col_a.selectbox("Document A", options=labels, index=0, key="cmp_a")
        doc_b_label = col_b.selectbox("Document B", options=labels, index=min(1, len(labels) - 1), key="cmp_b")

        if st.button("Compare documents", type="primary", use_container_width=True):
            if doc_options[doc_a_label] == doc_options[doc_b_label]:
                st.warning("Please select two different documents.")
            else:
                with st.spinner("Comparing documents..."):
                    result, error = api_post(
                        "compare",
                        json={
                            "document_id_a": doc_options[doc_a_label],
                            "document_id_b": doc_options[doc_b_label],
                        },
                    )
                if error:
                    _show_error(error)
                else:
                    st.session_state.compare_result = result

        if st.session_state.compare_result:
            res = st.session_state.compare_result
            st.divider()
            col_sim, col_diff = st.columns(2)
            with col_sim:
                st.markdown("#### Similarities")
                for s in res.get("similarities", []):
                    st.markdown(f"- {s}")
            with col_diff:
                st.markdown("#### Differences")
                for d in res.get("differences", []):
                    st.markdown(f"- {d}")
            st.divider()
            st.markdown("#### Recommendation")
            st.info(res.get("recommendation", ""))
            with st.expander("Detailed comparison"):
                st.write(res.get("detailed_comparison", ""))
