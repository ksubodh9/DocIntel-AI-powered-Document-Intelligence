"""
PDF Agent - Streamlit Frontend
Document Insights Dashboard + Conversational Chat Interface.

Redesigned UI:
  - Native chat bubbles (st.chat_message) with inline source citations
  - Insights dashboard: classification, summaries, topic/keyword/entity chips
  - Backend status that reflects "busy" vs "offline" (won't false-alarm while a
    long request is running, now that the backend no longer blocks /health)
  - Cleaner layout, consistent design tokens, accessible empty/landing states
"""

import os
import time

import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# API_BASE comes from the environment so it works both locally and in Docker.
#   Local:  http://localhost:8000/api/v1   (default)
#   Docker: http://backend:8000/api/v1     (set in docker-compose)
API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api/v1")
HEALTH_URL = API_BASE.replace("/api/v1", "") + "/health"

# Per-endpoint client timeouts (seconds). Local Ollama on CPU can be slow.
TIMEOUTS = {
    "upload": 600,     # extraction + embedding (first run loads the model)
    "classify": 120,
    "summarize": 300,  # multiple LLM calls
    "chat": 180,
    "default": 60,
}

st.set_page_config(
    page_title="PDF Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "doc_id": None,
    "doc_info": None,
    "chat_history": [],
    "insights_loaded": False,
    "classify_data": None,
    "summary_data": None,
    "questions_data": None,
    "pending_question": None,
    "active_tab": "insights",
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ──────────────────────────────────────────────────────────────────────────────
# Styling (design tokens kept in one place)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --pa-primary: #4c6ef5;
    --pa-primary-soft: #eef1ff;
    --pa-ink: #1f2430;
    --pa-muted: #6b7280;
    --pa-border: #e6e8ee;
}

/* Tighten the default top padding */
.block-container { padding-top: 2.2rem; }

/* Hero card on the landing screen */
.pa-hero {
    background: linear-gradient(135deg, #4c6ef5 0%, #7048e8 100%);
    color: #fff;
    border-radius: 18px;
    padding: 34px 38px;
    margin-bottom: 22px;
}
.pa-hero h1 { color: #fff; margin: 0 0 6px 0; font-size: 2.0rem; }
.pa-hero p  { color: #e8ebff; margin: 0; font-size: 1.02rem; }

/* Feature tiles */
.pa-tile {
    border: 1px solid var(--pa-border);
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
    background: #fff;
}
.pa-tile h4 { margin: 0 0 6px 0; color: var(--pa-ink); font-size: 1.02rem; }
.pa-tile p  { margin: 0; color: var(--pa-muted); font-size: 0.9rem; }
.pa-tile .pa-emoji { font-size: 1.5rem; }

/* Status badge */
.pa-status { display:flex; align-items:center; gap:8px; font-size:0.9rem; font-weight:600; }
.pa-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.pa-dot.ok   { background:#37b24d; box-shadow:0 0 0 3px rgba(55,178,77,.18); }
.pa-dot.off  { background:#e03131; box-shadow:0 0 0 3px rgba(224,49,49,.18); }

/* Chips */
.chip {
    display:inline-block; padding:3px 11px; border-radius:14px; margin:3px 3px 3px 0;
    font-size:0.8rem; font-weight:600;
}
.chip-topic   { background:#e7f5ff; color:#1971c2; }
.chip-keyword { background:#ebfbee; color:#2f9e44; }
.chip-entity  { background:#fff4e6; color:#e8590c; }

/* Citation card */
.cite {
    background:#f8f9fc; border:1px solid var(--pa-border); border-left:3px solid var(--pa-primary);
    border-radius:8px; padding:10px 13px; margin:6px 0; font-size:0.83rem; color:#3a3f4b;
}
.cite b { color: var(--pa-primary); }

/* Section label */
.pa-label { font-size:0.78rem; letter-spacing:.04em; text-transform:uppercase;
            color:var(--pa-muted); font-weight:700; margin-bottom:6px; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────
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
            return None, (
                "Cannot reach the backend. It may still be starting — wait ~30s "
                "and try again, or check that the backend container is running."
            )
        except requests.exceptions.Timeout:
            return None, (
                f"Request timed out after {timeout}s. Local models on CPU can be "
                "slow on first use. Consider a smaller model (e.g. OLLAMA_MODEL=phi3:mini)."
            )
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            return None, detail


def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}/{endpoint}", timeout=60)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the backend."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail


def backend_online() -> bool:
    # Generous timeout: the backend no longer blocks /health during long jobs,
    # but a cold start or a busy host can still take a few seconds to respond.
    try:
        return requests.get(HEALTH_URL, timeout=10).status_code == 200
    except Exception:
        return False


def reset_document_state():
    st.session_state.update(
        {
            "doc_id": None,
            "doc_info": None,
            "chat_history": [],
            "insights_loaded": False,
            "classify_data": None,
            "summary_data": None,
            "questions_data": None,
            "pending_question": None,
        }
    )


def chips(items, css_class) -> str:
    if not items:
        return "<span style='color:#9aa0ac;font-size:0.85rem;'>None found</span>"
    return " ".join(f"<span class='chip {css_class}'>{i}</span>" for i in items)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Agent")
    st.caption("AI-powered document intelligence")

    online = backend_online()
    dot = "ok" if online else "off"
    label = "Backend connected" if online else "Backend offline"
    st.markdown(
        f"<div class='pa-status'><span class='pa-dot {dot}'></span>{label}</div>",
        unsafe_allow_html=True,
    )
    if not online:
        st.caption("Start it with: `uvicorn app.main:app --reload`")

    st.divider()

    st.markdown("#### Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Max 50 MB. Text-based PDFs only (scanned PDFs not yet supported).",
        label_visibility="collapsed",
    )
    if uploaded_file and st.button("Process PDF", use_container_width=True, type="primary"):
        with st.spinner("Uploading & indexing… first run also loads the embedding model."):
            result, error = api_post(
                "upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            )
        if error:
            st.error(error)
        elif result.get("status") == "error":
            st.error(result.get("message", "Processing failed."))
        else:
            reset_document_state()
            st.session_state.doc_id = result["document_id"]
            st.session_state.doc_info = result
            st.success(f"Indexed {result.get('page_count', '?')} pages.")
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


# ──────────────────────────────────────────────────────────────────────────────
# Landing screen (no document yet)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.markdown(
        """
        <div class="pa-hero">
            <h1>Understand any PDF in seconds</h1>
            <p>Upload a document to classify it, summarize it, extract topics, and chat with it — every answer cited back to the page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    tiles = [
        (c1, "🧭", "Insights dashboard", "Automatic classification, summaries, topics, keywords and entities."),
        (c2, "💬", "Chat with citations", "Ask in natural language. Answers are grounded in the document and cite the page."),
        (c3, "🔎", "RAG retrieval", "Semantic chunk search over a local vector store — no data leaves your machine."),
    ]
    for col, emoji, title, body in tiles:
        with col:
            st.markdown(
                f"<div class='pa-tile'><div class='pa-emoji'>{emoji}</div>"
                f"<h4>{title}</h4><p>{body}</p></div>",
                unsafe_allow_html=True,
            )
    st.markdown("")
    st.info("⬅️  Upload a PDF from the sidebar to get started.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Document loaded → Insights + Chat
# ──────────────────────────────────────────────────────────────────────────────
doc_id = st.session_state.doc_id
tab_insights, tab_chat = st.tabs(["📊  Insights", "💬  Chat"])

# ── Insights ──────────────────────────────────────────────────────────────────
with tab_insights:
    if not st.session_state.insights_loaded:
        st.markdown("#### Generate insights")
        st.caption("Classify the document and produce summaries, topics and suggested questions.")
        col_a, col_b = st.columns(2)
        run_classify = col_a.button("Classify only", use_container_width=True)
        run_full = col_b.button("Analyze & summarize", use_container_width=True, type="primary")

        if run_classify or run_full:
            progress = st.progress(0, "Starting…")
            progress.progress(20, "Classifying…")
            classify_data, err = api_post(f"classify/{doc_id}")
            if err:
                st.error(f"Classification failed: {err}")

            summary_data = None
            if run_full:
                progress.progress(55, "Summarizing… (local models on CPU can take a few minutes)")
                summary_data, err = api_post(f"summarize/{doc_id}")
                if err:
                    st.error(f"Summarization failed: {err}")

            progress.progress(90, "Fetching suggested questions…")
            questions_data, _ = api_get(f"questions/{doc_id}")
            progress.progress(100, "Done")

            st.session_state.classify_data = classify_data
            st.session_state.summary_data = summary_data
            st.session_state.questions_data = questions_data
            st.session_state.insights_loaded = True
            st.rerun()
    else:
        classify_data = st.session_state.classify_data
        summary_data = st.session_state.summary_data
        questions_data = st.session_state.questions_data

        if classify_data:
            c1, c2 = st.columns([2, 1])
            c1.metric("Document type", classify_data.get("document_type", "Unknown"))
            conf = classify_data.get("confidence", 0) or 0
            c2.metric("Confidence", f"{conf:.0%}")
            c2.progress(min(max(conf, 0.0), 1.0))
            st.divider()

        if summary_data:
            st.markdown("<div class='pa-label'>Quick summary</div>", unsafe_allow_html=True)
            st.write(summary_data.get("short_summary", "") or "_No summary._")
            with st.expander("Detailed analysis"):
                st.write(summary_data.get("detailed_summary", "") or "_No detailed summary._")

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

        if questions_data and questions_data.get("questions"):
            st.markdown("<div class='pa-label'>Suggested questions</div>", unsafe_allow_html=True)
            st.caption("Click a question to ask it in the chat tab.")
            for i, q in enumerate(questions_data["questions"]):
                if st.button(f"💡 {q}", key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.toast("Question queued — open the Chat tab.")

        if not classify_data and not summary_data:
            st.info("No insights yet. Run an analysis above.")

        if st.button("↻ Re-analyze", help="Clear cached results and run again"):
            st.session_state.insights_loaded = False
            st.rerun()

# ── Chat ──────────────────────────────────────────────────────────────────────
with tab_chat:
    st.caption("Answers are grounded in the document only, with page-level citations.")

    # Render history as native chat bubbles
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"Sources · {len(msg['citations'])} citation(s)"):
                    for c in msg["citations"]:
                        st.markdown(
                            f"<div class='cite'><b>Page {c.get('page', '?')}</b><br>{c.get('text','')}</div>",
                            unsafe_allow_html=True,
                        )

    # Input. (A form is used instead of st.chat_input because chat_input cannot
    # be nested inside tabs.) A pending question from the Insights tab pre-fills it.
    prefill = st.session_state.pop("pending_question", None) or ""
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            value=prefill,
            placeholder="e.g. What is the main finding of this document?",
            label_visibility="collapsed",
        )
        col_send, col_clear = st.columns([5, 1])
        send = col_send.form_submit_button("Send", use_container_width=True, type="primary")
        clear = col_clear.form_submit_button("Clear", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Searching the document and generating an answer…"):
            result, error = api_post(
                "chat",
                json={"document_id": doc_id, "message": user_input, "include_history": True},
            )
        if error:
            st.session_state.chat_history.pop()  # roll back the unanswered question
            st.error(f"Chat error: {error}")
        else:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                }
            )
            st.rerun()
