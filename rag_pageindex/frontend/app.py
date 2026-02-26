import os
import time
import base64
import json
import requests
import streamlit as st
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="PageIndex RAG",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌿",
)

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────

for key, default in [
    ("chat_history", []),
    ("current_question", ""),
    ("active_doc_id", None),
    ("active_filename", None),
    ("tree_data", None),
    ("doc_list", []),
    ("current_tab", "chat"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────
# CSS  –  teal / emerald palette, Inter font
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── layout ── */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── title ── */
.pi-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.pi-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── sample question buttons ── */
.sample-questions-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3rem 2rem;
    min-height: 280px;
}

/* ── chat bubbles ── */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 1rem 0.5rem;
    margin-bottom: 1rem;
}

.user-message {
    background: #e0f2fe;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-right: 18%;
    float: left;
    clear: both;
    max-width: 72%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    color: #0f172a;
    font-size: 0.95rem;
}

.bot-message {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    margin-left: 18%;
    float: right;
    clear: both;
    max-width: 72%;
    box-shadow: 0 2px 6px rgba(13,148,136,0.3);
    font-size: 0.95rem;
}

.message-text { margin: 0; line-height: 1.6; }
.clearfix::after { content: ""; clear: both; display: table; }

/* ── node card ── */
.node-card {
    background: #f0fdfa;
    border-left: 4px solid #0d9488;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 6px;
    font-size: 0.88rem;
}

.node-id {
    font-family: 'JetBrains Mono', monospace;
    color: #0d9488;
    font-weight: 600;
    font-size: 0.82rem;
}

.node-title {
    font-weight: 600;
    color: #134e4a;
    margin-bottom: 4px;
}

.node-summary {
    color: #475569;
    font-size: 0.83rem;
    margin-top: 4px;
}

.node-preview {
    color: #64748b;
    font-size: 0.80rem;
    margin-top: 6px;
    border-top: 1px solid #ccfbf1;
    padding-top: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── thinking box ── */
.thinking-box {
    background: #fefce8;
    border-left: 4px solid #eab308;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #713f12;
    font-style: italic;
}

/* ── tree node ── */
.tree-node {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #0f172a;
}

.tree-node-id {
    color: #0d9488;
    font-weight: 600;
}

/* ── status badges ── */
.badge-completed  { color: #065f46; background: #d1fae5; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-processing { color: #92400e; background: #fef3c7; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-queued     { color: #1e3a5f; background: #dbeafe; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge-failed     { color: #7f1d1d; background: #fee2e2; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }

/* ── Streamlit button overrides ── */
.stButton>button {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.88; color: white; }

/* ── form submit ── */
.stFormSubmitButton>button {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def api_ok() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.ok
    except Exception:
        return False


def status_badge(status: str) -> str:
    cls = {
        "completed": "badge-completed",
        "processing": "badge-processing",
        "queued": "badge-queued",
        "failed": "badge-failed",
    }.get(status, "badge-queued")
    return f'<span class="{cls}">{status}</span>'


def render_tree_node(node: dict, depth: int = 0):
    """Recursively render a PageIndex tree node using st.expander."""
    nid = node.get("id", "?")
    title = node.get("title", nid)
    summary = node.get("summary", "")
    children = node.get("children", [])

    label = f"{'  ' * depth}📄 [{nid}] {title}"
    if children:
        with st.expander(label, expanded=(depth == 0)):
            if summary:
                st.markdown(f'<div class="node-summary">📝 {summary}</div>', unsafe_allow_html=True)
            for child in children:
                render_tree_node(child, depth + 1)
    else:
        with st.expander(label, expanded=False):
            if summary:
                st.markdown(f'<div class="node-summary">📝 {summary}</div>', unsafe_allow_html=True)
            text = node.get("text", "")
            if text:
                st.markdown(f'<div class="node-preview">{text[:500]}{"…" if len(text) > 500 else ""}</div>',
                            unsafe_allow_html=True)


def poll_until_ready(doc_id: str, placeholder) -> bool:
    """Poll /status until ready or failed. Returns True when ready."""
    for _ in range(60):  # max ~5 min
        try:
            r = requests.get(f"{API_BASE}/status/{doc_id}", timeout=10)
            if r.ok:
                data = r.json()
                status = data.get("status", "")
                ready = data.get("ready", False)
                placeholder.info(f"⏳ Status: **{status}** — waiting for index to be ready…")
                if ready or status == "completed":
                    placeholder.success("✅ Index ready!")
                    return True
                if status == "failed":
                    placeholder.error("❌ Indexing failed.")
                    return False
        except Exception:
            pass
        time.sleep(5)
    placeholder.warning("⚠️ Timed out waiting for index.")
    return False


def load_doc_list():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        if r.ok:
            st.session_state["doc_list"] = r.json().get("documents", [])
    except Exception:
        pass


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.markdown("## 🌿 PageIndex RAG")
st.sidebar.markdown("*Vectorless RAG powered by document trees*")
st.sidebar.markdown("---")

# ── upload ────────────────────────────────────
st.sidebar.markdown("### 📤 Upload PDF")
uploaded = st.sidebar.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded:
    st.sidebar.info(f"📄 {uploaded.name}")
    if st.sidebar.button("🚀 Index Document", use_container_width=True):
        with st.sidebar:
            status_ph = st.empty()
            status_ph.info("Uploading…")
            try:
                r = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded.name, uploaded.getbuffer(), "application/pdf")},
                    timeout=60,
                )
                if r.ok:
                    resp = r.json()
                    doc_id = resp["doc_id"]
                    status = resp.get("status", "")
                    status_ph.info(f"📥 Submitted: `{doc_id}`")

                    if status != "completed":
                        ready = poll_until_ready(doc_id, status_ph)
                    else:
                        ready = True
                        status_ph.success("✅ Already indexed!")

                    if ready:
                        st.session_state["active_doc_id"] = doc_id
                        st.session_state["active_filename"] = uploaded.name
                        st.session_state["tree_data"] = None
                        st.session_state["chat_history"] = []
                        load_doc_list()
                        st.rerun()
                else:
                    st.sidebar.error(f"❌ {r.status_code}: {r.text}")
            except Exception as e:
                st.sidebar.error(f"❌ {e}")

st.sidebar.markdown("---")

# ── indexed documents ─────────────────────────
st.sidebar.markdown("### 📚 Indexed Documents")

if st.sidebar.button("🔄 Refresh List", use_container_width=True):
    load_doc_list()

if not st.session_state["doc_list"]:
    load_doc_list()

completed_docs = [d for d in st.session_state["doc_list"] if d.get("status") == "completed"]

if not completed_docs:
    st.sidebar.caption("No completed documents yet. Upload one above.")
else:
    for doc in completed_docs:
        doc_id = doc.get("id", "")
        name = doc.get("cached_filename") or doc.get("name", doc_id)
        pages = doc.get("pageNum", "?")
        is_active = doc_id == st.session_state.get("active_doc_id")

        label = f"{'✅ ' if is_active else '📄 '}{name} ({pages}p)"
        if st.sidebar.button(label, key=f"sel_{doc_id}", use_container_width=True):
            st.session_state["active_doc_id"] = doc_id
            st.session_state["active_filename"] = name
            st.session_state["tree_data"] = None
            st.session_state["chat_history"] = []
            st.rerun()

st.sidebar.markdown("---")

# ── PDF preview ───────────────────────────────
if st.session_state.get("active_filename"):
    st.sidebar.markdown("### 📄 PDF Preview")
    fname = st.session_state["active_filename"]
    if st.sidebar.checkbox(f"Show `{fname}`", key="show_pdf"):
        try:
            r = requests.get(f"{API_BASE}/pdf/{fname}", timeout=10)
            if r.ok:
                b64 = base64.b64encode(r.content).decode()
                st.sidebar.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="420" type="application/pdf"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.sidebar.warning("PDF not available for preview.")
        except Exception as e:
            st.sidebar.error(str(e))
    st.sidebar.markdown("---")

# ── backend status ────────────────────────────
with st.sidebar.expander("🔌 Backend Status"):
    if api_ok():
        st.success(f"✅ Connected to `{API_BASE}`")
    else:
        st.error(f"❌ Cannot reach `{API_BASE}`\n\nStart with:\n```\nuv run uvicorn api:app --port 8001\n```")


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────

st.markdown('<h1 class="pi-title">🌿 PageIndex RAG</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="pi-subtitle">Vectorless retrieval · Tree-structured indexing · Groq LLM</p>',
    unsafe_allow_html=True,
)

active_doc_id = st.session_state.get("active_doc_id")
active_filename = st.session_state.get("active_filename")

if not active_doc_id:
    st.info("👈 Upload a PDF from the sidebar to get started, or select an already-indexed document.")
    st.stop()

st.markdown(
    f"**Active document:** `{active_filename}` &nbsp;·&nbsp; `{active_doc_id}`",
    unsafe_allow_html=True,
)

# ── tabs ──────────────────────────────────────
tab_chat, tab_tree = st.tabs(["💬 Chat", "🌲 Document Tree"])

# ═══════════════════════════════════════════════
# TAB: CHAT
# ═══════════════════════════════════════════════
with tab_chat:
    sample_questions = [
        "Summarize the key ideas in this document",
        "What are the main findings and conclusions?",
        "What methods or approaches are described?",
        "List any important equations or formulas",
    ]

    chat_area = st.container()

    with chat_area:
        if not st.session_state["chat_history"]:
            st.markdown("### 💡 Try asking:")
            for idx, q in enumerate(sample_questions):
                if st.button(q, key=f"sq_{idx}", use_container_width=True):
                    st.session_state["current_question"] = q
                    st.rerun()
        else:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            for i, item in enumerate(st.session_state["chat_history"]):
                question, answer, thinking, nodes = item

                # user bubble
                st.markdown(
                    f'<div class="user-message"><p class="message-text">{question}</p></div>'
                    '<div class="clearfix"></div>',
                    unsafe_allow_html=True,
                )

                # bot bubble
                st.markdown(
                    f'<div class="bot-message"><p class="message-text">{answer}</p></div>'
                    '<div class="clearfix"></div>',
                    unsafe_allow_html=True,
                )

                # expandable details
                col_think, col_nodes = st.columns(2)

                if thinking:
                    with col_think:
                        with st.expander(f"🧠 Reasoning", expanded=False):
                            st.markdown(
                                f'<div class="thinking-box">{thinking}</div>',
                                unsafe_allow_html=True,
                            )

                if nodes:
                    with col_nodes:
                        with st.expander(f"📌 Source Nodes ({len(nodes)})", expanded=False):
                            for node in nodes:
                                st.markdown(
                                    f"""<div class="node-card">
                                        <div class="node-id">{node['id']}</div>
                                        <div class="node-title">{node['title']}</div>
                                        <div class="node-summary">{node.get('summary','')}</div>
                                        <div class="node-preview">{node.get('text_preview','')[:250]}…</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                st.markdown("---")

            st.markdown('</div>', unsafe_allow_html=True)

    # ── input form ──────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([7, 1])
        with col_in:
            user_input = st.text_input(
                "Ask a question…",
                value=st.session_state.get("current_question", ""),
                placeholder="Ask anything about the document…",
                label_visibility="collapsed",
            )
        with col_btn:
            submit = st.form_submit_button("Send ➤", use_container_width=True)

        if submit and user_input:
            st.session_state["current_question"] = ""

            with st.spinner("🌿 Searching document tree…"):
                try:
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={"doc_id": active_doc_id, "question": user_input},
                        timeout=120,
                    )
                    if r.ok:
                        resp = r.json()
                        st.session_state["chat_history"].append((
                            user_input,
                            resp.get("answer", "No answer generated."),
                            resp.get("thinking", ""),
                            resp.get("selected_nodes", []),
                        ))
                        st.rerun()
                    else:
                        st.error(f"❌ API error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ═══════════════════════════════════════════════
# TAB: TREE VIEWER
# ═══════════════════════════════════════════════
with tab_tree:
    st.markdown("### 🌲 Document Tree")
    st.caption("Showing the hierarchical structure PageIndex built for this document.")

    col_fetch, col_fmt = st.columns([3, 1])
    with col_fetch:
        fetch_tree = st.button("📥 Load / Refresh Tree", use_container_width=True)
    with col_fmt:
        raw_view = st.checkbox("Raw JSON", value=False)

    if fetch_tree or st.session_state.get("tree_data"):
        if fetch_tree or not st.session_state["tree_data"]:
            with st.spinner("Fetching tree…"):
                try:
                    r = requests.get(f"{API_BASE}/tree/{active_doc_id}", timeout=30)
                    if r.ok:
                        st.session_state["tree_data"] = r.json()
                    else:
                        st.error(f"❌ {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(str(e))

        tree = st.session_state.get("tree_data")
        if tree:
            if raw_view:
                st.json(tree)
            else:
                # Render tree recursively
                if isinstance(tree, list):
                    for root in tree:
                        render_tree_node(root, depth=0)
                elif isinstance(tree, dict):
                    render_tree_node(tree, depth=0)
                else:
                    st.json(tree)
    else:
        st.info("Click **Load / Refresh Tree** to visualise the document structure.")
