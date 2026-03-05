import os
import base64
import time
import requests
import streamlit as st
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="SmartCache RAG™",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ─────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("current_question", ""),
    ("ingested_files", []),
    ("cache_hit_counts", {"exact": 0, "semantic": 0, "retrieval": 0, "miss": 0}),
    ("total_queries", 0),
    ("total_latency_ms", 0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    .bot-title {
        text-align: center; 
        font-size: 2.8rem; 
        font-weight: 900;
        color: #10b981;
        background: linear-gradient(120deg, #10b981, #14b8a6, #06b6d4, #0ea5e9);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center; color: #475569; font-size: 1rem; margin-bottom: 1.5rem;
        font-weight: 500; letter-spacing: 0.3px;
    }

    /* Cache tier badges */
    .badge { display: inline-block; padding: 4px 12px; border-radius: 16px;
             font-size: 0.75rem; font-weight: 700; margin-left: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .badge-t1  { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; }
    .badge-t2  { background: linear-gradient(135deg, #ccfbf1, #99f6e4); color: #0f766e; }
    .badge-t3  { background: linear-gradient(135deg, #e0f2fe, #bae6fd); color: #075985; }
    .badge-miss{ background: linear-gradient(135deg, #ffe4e6, #fecdd3); color: #be123c; }

    /* Chat bubbles */
    .user-msg {
        background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 16px; padding: 12px 18px;
        margin: 8px 0; max-width: 75%; float: left; clear: both;
        box-shadow: 0 2px 4px rgba(20, 184, 166, 0.1);
    }
    .bot-msg {
        background: linear-gradient(135deg, #10b981 0%, #14b8a6 50%, #06b6d4 100%);
        color: white; border-radius: 16px; padding: 12px 18px;
        margin: 8px 0; max-width: 85%; float: right; clear: both;
        box-shadow: 0 3px 8px rgba(16, 185, 129, 0.4);
    }
    .meta-row { font-size: 0.72rem; color: #94a3b8; margin-top: 4px; }
    .clearfix::after { content:""; display:table; clear:both; }

    /* Source cards */
    .src-card {
        background: #f0fdfa; border-left: 4px solid #14b8a6;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        font-size: 0.85rem; box-shadow: 0 1px 3px rgba(20, 184, 166, 0.1);
    }
    .src-title { font-weight: 600; color: #0f172a; }
    .src-section { color: #0d9488; font-size: 0.78rem; font-weight: 600; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        border: 1px solid #ccfbf1; border-radius: 12px; padding: 16px 18px;
        text-align: center; box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }
    .metric-val { font-size: 1.9rem; font-weight: 900; color: #10b981; }
    .metric-lbl { font-size: 0.78rem; color: #64748b; margin-top: 4px; font-weight: 600; }

    .sample-btn { width: 100%; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

TIER_BADGE = {
    1:    ('<span class="badge badge-t1">⚡ Tier-1 Exact</span>', "⚡ Exact Cache Hit"),
    2:    ('<span class="badge badge-t2">🌊 Tier-2 Semantic</span>', "🌊 Semantic Cache Hit"),
    3:    ('<span class="badge badge-t3">📦 Tier-3 Retrieval</span>', "📦 Retrieval Cache Hit"),
    None: ('<span class="badge badge-miss">🔴 Cache Miss</span>', "🔴 Full Pipeline"),
}


def _tier_color(tier):
    return {1: "#065f46", 2: "#0f766e", 3: "#075985", None: "#be123c"}.get(tier, "#be123c")


def _api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ SmartCache RAG™")
    st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-top: -10px;'>Intelligent Multi-Tier Caching</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Backend status
    ok, hdata = _api_health()
    if ok:
        redis_status = hdata.get("redis", "unknown")
        st.success(f"✅ API Connected | Redis: {redis_status}")
    else:
        st.error("❌ API not reachable — start `main.py` first")

    st.markdown("---")
    st.markdown("### 📁 Document Ingestion")

    uploaded = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        help="Files are saved to the `data/` directory and ingested into Pinecone.",
    )

    strategy = st.selectbox(
        "Chunking Strategy",
        ["parent_child", "structure_recursive", "pdf_rich"],
        format_func=lambda x: {
            "parent_child":         "🌳 Parent-Child",
            "structure_recursive":  "📐 Structure + Recursive",
            "pdf_rich":             "🖼️  PDF Rich (images + tables)",
        }.get(x, x),
        help=(
            "Parent-Child: small children indexed, full parent sent to LLM.\n"
            "Structure+Recursive: splits by Markdown headers.\n"
            "PDF Rich: extracts images via Groq vision + table-aware chunking."
        ),
    )

    if uploaded:
        st.info(f"📄 {uploaded.name}")
        if st.button("🚀 Ingest Document"):
            with st.spinner("Ingesting…"):
                content = uploaded.read()
                dest = Path(DOCS_DIR) / uploaded.name
                dest.write_bytes(content)
                try:
                    r = requests.post(
                        f"{API_BASE}/ingest",
                        json={"filepath": str(dest), "strategy": strategy},
                        timeout=120,
                    )
                    if r.ok:
                        resp = r.json()
                        breakdown = resp.get("breakdown")
                        chunk_info = f"{resp['chunk_count']} chunks upserted"
                        parent_info = (
                            f" ({resp['parent_count']} parents cached)"
                            if resp['parent_count'] else ""
                        )
                        bd_info = ""
                        if breakdown:
                            bd_info = (
                                f"  \n📊 text={breakdown.get('text',0)}, "
                                f"table={breakdown.get('table',0)}, "
                                f"image={breakdown.get('image',0)}"
                            )
                        st.success(f"✅ {chunk_info}{parent_info}{bd_info}")
                        if uploaded.name not in st.session_state["ingested_files"]:
                            st.session_state["ingested_files"].append(uploaded.name)
                    else:
                        st.error(f"❌ {r.status_code}: {r.text[:200]}")
                except Exception as e:
                    st.error(f"❌ {e}")

    if st.session_state["ingested_files"]:
        st.markdown("**Ingested files:**")
        for fn in st.session_state["ingested_files"]:
            st.markdown(f"  • {fn}")

    st.markdown("---")
    st.markdown("### ⚙️ Query Settings")
    use_reranker = st.checkbox("🔄 Enable Reranker", value=True)
    debug_mode = st.checkbox("🐛 Debug Mode (show raw chunks)", value=False)

    st.markdown("---")
    st.markdown("### 🗄️ Cache Management")

    # ── Live Redis tier stats ─────────────────────────────────────────────
    try:
        sr = requests.get(f"{API_BASE}/cache/stats", timeout=3)
        if sr.ok:
            sd = sr.json()
            redis_tiers = [
                ("⚡ Exact",     sd.get("tier1_exact",     {}).get("entries", "—")),
                ("🌊 Semantic",  sd.get("tier2_semantic",  {}).get("entries", "—")),
                ("📦 Retrieval", sd.get("tier3_retrieval", {}).get("entries", "—")),
                ("👪 Parents",   sd.get("parent_cache",    {}).get("entries", "—")),
            ]
            cols = st.columns(2)
            for i, (label, cnt) in enumerate(redis_tiers):
                cols[i % 2].metric(label, cnt)
    except Exception:
        st.caption("Redis stats unavailable")

    col_a, col_b = st.columns(2)
    if col_a.button("📊 Refresh Stats"):
        st.session_state["_refresh_stats"] = True
    if col_b.button("🗑️ Clear Cache"):
        try:
            r = requests.delete(f"{API_BASE}/cache/clear", timeout=10)
            if r.ok:
                st.success("Cache cleared!")
                st.session_state["cache_hit_counts"] = {
                    "exact": 0, "semantic": 0, "retrieval": 0, "miss": 0
                }
        except Exception as e:
            st.error(str(e))


# ── Main ───────────────────────────────────────────────────────────────────

# Centered title: place the title and subtitle in the middle column for visual balance
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    # Use an HTML wrapper to ensure the header appears centered while keeping the bot-title styling
    st.markdown(
        '<div style="text-align:center">'
        '<h1 class="bot-title">⚡ SmartCache RAG™</h1>'
        '<p class="subtitle">🎯 Advanced Multi-Strategy Chunking · 🚀 Lightning-Fast 3-Tier Intelligent Cache · 🔮 AI-Powered Retrieval</p>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Cache analytics strip ─────────────────────────────────────────────────
counts = st.session_state["cache_hit_counts"]
total_q = st.session_state["total_queries"]
total_lat = st.session_state["total_latency_ms"]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.markdown(
    f'<div class="metric-card"><div class="metric-val" style="color:#065f46">'
    f'{counts["exact"]}</div><div class="metric-lbl">⚡ Exact Hits</div></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    f'<div class="metric-card"><div class="metric-val" style="color:#0f766e">'
    f'{counts["semantic"]}</div><div class="metric-lbl">🌊 Semantic Hits</div></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    f'<div class="metric-card"><div class="metric-val" style="color:#075985">'
    f'{counts["retrieval"]}</div><div class="metric-lbl">📦 Retrieval Hits</div></div>',
    unsafe_allow_html=True,
)
m4.markdown(
    f'<div class="metric-card"><div class="metric-val" style="color:#be123c">'
    f'{counts["miss"]}</div><div class="metric-lbl">🔴 Full Pipeline</div></div>',
    unsafe_allow_html=True,
)
m5.markdown(
    f'<div class="metric-card"><div class="metric-val">{total_q}</div>'
    f'<div class="metric-lbl">Total Queries</div></div>',
    unsafe_allow_html=True,
)
avg_lat = round(total_lat / max(total_q, 1), 1)
m6.markdown(
    f'<div class="metric-card"><div class="metric-val">{avg_lat} ms</div>'
    f'<div class="metric-lbl">Avg Latency</div></div>',
    unsafe_allow_html=True,
)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Live cache tier breakdown (Redis) ─────────────────────────────────────
if st.session_state.get("_refresh_stats"):
    st.session_state.pop("_refresh_stats")
    try:
        sr = requests.get(f"{API_BASE}/cache/stats", timeout=5)
        if sr.ok:
            sd = sr.json()
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, key, label, color in [
                (sc1, "tier1_exact", "Tier-1 Exact entries", "#065f46"),
                (sc2, "tier2_semantic", "Tier-2 Semantic entries", "#0f766e"),
                (sc3, "tier3_retrieval", "Tier-3 Retrieval entries", "#075985"),
                (sc4, "parent_cache", "Parent chunks cached", "#10b981"),
            ]:
                entries = sd.get(key, {}).get("entries", "N/A")
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val" style="color:{color}">{entries}</div>'
                    f'<div class="metric-lbl">{label}</div></div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.warning(f"Could not fetch cache stats: {e}")

# ── CHAT AREA ──────────────────────────────────────────────────────────────
chat_area = st.container()

SAMPLES = [
    "Summarize the key points from the document",
    "What are the main findings and conclusions?",
    "List any recommendations mentioned",
    "What are the most important facts in the document?",
]

with chat_area:
    if not st.session_state["chat_history"]:
        st.markdown("### 💡 Try asking:")
        for idx, q in enumerate(SAMPLES):
            if st.button(q, key=f"sample_{idx}"):
                st.session_state["current_question"] = q
                st.rerun()
    else:
        for item in st.session_state["chat_history"]:
            question = item["question"]
            answer   = item["answer"]
            sources  = item.get("sources", [])
            tier     = item.get("cache_tier")
            sim      = item.get("cache_similarity")
            lat      = item.get("total_latency_ms", 0)
            t3_regen = item.get("tier3_regen", False)
            raw_chunks = item.get("raw_chunks", [])

            badge_html, badge_label = TIER_BADGE.get(tier, TIER_BADGE[None])

            # ── User bubble ──────────────────────────────────────────────
            st.markdown(
                f'<div class="user-msg">👤 {question}</div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

            # ── Bot bubble ───────────────────────────────────────────────
            tier3_note = " (chunks reused, answer re-generated)" if t3_regen else ""
            sim_note = f" · similarity {sim:.3f}" if sim else ""
            meta = f"{badge_label}{sim_note}{tier3_note} · {lat} ms"

            st.markdown(
                f'<div class="bot-msg">{answer}'
                f'<div class="meta-row">{meta}</div></div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(badge_html, unsafe_allow_html=True)

            # ── Expandable panels ────────────────────────────────────────
            exp_cols = st.columns(3 if debug_mode and raw_chunks else 2)

            if sources:
                with exp_cols[0]:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for src in sources:
                            section = src.get("section", "")
                            st.markdown(
                                f'<div class="src-card">'
                                f'<div class="src-title">{src["citation"]} {src["source"]}</div>'
                                + (f'<div class="src-section">§ {section}</div>' if section else "")
                                + f'<div style="color:#475569;font-size:.82rem;margin-top:4px">'
                                  f'Score: {src["score"]:.4f}</div>'
                                f'<div style="color:#94a3b8;font-size:.8rem;margin-top:4px">'
                                f'{src["chunk_preview"]}…</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            with exp_cols[1]:
                with st.expander("ℹ️ Pipeline Info"):
                    tier_names = {1: "⚡ Exact Cache", 2: "🌊 Semantic Cache",
                                  3: "📦 Retrieval Cache", None: "🔴 Full Pipeline"}
                    st.write(f"**Cache Tier:** {tier_names.get(tier, 'Unknown')}")
                    if sim:
                        st.write(f"**Similarity:** {sim:.4f}")
                    if t3_regen:
                        st.info("Chunks reused from cache — only LLM called.")
                    st.write(f"**Latency:** {lat} ms")

            if debug_mode and raw_chunks and len(exp_cols) > 2:
                with exp_cols[2]:
                    with st.expander(f"🔬 Raw Chunks ({len(raw_chunks)})"):
                        for ci, ch in enumerate(raw_chunks, 1):
                            st.markdown(
                                f"**[{ci}]** `{ch.get('id','?')}` "
                                f"score={ch.get('score',0):.4f}  \n"
                                f"{ch.get('chunk_text','')[:200]}…"
                            )

            st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([7, 1])
    user_input = c1.text_input(
        "Ask a question",
        value=st.session_state.get("current_question", ""),
    )
    submitted = c2.form_submit_button("Send 📤")

if submitted and user_input.strip():
    st.session_state["current_question"] = ""
    payload = {
        "query": user_input.strip(),
        "use_reranker": use_reranker,
        "debug": debug_mode,
    }
    with st.spinner("🤔 Processing your query through the cache pipeline…"):
        try:
            r = requests.post(f"{API_BASE}/chat", json=payload, timeout=90)
            if r.ok:
                resp = r.json()

                # Update session metrics
                tier = resp.get("cache_tier")
                tier_name = resp.get("cache_tier_name") or "miss"
                st.session_state["cache_hit_counts"][tier_name] = (
                    st.session_state["cache_hit_counts"].get(tier_name, 0) + 1
                )
                st.session_state["total_queries"] += 1
                st.session_state["total_latency_ms"] += resp.get("total_latency_ms", 0)

                # Append to chat history
                st.session_state["chat_history"].append(
                    {
                        "question": user_input,
                        "answer": resp.get("answer", ""),
                        "sources": resp.get("sources", []),
                        "cache_hit": resp.get("cache_hit"),
                        "cache_tier": tier,
                        "cache_similarity": resp.get("cache_similarity"),
                        "tier3_regen": resp.get("tier3_regen", False),
                        "total_latency_ms": resp.get("total_latency_ms", 0),
                        "pipeline_steps": resp.get("pipeline_steps", []),
                        "raw_chunks": resp.get("raw_chunks", []),
                    }
                )
                st.rerun()
            else:
                st.error(f"API error {r.status_code}: {r.text[:300]}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach API. Make sure `python main.py` is running.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ── Clear chat ─────────────────────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.session_state["cache_hit_counts"] = {
            "exact": 0, "semantic": 0, "retrieval": 0, "miss": 0
        }
        st.session_state["total_queries"] = 0
        st.session_state["total_latency_ms"] = 0.0
        st.rerun()
