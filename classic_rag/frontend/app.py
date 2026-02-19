import os
import base64
import requests
import streamlit as st
from pathlib import Path

# Configuration
API_BASE = "http://127.0.0.1:8000"
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Classic RAG Bot", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""
if 'ingested_files' not in st.session_state:
    st.session_state['ingested_files'] = []
if 'preview_file' not in st.session_state:
    st.session_state['preview_file'] = None

# Custom CSS for modern chat interface
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Bot title styling */
    .bot-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sample question boxes */
    .sample-questions-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 2rem;
        min-height: 300px;
    }
    
    .sample-question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 15px 25px;
        margin: 8px;
        color: white;
        font-weight: 500;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        min-width: 350px;
    }
    
    .sample-question-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    /* Chat bubbles */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: #e3f2fd;
        border-radius: 18px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-right: 20%;
        float: left;
        clear: both;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-left: 20%;
        float: right;
        clear: both;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .message-text {
        margin: 0;
        line-height: 1.5;
    }
    
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Input box at bottom */
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 2px solid #e0e0e0;
        z-index: 100;
    }
    
    /* Source cards */
    .source-card {
        background: #f5f5f5;
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    
    .source-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 4px;
    }
    
    .source-score {
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar: upload + options
st.sidebar.markdown("### 📁 Document Management")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"]) 

if uploaded_file:
    st.sidebar.info(f"📄 Selected: {uploaded_file.name}")
    if st.sidebar.button("🚀 Upload & Ingest", use_container_width=True):
        with st.spinner("Processing document..."):
            # Save file
            filename = uploaded_file.name
            save_path = os.path.join(DOCS_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Ingest via API
            try:
                r = requests.post(f"{API_BASE}/ingest", json={"filepath": save_path}, timeout=60)
                if r.ok:
                    resp = r.json()
                    st.sidebar.success(f"✅ Ingested {resp.get('chunks', 0)} chunks!")
                    st.session_state['ingested_files'].append(filename)
                else:
                    st.sidebar.error(f"❌ Error: {r.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ Connection error: {str(e)}")

if st.session_state.get('ingested_files'):
    st.sidebar.markdown("### 📚 Ingested Files")
    for f in st.session_state['ingested_files']:
        col_a, col_b = st.sidebar.columns([3, 1])
        col_a.markdown(f"📄 {f}")
        if col_b.button("👁️", key=f"preview_{f}", help="Preview"):
            st.session_state['preview_file'] = f

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Options")
use_reranker = st.sidebar.checkbox("🔄 Use Reranker", value=True)
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Document Preview")

# Preview uploaded file
if uploaded_file:
    if st.sidebar.checkbox("Preview Uploaded File", key="preview_upload"):
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="400" type="application/pdf"></iframe>'
            st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

# Preview ingested file
if st.session_state.get('preview_file'):
    preview_path = os.path.join(DOCS_DIR, st.session_state['preview_file'])
    if os.path.exists(preview_path):
        st.sidebar.markdown(f"**Previewing:** {st.session_state['preview_file']}")
        with open(preview_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="400" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
        if st.sidebar.button("Close Preview"):
            st.session_state['preview_file'] = None
            st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("🔌 Backend Status"):
    try:
        r = requests.get(f"{API_BASE}/docs", timeout=3)
        st.success("✅ Connected")
    except:
        st.error("❌ Disconnected")

# Main content area
st.markdown('<h1 class="bot-title">🤖 Classic RAG Bot</h1>', unsafe_allow_html=True)

# Sample questions (shown when no chat history)
sample_questions = [
    "Summarize the key points from the document",
    "What are the main findings and conclusions?",
    "List any recommendations mentioned",
    "What dates, timelines, or deadlines are discussed?",
]

# Chat display area
chat_container = st.container()

with chat_container:
    if len(st.session_state['chat_history']) == 0:
        # Show sample questions in center when no chat
        st.markdown('<div class="sample-questions-container">', unsafe_allow_html=True)
        st.markdown("### 💡 Try asking:")
        
        cols = st.columns(1)
        for idx, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{idx}", use_container_width=True):
                st.session_state['current_question'] = question
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, chat_item in enumerate(st.session_state['chat_history']):
            # Unpack with defaults for backward compatibility
            if len(chat_item) == 4:
                question, answer, sources, debug_data = chat_item
            elif len(chat_item) == 3:
                question, answer, sources = chat_item
                debug_data = {}
            else:
                question, answer = chat_item
                sources = []
                debug_data = {}
            # User message (left side)
            st.markdown(f'<div class="user-message"><p class="message-text">{question}</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
            
            # Bot message (right side)
            st.markdown(f'<div class="bot-message"><p class="message-text">{answer}</p></div>', unsafe_allow_html=True)
            st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
            
            # Create columns for expandable sections
            expand_cols = st.columns(3)
            
            # Sources (expandable) - Column 1
            if sources:
                with expand_cols[0]:
                    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">{src.get('citation', f'[{idx}]')} {src.get('source', 'Unknown')}</div>
                                <div>Score: <span class="source-score">{src.get('score', 0):.4f}</span> | Pages: {src.get('pages', 'N/A')}</div>
                                <div style="margin-top:8px; font-size:0.85rem; color:#666;">{src.get('chunk_text', '')[:250]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Retrieved results (expandable) - Column 2
            if debug_data and debug_data.get('retrieved'):
                with expand_cols[1]:
                    with st.expander(f"🔍 Retrieved ({len(debug_data['retrieved'])})", expanded=False):
                        for idx, ret in enumerate(debug_data['retrieved'], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">[{idx}] {ret.get('source', 'Unknown')}</div>
                                <div>Score: <span class="source-score">{ret.get('score', 0):.4f}</span> | ID: {ret.get('id', 'N/A')}</div>
                                <div style="margin-top:8px; font-size:0.85rem; color:#666;">{ret.get('chunk_text', '')[:200]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Reranked results (expandable) - Column 3
            if debug_data and debug_data.get('reranked'):
                with expand_cols[2]:
                    with st.expander(f"🎯 Reranked ({len(debug_data['reranked'])})", expanded=False):
                        for idx, rnk in enumerate(debug_data['reranked'], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div class="source-title">[{idx}] {rnk.get('source', 'Unknown')}</div>
                                <div>Score: <span class="source-score">{rnk.get('score', 0):.4f}</span> | ID: {rnk.get('id', 'N/A')}</div>
                                <div style="margin-top:8px; font-size:0.85rem; color:#666;">{rnk.get('chunk_text', '')[:200]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Input area at bottom (always visible)
st.markdown('<div class="input-container">', unsafe_allow_html=True)

with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Type your question here...",
            value=st.session_state.get('current_question', ''),
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed",
            key="user_input"
        )
    with col2:
        submit = st.form_submit_button("Send 📤", use_container_width=True)
    
    if submit and user_input:
        # Clear sample question
        st.session_state['current_question'] = ""
        
        # Call chat endpoint
        payload = {
            "question": user_input,
            "use_reranker": use_reranker,
            "debug": debug_mode
        }
        
        with st.spinner("🤔 Thinking..."):
            try:
                r = requests.post(f"{API_BASE}/chat", json=payload, timeout=60)
                if r.ok:
                    resp = r.json()
                    answer = resp.get('answer', 'No answer generated.')
                    sources = resp.get('sources', [])
                    
                    # Capture debug data (retrieved and reranked)
                    debug_data = {
                        'retrieved': resp.get('retrieved'),
                        'reranked': resp.get('reranked')
                    }
                    
                    # Add to chat history
                    st.session_state['chat_history'].append((user_input, answer, sources, debug_data))
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {r.status_code} - {r.text}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Clear chat button
if st.session_state['chat_history']:
    if st.button("🗑️ Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()
