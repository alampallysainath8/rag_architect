"""
Enhanced RAG Frontend - Streamlit UI with Username Support

Features:
- User authentication (username input)
- Document upload with versioning
- Incremental update visualization
- Chat with documents
- Search functionality
- Document management per user
"""

import os
import base64
import requests
import streamlit as st
from pathlib import Path
from typing import Dict, Any

# Configuration
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TEMP_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_uploads"))
Path(TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Enhanced RAG Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ""
if 'user_documents' not in st.session_state:
    st.session_state['user_documents'] = []
if 'last_upload_info' not in st.session_state:
    st.session_state['last_upload_info'] = None

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Title styling */
    .bot-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* User badge */
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    /* Sample questions */
    .sample-questions-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
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
        min-width: 400px;
    }
    
    .sample-question-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    
    /* Chat messages */
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
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
    
    /* Version badge */
    .version-badge {
        background: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    
    /* Update info box */
    .update-info {
        background: #e8f5e9;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    
    .update-info-title {
        font-weight: 600;
        color: #2e7d32;
        margin-bottom: 8px;
    }
    
    /* Login container */
    .login-container {
        max-width: 500px;
        margin: 100px auto;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============================
# Helper Functions
# ============================

def fetch_user_documents(username: str) -> list:
    """Fetch user's documents from API."""
    try:
        response = requests.get(f"{API_BASE}/documents/{username}", timeout=5)
        if response.ok:
            data = response.json()
            return data.get('documents', [])
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
    return []


def format_update_info(info: Dict[str, Any]) -> str:
    """Format upload/update information for display."""
    if info['status'] == 'duplicate':
        return f"⚠️ **Duplicate Document** - Same content already exists (Version {info.get('version', 'N/A')})"
    elif info['status'] == 'success':
        if info.get('old_version'):
            # Incremental update
            return f"""
            ✅ **Document Updated Successfully!**
            - **Version:** {info['old_version']} → {info['new_version']}
            - **Total Chunks:** {info['total_chunks']}
            - **Changes:**
              - ➕ Added: {info['chunks_added']} chunks
              - ➖ Deleted: {info['chunks_deleted']} chunks
              - ✓ Unchanged: {info['unchanged_chunks']} chunks
            """
        else:
            # New upload
            return f"""
            ✅ **Document Ingested Successfully!**
            - **Version:** {info.get('version', 1)}
            - **Total Chunks:** {info['total_chunks']}
            - **Text Chunks:** {info['text_chunks']}
            - **Table Chunks:** {info['table_chunks']}
            """
    return ""


# ============================
# Authentication Screen
# ============================

if not st.session_state['authenticated']:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="bot-title">🚀 Enhanced RAG Bot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sign in to access your documents</p>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            help="Your username is used to organize and isolate your documents"
        )
        submit = st.form_submit_button("Sign In 🔐", use_container_width=True)
        
        if submit:
            if username.strip():
                st.session_state['username'] = username.strip().lower()
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("Please enter a valid username")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show backend status
    st.markdown("---")
    with st.expander("🔌 Backend Status"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.ok:
                st.success("✅ Backend Connected")
                health_data = r.json()
                st.json(health_data)
            else:
                st.error("❌ Backend Error")
        except Exception as e:
            st.error(f"❌ Cannot connect to backend: {str(e)}")
    
    st.stop()


# ============================
# Main Application (Authenticated)
# ============================

# Sidebar: User info and document management
st.sidebar.markdown(f'<div class="user-badge">👤 {st.session_state["username"]}</div>', unsafe_allow_html=True)

if st.sidebar.button("🔓 Sign Out", use_container_width=True):
    st.session_state['authenticated'] = False
    st.session_state['username'] = ""
    st.session_state['chat_history'] = []
    st.session_state['user_documents'] = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Document Management")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="file_uploader")

if uploaded_file:
    st.sidebar.info(f"📄 Selected: {uploaded_file.name}")
    
    if st.sidebar.button("🚀 Upload & Process", use_container_width=True):
        with st.spinner("Processing document..."):
            try:
                # Save file temporarily
                temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{st.session_state['username']}_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Upload to API with username
                # Important: Specify the original filename in the tuple to preserve it
                file_handle = open(temp_path, 'rb')
                files = {'file': (uploaded_file.name, file_handle, 'application/pdf')}
                data = {
                    'username': st.session_state['username']
                }
                
                response = requests.post(
                    f"{API_BASE}/ingest",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                file_handle.close()
                os.remove(temp_path)  # Cleanup
                
                if response.ok:
                    result = response.json()
                    st.session_state['last_upload_info'] = result
                    
                    # Show result
                    if result['status'] == 'duplicate':
                        st.sidebar.warning(f"⚠️ Duplicate! Version {result.get('version')} already exists")
                    else:
                        if result.get('old_version'):
                            st.sidebar.success(f"✅ Updated! v{result['old_version']} → v{result['new_version']}")
                        else:
                            st.sidebar.success(f"✅ Uploaded! Version {result.get('version', 1)}")
                    
                    # Refresh documents
                    st.session_state['user_documents'] = fetch_user_documents(st.session_state['username'])
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# Show last upload info
if st.session_state.get('last_upload_info'):
    with st.sidebar.expander("📊 Last Upload Details", expanded=True):
        info = st.session_state['last_upload_info']
        st.markdown(format_update_info(info), unsafe_allow_html=True)
        
        if st.button("Clear Info", key="clear_upload_info"):
            st.session_state['last_upload_info'] = None
            st.rerun()

st.sidebar.markdown("---")

# Fetch and display user documents
if st.sidebar.button("🔄 Refresh Documents", use_container_width=True):
    st.session_state['user_documents'] = fetch_user_documents(st.session_state['username'])

if not st.session_state['user_documents']:
    st.session_state['user_documents'] = fetch_user_documents(st.session_state['username'])

if st.session_state['user_documents']:
    st.sidebar.markdown("### 📚 Your Documents")
    
    for doc in st.session_state['user_documents']:
        with st.sidebar.expander(f"📄 {doc['filename']}", expanded=False):
            st.markdown(f"**Version:** {doc['version']}")
            st.markdown(f"**Status:** {doc['status']}")
            st.markdown(f"**Uploaded:** {doc['uploaded_at'][:19]}")
            st.markdown(f"**Active:** {'✓ Yes' if doc['is_active'] else '✗ No'}")
else:
    st.sidebar.info("No documents uploaded yet")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Search Options")
use_reranker = st.sidebar.checkbox("🔄 Use Reranker", value=True, help="Use cross-encoder for better relevance")
top_k = st.sidebar.slider("Top K Results", min_value=5, max_value=20, value=10, help="Number of candidates to retrieve")
top_n = st.sidebar.slider("Top N After Rerank", min_value=3, max_value=10, value=5, help="Number of results after reranking")

st.sidebar.markdown("---")
with st.sidebar.expander("🔌 Backend Status"):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.ok:
            st.success("✅ Connected")
            health = r.json()
            st.json(health)
        else:
            st.error("❌ Error")
    except:
        st.error("❌ Disconnected")

# ============================
# Main Content Area
# ============================

st.markdown('<h1 class="bot-title">🚀 Enhanced RAG Bot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about your documents with intelligent retrieval</p>', unsafe_allow_html=True)

# Sample questions
sample_questions = [
    "Summarize the key points from my documents",
    "What are the main findings and conclusions?",
    "List any recommendations mentioned",
    "What dates, timelines, or deadlines are discussed?",
    "Compare the information across different document versions"
]

# Chat display area
chat_container = st.container()

with chat_container:
    if len(st.session_state['chat_history']) == 0:
        # Show sample questions
        st.markdown('<div class="sample-questions-container">', unsafe_allow_html=True)
        st.markdown("### 💡 Try asking:")
        
        for idx, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{idx}", use_container_width=False):
                st.session_state['current_question'] = question
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for i, chat_item in enumerate(st.session_state['chat_history']):
            question, answer, citations = chat_item
            
            # User message
            st.markdown(f'<div class="user-message">{question}</div>', unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f'<div class="bot-message">{answer}</div>', unsafe_allow_html=True)
            
            # Citations
            if citations:
                with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">
                                [{cit['number']}] {cit['source']}
                                <span class="version-badge">v{cit['version']}</span>
                            </div>
                            <div>
                                Page {cit['page']} | 
                                Type: {cit['content_type']} | 
                                Score: <span class="source-score">{cit['score']:.4f}</span>
                            </div>
                            <div style="margin-top:8px; font-size:0.85rem; color:#666;">
                                {cit['text_preview'][:200]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Input area at bottom
st.markdown("---")
with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Question",
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
            "username": st.session_state['username'],
            "use_reranker": use_reranker,
            "top_k": top_k,
            "top_n": top_n
        }
        
        with st.spinner("🤔 Thinking..."):
            try:
                response = requests.post(f"{API_BASE}/chat", json=payload, timeout=60)
                
                if response.ok:
                    result = response.json()
                    answer = result.get('answer', 'No answer generated.')
                    citations = result.get('citations', [])
                    
                    # Add to chat history
                    st.session_state['chat_history'].append((user_input, answer, citations))
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Clear chat button
if st.session_state['chat_history']:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state['chat_history'] = []
            st.rerun()
