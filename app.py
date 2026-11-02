import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.rag.config import get_groq_api_key, DEFAULT_LLM_MODEL
from src.rag.engine import RagEngine
from src.rag.embeddings import EmbeddingManager
from src.rag.vectorstore import VectorStore
from src.rag.chat_history import (
    load_all_sessions,
    save_session,
    create_new_session,
    delete_session,
    MEDDIE_WELCOME_MESSAGE,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Meddie - Dr. Kallu's Medical AI Mentor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for chat layout & styling
st.markdown("""
<style>
    /* Global styles */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }
    
    /* Header Card */
    .meddie-header {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 50%, #42a5f5 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.15);
    }
    .meddie-header h2 {
        color: white !important;
        margin: 0;
        font-size: 1.7rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .meddie-header p {
        color: #e3f2fd !important;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }
    
    /* User Message Container (Aligned Right) */
    .user-container {
        display: flex;
        justify-content: flex-end;
        margin: 14px 0;
    }
    .user-bubble {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        color: white !important;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.2);
        font-size: 0.98rem;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #bbdefb;
        margin-bottom: 4px;
        text-align: right;
    }

    /* Source citations */
    .source-card {
        border-left: 3px solid #1976d2;
        background-color: rgba(25, 118, 210, 0.05);
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .source-title {
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 4px;
    }

    /* Sidebar History Button Styling */
    .stSidebar .stButton button {
        border-radius: 8px;
        text-align: left;
    }
    
    /* Sticky chat input */
    .stChatInput {
        padding-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Motivational quotes for Dr. Kallu
MOTIVATIONAL_QUOTES = [
    "“Wherever the art of Medicine is loved, there is also a love of Humanity.” — Hippocrates",
    "“The good physician treats the disease; the great physician treats the patient who has the disease.” — Sir William Osler",
    "“Every case sheet you study, Dr. Kallu, is a life you will one day touch.”",
    "“Medicine is a journey of persistence. Take a deep breath; you are doing exceptional work, Dr. Kallu!”",
]

import random

# Cache RAG Engine
@st.cache_resource(show_spinner=False)
def get_engine() -> RagEngine:
    return RagEngine()

# Verify API key
try:
    _ = get_groq_api_key()
except ValueError as e:
    st.error(f"⚠️ Configuration error: {e}")
    st.stop()

engine = get_engine()

# Initialize session management
all_sessions = load_all_sessions()

if "current_session_id" not in st.session_state or st.session_state.current_session_id not in all_sessions:
    if all_sessions:
        first_id = next(iter(all_sessions))
        st.session_state.current_session_id = first_id
    else:
        new_id, _ = create_new_session()
        st.session_state.current_session_id = new_id
        all_sessions = load_all_sessions()

current_session = all_sessions[st.session_state.current_session_id]

# ================= SIDE DRAWER =================
with st.sidebar:
    st.markdown("### 🩺 **Meddie**")
    st.caption("AI Medical Study Partner for **Dr. Kallu**")

    # + New Chat Button
    if st.button("➕ **New Chat**", use_container_width=True, type="primary"):
        new_id, _ = create_new_session()
        st.session_state.current_session_id = new_id
        st.rerun()

    st.markdown("---")
    st.subheader("📜 **Consultation History**")

    if not all_sessions:
        st.caption("No saved chats yet.")
    else:
        for sess_id, sess_info in list(all_sessions.items()):
            col_sess, col_del = st.columns([0.85, 0.15])
            is_active = (sess_id == st.session_state.current_session_id)
            title = sess_info.get("title", "Case")
            time_str = sess_info.get("updated_at", "")
            
            # Display title with active marker
            button_label = f"📌 {title}" if is_active else f"💬 {title}"
            with col_sess:
                if st.button(button_label, key=f"sess_{sess_id}", use_container_width=True):
                    st.session_state.current_session_id = sess_id
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{sess_id}", help="Delete this chat"):
                    delete_session(sess_id)
                    all_sessions = load_all_sessions()
                    if st.session_state.current_session_id == sess_id:
                        if all_sessions:
                            st.session_state.current_session_id = next(iter(all_sessions))
                        else:
                            new_id, _ = create_new_session()
                            st.session_state.current_session_id = new_id
                    st.rerun()

    st.markdown("---")
    # Quick medical knowledge base stats
    kb_count = engine.vector_store.count()
    st.caption(f"📚 Knowledge Base: **{kb_count} clinical chunks** indexed")
    st.caption("✨ Dedicated to **Dr. Kallu's** medical excellence")

# ================= MAIN CHAT DISPLAY =================

# Header Banner
quote_index = hash(st.session_state.current_session_id) % len(MOTIVATIONAL_QUOTES)
daily_quote = MOTIVATIONAL_QUOTES[quote_index]

st.markdown(f"""
<div class="meddie-header">
    <h2>🩺 Meddie</h2>
    <p><em>{daily_quote}</em></p>
</div>
""", unsafe_allow_html=True)

# Render Messages: User on RIGHT, Meddie on LEFT
messages = current_session.get("messages", [])

for msg in messages:
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])

    if role == "user":
        # User message: aligned to the RIGHT
        st.markdown(f"""
        <div class="user-container">
            <div class="user-bubble">
                <div class="user-label">Dr. Kallu 👨‍⚕️</div>
                <div>{content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Meddie message: aligned to the LEFT
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(content)
            
            # Show sources if any
            if sources:
                with st.expander(f"📚 Clinical References ({len(sources)} sources cited)", expanded=False):
                    for s in sources:
                        meta = s.get("metadata", {})
                        src_name = meta.get("file_name") or meta.get("source", "Medical Reference")
                        page = meta.get("page")
                        page_str = f" • Page {page}" if page is not None else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">📄 {src_name}{page_str}</div>
                            <div>{s.get('content', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)

# Chat Input Box
if prompt := st.chat_input("Ask Meddie about cases, symptoms, differentials, or exam topics..."):
    # 1. Add User Message
    messages.append({
        "role": "user",
        "content": prompt,
    })

    # Render user message right away
    st.markdown(f"""
    <div class="user-container">
        <div class="user-bubble">
            <div class="user-label">Dr. Kallu 👨‍⚕️</div>
            <div>{prompt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Meddie Response on Left
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Meddie is consulting clinical records and notes..."):
            retrieved_docs = engine.retrieve(query=prompt, top_k=4)

        response_placeholder = st.empty()
        full_response = ""

        try:
            stream_gen = engine.generate_answer_stream(
                query=prompt,
                context_docs=retrieved_docs,
                chat_history=messages[:-1],
                model_name=DEFAULT_LLM_MODEL,
                temperature=0.3,
            )
            for token in stream_gen:
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as err:
            full_response = (
                f"Hello Dr. Kallu! I encountered a small hiccup retrieving that detail: `{err}`. "
                "Let's try asking that again or check another case together!"
            )
            response_placeholder.error(full_response)

        # Show sources
        if retrieved_docs:
            with st.expander(f"📚 Clinical References ({len(retrieved_docs)} sources cited)", expanded=False):
                for s in retrieved_docs:
                    meta = s.get("metadata", {})
                    src_name = meta.get("file_name") or meta.get("source", "Medical Reference")
                    page = meta.get("page")
                    page_str = f" • Page {page}" if page is not None else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {src_name}{page_str}</div>
                        <div>{s.get('content', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # 3. Save assistant message and update session history on disk
    messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": retrieved_docs,
    })

    save_session(
        session_id=st.session_state.current_session_id,
        title=current_session.get("title", "Consultation"),
        messages=messages,
    )
