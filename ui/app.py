import streamlit as st
import requests
import time
import uuid
from typing import Optional, List, Dict, Any

# Backend API URL
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Custom CSS
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2d3748;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Citations */
    .citation {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin: 0.3rem;
        font-size: 0.85em;
        display: inline-block;
        border-left: 3px solid #f59e0b;
    }
    
    /* Context chunks */
    .context-chunk {
        background: linear-gradient(to right, #fdf4ff 0%, #f3e8ff 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #a855f7;
        color: #4c1d95;
    }
    
    /* Document cards */
    .doc-card {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-indexed { background-color: #d1fae5; color: #065f46; }
    .status-queued { background-color: #fef3c7; color: #92400e; }
    .status-parsing { background-color: #dbeafe; color: #1e40af; }
    .status-embedding { background-color: #e0e7ff; color: #3730a3; }
    .status-failed { background-color: #fee2e2; color: #991b1b; }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    /* Section headers */
    .section-header {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False


def fetch_documents():
    """Fetch all documents from backend."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost", port=5432, dbname="ragdb", user="rag", password="ragpass"
        )
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, file_hash, status, page_count, created_at
            FROM documents
            ORDER BY created_at DESC
        """
        )
        docs = []
        for row in cur.fetchall():
            docs.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "file_hash": row[2],
                    "status": row[3],
                    "page_count": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                }
            )
        cur.close()
        conn.close()
        return docs
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []


def upload_document(file):
    """Upload a document to the backend."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        response.raise_for_status()
        result = response.json()

        # Check if document was already indexed
        if result.get("status") == "indexed":
            return result, True
        return result, False
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None, False


def send_chat_message(
    message: str,
    top_k: int,
    filter_doc_ids: Optional[List[int]],
    model: str,
    only_if_sources: bool,
    temperature: float,
):
    """Send a chat message to the backend."""
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "message": message,
            "top_k": top_k,
            "model": model,
            "only_if_sources": only_if_sources,
            "temperature": temperature,
        }
        if filter_doc_ids:
            payload["filter_doc_ids"] = filter_doc_ids

        response = requests.post(f"{API_BASE_URL}/chat/", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        st.error(
            "⏱️ Request timed out. The model might be downloading (first time only). Please try again."
        )
        return None
    except Exception as e:
        st.error(f"❌ Chat failed: {e}")
        return None


def clear_chat():
    """Clear chat history and create new session."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()


# Header
st.markdown("# 📚 RAG Document Chat")
st.markdown("Ask questions about your documents with AI-powered retrieval")
st.divider()

# Main layout
sidebar_col, main_col = st.columns([1, 3])

# ============= SIDEBAR =============
with sidebar_col:
    # Documents Section
    st.markdown("### 📁 Documents")

    # Upload in expander
    with st.expander("📤 Upload New PDF", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file:
            if st.button("⬆️ Upload", use_container_width=True, type="primary"):
                with st.spinner("Uploading..."):
                    result, already_indexed = upload_document(uploaded_file)
                    if result:
                        if already_indexed:
                            st.info(f"📄 Already indexed: {result['title']}")
                        else:
                            st.success(f"✅ Uploaded: {result['title']}")
                        time.sleep(1)
                        st.rerun()

    # Refresh button
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"**Total:** {len(st.session_state.documents)}")
    with col2:
        if st.button("🔄", use_container_width=True, help="Refresh document list"):
            st.rerun()

    # Fetch and display documents
    st.session_state.documents = fetch_documents()
    indexed_docs = [
        doc for doc in st.session_state.documents if doc["status"] == "indexed"
    ]

    # Document list with status
    if st.session_state.documents:
        for doc in st.session_state.documents[:10]:  # Show top 10
            status_class = f"status-{doc['status']}"
            status_emoji = {
                "indexed": "✅",
                "queued": "⏳",
                "parsing": "📄",
                "embedding": "🔢",
                "failed": "❌",
            }.get(doc["status"], "❓")

            st.markdown(
                f"""
                <div class="doc-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #1f2937; margin-bottom: 4px;">
                                {status_emoji} {doc['title'][:30]}{'...' if len(doc['title']) > 30 else ''}
                            </div>
                            <div style="font-size: 0.75rem; color: #6b7280;">
                                Pages: {doc['page_count'] or 'N/A'} • ID: {doc['id']}
                            </div>
                        </div>
                        <span class="status-badge {status_class}">{doc['status']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No documents yet. Upload one to get started!")

    st.divider()

    # Settings Toggle
    if st.button(
        "⚙️ Settings" if not st.session_state.show_settings else "⚙️ Hide Settings",
        use_container_width=True,
    ):
        st.session_state.show_settings = not st.session_state.show_settings

    # Settings Panel (collapsible)
    if st.session_state.show_settings:
        st.markdown("### ⚙️ Chat Settings")

        # Document Filter
        with st.expander("📄 Document Filter", expanded=True):
            if indexed_docs:
                selected_docs = st.multiselect(
                    "Select documents",
                    options=[doc["id"] for doc in indexed_docs],
                    format_func=lambda x: next(
                        (doc["title"] for doc in indexed_docs if doc["id"] == x), str(x)
                    ),
                    label_visibility="collapsed",
                )
                st.session_state.filter_doc_ids = (
                    selected_docs if selected_docs else None
                )

                if selected_docs:
                    st.success(f"✓ {len(selected_docs)} document(s)")
                else:
                    st.info("All documents")
            else:
                st.warning("No indexed documents")
                st.session_state.filter_doc_ids = None

        # Model Settings
        with st.expander("🤖 Model", expanded=False):
            st.session_state.model = st.selectbox(
                "Model",
                options=[
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                ],
                index=0,
                label_visibility="collapsed",
            )

            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1,
                help="Lower = focused, Higher = creative",
            )

        # Retrieval Settings
        with st.expander("🔍 Retrieval", expanded=False):
            st.session_state.top_k = st.slider(
                "Top K chunks",
                min_value=1,
                max_value=20,
                value=5,
            )

            st.session_state.only_if_sources = st.checkbox(
                "Require sources",
                value=False,
                help="Only answer if sources are found",
            )

        # Session info
        st.caption(f"📊 Session: `{st.session_state.session_id[:8]}...`")
        st.caption(f"💬 Messages: {len(st.session_state.messages)}")


# ============= MAIN CHAT AREA =============
with main_col:
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state.get("filter_doc_ids"):
            filtered_names = [
                doc["title"][:20] + "..." if len(doc["title"]) > 20 else doc["title"]
                for doc in st.session_state.documents
                if doc["id"] in st.session_state.filter_doc_ids
            ]
            st.info(f"🔍 Filtering: {', '.join(filtered_names)}")
    with col3:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            clear_chat()

    # Chat messages
    chat_container = st.container(height=500)

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
            <div style="text-align: center; padding: 3rem; color: #9ca3af;">
                <h3>👋 Welcome!</h3>
                <p>Upload documents and start asking questions</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-message"><strong>You:</strong><br/>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-message"><strong>Assistant:</strong><br/>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

                # Citations
                if "citations" in msg and msg["citations"]:
                    st.markdown("**📚 Sources:**")
                    citation_html = " ".join(
                        [
                            f'<span class="citation">📄 {c["doc_title"]}, p. {c["page"]}</span>'
                            for c in msg["citations"]
                        ]
                    )
                    st.markdown(citation_html, unsafe_allow_html=True)

                # Context (collapsible)
                if "context_chunks" in msg and msg["context_chunks"]:
                    with st.expander(
                        f"🔍 Context ({len(msg['context_chunks'])} chunks)"
                    ):
                        for idx, chunk in enumerate(msg["context_chunks"], 1):
                            st.markdown(
                                f"""
                                <div class="context-chunk">
                                    <strong>#{idx}</strong> · Score: {chunk['score']:.3f}<br/>
                                    📄 <strong>{chunk['doc_title']}</strong>, Page {chunk['page']}<br/>
                                    <div style="margin-top: 0.5rem; font-size: 0.9em;">
                                        {chunk['text'][:250]}{'...' if len(chunk['text']) > 250 else ''}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Metrics
                if "metrics" in msg and msg["metrics"]:
                    with st.expander("⏱️ Performance"):
                        m = msg["metrics"]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total", f"{m['latency_ms_total']}ms")
                        col2.metric("Embed", f"{m['latency_ms_embed']}ms")
                        col3.metric("Search", f"{m['latency_ms_qdrant']}ms")
                        col4.metric("LLM", f"{m['latency_ms_llm']}ms")

    # Chat input
    user_input = st.chat_input("💬 Ask a question about your documents...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response
        with st.spinner("🤔 Thinking..."):
            response = send_chat_message(
                message=user_input,
                top_k=st.session_state.get("top_k", 5),
                filter_doc_ids=st.session_state.get("filter_doc_ids", None),
                model=st.session_state.get("model", "openai/gpt-oss-20b"),
                only_if_sources=st.session_state.get("only_if_sources", False),
                temperature=st.session_state.get("temperature", 0.2),
            )

        if response:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["answer"],
                    "citations": response.get("citations", []),
                    "context_chunks": response.get("context_chunks", []),
                    "metrics": response.get("metrics", {}),
                }
            )

        st.rerun()
