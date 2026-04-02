import streamlit as st
import os
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="GitLab Handbook RAG Chatbot",
    page_icon="📖",
    layout="wide"
)


st.markdown("""
<style>

/* Theme Variables */
:root {
    --bg-color: #ffffff;
    --text-color: #1c1c1e;
    --card-bg: #f5f5f7;
    --header-bg: #ffffff;
    --border-color: rgba(0,0,0,0.08);
    --accent: #A0522D;
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-color: #0f0f1a;
        --text-color: #e6e6eb;
        --card-bg: #1e1e2e;
        --header-bg: #0f0f1a;
        --border-color: rgba(255,255,255,0.05);
    }
}

/* App */
.stApp {
    background: var(--bg-color);
    color: var(--text-color);
}

/* Header */
.block-container {
    background-color: var(--header-bg);
    padding-top: 2rem;
}

/* Title */
h1 {
    color: var(--accent);
    font-weight: 700;
}

/* Chat Messages */
.stChatMessage {
    background-color: var(--card-bg) !important;
    border-radius: 12px;
    padding: 12px;
    border: 1px solid var(--border-color);
}

/* Input Box (FIXED) */
section[data-testid="stChatInput"] textarea {
    background-color: var(--card-bg) !important;
    color: var(--text-color) !important;
    border-radius: 10px;
}

/* Input Container */
section[data-testid="stChatInput"] {
    background-color: var(--header-bg);
    border-top: 1px solid var(--border-color);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--card-bg);
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    background-color: var(--accent);
    color: white;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

/* Source Box */
.source-box {
    background: var(--card-bg);
    border-left: 3px solid var(--accent);
    padding: 8px;
    border-radius: 8px;
    font-size: 0.8rem;
    margin-top: 6px;
}

</style>
""", unsafe_allow_html=True)




# Sidebar 
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.35, 0.05,
                                     help="Lower = retrieve more (possibly less relevant) chunks")
    n_results = st.slider("Chunks to retrieve", 1, 10, 5)
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**About**\n\nRAG chatbot over the GitLab Handbook using ChromaDB + Gemini.\n By: Saumya Bhardwaj")

# Init 
@st.cache_resource(show_spinner="Loading vector database…")
def load_pipeline():
    return RAGPipeline(chroma_path="chroma_db", gitlab_path="Gitlab")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header 
st.title("GitLab Handbook AI Assistant")
st.markdown(
    "<span style='color:#aaa;'>Ask about GitLab engineering, remote work, and company culture</span>",
    unsafe_allow_html=True
)
# Build / load DB button 
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("⚡ Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Ingesting documents and building ChromaDB…"):
            pipeline = load_pipeline()
            n = pipeline.ingest()
            st.success(f"✅ Ingested {n} chunks into ChromaDB")
            st.cache_resource.clear()

# Chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown(
                "<div class='source-box'>📄 Sources: " +
                " · ".join(f"<code>{s}</code>" for s in msg["sources"]) +
                "</div>",
                unsafe_allow_html=True
            )

# Input 
if prompt := st.chat_input("Ask about the GitLab Handbook…"):
    if not api_key:
        st.error("Missing GEMINI_API_KEY. Set it in environment variables.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    pipeline = load_pipeline()

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching handbook…"):
            response, sources = pipeline.query(
                question=prompt,
                api_key=api_key,
                chat_history=st.session_state.messages[:-1],
                n_results=n_results,
                relevance_threshold=relevance_threshold,
            )
        st.markdown(response)
        if sources:
            st.markdown(
                "<div class='source-box'>📄 Sources: " +
                " · ".join(f"<code>{s}</code>" for s in sources) +
                "</div>",
                unsafe_allow_html=True
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })
