"""
Document-Based RAG Chatbot — Powered by Google Gemini
Assignment 12 — Data Science and AI (Batch-06), Analytix Camp

Pipeline: Load document -> Split into chunks -> Create embeddings ->
Store in vector database (FAISS) -> Retrieve relevant chunks -> Generate answer (LLM)
"""

import os
import time
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DocuMind AI — Chat With Your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING — background, chat bubbles, fonts
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #93c5fd !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .user-bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    .bot-bubble {
        background: #1e293b;
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 75%;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .source-box {
        background: #0f172a;
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
        color: #94a3b8;
        border-radius: 4px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }
    section[data-testid="stFileUploaderDropzone"] {
        background: #1e293b;
        border: 2px dashed #3b82f6;
        border-radius: 12px;
    }
    .status-ready {
        display: inline-block;
        background: #065f46;
        color: #6ee7b7;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-empty {
        display: inline-block;
        background: #7c2d12;
        color: #fdba74;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📚 DocuMind AI")
    st.caption("RAG-based document Q&A — Assignment 12 (Powered by Gemini)")
    st.divider()

    st.markdown("### 🔑 Google Gemini API Key")
    api_key = st.text_input(
        "Enter your API key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at aistudio.google.com/app/apikey. Never hardcode this in your code.",
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or TXT file", type=["pdf", "txt"], label_visibility="collapsed"
    )

    st.markdown("### ⚙️ Chunking Settings")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=50)
    top_k = st.slider("Chunks retrieved per question", 1, 8, 4)

    st.divider()
    process_btn = st.button("🚀 Process Document", use_container_width=True)
    st.divider()

    st.markdown("### 📊 Pipeline Status")
    if st.session_state.vectorstore is not None:
        st.markdown(
            f'<span class="status-ready">● Ready</span> — '
            f'**{st.session_state.doc_name}** '
            f'({st.session_state.chunk_count} chunks indexed)',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-empty">● No document loaded</span>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# DOCUMENT PROCESSING PIPELINE
# =========================================================
def process_document(file, chunk_size, chunk_overlap):
    """Load -> Split -> Embed -> Store in FAISS vector database."""
    suffix = ".pdf" if file.name.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)
    return vectorstore, len(chunks)


if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your Google Gemini API key first.")
    elif not uploaded_file:
        st.sidebar.error("Please upload a document first.")
    else:
        with st.spinner("Processing document — loading, chunking, embedding..."):
            try:
                vectorstore, n_chunks = process_document(uploaded_file, chunk_size, chunk_overlap)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chunk_count = n_chunks
                st.session_state.chat_history = []
                st.sidebar.success(f"Indexed {n_chunks} chunks from {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error processing document: {e}")


# =========================================================
# MAIN AREA
# =========================================================
st.title("📚 DocuMind AI")
st.markdown("Ask questions about your uploaded document. Answers are grounded strictly in its content.")
st.divider()

if st.session_state.vectorstore is None:
    st.info("👈 Upload a document and click **Process Document** in the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 View source chunks used"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "N/A")
                        st.markdown(
                            f'<div class="source-box"><b>Chunk {i}</b> (page {page}): '
                            f'{src.page_content[:300]}...</div>',
                            unsafe_allow_html=True,
                        )

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="user-bubble">🧑 {question}</div>', unsafe_allow_html=True)

        if not api_key:
            st.error("Please enter your Google Gemini API key in the sidebar.")
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})

            prompt_template = PromptTemplate(
                template=(
                    "Answer the question using ONLY the context below. "
                    "If the answer is not in the context, say you don't know based on the document.\n\n"
                    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                ),
                input_variables=["context", "question"],
            )

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

            with st.spinner("Thinking..."):
                # Step 5: Retrieve relevant chunks, built manually (avoids relying on
                # legacy chain classes that move between LangChain versions)
                sources = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in sources)

                # Step 6: Generate the answer, grounded only in the retrieved context
                final_prompt = prompt_template.format(context=context_text, question=question)
                response = llm.invoke(final_prompt)
                answer = response.content

            placeholder = st.empty()
            displayed_text = ""
            for char in answer:
                displayed_text += char
                placeholder.markdown(
                    f'<div class="bot-bubble">🤖 {displayed_text}▌</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.012)
            placeholder.markdown(f'<div class="bot-bubble">🤖 {displayed_text}</div>', unsafe_allow_html=True)

            with st.expander("📎 View source chunks used"):
                for i, src in enumerate(sources, 1):
                    page = src.metadata.get("page", "N/A")
                    st.markdown(
                        f'<div class="source-box"><b>Chunk {i}</b> (page {page}): '
                        f'{src.page_content[:300]}...</div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )