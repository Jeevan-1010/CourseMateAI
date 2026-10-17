import hashlib
import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="CourseMate AI",
    page_icon="📚",
    layout="wide",
)

# Each browser session gets a unique ID. The uploaded documents and their
# vector store live only in this Streamlit session; no shared persistent DB.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )


@st.cache_resource
def load_llm():
    # Configure GROQ_API_KEY in Streamlit secrets or environment variables.
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are CourseMate AI, a helpful study assistant.
Answer using only the supplied PDF context. Explain concepts simply and clearly.
If the answer is not supported by the supplied context, say:
"I could not find the answer in the uploaded documents."
Do not invent facts or page references.
The context includes source labels and page numbers; cite relevant sources in your answer
using the exact labels supplied, such as [filename.pdf, p. 3].""",
        ),
        (
            "human",
            """Retrieved context:
{context}

Question:
{question}""",
        ),
    ]
)


def make_document_signature(files):
    """Hash file names and contents so the index rebuilds only when uploads change."""
    digest = hashlib.sha256()
    for uploaded in sorted(files, key=lambda f: (f.name, f.size)):
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(uploaded.getvalue())
        digest.update(b"\0")
    return digest.hexdigest()


def build_vectorstore(uploaded_files):
    all_chunks = []
    uploaded_names = []

    for uploaded in uploaded_files:
        file_bytes = uploaded.getvalue()
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name

            pages = PyPDFLoader(temp_path).load()

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        # Keep the original filename and human-readable, 1-based page number
        # in metadata so source citations can be displayed later.
        for page in pages:
            page.metadata["source"] = uploaded.name
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        uploaded_names.append(uploaded.name)

    if not all_chunks:
        raise ValueError(
            "No readable text was extracted from these PDFs. "
            "If they are scanned images, OCR may be required."
        )

    embedding_model = load_embedding_model()

    # In-memory Chroma store, created per session and per upload set.
    # No persist_directory: it will not read/write the project's existing Chroma-db.
    collection_name = "cm_" + st.session_state.session_id[:12] + "_" + uuid.uuid4().hex[:12]
    store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        collection_name=collection_name,
    )

    return store, uploaded_names, len(all_chunks)


with st.sidebar:
    st.title("📚 CourseMate AI")
    st.caption("Your AI study companion")
    st.divider()
    st.markdown("**How to use it**")
    st.markdown("1. Upload one or more PDFs")
    st.markdown("2. Wait for indexing to finish")
    st.markdown("3. Ask questions across your notes")
    st.divider()
    st.caption("Python · LangChain · Chroma · Hugging Face · Groq")
    if st.button("🗑️ Clear documents and chat", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.uploaded_signature = None
        st.session_state.uploaded_names = []
        st.session_state.messages = []
        st.rerun()


st.title("CourseMate AI")
st.write("Upload your notes or textbook chapters, then ask questions grounded in those documents.")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select multiple PDFs at once. Only the currently selected PDFs are indexed.",
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.markdown(
        """
        **What you can do**
        - Ask questions across multiple PDFs.
        - Get answers with source PDF names and page numbers.
        - Test concepts and request simple explanations.
        """
    )
    # Avoid continuing to use an index after the uploader has been cleared.
    if st.session_state.vectorstore is not None:
        st.session_state.vectorstore = None
        st.session_state.uploaded_signature = None
        st.session_state.uploaded_names = []
        st.session_state.messages = []
    st.stop()

signature = make_document_signature(uploaded_files)

if st.session_state.uploaded_signature != signature:
    try:
        with st.spinner("Reading PDFs, splitting text, and building your private study index..."):
            store, names, chunk_count = build_vectorstore(uploaded_files)
            st.session_state.vectorstore = store
            st.session_state.uploaded_signature = signature
            st.session_state.uploaded_names = names
            st.session_state.messages = []

        st.success(
            f"Indexed {len(names)} PDF(s) into {chunk_count} searchable chunks."
        )
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.uploaded_signature = None
        st.error(f"Could not process the uploaded PDFs: {exc}")
        st.stop()
else:
    st.success(f"Ready to study {len(st.session_state.uploaded_names)} PDF(s).")

with st.expander("📄 Uploaded documents", expanded=False):
    for name in st.session_state.uploaded_names:
        st.write(f"• {name}")

st.subheader("💬 Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your uploaded PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDFs and writing an answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 15,
                        "lambda_mult": 0.5,
                    },
                )
                docs = retriever.invoke(question)

                context_parts = []
                seen_sources = set()

                for doc in docs:
                    source = os.path.basename(str(doc.metadata.get("source", "uploaded PDF")))
                    page = doc.metadata.get("page", "?")
                    label = f"[{source}, p. {page}]"
                    context_parts.append(f"Source: {label}\n{doc.page_content}")
                    seen_sources.add((source, str(page)))

                context = "\n\n---\n\n".join(context_parts)

                final_prompt = prompt.invoke(
                    {"context": context, "question": question}
                )
                response = load_llm().invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

                # Show retrieved source pages independently of model-generated citations.
                if docs:
                    with st.expander("🔎 Retrieved source pages", expanded=True):
                        for source, page in sorted(seen_sources):
                            st.markdown(f"- **{source}**, page {page}")
                else:
                    st.caption("No relevant document chunks were retrieved.")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    except Exception as exc:
        st.error(f"Could not generate an answer: {exc}")


st.divider()
st.caption(
    "Privacy note: uploaded PDFs are processed in this app session and indexed in an "
    "in-memory vector store. This demo does not save the uploaded files or persist their "
    "vectors to your project's Chroma-db. Hosting providers may have their own logging "
    "and session-retention behavior."
)
