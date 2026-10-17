import hashlib
import os
import tempfile

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

# Keep the current chat and uploaded document across Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )


@st.cache_resource
def load_llm():
    # Uses GROQ_API_KEY from the .env file.
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are CourseMate AI, a helpful study assistant.
Use only the provided PDF context to answer the question.
Explain concepts simply and clearly.
If the answer is not present in the context, say:
"I could not find the answer in the document."
Do not invent information.""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)


with st.sidebar:
    st.title("📚 CourseMate AI")
    st.caption("Your PDF Study Companion")
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Let CourseMate process it")
    st.markdown("3. Ask questions about the document")
    st.divider()
    st.caption("Powered by LangChain, Chroma, Hugging Face and Groq")


st.title("Welcome to CourseMate AI 👋")
st.write(
    "Upload your study PDF and ask questions. "
    "I'll find relevant information and explain it simply."
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a text-based PDF to start asking questions.",
)

if uploaded_file is None:
    st.info("Upload a PDF to begin. Your existing Chroma-db is left untouched.")
    st.stop()

file_bytes = uploaded_file.getvalue()
current_document_id = hashlib.sha256(file_bytes).hexdigest()

# Build a new vector store only when the uploaded PDF changes.
if st.session_state.document_id != current_document_id:
    try:
        with st.spinner("Reading your PDF and creating its searchable index..."):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                documents = PyPDFLoader(temp_path).load()
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            if not documents:
                st.error("No readable pages were found in this PDF.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(documents)

            if not chunks:
                st.error(
                    "No readable text was extracted. "
                    "This may be a scanned PDF that needs OCR."
                )
                st.stop()

            embedding_model = load_embedding_model()

            # No persist_directory is specified, so this uploaded PDF's
            # vector store is separate from the existing project Chroma-db.
            # The collection name is unique to the file contents.
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name="course_" + current_document_id[:20],
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.document_id = current_document_id
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []

        st.success(
            f"Ready! Processed {len(documents)} pages into "
            f"{len(chunks)} searchable chunks."
        )
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()
else:
    st.success(f"Document ready: {st.session_state.document_name}")

st.subheader("💬 Chat with your PDF")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF and preparing an answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5,
                    },
                )

                docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = prompt.invoke(
                    {"context": context, "question": question}
                )
                response = load_llm().invoke(final_prompt)
                answer = response.content

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    except Exception as exc:
        st.error(f"Something went wrong while generating the answer: {exc}")
