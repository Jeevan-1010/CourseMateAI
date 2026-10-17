
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".coursemate_cache"
PARSED_DIR = CACHE_DIR / "parsed"
VECTOR_DIR = CACHE_DIR / "vectors"

for directory in (PARSED_DIR, VECTOR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# --------------------------------------------------
# 2. Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="CourseMate AI",
    page_icon=None,
    layout="wide",
)

st.title("CourseMate AI")
st.caption(
    "Ask questions about study notes, textbooks, "
    "research papers, tables and academic PDFs."
)

with st.sidebar:
    st.header("Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "Clear current session",
        use_container_width=True,
    )

    st.divider()
    st.caption("Document parsing: Docling")
    st.caption("Embeddings: Hugging Face")
    st.caption("Vector database: ChromaDB")
    st.caption("Answer generation: Groq")


# --------------------------------------------------
# 3. Session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "active_signature" not in st.session_state:
    st.session_state.active_signature = None


if clear_button:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    st.session_state.active_signature = None
    st.rerun()


# --------------------------------------------------
# 4. Cached embedding model and LLM
# --------------------------------------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
    )


# --------------------------------------------------
# 5. File fingerprints and duplicate detection
# --------------------------------------------------

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def get_upload_signature(files):
    unique_files = {}

    for uploaded_file in files:
        data = uploaded_file.getvalue()
        digest = get_file_hash(data)

        # Identical file contents are processed only once.
        if digest not in unique_files:
            unique_files[digest] = {
                "name": Path(uploaded_file.name).name,
                "bytes": data,
                "hash": digest,
            }

    combined = hashlib.sha256()

    for digest in sorted(unique_files):
        combined.update(digest.encode("utf-8"))

    return combined.hexdigest(), list(unique_files.values())


# --------------------------------------------------
# 6. Parse-cache helpers
# --------------------------------------------------

def parsed_cache_path(file_hash):
    return PARSED_DIR / f"{file_hash}.json"


def load_parsed_documents(file_hash, source_name):
    cache_path = parsed_cache_path(file_hash)

    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        documents = []

        for record in records:
            metadata = record.get("metadata", {})
            metadata["source"] = source_name

            documents.append(
                Document(
                    page_content=record["page_content"],
                    metadata=metadata,
                )
            )

        if documents:
            return documents

    except (OSError, ValueError, KeyError, TypeError):
        pass

    return None


def save_parsed_documents(file_hash, documents):
    cache_path = parsed_cache_path(file_hash)
    temporary_path = cache_path.with_suffix(".tmp")

    records = [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)

    temporary_path.replace(cache_path)


# --------------------------------------------------
# 7. Extract provenance information
# --------------------------------------------------

def find_page_numbers(value):
    pages = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("page_no", "page_number"):
                try:
                    pages.add(int(item))
                except (TypeError, ValueError):
                    pass
            else:
                pages.update(find_page_numbers(item))

    elif isinstance(value, list):
        for item in value:
            pages.update(find_page_numbers(item))

    return pages


def normalize_metadata(document, source_name):
    metadata = document.metadata or {}

    pages = find_page_numbers(metadata)

    # Keep metadata JSON-serializable for the disk cache.
    clean_metadata = {
        "source": source_name,
        "page_numbers": ",".join(
            str(page) for page in sorted(pages)
        ) if pages else "unknown",
    }

    headings = metadata.get("headings")

    if isinstance(headings, list):
        clean_metadata["headings"] = " > ".join(
            str(heading) for heading in headings
        )
    elif headings:
        clean_metadata["headings"] = str(headings)

    document.metadata = clean_metadata
    return document


# --------------------------------------------------
# 8. Parse a PDF, reusing the disk cache when possible
# --------------------------------------------------

def parse_pdf(file_record):
    file_hash = file_record["hash"]
    source_name = file_record["name"]

    cached_documents = load_parsed_documents(
        file_hash,
        source_name,
    )

    if cached_documents is not None:
        return cached_documents, True

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(file_record["bytes"])
            temp_path = temp_file.name

        loader = DoclingLoader(file_path=temp_path)
        extracted_documents = loader.load()

        documents = []

        for document in extracted_documents:
            document = normalize_metadata(document, source_name)

            if document.page_content.strip():
                documents.append(document)

        if not documents:
            raise ValueError(
                f"No readable text was extracted from {source_name}."
            )

        save_parsed_documents(file_hash, documents)

        return documents, False

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# 9. Create or reuse the persistent vector database
# --------------------------------------------------

def vector_database_path(signature):
    return VECTOR_DIR / signature


def build_or_load_vectorstore(file_records, signature):
    db_path = vector_database_path(signature)
    sqlite_path = db_path / "chroma.sqlite3"

    embeddings = get_embeddings()

    # Reuse a previously built index for this exact upload set.
    if sqlite_path.exists():
        vectorstore = Chroma(
            collection_name="coursemate_documents",
            embedding_function=embeddings,
            persist_directory=str(db_path),
        )

        if vectorstore._collection.count() > 0:
            return vectorstore, True, 0, 0

    parsed_documents = []
    cache_hits = 0
    parse_seconds = 0.0

    progress = st.progress(0)
    status = st.empty()

    for index, file_record in enumerate(file_records):
        status.info(
            f"Reading PDF {index + 1}/{len(file_records)}: "
            f"{file_record['name']}"
        )

        start_time = time.perf_counter()

        documents, was_cached = parse_pdf(file_record)

        parse_seconds += time.perf_counter() - start_time

        if was_cached:
            cache_hits += 1

        parsed_documents.extend(documents)
        progress.progress((index + 1) / len(file_records))

    progress.empty()
    status.empty()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(parsed_documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded PDFs."
        )

    # Keep the collection dedicated to this upload signature.
    db_path.mkdir(parents=True, exist_ok=True)

    embedding_start = time.perf_counter()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="coursemate_documents",
        persist_directory=str(db_path),
    )

    embedding_seconds = time.perf_counter() - embedding_start

    return (
        vectorstore,
        False,
        parse_seconds,
        embedding_seconds,
        len(chunks),
        cache_hits,
    )


# --------------------------------------------------
# 10. Process uploaded files
# --------------------------------------------------

if process_button:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")

    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY was not found. "
            "Check the .env file in your project folder."
        )

    else:
        try:
            signature, file_records = get_upload_signature(
                uploaded_files
            )

            with st.spinner("Preparing your document collection..."):
                result = build_or_load_vectorstore(
                    file_records,
                    signature,
                )

            vectorstore, reused, *metrics = result

            st.session_state.vectorstore = vectorstore
            st.session_state.active_signature = signature
            st.session_state.processed_files = [
                record["name"] for record in file_records
            ]
            st.session_state.chat_history = []

            if reused:
                st.success(
                    "Loaded the existing document index. "
                    "No PDF parsing or embedding was required."
                )
            else:
                parse_seconds, embedding_seconds, chunk_count, cache_hits = (
                    metrics
                )

                st.success(
                    f"Prepared {len(file_records)} unique PDF(s) "
                    f"and {chunk_count} text chunks."
                )

                st.write(
                    f"Document parsing time: {parse_seconds:.2f} seconds"
                )
                st.write(
                    f"Embedding and indexing time: "
                    f"{embedding_seconds:.2f} seconds"
                )
                st.write(
                    f"PDFs loaded from parsing cache: {cache_hits}"
                )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


# --------------------------------------------------
# 11. Display loaded documents
# --------------------------------------------------

if st.session_state.processed_files:
    with st.expander("Processed documents"):
        for filename in st.session_state.processed_files:
            st.write(filename)


# --------------------------------------------------
# 12. Grounded question-answering prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are CourseMate AI, a study assistant for notes,
textbooks, research papers, and academic documents.

Answer using the supplied document context.

Rules:
- Ground factual answers in the supplied context.
- Do not invent facts, results, quotations, equations, or citations.
- If the documents do not contain enough information, say so.
- Explain difficult concepts clearly while preserving technical terms.
- For research papers, discuss methods, findings, and limitations
  only when supported by the context.
- Treat extracted tables carefully; do not guess missing values.
- Cite sources using the exact source labels in the context.
- Never invent a page number.
- Distinguish information from different documents.

Context:
{context}
""",
        ),
        (
            "human",
            """
Recent conversation:
{chat_history}

Question:
{question}

Provide a clear answer with relevant source citations.
""",
        ),
    ]
)


# --------------------------------------------------
# 13. Chat interface
# --------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.vectorstore is None:
    st.info(
        "Upload your PDFs and select Process documents "
        "to start asking questions."
    )

else:
    question = st.chat_input(
        "Ask a question about your uploaded documents..."
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    retriever = (
                        st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 6,
                                "fetch_k": 15,
                                "lambda_mult": 0.5,
                            },
                        )
                    )

                    retrieved_documents = retriever.invoke(question)

                    if not retrieved_documents:
                        answer = (
                            "I could not find relevant information "
                            "in your uploaded documents."
                        )

                    else:
                        context_parts = []

                        for document in retrieved_documents:
                            source = document.metadata.get(
                                "source",
                                "Unknown document",
                            )

                            pages = document.metadata.get(
                                "page_numbers",
                                "unknown",
                            )

                            if pages != "unknown":
                                page_labels = ", ".join(
                                    f"p. {page.strip()}"
                                    for page in pages.split(",")
                                    if page.strip()
                                )
                                source_label = (
                                    f"[{source}, {page_labels}]"
                                    if page_labels
                                    else f"[{source}]"
                                )
                            else:
                                source_label = f"[{source}]"

                            context_parts.append(
                                f"Source: {source_label}\n"
                                f"{document.page_content}"
                            )

                        context = "\n\n---\n\n".join(context_parts)

                        recent_history = (
                            st.session_state.chat_history[-8:-1]
                        )

                        history_text = "\n".join(
                            f"{message['role'].capitalize()}: "
                            f"{message['content']}"
                            for message in recent_history
                        )

                        response = (prompt | get_llm()).invoke(
                            {
                                "context": context,
                                "chat_history": history_text
                                or "No previous conversation.",
                                "question": question,
                            }
                        )

                        answer = response.content

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                st.error(f"Could not answer the question: {error}")
