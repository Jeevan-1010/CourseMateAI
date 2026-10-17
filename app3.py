
import os
import json
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="CourseMate AI",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


# --------------------------------------------------
# 2. Page design
# --------------------------------------------------

st.title("📚 CourseMate AI")
st.caption(
    "Your AI study companion for notes, textbooks, "
    "research papers, tables and academic PDFs."
)

with st.sidebar:
    st.header("📄 Your documents")
    st.write(
        "Upload your study materials and ask questions "
        "about their contents."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload notes, textbooks and research papers.",
    )

    process_button = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "Clear documents and chat",
        use_container_width=True,
    )

    st.divider()
    st.caption("Document processing: Docling")
    st.caption("Embeddings: Hugging Face")
    st.caption("Vector database: ChromaDB")
    st.caption("Answer generation: Groq")


# --------------------------------------------------
# 3. Initialize session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# --------------------------------------------------
# 4. Extract page numbers from Docling metadata
# --------------------------------------------------

def find_page_numbers(value):
    """Find page numbers in nested Docling metadata."""

    pages = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if key == "page_no":
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


# --------------------------------------------------
# 5. Prepare metadata for ChromaDB
# --------------------------------------------------

def prepare_document_metadata(document, original_name):
    metadata = document.metadata or {}

    # Docling stores structural and provenance metadata here.
    docling_metadata = metadata.get("dl_meta", {})
    pages = find_page_numbers(docling_metadata)

    # Also check other metadata fields if needed.
    pages.update(find_page_numbers(metadata.get("provenance", {})))

    # Replace temporary file paths with original PDF names.
    clean_metadata = {
        "source": original_name,
        "page_numbers": ",".join(map(str, sorted(pages)))
        if pages
        else "unknown",
    }

    headings = metadata.get("headings")
    if headings:
        if isinstance(headings, list):
            clean_metadata["headings"] = " > ".join(
                str(heading) for heading in headings
            )
        else:
            clean_metadata["headings"] = str(headings)

    # Chroma metadata should contain simple scalar values.
    document.metadata = clean_metadata
    return document


# --------------------------------------------------
# 6. Process uploaded PDFs using Docling
# --------------------------------------------------

def build_vectorstore(files):
    all_documents = []
    total_files = len(files)

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(files):
        original_name = Path(uploaded_file.name).name

        status.info(
            f"Processing {index + 1}/{total_files}: {original_name}"
        )

        temp_path = None

        try:
            # Docling receives a real PDF file path.
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = DoclingLoader(file_path=temp_path)
            documents = loader.load()

            if not documents:
                st.warning(
                    f"No extractable content found in {original_name}."
                )
                continue

            for document in documents:
                document = prepare_document_metadata(
                    document,
                    original_name,
                )

                if document.page_content.strip():
                    all_documents.append(document)

        except Exception as error:
            st.error(
                f"Could not process {original_name}: {error}"
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        progress.progress((index + 1) / total_files)

    progress.empty()
    status.empty()

    if not all_documents:
        raise ValueError(
            "No readable document content was extracted. "
            "Check the PDFs and try again."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
    )

    return vectorstore, len(all_documents)


# --------------------------------------------------
# 7. Process button
# --------------------------------------------------

if clear_button:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.session_state.processed_files = []

    st.rerun()


if process_button:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")

    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY was not found. "
            "Check your .env file in the project folder."
        )

    else:
        try:
            # Avoid processing the same set of uploads repeatedly.
            file_signature = hashlib.sha256(
                b"".join(
                    (
                        uploaded_file.name.encode("utf-8")
                        + uploaded_file.getvalue()
                    )
                    for uploaded_file in uploaded_files
                )
            ).hexdigest()

            with st.spinner(
                "Reading PDFs and building your knowledge base..."
            ):
                vectorstore, chunk_count = build_vectorstore(
                    uploaded_files
                )

            st.session_state.vectorstore = vectorstore
            st.session_state.chat_history = []
            st.session_state.processed_files = [
                uploaded_file.name
                for uploaded_file in uploaded_files
            ]
            st.session_state.file_signature = file_signature

            st.success(
                f"Successfully processed {len(uploaded_files)} PDF(s) "
                f"into {chunk_count} document chunks!"
            )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


# --------------------------------------------------
# 8. Display loaded documents
# --------------------------------------------------

if st.session_state.processed_files:
    with st.expander("View processed documents", expanded=False):
        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")


# --------------------------------------------------
# 9. Initialize Groq model
# --------------------------------------------------

@st.cache_resource
def get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
    )


# --------------------------------------------------
# 10. Build grounded answer prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are CourseMate AI, a helpful academic study assistant.

Answer the student's question using the supplied document context.

Rules:
1. Use the provided context as your primary source.
2. Do not invent facts, quotations, results, equations, or citations.
3. If the documents do not contain enough information, say so clearly.
4. Explain difficult concepts in simple language when appropriate.
5. Preserve important technical terminology from the documents.
6. For research papers, explain methodology, findings, limitations,
   and comparisons only when the context supports them.
7. For tables, use the extracted values carefully and do not guess
   missing values.
8. Cite relevant sources using the exact source labels provided
   in the context, such as [paper.pdf, p. 4].
9. If page information is unavailable, cite the filename without
   inventing a page number.
10. Distinguish information from different documents clearly.

Document context:
{context}
""",
        ),
        (
            "human",
            """
Previous conversation:
{chat_history}

Student's question:
{question}

Give a clear, useful answer with source citations.
""",
        ),
    ]
)


# --------------------------------------------------
# 11. Chat interface
# --------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.vectorstore is None:
    st.info(
        "👈 Upload your PDFs in the sidebar and click "
        "**Process documents** to begin."
    )

else:
    question = st.chat_input(
        "Ask a question about your notes or research papers..."
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

                    retrieved_docs = retriever.invoke(question)

                    if not retrieved_docs:
                        answer = (
                            "I couldn't find relevant information "
                            "in your uploaded documents."
                        )

                    else:
                        context_parts = []

                        for document in retrieved_docs:
                            source = document.metadata.get(
                                "source", "Unknown document"
                            )

                            raw_pages = document.metadata.get(
                                "page_numbers", "unknown"
                            )

                            if raw_pages != "unknown":
                                page_list = [
                                    page.strip()
                                    for page in raw_pages.split(",")
                                    if page.strip()
                                ]

                                if page_list:
                                    page_label = ", ".join(
                                        f"p. {page}"
                                        for page in page_list
                                    )
                                    source_label = (
                                        f"[{source}, {page_label}]"
                                    )
                                else:
                                    source_label = f"[{source}]"
                            else:
                                source_label = f"[{source}]"

                            context_parts.append(
                                f"Source: {source_label}\n"
                                f"{document.page_content}"
                            )

                        context = "\n\n---\n\n".join(
                            context_parts
                        )

                        recent_history = (
                            st.session_state.chat_history[-8:-1]
                        )

                        history_text = "\n".join(
                            f"{item['role'].capitalize()}: "
                            f"{item['content']}"
                            for item in recent_history
                        )

                        llm = get_llm()

                        chain = prompt | llm

                        response = chain.invoke(
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
                st.error(
                    f"Something went wrong while answering: {error}"
                )
