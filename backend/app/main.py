
import hashlib
import re
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document as WordDocument
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from backend.app.database import check_database_connection, initialize_database
from backend.app.persistence import (
    conversation_belongs_to_document,
    create_conversation,
    get_document_id,
    list_conversations,
    list_documents,
    load_conversation_context,
    save_document,
    save_exchange,
    get_conversation_messages,
    get_messages_to_summarize,
    save_conversation_summary,
)


BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
CHROMA_DIR = BASE_DIR / "backend" / "Chroma-db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024

app = FastAPI(
    title="CourseMate AI API",
    version="0.6.0",
)

@app.on_event("startup")
def startup_initialize_database():
    initialize_database()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATE_K = 20
RERANK_FINAL_K = 4
RERANK_MAX_LENGTH = 384
RERANK_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Study chunks are limited to 1,000 characters. A 384-token cap avoids
    # unnecessary transformer work while retaining room for most chunks.
    return CrossEncoder(
        RERANKER_MODEL_NAME,
        max_length=RERANK_MAX_LENGTH,
    )


def rerank_documents(query: str, documents: list[Document]):
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [(query, document.page_content) for document in documents]
    scores = reranker.predict(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        (document, float(score))
        for document, score in ranked[:RERANK_FINAL_K]
    ]


def _normalized_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _chunk_token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", _normalized_chunk_text(text)))


def deduplicate_document_matches(matches, similarity_threshold: float = 0.92):
    """Keep the first/highest-ranked copy of exact or near-identical chunks."""
    unique = []
    exact_seen = set()
    token_sets = []

    for document, distance in matches:
        normalized = _normalized_chunk_text(document.page_content)
        if not normalized:
            continue

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in exact_seen:
            continue

        tokens = _chunk_token_set(normalized)
        is_near_duplicate = False
        for existing_tokens in token_sets:
            if not tokens or not existing_tokens:
                continue
            union = tokens | existing_tokens
            similarity = len(tokens & existing_tokens) / len(union)
            if similarity >= similarity_threshold:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            continue

        exact_seen.add(digest)
        token_sets.append(tokens)
        unique.append((document, distance))

    return unique


def split_query_aspects(query: str) -> list[str]:
    """Extract searchable concepts from comma- and conjunction-separated questions."""
    cleaned = re.sub(r"\s+", " ", query).strip()
    parts = re.split(
        r"\s*(?:,|;|\b(?:and|vs\.?|versus|plus|as well as|along with)\b)\s*",
        cleaned,
        flags=re.IGNORECASE,
    )

    aspects = []
    seen = set()
    for part in parts:
        aspect = re.sub(
            r"^(?:please\s+)?(?:explain|describe|compare|discuss|define|what is|what are|tell me about)\s+",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        ).strip(" ?.:")
        normalized = aspect.casefold()
        if len(aspect.split()) < 1 or len(normalized) < 3:
            continue
        if normalized not in seen:
            seen.add(normalized)
            aspects.append(aspect)

    return aspects if len(aspects) > 1 else []


def retrieve_candidate_matches(vectorstore: Chroma, query: str):
    """Merge whole-question, aspect-level, and diversity retrieval candidates."""
    similarity_matches = vectorstore.similarity_search_with_score(
        query,
        k=RERANK_CANDIDATE_K,
    )

    aspects = split_query_aspects(query)
    aspect_matches = []
    for aspect in aspects:
        aspect_matches.append(
            vectorstore.similarity_search_with_score(
                aspect,
                k=max(6, RERANK_CANDIDATE_K // 3),
            )
        )

    query_is_broad = bool(
        aspects
        or re.search(
            r"\b(all|types|different|compare|difference|and|examples|various|multiple)\b",
            query,
            flags=re.IGNORECASE,
        )
    )

    ranked_sources = [similarity_matches, *aspect_matches]
    if query_is_broad:
        diverse_documents = vectorstore.max_marginal_relevance_search(
            query,
            k=RERANK_CANDIDATE_K,
            fetch_k=max(RERANK_CANDIDATE_K * 3, 40),
            lambda_mult=0.65,
        )
        ranked_sources.append([(document, None) for document in diverse_documents])

    # Round-robin source rankings so each question aspect contributes candidates.
    raw_matches = []
    max_depth = max((len(source) for source in ranked_sources), default=0)
    for index in range(max_depth):
        for source in ranked_sources:
            if index < len(source):
                raw_matches.append(source[index])

    unique_matches = deduplicate_document_matches(raw_matches)
    return raw_matches, unique_matches


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QuestionRequest(BaseModel):
    filename: str
    question: str
    conversation_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    filename: str


def safe_document_path(filename: str) -> Path:
    filename = Path(PureWindowsPath(filename).name).name

    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported formats are PDF, TXT, and DOCX.",
        )

    path = UPLOAD_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload the file first.",
        )

    return path


def collection_name(filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24]
    return f"doc_{digest}"


def invalidate_document_index(filename: str) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(filename)

    try:
        client.delete_collection(name=name)
    except Exception:
        pass


def load_document_pages(filename: str) -> list[dict]:
    path = safe_document_path(filename)
    extension = path.suffix.lower()

    if extension == ".pdf":
        try:
            reader = PdfReader(str(path))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported.",
                )

            return [
                {"page": number, "text": page.extract_text() or ""}
                for number, page in enumerate(reader.pages, start=1)
            ]

        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read PDF: {type(error).__name__}",
            )

    if extension == ".txt":
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="TXT files must use UTF-8 text encoding.",
            )

        return [{"page": 1, "text": text}]

    try:
        word_document = WordDocument(str(path))

        paragraphs = [
            paragraph.text
            for paragraph in word_document.paragraphs
            if paragraph.text.strip()
        ]

        tables = []
        for table in word_document.tables:
            for row in table.rows:
                tables.append(" | ".join(cell.text for cell in row.cells))

        text = "\n".join(paragraphs + tables)
        return [{"page": 1, "text": text}]

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read DOCX: {type(error).__name__}",
        )


def get_vectorstore(filename: str) -> Chroma:
    safe_document_path(filename)
    name = collection_name(filename)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=name)

        if collection.count() > 0:
            return Chroma(
                client=client,
                collection_name=name,
                embedding_function=embeddings,
            )

    except Exception:
        pass

    pages = load_document_pages(filename)

    documents = [
        Document(
            page_content=page["text"],
            metadata={
                "source": filename,
                "page": page["page"] - 1,
            },
        )
        for page in pages
        if page["text"].strip()
    ]

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in this document.",
        )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No usable text chunks were created.",
        )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=name,
        persist_directory=str(CHROMA_DIR),
    )


def get_language_model():
    import os

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing from the environment.",
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )


def update_conversation_summary(conversation_id: str, model) -> None:
    previous_summary, older_messages = get_messages_to_summarize(
        conversation_id, keep_recent=8
    )

    if not older_messages:
        return

    messages_text = "\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in older_messages
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Update a compact memory of this student's conversation. "
            "Combine the existing summary with the newly supplied older "
            "messages. Preserve the student's goals, preferences, key "
            "questions, established context, important explanations, and "
            "unresolved references needed to understand future follow-ups. "
            "Remove repetition and outdated details. Do not invent facts. "
            "Return only the updated summary.",
        ),
        (
            "human",
            "Existing summary:\n{previous_summary}\n\n"
            "New older messages to incorporate:\n{messages}",
        ),
    ])

    result = (summary_prompt | model).invoke({
        "previous_summary": previous_summary or "No previous summary.",
        "messages": messages_text,
    })
    updated_summary = result.content.strip()

    if updated_summary:
        save_conversation_summary(
            conversation_id,
            updated_summary,
            older_messages[-1]["id"],
        )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CourseMate AI API",
    }

@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


@app.get("/documents/{filename}/conversations")
def get_document_conversations(filename: str):
    document_id = get_document_id(filename)
    if document_id is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"conversations": list_conversations(document_id)}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    messages = get_conversation_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/db-health")
def database_health():
    try:
        check_database_connection()
        return {"status": "connected", "database": "Neon PostgreSQL"}
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Neon database health check failed."
        )
        raise HTTPException(
            status_code=503,
            detail="Database connection failed. Check the FastAPI terminal for the underlying error.",
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = Path(PureWindowsPath(file.filename).name).name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF, TXT, or DOCX file.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The maximum file size is 20 MB.",
        )

    temporary = UPLOAD_DIR / f".uploading_{filename}"

    try:
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file.",
                )

            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported.",
                )

        elif extension == ".docx":
            WordDocument(BytesIO(content))

        elif extension == ".txt":
            content.decode("utf-8-sig")

        destination = UPLOAD_DIR / filename
        temporary.write_bytes(content)

        if extension == ".pdf":
            test_reader = PdfReader(str(temporary))

            if test_reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported.",
                )

        temporary.replace(destination)
        invalidate_document_index(filename)

        pages = load_document_pages(filename)
        has_text = any(page["text"].strip() for page in pages)

        save_document(
            filename=filename,
            file_type=extension[1:].upper(),
            file_size_bytes=len(content),
        )

        return {
            "message": "Document uploaded successfully.",
            "filename": filename,
            "file_type": extension[1:].upper(),
            "size_bytes": len(content),
            "page_count": len(pages) if extension == ".pdf" else None,
            "has_extractable_text": has_text,
            "replaced_existing_file": True,
        }

    except HTTPException:
        temporary.unlink(missing_ok=True)
        raise

    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the upload: {type(error).__name__}",
        )


@app.get("/extract/{filename}")
def extract_document(filename: str):
    pages = load_document_pages(filename)
    extension = Path(filename).suffix.lower()

    return {
        "filename": Path(PureWindowsPath(filename).name).name,
        "page_count": len(pages) if extension == ".pdf" else None,
        "pages": pages,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    request_start = time.perf_counter()
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Enter a question.",
        )

    try:
        document_id = get_document_id(request.filename)
        if document_id is None:
            raise HTTPException(
                status_code=404,
                detail="Document record not found. Upload the document again.",
            )

        if request.conversation_id:
            conversation_id = request.conversation_id
            if not conversation_belongs_to_document(
                conversation_id, document_id
            ):
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found for this document.",
                )
        else:
            conversation_id = create_conversation(
                document_id, title=question[:80]
            )

        stored_context = load_conversation_context(conversation_id)
        if stored_context is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        summary, stored_history = stored_context
        # The database is the source of truth; keep only the latest 8 messages.
        history = stored_history
        if not history and request.history:
            history = [
                {"role": item.role, "content": item.content}
                for item in request.history[-8:]
                if item.content.strip()
            ]

        chat_history = [
            ("human" if item["role"] == "user" else "ai", item["content"])
            for item in history
        ]
        if summary:
            chat_history.insert(
                0,
                ("system", "Earlier conversation summary: " + summary),
            )

        model = get_language_model()
        retrieval_query = question

        # Rewrite follow-up questions into standalone search queries.
        if chat_history:
            contextualize_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Rewrite the latest student question as a standalone "
                    "search query for the uploaded study document. Use the "
                    "conversation only to resolve references and omitted "
                    "details. Preserve the student's intent. Do not answer "
                    "the question. If it is already standalone, return it "
                    "unchanged.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ])

            rewritten = (contextualize_prompt | model).invoke({
                "chat_history": chat_history,
                "question": question,
            })
            retrieval_query = rewritten.content.strip() or question

        vectorstore = get_vectorstore(request.filename)
        retrieval_start = time.perf_counter()

        # Combine semantic similarity with MMR diversity for broad questions,
        # then remove redundant chunks before expensive cross-encoder scoring.
        raw_candidate_matches, unique_candidate_matches = retrieve_candidate_matches(
            vectorstore,
            retrieval_query,
        )
        candidate_matches = unique_candidate_matches[:RERANK_CANDIDATE_K]

        retrieval_latency_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        reranking_start = time.perf_counter()
        reranked_matches = rerank_documents(
            retrieval_query,
            [doc for doc, distance in candidate_matches],
        )
        candidate_distances = {
            id(doc): float(distance) if distance is not None else None
            for doc, distance in candidate_matches
        }
        selected_matches = [
            (doc, rerank_score, candidate_distances[id(doc)])
            for doc, rerank_score in reranked_matches
        ]
        reranking_latency_ms = (
            time.perf_counter() - reranking_start
        ) * 1000

        print("\nReranking results for:", retrieval_query)
        for doc, rerank_score, distance in selected_matches:
            distance_label = (
                f"{distance:.4f}" if distance is not None else "MMR-only"
            )
            print(
                f"Page: {doc.metadata.get('page', 0) + 1}, "
                f"Rerank score: {rerank_score:.4f}, "
                f"Vector distance: {distance_label}"
            )

        matches = [doc for doc, rerank_score, distance in selected_matches]

        if not matches:
            raise HTTPException(
                status_code=422,
                detail="No relevant text was found.",
            )

        context = "\n\n".join(
            f"Source section {doc.metadata.get('page', 0) + 1}:\n"
            f"{doc.page_content}"
            for doc in matches
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are CourseMate AI, a study assistant. Answer the "
                "student's latest question using the uploaded document "
                "context as the source of factual information. Use the "
                "conversation history only to understand references and "
                "the ongoing discussion. Do not treat earlier assistant "
                "answers as authoritative evidence. If the answer is not "
                "supported by the document context, clearly say that the "
                "uploaded document does not provide enough information. "
                "Do not invent facts.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "Uploaded document context:\n{context}\n\n"
                "Latest student question: {question}",
            ),
        ])

        generation_start = time.perf_counter()

        response = (prompt | model).invoke({
            "chat_history": chat_history,
            "context": context,
            "question": question,
        })

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        source_sections = sorted({
            doc.metadata.get("page", 0) + 1
            for doc in matches
        })

        end_to_end_latency_ms = (
            time.perf_counter() - request_start
        ) * 1000

        save_exchange(conversation_id, question, response.content)

        # Summarize older turns while retaining the latest 8 messages verbatim.
        try:
            update_conversation_summary(conversation_id, model)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Conversation summary update failed."
            )

        end_to_end_latency_ms = (
            time.perf_counter() - request_start
        ) * 1000

        return {
            "filename": request.filename,
            "conversation_id": conversation_id,
            "question": question,
            "retrieval_query": retrieval_query,
            "answer": response.content,
            "retrieved_context": [
                {
                    "page": doc.metadata.get("page", 0) + 1,
                    "text": doc.page_content,
                    "distance": round(distance, 4) if distance is not None else None,
                    "rerank_score": round(rerank_score, 4),
                }
                for doc, rerank_score, distance in selected_matches
            ],
            "reranker_model": RERANKER_MODEL_NAME,
            "reranker_max_length": RERANK_MAX_LENGTH,
            "reranker_batch_size": RERANK_BATCH_SIZE,
            "candidate_chunks_count": len(raw_candidate_matches),
            "unique_candidate_chunks_count": len(unique_candidate_matches),
            "deduplicated_chunks_count": (
                len(raw_candidate_matches) - len(unique_candidate_matches)
            ),
            "reranked_chunks_count": len(selected_matches),
            "source_pages_or_sections": [
                {"page": section}
                for section in source_sections
            ],
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "reranking_latency_ms": round(reranking_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {type(error).__name__}",
        )

@app.post("/summarize")
def summarize_document(request: SummarizeRequest):
    pages = load_document_pages(request.filename)

    full_text = "\n\n".join(
        f"Section {page['page']}:\n{page['text']}"
        for page in pages
        if page["text"].strip()
    )

    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found.",
        )

    try:
        model = get_language_model()

        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Create a clear study summary using only the provided text. "
                "Preserve important concepts, definitions, and facts. "
                "Do not invent information.",
            ),
            ("human", "Summarize this study material:\n{text}"),
        ])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=300,
        )

        sections = text_splitter.split_text(full_text)
        section_summaries = []

        for section in sections:
            result = (summary_prompt | model).invoke({"text": section})
            section_summaries.append(result.content)

        if len(section_summaries) == 1:
            final_summary = section_summaries[0]

        else:
            combine_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Combine the supplied section summaries into one "
                    "coherent study summary. Keep important facts and "
                    "avoid repetition. Do not invent information.",
                ),
                ("human", "Section summaries:\n{summaries}"),
            ])

            result = (combine_prompt | model).invoke({
                "summaries": "\n\n".join(section_summaries)
            })

            final_summary = result.content

        extension = Path(request.filename).suffix.lower()

        return {
            "filename": request.filename,
            "file_type": extension[1:].upper(),
            "page_count": len(pages) if extension == ".pdf" else None,
            "summary": final_summary,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {type(error).__name__}",
        )
