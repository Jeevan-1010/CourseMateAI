
import hashlib
from io import BytesIO
from pathlib import Path, PureWindowsPath

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as WordDocument

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


class QuestionRequest(BaseModel):
    filename: str
    question: str


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
        # The collection may not exist yet.
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


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CourseMate AI API",
    }


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
        temporary = UPLOAD_DIR / f".uploading_{filename}"

        # Write and validate the replacement before replacing the old file.
        temporary.write_bytes(content)

        if extension == ".pdf":
            test_reader = PdfReader(str(temporary))
            if test_reader.is_encrypted:
                temporary.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted PDFs are not supported.",
                )

        # Replace an existing filename with the new upload.
        temporary.replace(destination)

        # Remove old embeddings so the replacement is indexed afresh.
        invalidate_document_index(filename)

        pages = load_document_pages(filename)
        has_text = any(page["text"].strip() for page in pages)

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
        raise
    except Exception as error:
        temporary = UPLOAD_DIR / f".uploading_{filename}"
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
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Enter a question.",
        )

    try:
        vectorstore = get_vectorstore(request.filename)
        matches = vectorstore.similarity_search(question, k=4)

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
                "Answer the student's question using only the supplied "
                "document context. If the answer is not present, say so. "
                "Do not invent facts.",
            ),
            (
                "human",
                "Document context:\n{context}\n\n"
                "Student question: {question}",
            ),
        ])

        response = (prompt | get_language_model()).invoke({
            "context": context,
            "question": question,
        })

        source_sections = sorted({
            doc.metadata.get("page", 0) + 1
            for doc in matches
        })

        return {
            "filename": request.filename,
            "question": question,
            "answer": response.content,
            "source_pages_or_sections": [
                {"page": section}
                for section in source_sections
            ],
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
