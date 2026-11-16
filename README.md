# CourseMateAI

An AI-powered Retrieval-Augmented Generation (RAG) study assistant that lets students upload study materials, ask document-grounded questions, view extracted text, and generate study summaries.

## Overview

CourseMateAI combines a Streamlit interface with a FastAPI backend. Uploaded documents are processed, split into smaller chunks, embedded, and stored in a persistent ChromaDB vector database. When a student asks a question, the application retrieves relevant chunks and supplies them as context to a language model hosted by Groq.

The application is designed to answer from the selected document context and to say when the requested information is not available in that context.

## Features

- Upload multiple study documents in PDF, TXT, and DOCX formats.
- Extract text from uploaded documents.
- Ask questions about a selected document using retrieval-augmented generation.
- Generate study summaries from uploaded material.
- Keep document embeddings in a persistent ChromaDB store.
- Return source page or section references with question-answering responses.
- Validate request bodies with Pydantic models and validate uploaded file types and sizes.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| RAG orchestration | LangChain |
| Vector database | ChromaDB |
| Embedding model | Sentence Transformers - `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Language model | `openai/gpt-oss-20b`, accessed through the Groq API |
| PDF extraction | pypdf |
| DOCX extraction | python-docx |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Configuration | python-dotenv |

## RAG Workflow

1. **Upload:** The student uploads a PDF, TXT, or DOCX file through the Streamlit interface.
2. **Extract:** The FastAPI backend extracts text and preserves page information where available.
3. **Chunk:** LangChain's `RecursiveCharacterTextSplitter` divides the text into chunks using a configured chunk size of 1,000 characters and an overlap of 200 characters.
4. **Embed:** Sentence Transformers converts the chunks into numerical embeddings.
5. **Store:** ChromaDB stores the embeddings, text, and document metadata persistently.
6. **Retrieve:** The student's question is embedded with the same embedding model. ChromaDB similarity search retrieves the top 4 chunks.
7. **Generate:** The retrieved context and question are passed to the Groq-hosted `openai/gpt-oss-20b` model. The prompt instructs the model to answer from the supplied context and not invent missing information.
8. **Respond:** The generated answer and source page or section references are returned to the frontend.

### Architecture

```text
Student
   |
   v
Streamlit frontend (frontend.py)
   |
   | HTTP requests
   v
FastAPI backend (backend/app/main.py)
   |
   +--> Document extraction (PDF / TXT / DOCX)
   |
   +--> Recursive character text splitting
   |
   +--> Sentence Transformers embeddings
   |
   +--> ChromaDB persistent vector store
   |
   +--> Similarity search (top 4 chunks)
   |
   +--> Groq API: openai/gpt-oss-20b
   |
   v
Answer / summary returned to Streamlit
```

## API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Returns the API health status. |
| `POST` | `/upload` | Uploads and validates a PDF, TXT, or DOCX file. Send the file as multipart form data using the `file` field. |
| `GET` | `/extract/{filename}` | Extracts the document text and returns its pages or sections. |
| `POST` | `/ask` | Answers a question using retrieved document context. |
| `POST` | `/summarize` | Generates a study summary for an uploaded document. |

### Request examples

Question answering:

```json
{
  "filename": "lecture-notes.pdf",
  "question": "What are the main concepts covered in this chapter?"
}
```

Summarization:

```json
{
  "filename": "lecture-notes.pdf"
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs` when the backend is running.

## Project Structure

The main application files are organized as follows:

```text
CourseMateAI/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI application and API endpoints
│   ├── uploads/             # Uploaded documents (created at runtime)
│   ├── Chroma-db/           # Persistent vector database (created at runtime)
│   └── requirements.txt     # Backend dependencies
├── frontend.py              # Streamlit user interface
├── requirements.txt          # Project dependencies
├── pyproject.toml            # Python project configuration
└── .gitignore
```

Uploaded documents, the local ChromaDB database, and environment files are intended to remain local and are excluded from version control.

## Getting Started

### Prerequisites

- Python 3.12 or later.
- A Groq API key.
- Git.

### 1. Clone the repository

```powershell
git clone https://github.com/Jeevan-1010/CourseMateAI.git
cd CourseMateAI
```

### 2. Create and activate a virtual environment

Using `uv` on Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use the Python executable from the virtual environment directly or adjust the current user's PowerShell execution policy according to your environment.

### 3. Install dependencies

From the repository root:

```powershell
uv pip install -r requirements.txt
```

Alternatively, after activating the virtual environment:

```powershell
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the repository root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit your `.env` file or expose your API key.

### 5. Start the FastAPI backend

Open a PowerShell terminal in the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload --reload-dir backend/app
```

The API will be available at:

- API: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs

### 6. Start the Streamlit frontend

Open a second PowerShell terminal in the repository root, activate the same virtual environment, and run:

```powershell
streamlit run frontend.py
```

Streamlit will display the local URL for the application, usually http://localhost:8501.

Keep the FastAPI backend running while using the Streamlit frontend.

## Configuration

The current backend configuration includes:

| Setting | Value |
|---|---|
| Allowed document formats | PDF, TXT, DOCX |
| Maximum upload size | 20 MB |
| Embedding model | `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Chunk size for indexing | 1,000 characters |
| Chunk overlap for indexing | 200 characters |
| Retrieved chunks for Q&A | 4 |
| Vector database | Persistent ChromaDB |
| LLM | `openai/gpt-oss-20b` through Groq |

The summarization pipeline uses a separate text-splitting configuration to process longer documents before combining section summaries.

## Evaluation

The project was evaluated using a 10-question test set containing 8 answerable and 2 unanswerable questions. The recorded results from the final evaluation run were:

| Evaluation area | Metric | Recorded result |
|---|---|---:|
| Retrieval | Page precision | 45.83% |
| Retrieval | Expected-page recall | 100% |
| Retrieval performance | Average retrieval latency | 79.624 ms |
| Generation | Correctness | 60.72% |
| Generation | Faithfulness | 91.1% |
| Generation | Answer relevancy | 84.71% |
| Refusal behavior | Correct refusal of unanswerable questions | 100% (2 of 2) |
| Overall performance | Average end-to-end latency | 2,312.814 ms (about 2.31 seconds) |

These are results from the project's small evaluation set and evaluation method, not results from a large external benchmark. The page precision result indicates that some retrieved pages were not relevant to the corresponding questions. The evaluation also showed that generation time was the largest contributor to end-to-end latency.

## Limitations

- Scanned or image-only PDFs may not yield extractable text; OCR is not currently part of the documented upload pipeline.
- Retrieval can include irrelevant passages, so page precision can be improved.
- Answer quality depends on the content and quality of the uploaded material and on the relevance of retrieved chunks.
- LLM generation accounts for most of the measured response time.
- The evaluation set contains only 10 questions, so the recorded metrics should be treated as preliminary.

## Future Improvements

- Improve retrieval precision through better chunking experiments, metadata filtering, similarity thresholds, or reranking.
- Expand the evaluation dataset with more documents and questions.
- Add OCR support for scanned documents.
- Improve response latency and generation efficiency.
- Improve source attribution and document-level traceability.

## Security Notes

- Keep API keys in the local `.env` file.
- Do not commit uploaded study documents, local vector database files, or secrets.
- The backend currently stores uploaded files and vector data on the machine where it runs. This project should be deployed with appropriate authentication, access controls, and storage protections before being exposed to untrusted users.

## License

No license information is currently specified in this repository.
