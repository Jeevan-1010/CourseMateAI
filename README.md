# CourseMateAI

An AI-powered Retrieval-Augmented Generation (RAG) study assistant that lets students upload study materials, ask document-grounded questions, view extracted text, and generate study summaries.

## Overview

CourseMateAI combines a Streamlit interface with a FastAPI backend. Uploaded documents are processed, split into smaller chunks, embedded, and stored in a persistent ChromaDB vector database. Neon PostgreSQL, accessed through SQLAlchemy, stores document metadata, conversations, messages, and rolling conversation summaries so chat history and conversational context can be restored across application restarts. When a student asks a question, the application retrieves relevant chunks and supplies them as context to a language model hosted by Groq.

The application retrieves the top 4 chunks for each question and passes them to the language model as document context. It does not apply a similarity-distance threshold to filter those results. The model is instructed to answer only when the supplied context supports an answer; otherwise, it should state that the uploaded document does not provide enough information. This refusal behavior happens at the answer-generation stage, not because retrieval returns zero chunks.

## Features

- Upload multiple study documents in PDF, TXT, and DOCX formats.
- Extract text from uploaded documents.
- Ask questions about a selected document using retrieval-augmented generation.
- Maintain conversational context across follow-up questions within the current chat session.
- Rewrite context-dependent follow-up questions into standalone retrieval queries before searching the document.
- Generate study summaries from uploaded material.
- Keep document embeddings in a persistent ChromaDB vector store.
- Persist document metadata, conversations, messages, and rolling conversation summaries in Neon PostgreSQL.
- Restore saved conversations and their context, combining a stored summary with the latest 8 messages to support longer-running chats.
- Return source page or section references with question-answering responses.
- Inspect the retrieved chunks, including their page numbers, text, and vector-search distances.
- Measure retrieval, reranking, generation, and end-to-end question-answering latency.
- Deduplicate redundant retrieval candidates and use MMR diversity retrieval for broad or multi-part questions.
- Rerank up to 12 candidate chunks with `BAAI/bge-reranker-v2-m3`, then send the top 4 chunks to the language model.
- Support ONNX Runtime dynamic INT8 reranking on CPU, with automatic PyTorch fallback when the ONNX model file is unavailable.
- Validate request bodies with Pydantic models and validate uploaded file types and sizes.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| RAG orchestration | LangChain |
| Vector database | ChromaDB |
| Persistent relational database | Neon PostgreSQL, accessed through SQLAlchemy |
| Embedding model | Sentence Transformers - `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Reranker | `BAAI/bge-reranker-v2-m3` (ONNX Runtime dynamic INT8 or PyTorch FP32 fallback) |
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
6. **Understand the conversation:** When prior messages are available, the backend uses them to rewrite a follow-up question as a standalone retrieval query. Conversation history is used to resolve references, not as factual evidence.
7. **Retrieve and rerank:** ChromaDB similarity search, aspect-level retrieval, and MMR diversity retrieval produce candidates. Exact and near-duplicate chunks are removed, up to 12 unique candidates are scored by `BAAI/bge-reranker-v2-m3`, and the top 4 are passed to the language model.
8. **Persist conversational memory:** Neon PostgreSQL stores conversations and message exchanges. Older messages are periodically condensed into a rolling summary, while the latest 8 messages are retained as recent context. On a returning session, the summary and recent messages are loaded from the database.
9. **Generate:** The retrieved context, latest question, and conversation history are passed to the Groq-hosted `openai/gpt-oss-20b` model. The prompt instructs the model to answer from the supplied document context and not invent missing information.
10. **Respond:** The generated answer, retrieval query, retrieved chunks, source page or section references, and latency measurements are returned to the frontend.

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
   +--> MMR retrieval, deduplication, and reranking
   |
   +--> Neon PostgreSQL (SQLAlchemy)
   |      +--> Document metadata
   |      +--> Conversations and messages
   |      +--> Rolling conversation summaries
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
| `GET` | `/health` | Returns the API health status, including the Neon PostgreSQL connection check. |
| `GET` | `/documents` | Lists persisted document metadata. |
| `GET` | `/documents/{filename}/conversations` | Lists saved conversations associated with a document. |
| `GET` | `/conversations/{conversation_id}` | Restores the messages for a saved conversation. |
| `POST` | `/upload` | Uploads and validates a PDF, TXT, or DOCX file. Send the file as multipart form data using the `file` field. |
| `GET` | `/extract/{filename}` | Extracts the document text and returns its pages or sections. |
| `POST` | `/ask` | Answers a question using retrieved document context. |
| `POST` | `/retrieve` | Runs retrieval, deduplication, and reranking without calling the language model; useful for retrieval evaluation and diagnostics. |
| `POST` | `/summarize` | Generates a study summary for an uploaded document. |

### Request examples

Question answering:

```json
{
  "filename": "lecture-notes.pdf",
  "question": "What are the main concepts covered in this chapter?",
  "history": [
    {"role": "user", "content": "What is a primary key?"},
    {"role": "assistant", "content": "A primary key uniquely identifies each row in a table."}
  ]
}
```

The `history` field is optional and defaults to an empty list. Each message must have a `role` of `user` or `assistant` and a `content` string. The frontend sends recent messages from the current chat session; the backend uses up to the latest 24 messages. Starting a new chat clears that conversation history. When history is supplied, the backend may make an additional model call to rewrite a follow-up question before retrieval.

The `/ask` response includes `retrieval_query`, `retrieved_context` (page, text, vector distance, and reranker score for each returned chunk), `source_pages_or_sections`, candidate and deduplication counts, the active `reranker_backend`, and retrieval, reranking, generation, and end-to-end latency measurements. Retrieval distances are diagnostic information; the current implementation does not apply a distance threshold to filter chunks. `/retrieve` exposes retrieval and reranking diagnostics without making an LLM generation call.

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

Uploaded document files, the local ChromaDB database, generated ONNX model artifacts, and environment files are intended to remain local and are excluded from version control. Neon PostgreSQL is used for relational persistence; the uploaded file contents and ChromaDB vector index are not stored in Neon.

## Getting Started

### Prerequisites

- Python 3.12 or later.
- A Groq API key.
- A Neon PostgreSQL database and its connection URL.
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
DATABASE_URL=postgresql+psycopg://username:password@host/database?sslmode=require
```

Set `DATABASE_URL` to your Neon connection string using the SQLAlchemy-compatible `postgresql+psycopg://` scheme. Do not commit your `.env` file, database credentials, or API key.

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
| Unique reranker candidates for Q&A | Up to 12 |
| Context chunks sent to the LLM | 4 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Reranker max sequence length | 256 tokens |
| Reranker batch size | 32 |
| Reranker backend | `auto` (ONNX INT8 when the local model exists; otherwise PyTorch FP32) |
| Vector database | Persistent ChromaDB |
| Relational persistence | Neon PostgreSQL via SQLAlchemy |
| Conversation memory | Persisted message history and rolling summaries; latest 8 messages loaded with the summary |
| LLM | `openai/gpt-oss-20b` through Groq |

The summarization pipeline uses a separate text-splitting configuration to process longer documents before combining section summaries.

### ONNX INT8 reranker

The backend automatically uses the locally exported ONNX dynamic-INT8 model when this file exists:

`evaluation/models/bge-reranker-v2-m3-onnx/int8/model_quantized.onnx`

The model is excluded from Git because it is a generated binary artifact. Generate it with the benchmark workflow in `evaluation/benchmark_reranker.py` and keep it at the path above. The backend then loads the ONNX model with ONNX Runtime and downloads the matching tokenizer from Hugging Face on first use.

Optional environment settings:

| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_BACKEND` | `auto` | `auto`, `onnx_int8`, or `pytorch` |
| `RERANKER_ONNX_MODEL_PATH` | `evaluation/models/bge-reranker-v2-m3-onnx/int8/model_quantized.onnx` | Override the ONNX model path |
| `RERANKER_ONNX_THREADS` | `4` | ONNX Runtime CPU intra-op threads |

Use `RERANKER_BACKEND=onnx_int8` to require the quantized model (the first reranking request returns an error if it is missing). Use `pytorch` to force the Sentence Transformers implementation. The default `auto` mode falls back to PyTorch if the ONNX file or its runtime cannot be loaded. API responses expose the active backend as `reranker_backend`.


## Evaluation

### Retrieval quality

The current retrieval evaluation uses 24 SQL questions: 21 answerable and 3 unanswerable. The reranking candidate study compared the top 4 results before and after reranking.

| Metric | Similarity baseline | Reranked (12 candidates) |
|---|---:|---:|
| Precision@4 | 38.10% | 42.86% |
| Recall@4 | 77.70% | 84.60% |
| Mean Reciprocal Rank (MRR) | 82.94% | 97.62% |
| Hit@1 | 71.43% | 95.24% |
| NDCG@4 | 73.94% | 85.87% |

The candidate-count comparison found that increasing the reranker input from 12 to 16 candidates did not improve these quality metrics. The application therefore uses up to 12 unique candidates before selecting the final 4 context chunks.

### Reranker inference benchmark

A controlled benchmark used the same 24 questions and 288 query-passage pairs, with 3 warm-up runs and 5 measured repeats. Timing includes tokenization and model inference, but excludes model loading, ONNX export, and quantization.

| Backend | Average reranking latency | Precision@4 | Recall@4 | MRR |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 5,039.96 ms | 42.86% | 84.60% | 97.62% |
| ONNX Runtime FP32 | 4,362.21 ms | 42.86% | 84.60% | 97.62% |
| ONNX Runtime dynamic INT8 | 1,404.96 ms | 42.86% | 84.60% | 97.62% |

ONNX dynamic INT8 reduced average reranker latency by approximately 72.1% compared with PyTorch FP32 (about 3.59x faster) while preserving the measured retrieval metrics on this evaluation set.

### Live integration check

The integrated FastAPI endpoint was tested with the ONNX backend. The response reported `reranker_backend: onnxruntime_dynamic_int8`. A repeated live request recorded 5,459.63 ms reranking latency and 10,093.32 ms end-to-end latency. This single live request is an integration check, not a replacement for the controlled benchmark; live timings include application and runtime overhead.

All results are from a small, document-specific evaluation set and should be treated as experimental rather than as a general performance guarantee.

## Limitations

- Scanned or image-only PDFs may not yield extractable text; OCR is not currently part of the documented upload pipeline.
- Retrieval can still include irrelevant passages; evaluation on additional subjects and document types is needed.
- Answer quality depends on the content and quality of the uploaded material and on the relevance of retrieved chunks.
- LLM generation accounts for most of the measured response time.
- The evaluation set contains only 10 questions, so the recorded metrics should be treated as preliminary.

## Future Improvements

- Expand the evaluation dataset across more subjects, document formats, and question types.
- Add OCR support for scanned documents.
- Profile live reranking and end-to-end latency under repeated requests and different hardware.
- Improve source attribution and document-level traceability.

## Security Notes

- Keep API keys in the local `.env` file.
- Do not commit uploaded study documents, local vector database files, or secrets.
- The backend currently stores uploaded files and vector data on the machine where it runs. This project should be deployed with appropriate authentication, access controls, and storage protections before being exposed to untrusted users.

## License

No license information is currently specified in this repository.

