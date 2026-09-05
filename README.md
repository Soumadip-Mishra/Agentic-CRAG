# Agentic CRAG 🤖

An autonomous **Corrective Retrieval-Augmented Generation (CRAG)** system powered by **LangGraph**, **FastAPI**, and **Qdrant**, featuring self-correcting retrieval guardrails, hybrid search, and automated web fallback.

---

## ✨ Features

- **🤖 Autonomous State Machine** – Orchestrated via LangGraph to dynamically route queries between local generation, query rewriting, and web search fallback.
- **🔍 Hybrid Retrieval Engine** – Combines dense semantic vector search (Qdrant ANN) and sparse keyword retrieval (Okapi BM25) for high recall.
- **⚖️ Reciprocal Rank Fusion (RRF)** – Blends dense and sparse rankings using arithmetic score fusion without requiring manual calibration.
- **📄 Parent-Child Chunking** – Hierarchical indexing that searches granular child chunks (500 chars) while supplying comprehensive parent context (2,000 chars) to the generator.
- **🛡️ Document Relevance Grader** – LLM-powered evaluator that scores retrieved documents and strips out irrelevant context before generation.
- **🔍 Hallucination Guardrail** – Post-generation verification checking factual groundedness against source documents before returning an answer.
- **🌐 Web Search Fallback** – Automatically invokes Tavily Search when local knowledge base documents are missing, ambiguous, or irrelevant.
- **📁 Multi-Format Ingestion** – Automated parsers for PDF, DOCX, Markdown, and plain-text documents.
- **⚡ Production-Ready API** – Fully containerized FastAPI application with Pydantic v2 validation, healthchecks, and Redis caching infrastructure.

---

## 🛠️ Tech Stack

### 🧠 AI & Agent Orchestration
- **LangGraph** – State machine definition, conditional branching, and self-correction cycles
- **LangChain** – Prompt templates, document loaders, and output parsing
- **OpenAI (GPT-4o-mini)** – Generation, query rewriting, document grading, and groundedness checking
- **OpenAI (`text-embedding-3-small`)** – High-density vector embeddings
- **Tavily Search API** – Web search fallback when local retrieval is insufficient

### 🔍 Retrieval & Storage
- **Qdrant** – Vector database for high-performance approximate nearest neighbor (ANN) dense search
- **Rank-BM25** – In-memory sparse retrieval engine for keyword matching
- **NumPy** – Vector math and reciprocal rank fusion computation
- **Redis 7** – High-throughput cache layer for fast query responses

### 📄 Document Processing
- **PyMuPDF (`pymupdf`)** – High-speed PDF text and layout extraction
- **python-docx** – Document parsing for DOCX files
- **LangChain Text Splitters** – Recursive character chunking for parent and child documents

### ⚡ Backend & Infrastructure
- **Python 3.10+** – Core programming language
- **FastAPI** – High-performance async REST API framework
- **Uvicorn** – Lightweight ASGI web server
- **Pydantic v2 & Pydantic Settings** – Strict environment variable and schema validation
- **Docker & Docker Compose** – Containerized multi-service deployment

### 🧪 Testing & Evaluation
- **Ragas** – Evaluation framework measuring Faithfulness and Answer Relevancy
- **Pytest & pytest-asyncio** – Unit testing suite for agent routing and retrieval math
- **Loguru** – Structured colored terminal logging

---

## 🏗️ Architecture

```text
       User Query
           │
           ▼
    ┌──────────────┐
    │ Hybrid Search│ (Dense ANN + BM25 Sparse)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  RRF Fusion  │ (Reciprocal Rank Fusion)
    └──────┬───────┘
           │
           ▼
  ┌──────────────────┐
  │ Relevance Grader │ (LLM)
  └────────┬─────────┘
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
[Rel.]  [Ambig.]  [Irrel.]
  │        │        │
  ▼        ▼        ▼
 Gen.   Rewrite  Web Search
  │        │        │
  └────────┼────────┘
           ▼
  ┌──────────────────┐
  │  Hallucination   │ (LLM)
  │      Guard       │
  └────────┬─────────┘
           │
           ▼
        Response
```

---

## 🚀 Getting Started

Follow either of the execution methods below to get a local instance up and running.

### Prerequisites

- **Git** installed on your system
- **Docker & Docker Compose** (for Docker deployment or to host Qdrant & Redis)
- **Python 3.10+** (if running via local virtual environment)
- **OpenAI API Key** (for embeddings, grading, and answer generation)
- **Tavily API Key** (optional but recommended for web search fallback)

---

### Environment Variables

Clone the repository and set up your `.env` configuration:

```bash
git clone https://github.com/Soumadip-Mishra/Agentic-CRAG.git
cd Agentic-CRAG
cp .env.example .env
```

Open `.env` and fill in your keys:

```dotenv
# ──────────────────────────────────────────────
# LLM Provider
# ──────────────────────────────────────────────
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# ──────────────────────────────────────────────
# Vector Database (Qdrant)
# ──────────────────────────────────────────────
# Use 'localhost' for local development. Docker Compose automatically overrides to 'qdrant'.
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=agentic_crag
QDRANT_API_KEY=

# ──────────────────────────────────────────────
# Redis Cache
# ──────────────────────────────────────────────
# Use 'localhost' for local development. Docker Compose automatically overrides to 'redis'.
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# ──────────────────────────────────────────────
# Web Search Fallback (Tavily)
# ──────────────────────────────────────────────
TAVILY_API_KEY=your_tavily_api_key_here

# ──────────────────────────────────────────────
# Agent Thresholds & Tuning
# ──────────────────────────────────────────────
RELEVANCE_THRESHOLD=0.7
HALLUCINATION_THRESHOLD=0.8
MAX_RETRIES=3
RETRIEVAL_TOP_N=5

# ──────────────────────────────────────────────
# FastAPI Server Configuration
# ──────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

---

## 📦 Execution Method 1: Docker (Recommended)

This runs the entire stack (FastAPI application, Qdrant vector database, and Redis) inside isolated Docker containers.

### 1. Build and Start Services
```bash
docker compose up -d --build
```

### 2. Ingest Documents
Place your source files (`.pdf`, `.docx`, `.md`, or `.txt`) into the `data/raw/` directory on your host machine, then trigger ingestion inside the running container:

```bash
docker compose exec app python -m src.ingestion.pipeline
```

### 3. Check Health Status
```bash
curl http://localhost:8000/health
```

### 4. Query the System
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the primary routing algorithms in computer networks?"}'
```

### 5. Stop Containers & Clean Volumes (Optional)
```bash
# Stop containers
docker compose down

# To also wipe Qdrant vector storage and Redis cache:
docker compose down -v
```

---

## 💻 Execution Method 2: Local Virtual Environment (`venv`)

If you prefer developing and debugging directly on your host machine across **Linux**, **macOS**, or **Windows**:

### 1. Start Vector Database & Cache
Launch Qdrant and Redis in the background:
```bash
docker compose up -d qdrant redis
```

### 2. Create and Activate Virtual Environment

#### On Linux & macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If script execution is restricted on Windows, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first)*.

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. Ingest Documents
Add documents into `data/raw/`, then execute the pipeline:
```bash
python -m src.ingestion.pipeline
```

### 5. Launch the FastAPI Development Server
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Reference

### `POST /api/v1/chat`
Query the CRAG state machine.

**Request:**
```json
{
  "query": "How does the Sliding Window protocol manage flow control?"
}
```

**Response:**
```json
{
  "answer": "The Sliding Window protocol is a flow control mechanism...",
  "sources": [
    "data/raw/Lecture-Notes-Computer-Networks.pdf"
  ],
  "relevance_decision": "relevant",
  "hallucination_score": 0.92,
  "retries": 0
}
```

### `POST /api/v1/ingest`
Trigger ingestion over documents in a specified folder.

**Request:**
```json
{
  "data_dir": "data/raw"
}
```

### `GET /health`
Verify API service availability.

---

## 🧪 Evaluation & Testing

The repository separates logic unit tests from AI benchmark evaluation.

### 1. Unit & State Routing Tests (Fast & Free)
Runs all unit tests for state machine routing, parent-child chunking, and RRF math without calling external LLM APIs:
```bash
# In local venv:
pytest tests/ -v

# Or inside Docker:
docker compose exec app pytest tests/ -v
```

### 2. AI Benchmark Evaluation (Ragas)
Evaluates response accuracy against the curated benchmark dataset ([data/eval_dataset.json](file:///home/soumadip/Desktop/.NILU-Projects/Agentic-CRAG/data/eval_dataset.json)) measuring **Faithfulness** and **Answer Relevancy**:

```bash
# In local venv:
python -m eval.eval_ragas

# Or inside Docker:
docker compose exec app python -m eval.eval_ragas
```

### 3. Pytest Regression Benchmark Suite
Verifies minimum quality thresholds, non-empty outputs, and retry limits:
```bash
pytest eval/test_benchmark.py -v
```

---

## 📋 Roadmap & Next Implementations

- [ ] **Incremental Ingestion & Deduplication Pipeline**:
  - Track document checksums (SHA-256) in an `ingested_files.json` registry to skip re-embedding already processed files.
  - Deterministic Qdrant point IDs (e.g., UUIDv5 from file path and chunk index) for idempotent vector upserts.
  - Incremental BM25 corpus extension to append new documents while keeping existing indices intact.
- [ ] **Multipart File Upload API**:
  - Add a dedicated `POST /api/v1/upload` endpoint supporting drag-and-drop multipart document uploads.
- [ ] **Redis Semantic & Query Caching**:
  - Implement Redis caching layer to short-circuit identical user questions and cache frequent query responses.
- [ ] **API Security & Rate Limiting**:
  - Add API key authentication headers (`X-API-Key`) and token-bucket rate limiting middleware.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
