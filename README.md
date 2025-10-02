# Chatbot ML / NLP

> Production-ready Chatbot built with classical NLP, retrieval-based components, and modular serving layers.


## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Value](#business-value)
3. [Key Features](#key-features)
4. [Repository Layout](#repository-layout)
5. [Architecture & Data Flow](#architecture--data-flow)
6. [Technologies & Why They Matter](#technologies--why-they-matter)
7. [Quick Start (Developer)](#quick-start-developer)
8. [Usage & Examples](#usage--examples)
9. [Training & Building the KB Index](#training--building-the-kb-index)
10. [Testing & CI](#testing--ci)
11. [Observability & Monitoring](#observability--monitoring)
12. [Roadmap & Suggested Improvements](#roadmap--suggested-improvements)
13. [Troubleshooting Tips](#troubleshooting-tips)
14. [Contributing](#contributing)
15. [License](#license)

---

## Project Overview
This repository contains a modular Chatbot built with classical Machine Learning / NLP pipelines and semantic retrieval components. The bot supports intent classification, knowledge-base (KB) retrieval using embeddings, response generation, and multiple serving interfaces (REST API, Streamlit demo, desktop GUI). The codebase is organized to separate data preparation, model training, retrieval/indexing, and serving.

Use cases include: customer support automation, FAQ answering, internal knowledge assistant, proof-of-concept demos for conversational AI, and research/experimentation with retrieval-augmented generation.

---

## Business Value
- **Lower cost & higher efficiency:** Automate frequent customer interactions and triage issues, reducing workload on support teams.
- **Faster response times:** Provide 24/7 instant answers for common queries and reduce SLA time-to-first-response.
- **Knowledge centralization:** Maintain a single KB that can be updated and reflected in bot responses quickly.
- **Actionable analytics:** Track intent distribution, model performance, and drift to prioritize retraining and content improvements.
- **Extendibility:** Swap in new NLU models, vectorstores, or LLMs without major rewrites.

---

## Key Features
- Intent classification training pipeline and inference
- Text preprocessing (tokenization, normalization, PII scrubbing hooks)
- Embeddings-based semantic retrieval and local vector store helpers
- Hybrid retrieval + generator flow (retrieve KB docs → condition generator)
- Multiple serving options: Flask (sync), FastAPI (async), Streamlit demo, Desktop GUI
- Scripts for building and updating KB index
- Dockerized for reproducible environments
- Basic monitoring and drift detection stubs
- Unit & integration tests with pytest

---

## Repository Layout
A high-level view of the most important folders and files:

```
chatbot_ml_nlp/
├─ src/
│  ├─ nlp/                    # preprocessing pipeline & helpers
│  ├─ classification/         # intent classifier training & inference
│  ├─ retrieval/              # embeddings, index builders, vectorstore helpers
│  ├─ response/               # response ranking & templating
│  ├─ serving/                # FastAPI/Flask apps, schemas, auth
│  ├─ training/               # training scripts and configs
│  └─ data/                   # raw and processed datasets
├─ scripts/                   # utilities (e.g., build_kb_index.py)
├─ tests/                     # pytest tests
├─ observability/             # prometheus, grafana, drift checks
├─ Dockerfile
├─ docker-compose*.yaml
├─ requirements.txt
└─ README.md
```

---

## Architecture & Data Flow
1. **User message** arrives at API endpoint (Flask/FastAPI).
2. **Preprocessing**: clean text, remove PII, apply normalization and tokenization.
3. **Intent classification**: predict intent + confidence (fallback to retrieval/generation if low confidence).
4. **Retrieval**: embed user message and search vector store (local FAISS-like helper or external vectorstore) for relevant KB docs.
5. **Response generation**: either template-based, retrieval-ranked answer, or generator-conditioned answer (if integrated).
6. **Logging & feedback**: store conversation metadata for analytics and offline retraining.

This separation makes it easy to upgrade single components (e.g., replace classifier with a new model, swap vectorstore to Milvus).

---

## Technologies & Why They Matter
- **Python**: Rich ML ecosystem (scikit-learn, PyTorch/TensorFlow, Hugging Face).
- **FastAPI / Flask**: lightweight, production-ready HTTP APIs; FastAPI provides OpenAPI docs and async support.
- **Streamlit / Desktop UI**: simple UI for demos and user testing.
- **Embeddings & Vector Store**: enables semantic search and retrieval-augmented answers.
- **Docker & Docker Compose**: reproducible environments and easy local deployments.
- **MLflow**: experiments, model tracking and versioning (optional but recommended).
- **Pytest**: testing to avoid regressions.
- **Prometheus & Grafana**: metrics, dashboards, and alerts for production monitoring.

---

## Quick Start (Developer)
> Prerequisites: Python 3.10+, Docker (optional but recommended), Git

1. **Clone the repository**

```bash
git clone <repo-url>
cd chatbot_ml_nlp
```

2. **(Optional) Create virtual environment & install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. **Run with Docker Compose (recommended)**

```bash
docker compose -f docker-compose.dev.yaml up --build
```

4. **Run locally without Docker**
- Start FastAPI app for development:

```bash
uvicorn src.serving.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

- Or start Flask demo:

```bash
python src/web/flask_app.py
```

5. **Open demo UI**
- Streamlit demo: `streamlit run streamlit_app.py`
- FastAPI docs: `http://localhost:8000/docs`

---

## Usage & Examples
**Chat endpoint (example)**

`POST /api/chat`

Body:
```json
{
  "user_id": "user_123",
  "text": "How do I reset my password?",
  "context": {}
}
```

Response (example):
```json
{
  "reply": "To reset your password, ...",
  "intent": "password_reset",
  "confidence": 0.92,
  "sources": ["kb/doc_123"]
}
```

**Health check**: `GET /health` — returns service health and version.

---

## Training & Building the KB Index
**Train intent classifier (example)**

```bash
python src/training/train_intent.py --config configs/intent/config.json
```

**Build KB index**

Preprocess KB (clean & split into passages), produce embeddings, and persist vector index:

```bash
python scripts/build_kb_index.py --kb data/kb/faq.jsonl --out data/kb/index
```

Notes:
- Use `configs/` to tune embedding model, chunk size, nearest-neighbors, and distance metric.
- Store embeddings and index artifacts (e.g., in `data/kb/index/`) so serving code can load at startup.

---

## Testing & CI
- Unit & integration tests: `pytest -q`
- Add tests for new components and update CI config in `.github/workflows/ci.yml`.
- Use pre-commit hooks to maintain code quality.

---

## Observability & Monitoring
- Export metrics (request rate, latencies, intent distribution) via Prometheus client.
- Grafana dashboards in `observability/` provide charts for key metrics.
- Drift detection script (`monitoring/drift_check.py`) compares recent intent/vocabulary distributions to baseline and flags anomalies.

