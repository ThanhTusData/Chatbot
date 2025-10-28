# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Clients                             │
│   Web Browser │ Mobile App │ API Client │ Desktop App       │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                   Load Balancer / Ingress                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼────────┐  ┌────────▼────────┐
│  Flask Web UI  │  │   FastAPI API   │
└────────────────┘  └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
         ┌─────────▼──────┐  ┌────────▼───────┐
         │ Intent Classifier│  │ Semantic Search│
         │  (LSTM+Attention)│  │ (FAISS + BERT) │
         └──────────────────┘  └────────────────┘
```

## Component Overview

### 1. Frontend Layer

#### Flask Web Application
- User-friendly chat interface
- Session management
- Real-time messaging
- Chat history

#### Streamlit Dashboard
- Analytics and visualization
- Model performance monitoring
- Interactive testing

#### PyQt5 Desktop App
- Native desktop experience
- Offline capability
- Voice integration

### 2. API Layer

#### FastAPI Server
- RESTful endpoints
- Async request handling
- Request validation (Pydantic)
- API documentation (Swagger)

**Key Endpoints:**
- `/chat` - Main chat endpoint
- `/predict` - Intent classification
- `/retrieve` - Document search
- `/health` - Health check

### 3. NLP Processing

#### Text Preprocessing
- Tokenization (spaCy)
- Lemmatization
- Stop word removal
- Entity extraction

#### Intent Classification
- Architecture: Bidirectional LSTM + Attention
- Input: Tokenized sequences
- Output: Intent probabilities
- Features:
  - Embedding layer (trainable)
  - Spatial dropout
  - Attention mechanism
  - Softmax classification

#### Semantic Retrieval
- Embedding: Sentence-BERT
- Vector Store: FAISS
- Similarity: Cosine similarity
- Top-K retrieval with threshold

### 4. Data Layer

#### Training Data
```
data/
├── raw/              # Original datasets
├── processed/        # Preprocessed data
└── kb/              # Knowledge base
    ├── faq.jsonl
    ├── product_info.jsonl
    └── technical_docs.jsonl
```

#### Model Artifacts
```
models/
├── intent/
│   ├── model.h5       # Trained model
│   ├── tokenizer.pkl  # Tokenizer
│   └── metadata.json  # Training info
└── embeddings/        # Cached embeddings
```

#### Vector Indexes
```
indexes/
└── kb/
    ├── index.faiss    # FAISS index
    ├── documents.json # Document store
    └── metadata.json  # Index metadata
```

### 5. Monitoring & Observability

#### Prometheus
- Request metrics
- Latency tracking
- Prediction distribution
- Error rates

#### Grafana
- Real-time dashboards
- Alert management
- Performance visualization

#### MLflow
- Experiment tracking
- Model versioning
- Parameter logging

### 6. Deployment

#### Docker
- Containerized services
- Multi-stage builds
- Layer caching

#### Kubernetes
- Horizontal scaling
- Load balancing
- Health checks
- Rolling updates

## Data Flow

### Chat Request Flow

```
1. User Input
   ↓
2. Web/API Interface
   ↓
3. Text Preprocessing (NLP)
   ↓
4. Intent Classification (LSTM)
   ↓
5. Semantic Search (FAISS)
   ↓
6. Response Generation
   ↓
7. Response to User
```

### Training Pipeline

```
1. Load Raw Data
   ↓
2. Preprocessing
   ↓
3. Train/Val/Test Split
   ↓
4. Tokenization
   ↓
5. Model Training
   ↓
6. Evaluation
   ↓
7. Model Saving
   ↓
8. MLflow Logging
```

## Technology Stack

### Backend
- **FastAPI**: API framework
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning
- **spaCy**: NLP processing
- **Transformers**: BERT models

### ML/NLP
- **Sentence-BERT**: Embeddings
- **FAISS**: Vector search
- **NLTK**: Text processing

### Storage
- **FAISS**: In-memory vector DB
- **JSON**: Document storage
- **Pickle**: Model serialization

### Monitoring
- **Prometheus**: Metrics
- **Grafana**: Visualization
- **MLflow**: ML tracking

### Deployment
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Nginx**: Load balancing

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Load balancer distribution
- Multiple worker processes

### Caching
- Model caching in memory
- Response caching (Redis)
- Vector index caching

### Optimization
- Batch prediction
- Model quantization
- Async processing

## Security

### API Security
- Rate limiting
- Input validation
- HTTPS/TLS
- API key authentication

### Data Security
- Encrypted storage
- Secure model artifacts
- Environment variables

## Performance Metrics

### Target SLAs
- API Response Time: < 100ms (p95)
- Model Inference: < 50ms
- Throughput: > 1000 req/s
- Availability: > 99.9%

### Monitoring
- Request latency
- Error rates
- Model accuracy
- System resources