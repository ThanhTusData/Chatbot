# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production, implement API keys or OAuth2.

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true
}
```

### Chat

**POST** `/chat`

Send a message and get a response.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "response": "I'm doing great! How can I help you today?",
  "intent": {
    "intent": "greeting",
    "confidence": 0.98
  },
  "retrieved_docs": [],
  "timestamp": "2025-01-20T10:30:00"
}
```

### Predict Intent

**POST** `/predict`

Classify the intent of a text.

**Request:**
```json
{
  "text": "I need help with my account",
  "top_k": 3
}
```

**Response:**
```json
{
  "predictions": [
    {"intent": "support", "confidence": 0.92},
    {"intent": "account_inquiry", "confidence": 0.85},
    {"intent": "technical_support", "confidence": 0.78}
  ],
  "processing_time": 0.015
}
```

### Retrieve Documents

**POST** `/retrieve`

Search for relevant documents.

**Request:**
```json
{
  "query": "shipping information",
  "top_k": 5,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Standard shipping takes 5-7 business days...",
      "score": 0.92,
      "metadata": {"source": "faq", "category": "shipping"}
    }
  ],
  "query": "shipping information"
}
```

## Error Responses

**400 Bad Request**
```json
{
  "detail": "Invalid request format"
}
```

**503 Service Unavailable**
```json
{
  "detail": "Model not loaded"
}
```

## Rate Limiting

- Default: 100 requests per minute
- Burst: 200 requests per minute

## Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I need help", "top_k": 3}'
```

### Python

```python
import requests

# Chat
response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hello"}
)
data = response.json()
print(data['response'])

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I need help", "top_k": 3}
)
predictions = response.json()['predictions']
```

### JavaScript

```javascript
// Chat
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: 'Hello'})
})
.then(res => res.json())
.then(data => console.log(data.response));
```