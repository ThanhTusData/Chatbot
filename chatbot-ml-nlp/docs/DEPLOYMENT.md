# Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Python 3.9+

## Local Development

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download models
python -m spacy download en_core_web_sm

# Train model
make train

# Build knowledge base
make build-kb

# Run API server
make serve-api

# Run web interface (in another terminal)
make serve-web
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -f deployment/docker/Dockerfile -t chatbot:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/indexes:/app/indexes \
  --name chatbot-api \
  chatbot:latest
```

### Docker Compose

```bash
# Start all services
docker-compose -f deployment/compose/docker-compose.yaml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Configure cluster access
kubectl config use-context your-cluster
```

### Deploy

```bash
# Create namespace
kubectl create namespace chatbot

# Apply configurations
kubectl apply -f deployment/kubernetes/ -n chatbot

# Check status
kubectl get pods -n chatbot
kubectl get svc -n chatbot

# View logs
kubectl logs -f deployment/chatbot-api -n chatbot
```

### Scale

```bash
# Scale API deployment
kubectl scale deployment chatbot-api --replicas=5 -n chatbot

# Auto-scaling
kubectl autoscale deployment chatbot-api \
  --min=2 --max=10 \
  --cpu-percent=80 \
  -n chatbot
```

## Cloud Deployment

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URL
docker tag chatbot:latest YOUR_ECR_URL/chatbot:latest
docker push YOUR_ECR_URL/chatbot:latest

# Deploy to ECS
aws ecs update-service \
  --cluster production \
  --service chatbot-api \
  --force-new-deployment
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/chatbot

# Deploy
gcloud run deploy chatbot-api \
  --image gcr.io/PROJECT_ID/chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Login to ACR
az acr login --name yourregistry

# Build and push
docker tag chatbot:latest yourregistry.azurecr.io/chatbot:latest
docker push yourregistry.azurecr.io/chatbot:latest

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name chatbot-api \
  --image yourregistry.azurecr.io/chatbot:latest \
  --ports 8000
```

## Environment Variables

Create a `.env` file:

```bash
APP_ENV=production
DEBUG=False
LOG_LEVEL=INFO

API_HOST=0.0.0.0
API_PORT=8000

INTENT_MODEL_PATH=/app/models/intent/model.h5
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
KB_INDEX_PATH=/app/indexes/kb/

ENABLE_PROMETHEUS=True
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Health Checks

```bash
# API health
curl http://localhost:8000/health

# Detailed check
curl http://localhost:8000/health | jq
```

## Monitoring

Access monitoring dashboards:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs chatbot-api

# Check resource usage
docker stats chatbot-api
```

### Model loading issues

```bash
# Verify model files
docker exec chatbot-api ls -lh /app/models/intent/

# Check permissions
docker exec chatbot-api ls -la /app/models/
```

### High memory usage

```bash
# Set memory limits
docker run -d \
  --memory="2g" \
  --memory-swap="2g" \
  chatbot:latest
```

## Performance Tuning

### Gunicorn Configuration

```bash
gunicorn src.serving.fastapi_app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --keep-alive 5
```

### Resource Limits (K8s)

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```