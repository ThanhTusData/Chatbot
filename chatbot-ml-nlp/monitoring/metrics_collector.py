from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps
from typing import Callable

# Metrics
REQUEST_COUNT = Counter('chatbot_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('chatbot_request_latency_seconds', 'Request latency', ['endpoint'])
MODEL_PREDICTIONS = Counter('chatbot_predictions_total', 'Total predictions', ['intent'])
MODEL_CONFIDENCE = Histogram('chatbot_prediction_confidence', 'Prediction confidence scores')
ACTIVE_SESSIONS = Gauge('chatbot_active_sessions', 'Active chat sessions')

def track_request(endpoint: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
        return wrapper
    return decorator

def track_prediction(intent: str, confidence: float):
    MODEL_PREDICTIONS.labels(intent=intent).inc()
    MODEL_CONFIDENCE.observe(confidence)

def start_metrics_server(port: int = 8001):
    start_http_server(port)