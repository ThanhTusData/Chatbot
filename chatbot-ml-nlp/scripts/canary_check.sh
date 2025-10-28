#!/bin/bash

# Canary deployment health check script

set -e

API_URL="${1:-http://localhost:8000}"
MAX_RETRIES=10
RETRY_DELAY=5

echo "Starting canary deployment check for $API_URL"

# Function to check health
check_health() {
    local url=$1
    local response=$(curl -s -o /dev/null -w "%{http_code}" "$url/health")
    echo $response
}

# Function to test predictions
test_prediction() {
    local url=$1
    local response=$(curl -s -X POST "$url/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "hello", "top_k": 3}')
    echo $response
}

# Wait for service to be ready
echo "Waiting for service to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    status=$(check_health "$API_URL")
    
    if [ "$status" == "200" ]; then
        echo "✓ Service is healthy"
        break
    else
        echo "Attempt $i/$MAX_RETRIES: Service not ready (status: $status)"
        
        if [ $i -eq $MAX_RETRIES ]; then
            echo "✗ Service failed to become healthy"
            exit 1
        fi
        
        sleep $RETRY_DELAY
    fi
done

# Test prediction endpoint
echo "Testing prediction endpoint..."
prediction_result=$(test_prediction "$API_URL")

if echo "$prediction_result" | grep -q "predictions"; then
    echo "✓ Prediction endpoint working"
else
    echo "✗ Prediction endpoint failed"
    echo "Response: $prediction_result"
    exit 1
fi

# Test chat endpoint
echo "Testing chat endpoint..."
chat_result=$(curl -s -X POST "$API_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "hello"}')

if echo "$chat_result" | grep -q "response"; then
    echo "✓ Chat endpoint working"
else
    echo "✗ Chat endpoint failed"
    echo "Response: $chat_result"
    exit 1
fi

echo "✓ All canary checks passed!"
exit 0