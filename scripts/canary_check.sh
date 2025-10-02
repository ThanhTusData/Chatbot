#!/usr/bin/env bash
# scripts/canary_check.sh
# Simple canary health + predict + retrieve checks.
# Usage: ./scripts/canary_check.sh [BASE_URL]
# Optional env: CANARY_AUTH_HEADER (e.g. "Bearer dev-token")
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
AUTH_HEADER="${CANARY_AUTH_HEADER:-}"

curl_opts=("-sS" "--fail")

jq_exists=$(command -v jq || true)
if [ -z "$jq_exists" ]; then
  echo "Warning: jq not found. Install jq for JSON parsing. Exiting."
  exit 2
fi

echo "Canary check against $BASE_URL"

# helper to add auth if provided
_auth_header() {
  if [ -n "$AUTH_HEADER" ]; then
    echo "-H" "Authorization: $AUTH_HEADER"
  fi
}

echo "1) Checking /health..."
health_raw=$(curl "${curl_opts[@]}" $(_auth_header) "${BASE_URL%/}/health" || true)
if [ -z "$health_raw" ]; then
  echo "Health endpoint unreachable at ${BASE_URL%/}/health"
  exit 2
fi
echo "Health response: $health_raw"
model_loaded=$(echo "$health_raw" | jq -r '.model_loaded // "false"')
retrieval=$(echo "$health_raw" | jq -r '.retrieval // "false"')

if [ "$model_loaded" != "true" ] || [ "$retrieval" != "true" ]; then
  echo "Canary fail: model_loaded=$model_loaded retrieval=$retrieval"
  exit 3
fi
echo "Health OK (model loaded & retrieval initialized)."

echo "2) Checking /predict..."
predict_payload='{"texts":["canary test"]}'
predict_raw=$(curl "${curl_opts[@]}" -H "Content-Type: application/json" $(_auth_header) -d "$predict_payload" "${BASE_URL%/}/predict")
echo "Predict response: $predict_raw"
pred_count=$(echo "$predict_raw" | jq '.predictions | length' || echo "0")
if [ -z "$pred_count" ] || [ "$pred_count" -eq 0 ]; then
  echo "Canary fail: /predict returned no predictions"
  exit 4
fi
echo "/predict OK (predictions returned)."

echo "3) Checking /retrieve..."
retrieve_payload='{"query":"reset password","top_k":3}'
retrieve_raw=$(curl "${curl_opts[@]}" -H "Content-Type: application/json" $(_auth_header) -d "$retrieve_payload" "${BASE_URL%/}/retrieve" || true)
echo "Retrieve response: $retrieve_raw"
ret_count=$(echo "$retrieve_raw" | jq '.results | length' || echo "0")
# it's acceptable for results to be 0, but we want to ensure service responds; treat 0 as warning, not failure
if [ -z "$retrieve_raw" ]; then
  echo "Canary fail: /retrieve unreachable"
  exit 5
fi
echo "/retrieve responded (results count: $ret_count)."

echo "CANARY: all checks passed."
exit 0
