#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-http://127.0.0.1:8000}"
if command -v jq >/dev/null 2>&1; then
  PARSER=(jq ".")
else
  PARSER=(cat)
fi

echo "# Checking health"
curl -sS "${HOST}/health" | "${PARSER[@]}"
echo "# Sample answer"
curl -sS -X POST "${HOST}/answer" \
  -H "Content-Type: application/json" \
  -d '{"query":"Need to reset a workspace password","k":3}' | "${PARSER[@]}"