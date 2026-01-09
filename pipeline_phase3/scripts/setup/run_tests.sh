#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Running EDMO Pipeline Tests..."

# Unit tests
echo "Running unit tests..."
pytest tests/unit -v --cov=src --cov-report=html

# Integration tests
echo "Running integration tests..."
pytest tests/integration -v

# Go tests
echo "Running Go tests..."
cd src/go_core
go test ./... -v
cd ../..

echo "✅ All tests completed!"
