#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Starting EDMO Pipeline Services..."

# Check if venv is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    source venv/bin/activate
fi

# Start services in background
echo "Starting NLP service on port 8001..."
uvicorn src.python_services.nlp.app:app --host 0.0.0.0 --port 8001 &

echo "Starting ASR service on port 8002..."
uvicorn src.python_services.asr.app:app --host 0.0.0.0 --port 8002 &

echo "Starting Diarization service on port 8003..."
uvicorn src.python_services.diarization.app:app --host 0.0.0.0 --port 8003 &

echo "Starting Emotion service on port 8004..."
uvicorn src.python_services.emotion.app:app --host 0.0.0.0 --port 8004 &

echo "Starting Clustering service on port 8005..."
uvicorn src.python_services.clustering.app:app --host 0.0.0.0 --port 8005 &

echo "Starting Visualization service on port 8006..."
uvicorn src.python_services.visualization.app:app --host 0.0.0.0 --port 8006 &

echo "✅ All services started!"
echo "Use 'pkill -f uvicorn' to stop all services"
