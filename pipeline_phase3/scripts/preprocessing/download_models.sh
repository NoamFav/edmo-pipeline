#!/usr/bin/env bash
set -euo pipefail

echo "📦 Downloading required models..."

# SpaCy Dutch model
python -m spacy download nl_core_news_sm

# Download Whisper model (optional - done at runtime)
# whisper --model base --download-root data/models dummy.wav

echo "✅ Models downloaded!"
