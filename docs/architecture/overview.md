# EDMO Pipeline Architecture

## Overview
The EDMO pipeline analyzes student communication and collaboration during educational robotics tasks.

## Components

### 1. Data Preprocessing
- Audio normalization (16kHz mono WAV)
- Video frame extraction
- Robot log parsing and cleaning

### 2. Feature Extraction
- **Speech-to-Text**: Whisper ASR with timestamps
- **Speaker Diarization**: Pyannote.audio
- **Emotion Detection**: Fine-tuned transformers
- **Prosodic Features**: Pitch, energy, rate extraction
- **Sentence Embeddings**: E5/BERT embeddings
- **Robot Features**: Movement metrics and action patterns

### 3. Analysis Pipeline
- Dimensionality reduction (PCA/t-SNE)
- Fuzzy C-means clustering
- Pattern identification
- PISA CPS framework validation

### 4. Feedback Generation
- Per-student psychological profiles
- Timeline visualization
- Teacher-facing reports

## Technology Stack
- **Go**: Pipeline orchestration and data synchronization
- **Python**: ML services (ASR, NLP, clustering, visualization)
- **C++**: High-performance ASR integration (Whisper.cpp)
- **FastAPI**: Microservice communication

## Data Flow
```
Audio/Video/Robot Logs → Preprocessing → Feature Extraction → 
  Aggregation → Clustering → Visualization → Feedback
```
