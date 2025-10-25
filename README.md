# EDMO Communication Analysis Pipeline

A modular, multimodal analytics pipeline for measuring collaboration and communication patterns in educational robotics sessions. It processes audio, robot interaction logs, and derived linguistic and behavioral features to generate interpretable, actionable teacher feedback.

---

## Overview

The pipeline detects speaking patterns, turn-taking dynamics, emotional tone, and collaboration strategies during group problem-solving sessions. Outputs are aggregated into visualization dashboards and teacher-oriented summaries that support non-evaluative feedback.

### Core Features

| Component                          | Description                                                          |
| ---------------------------------- | -------------------------------------------------------------------- |
| **Speech Processing**              | Whisper-based ASR, PyAnnote diarization, per-speaker segmentation    |
| **Linguistic Analysis**            | Embeddings, keyword extraction, sentiment classification             |
| **Non-Verbal Dynamics**            | Turn duration statistics, overlap rate, interruption patterns        |
| **Clustering & Pattern Discovery** | PCA / Sparse PCA dimensionality reduction + fuzzy C-means clustering |
| **Visualization Suite**            | Timeline plots, explained-variance charts, radar skill profiles      |
| **Teacher-Focused Output**         | Structured summaries suitable for formative assessment contexts      |

---

## Documentation & Resources

- **📄 Technical Report**: [EDMO Pipeline Documentation (PDF)](./docs/EDMO_Pipeline_Documentation.pdf) — Comprehensive system architecture, methodology, and evaluation
- **🎥 Demo Video**: [Pipeline Demonstration (MP4)](./docs/EDMO_Demo.mp4) — End-to-end walkthrough of the analysis pipeline

---

## Architecture

| Layer                | Technology        | Purpose                                                       |
| -------------------- | ----------------- | ------------------------------------------------------------- |
| **Orchestration**    | Go                | Pipeline control, scheduling, data flow, service coordination |
| **ML/NLP Services**  | Python (FastAPI)  | ASR, diarization, sentiment, clustering, visualization        |
| **Optional Backend** | C++ (Whisper.cpp) | Efficient ASR on edge hardware                                |

### Service Layout

```
src/python_services/
├── asr/                # Whisper inference
├── diarization/        # Speaker segmentation
├── emotion/            # Sentiment / tone classification
├── nlp/                # Keyword + embedding extraction
├── clustering/         # PCA / Fuzzy C-Means / dimensionality reduction
├── nonverb_features/   # Turn-taking + overlap metrics
├── robot_data/         # Robot log parsing + control metrics
└── visualization/      # Plot generation (timeline, radar, variance)
```

---

## Project Structure

```
edmo-pipeline/
├── config/                      # Pipeline + service configuration
│   └── dev/
│       └── config.yaml
│
├── data/                        # Runtime data (gitignored in production)
│   ├── models/                  # Local model cache (HuggingFace / safetensors)
│   │   └── hf/                  # HuggingFace source-style structure
│   └── outputs/                 # Pipeline session outputs (clusters, plots, reports)
│
├── docs/                        # Documentation, presentations, showcase assets
│   ├── EDMO_Pipeline_Documentation.pdf  # Technical report
│   └── EDMO_Demo.mp4                    # Demo video
│
├── notebooks/                   # Research + exploration notebooks
│
├── scripts/                     # Helper shell scripts
│   ├── convert_audio.sh
│   └── extract_frames.sh
│
├── src/
│   ├── data_pipeline/           # Preprocessing utilities (normalization, extraction)
│   │   ├── convert_audio.sh
│   │   └── extract_frames.sh
│   │
│   ├── go_core/                 # Go orchestration layer (main pipeline)
│   │   ├── clients/             # HTTP clients for each microservice
│   │   ├── config/
│   │   ├── orchestrator/        # Pipeline control + analysis logic
│   │   ├── main.go              # Entry point
│   │   └── go.mod / go.sum
│   │
│   ├── python_services/         # ML/NLP microservice suite (FastAPI)
│   │   ├── asr/                 # Whisper speech recognition
│   │   ├── diarization/         # Speaker segmentation
│   │   ├── emotion/             # Emotion classification
│   │   ├── nlp/                 # Keywords / embeddings
│   │   ├── clustering/          # PCA / Fuzzy C-Means / Dim-red
│   │   ├── nonverb_features/    # Turn-taking + overlap metrics
│   │   ├── movement_tracker/    # (optional) physical robot movement analytics
│   │   ├── robot_data/          # Robot log parsing + derived control metrics
│   │   └── visualization/       # Timeline + radar + variance plots
│   │
│   └── nlp/                     # Legacy / development NLP utilities
│       ├── app.py
│       ├── processor.py
│       └── strategies.py
│
├── docker-compose.yml           # Multi-service runtime config
├── Dockerfile                   # Base build image
├── Makefile                     # Dev automation commands
├── requirements.txt             # Python dependency root
├── go.work                      # Go multi-module workspace config
├── LICENSE
└── README.md
```

---

## Quick Start

### Requirements

- Python 3.10+
- Go 1.21+
- Docker & Docker Compose
- FFmpeg (for audio normalization)

### Setup

```bash
git clone <repo>
cd <repo-name>
```

### Running the Pipeline

#### 1. Start All Microservices

```bash
docker compose up -d --build
```

This will start all Python FastAPI services (ASR, clustering, emotion, etc.) in detached mode.

#### 2. Run the Go Orchestrator

```bash
CONFIG_ENV=dev go run ./src/go_core <path/to/audio.wav>
```

**Example:**

```bash
CONFIG_ENV=dev go run ./src/go_core ./docs/jfk.wav
```

The orchestrator will:

1. Send audio to ASR service for transcription
2. Optionally run diarization for speaker identification
3. Extract linguistic features (emotion, keywords)
4. Compute non-verbal metrics (turn-taking, overlap)
5. Perform clustering on time-windowed features
6. Generate visualizations (timeline, radar, variance charts)
7. Save outputs to `data/outputs/session_YYYYMMDD-HHMMSS/`

---

## Output Examples

| Output                   | Format        | Description                                                |
| ------------------------ | ------------- | ---------------------------------------------------------- |
| `clusters.json`          | JSON          | Collaboration pattern assignments with membership matrix   |
| `windows.json`           | JSON          | Time-windowed features (emotions, overlap, speaking share) |
| `timeline.png`           | Visualization | Speaking timeline by cluster assignment                    |
| `ExplainedVariances.png` | Visualization | Variance retained per PCA/SPCA dimension                   |
| `radar.png`              | Visualization | Multi-dimensional skill profile                            |

### Sample Output Structure

```json
{
  "session_id": "session_20250124-143052",
  "audio_path": "/data/inputs/session.wav",
  "generated_at": "2025-01-24T14:35:18Z",
  "clusters": [0, 1, 1, 2, 0, 1, ...],
  "membership_matrix": [[0.85, 0.10, 0.05], ...]
}
```

### Viewing Results

After processing completes, check the session output directory:

```bash
ls -la data/outputs/session_20250124-143052/
# clusters.json          # Cluster assignments and membership
# windows.json           # Time-windowed feature vectors
# timeline.png           # Visual timeline of collaboration patterns
# ExplainedVariances.png # PCA/SPCA variance analysis
# radar.png              # Multi-dimensional profile chart
```

For a complete walkthrough of output interpretation, see the [demo video](./docs/EDMO_Demo.mp4).

---

## Configuration

Edit `config/dev/config.yaml` to adjust:

- Service URLs and ports
- Feature extraction windows (default: 30s window, 15s overlap)
- Clustering parameters (default: 5 clusters, 3 PCA components)
- Dimensionality reduction method (PCA or SparsePCA)

Example configuration snippet:

```yaml
services:
    asr:
        url: "http://127.0.0.1:8002"
    clustering:
        url: "http://127.0.0.1:8005"

features:
    time_window: 30 # seconds
    overlap: 15 # seconds

clustering:
    algorithm: "fuzzy_cmeans"
    n_clusters: 5
    dimensionality_reduction:
        method: "PCA" # or "SparsePCA"
        n_components: 3
```

For detailed configuration options and system design rationale, refer to the [technical documentation](./docs/EDMO_Pipeline_Documentation.pdf).

---

## Development

### Service Health Checks

```bash
# Check if all services are running
docker compose ps

# View logs for a specific service
docker compose logs -f asr
docker compose logs -f clustering

# Test individual service endpoints
curl http://localhost:8002/health  # ASR
curl http://localhost:8005/health  # Clustering
```

### Adding a New Service

1. Create service directory under `src/python_services/`
2. Implement FastAPI app with required endpoints
3. Add client code in `src/go_core/clients/`
4. Register service URL in `config/dev/config.yaml`
5. Add service to `docker-compose.yml`
6. Update orchestrator to call new service

### Running Tests

```bash
# Python service tests
pytest src/python_services/

# Go orchestrator tests
cd src/go_core
go test ./...
```

### Code Quality

```bash
# Python linting
ruff check src/python_services/
black src/python_services/

# Go formatting
cd src/go_core
go fmt ./...
```

---

## Troubleshooting

### Services won't start

```bash
# Rebuild containers from scratch
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Pipeline can't connect to services

Check that services are listening on expected ports:

```bash
docker compose ps
# Verify all services show "Up" status

# Test connectivity
curl http://localhost:8002/health
```

### Go module errors

```bash
cd src/go_core
go mod tidy
go mod download
```

---

## Project Context

Developed at the **Department of Advanced Computing Sciences, Maastricht University** as part of the Educational Robotics **EDMO** program.

### Team Members

- Noam Favier
- Daan Vankan
- Alexandros Ntoouz/Dawes
- Anne Katarina Zambare
- Paul Elfering
- Vladislav Snytko
- Evi Levels

### Academic Context

This pipeline supports research on collaborative learning analytics in educational robotics contexts. For methodology, evaluation results, and pedagogical implications, see the [technical report](./docs/EDMO_Pipeline_Documentation.pdf).

---

## License

MIT — See [LICENSE](LICENSE)

---

## Status

**Development Phase** — actively integrating visualization + teacher reporting modules.

### Roadmap

- [ ] Complete robot_data service integration
- [ ] Add real-time streaming support
- [ ] Implement teacher dashboard UI
- [ ] Add multi-session comparison tools
- [ ] Deploy containerized production environment

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

---

## Support

For questions or issues:

- Open a GitHub issue
- Contact: [project maintainer email]

---

## Additional Resources

- **Documentation**: [Technical Report (PDF)](./docs/EDMO_Pipeline_Documentation.pdf)
- **Demo**: [Pipeline Walkthrough (MP4)](./docs/EDMO_Demo.mp4)
- **Research Context**: Educational Robotics & Collaborative Learning Analytics
- **Institution**: Maastricht University, Department of Advanced Computing Sciences
