# EDMO Communication Analysis Pipeline

**Data-Driven Insights into Communication and Collaboration: Enhancing Soft Skill Development in Educational Robotics**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://golang.org/)

## Overview

The EDMO Pipeline analyzes student communication and collaboration during educational robotics tasks, providing teachers with data-driven insights into soft skills development.

### Key Features

- 🎤 **Multimodal Analysis**: Audio, video, and robot interaction data
- 🤖 **Advanced NLP**: Speech-to-text, diarization, emotion detection, embeddings
- 📊 **Clustering Analysis**: Discover communication patterns and collaboration strategies
- 📈 **Visualization**: Timeline graphs, radar charts, correlation analysis
- 👨‍🏫 **Teacher Feedback**: Non-evaluative, constructive insights for educators
- 🔒 **Privacy-First**: GDPR-compliant, anonymized data handling

## Project Information

- **Institution**: Department of Advanced Computing Sciences, Maastricht University
- **Coordinators**: Katharina Schüller, Rico Möckel
- **Duration**: September 9, 2025 - January 23, 2026
- **Program**: BSc Data Science and AI

### Team Members

1. Noam Favier (i6349778)
2. Daan Vankan (i6331424)
3. Alexandros Ntoouz/Dawes (i6323296)
4. Anne Katarina Zambare (i6365696)
5. Paul Elfering (i6365814)
6. Vladislav Snytko (i6363101)
7. Evi Levels (i6368803)

## Quick Start

### Prerequisites

- Python >= 3.10
- Go >= 1.21
- FFmpeg
- CMake >= 3.15

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd edmo-pipeline

# Run initialization script
./scripts/setup/init_repo.sh

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
make install

# Build the pipeline
make setup
```

### Running the Pipeline

```bash
# Start all microservices
make run

# Or start services individually
uvicorn src.python_services.nlp.app:app --port 8001

# Process an experiment
./bin/edmo-pipeline process --experiment experiment_001
```

## Architecture

### Pipeline Components

1. **Data Preprocessing**

    - Audio normalization (16kHz mono WAV)
    - Video frame extraction
    - Robot log parsing and synchronization

2. **Feature Extraction**

    - Speech-to-Text (Whisper)
    - Speaker Diarization (Pyannote.audio)
    - Emotion Detection (Fine-tuned Transformers)
    - Prosodic Features (Librosa)
    - Sentence Embeddings (E5/BERT)
    - Robot Interaction Metrics

3. **Analysis**

    - Dimensionality Reduction (PCA/t-SNE)
    - Fuzzy C-Means Clustering
    - Pattern Discovery
    - PISA CPS Framework Validation

4. **Output Generation**
    - Student psychological profiles
    - Timeline visualizations
    - Teacher feedback reports

### Technology Stack

- **Go**: Pipeline orchestration, data synchronization
- **Python**: ML services (FastAPI microservices)
- **C++**: High-performance audio processing (Whisper.cpp)
- **Libraries**: PyTorch, Transformers, scikit-learn, librosa

## Project Structure

```
edmo-pipeline/
├── config/              # Configuration files
├── data/                # Data storage (gitignored)
│   ├── raw/            # Original data
│   ├── processed/      # Preprocessed data
│   ├── features/       # Extracted features
│   ├── models/         # Model weights
│   └── outputs/        # Results and visualizations
├── docs/               # Documentation
├── experiments/        # Research experiments
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
├── src/
│   ├── cpp_native/    # C++ modules
│   ├── go_core/       # Go pipeline orchestrator
│   ├── python_services/ # Microservices
│   │   ├── asr/
│   │   ├── diarization/
│   │   ├── emotion/
│   │   ├── nlp/
│   │   ├── clustering/
│   │   └── visualization/
│   └── shared/        # Shared constants and schemas
└── tests/             # Test suite

```

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Installation Guide](docs/guides/setup/installation.md)
- [Quick Start Guide](docs/guides/usage/quickstart.md)
- [API Documentation](docs/api/services.md)

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
make format
```

### Building Documentation

```bash
make docs
```

## Research Background

This project builds on research in:

- **Multimodal Learning Analytics** (Ma et al., 2022, 2024; Spikol et al., 2017)
- **Communication Analysis** (Tsan et al., 2021; D'Angelo & Rajarathnam, 2024)
- **Educational Robotics** (Malinverni et al., 2021; Möckel et al., 2020)
- **Collaborative Problem Solving** (PISA 2015 CPS Framework)

## Objectives

### O1: Discover Communication Patterns

Identify patterns in student communication features using data-driven methods (clustering) and validate against the PISA CPS framework.

**KPI**: >75% accuracy in detecting collaborative skills

### O2: Analyze Pattern-Success Relationships

Explore how communication patterns relate to task success (robot performance metrics).

**KPI**: Clear visualizations showing dependencies between CPS skills and performance

### O3: Develop Teacher Feedback System

Generate constructive, non-evaluative feedback for teachers on students' soft skills.

**KPI**: >75% alignment with teacher assessments

## Privacy & Ethics

- ✅ GDPR-compliant data handling
- ✅ Anonymized student data
- ✅ No personally identifiable information (PII)
- ✅ Restricted data access
- ✅ Non-evaluative feedback (formative, not summative)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Department of Advanced Computing Sciences, Maastricht University
- EDMO Educational Robotics Project Team
- All participating teachers and students

## Contact

For questions or collaboration inquiries, please contact the project coordinators:

- Katharina Schüller
- Rico Möckel

---

**Status**: Phase 1 Complete (Prototyping) | Phase 2 In Progress (Development)

```mermaid
graph TD
    A[Project Root] --> B[config]
    B --> B1[dev/config.yaml]
    B --> B2[prod/config.yaml]
    B --> B3[test/]

    A --> C[data]
    C --> C1[features → {emotion, robot, speech}]
    C --> C2[models/]
    C --> C3[outputs → {clusters, feedback, visualizations}]
    C --> C4[processed → {audio, diarization, transcripts}]
    C --> C5[raw → {audio, robot_logs, video}]

    A --> D[docs]
    D --> D1[agenda.md]
    D --> D2[api/services.md]
    D --> D3[architecture/overview.md]
    D --> D4[guides → {development, setup/installation.md, usage/quickstart.md}]
    D --> D5[PROJECT_SUMMARY.md]
    D --> D6[research/]

    A --> E[experiments]
    E --> E1[exploratory/]
    E --> E2[prototypes/]
    E --> E3[validation/]

    A --> F[notebooks]
    F --> F1[clustering/]
    F --> F2[eda/]
    F --> F3[feature_analysis/]

    A --> G[scripts]
    G --> G1[analysis/]
    G --> G2[deployment/start_services.sh]
    G --> G3[preprocessing/download_models.sh]
    G --> G4[setup/run_tests.sh]

    A --> H[src]
    H --> H1[cpp_native → {audio, video, whisper, video_tools.cpp, whisper_runner.cpp}]
    H --> H2[data_pipeline → {convert_audio.sh, extract_frames.sh, preprocess.py}]
    H --> H3[go_core → {audio.go, pipeline.go, utils.go, video.go}]
    H --> H4[interfaces → {data_format.md, nlp_api.md}]
    H --> H5[nlp → {app.py, processor.py, strategies.py}]
    H --> H6[python_services]
    H6 --> H6a[asr/app.py]
    H6 --> H6b[clustering/app.py]
    H6 --> H6c[diarization/app.py]
    H6 --> H6d[emotion/app.py]
    H6 --> H6e[nlp/]
    H6 --> H6f[visualization/app.py]
    H --> H7[shared → {config.yaml, constants.go, schema.json}]
    H --> H8[main.py]

    A --> I[tests]
    I --> I1[e2e/]
    I --> I2[integration → {pipeline, services}]
    I --> I3[unit → {cpp, go, python}]
    I --> I4[fixtures → {audio, logs, video}]

    A --> J[Root Files]
    J --> J1[CHANGELOG.md]
    J --> J2[CONTRIBUTING.md]
    J --> J3[CODE_OF_CONDUCT.md]
    J --> J4[LICENSE]
    J --> J5[README.md]
    J --> J6[Dockerfile]
    J --> J7[docker-compose.yml]
    J --> J8[Makefile]
    J --> J9[requirements.txt]
    J --> J10[requirements-dev.txt]
    J --> J11[SECURITY.md]
```
