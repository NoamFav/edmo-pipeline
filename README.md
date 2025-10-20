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
