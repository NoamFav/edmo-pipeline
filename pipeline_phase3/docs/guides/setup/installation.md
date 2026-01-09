# Installation Guide

## Prerequisites
- Python >= 3.10
- Go >= 1.21
- FFmpeg
- CMake >= 3.15
- Git

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd edmo-pipeline
```

2. Run the initialization script:
```bash
./scripts/setup/init_repo.sh
```

3. Install Python dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Download spaCy Dutch model:
```bash
python -m spacy download nl_core_news_sm
```

5. Build Go core:
```bash
cd src/go_core
go build -o ../../bin/edmo-pipeline
```

6. (Optional) Build C++ modules:
```bash
cd src/cpp_native
mkdir build && cd build
cmake ..
make
```

## Running Services

Start all microservices:
```bash
./scripts/deployment/start_services.sh
```

Or start individually:
```bash
# NLP service
uvicorn src.python_services.nlp.app:app --port 8001

# ASR service
uvicorn src.python_services.asr.app:app --port 8002
```
