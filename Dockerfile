FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -m spacy download nl_core_news_sm

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8001 8002 8003 8004 8005 8006 8007

CMD ["bash"]
