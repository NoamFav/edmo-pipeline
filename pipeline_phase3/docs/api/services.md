# Microservices API Documentation

## NLP Service (Port 8001)

### POST /analyze
Analyze text for communication strategies

**Request:**
```json
{
  "text": "string",
  "language": "nl"
}
```

**Response:**
```json
{
  "embedding": [float],
  "strategies": ["question", "explanation"],
  "metadata": {}
}
```

## ASR Service (Port 8002)

### POST /transcribe
Transcribe audio with timestamps

**Request:**
- File upload (WAV, 16kHz, mono)

**Response:**
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "string"
    }
  ]
}
```

## Diarization Service (Port 8003)

### POST /diarize
Identify speakers in audio

## Emotion Service (Port 8004)

### POST /detect
Detect emotions from text/audio

## Clustering Service (Port 8005)

### POST /cluster
Cluster features using fuzzy c-means

## Visualization Service (Port 8006)

### POST /generate-report
Generate feedback reports
