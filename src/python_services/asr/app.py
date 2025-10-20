from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import whisper
import tempfile
import os

app = FastAPI(title="ASR Service", version="0.1.0")

# Load model on startup
model = None


@app.on_event("startup")
async def load_model():
    global model
    model_size = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_size)


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    segments: list[TranscriptionSegment]
    language: str


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using Whisper."""
    if not file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Unsupported file format")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="nl")
        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
            )
            for seg in result["segments"]
        ]
        return TranscriptionResponse(
            segments=segments,
            language=result["language"],
        )
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}
