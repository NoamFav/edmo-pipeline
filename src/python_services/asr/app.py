from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import whisper
import tempfile
import os

app = FastAPI(title="ASR Service", version="0.1.0")

model = None
force_lang = os.getenv("WHISPER_LANG", "")


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
async def transcribe_audio(audio_path: str = Body(..., description="Audio path")):
    """Transcribe audio file using Whisper."""
    if not audio_path.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(400, "Unsupported file format")

    kwargs = {"fp16": False}
    if force_lang:
        kwargs["language"] = force_lang  # e.g. "en", "nl"
    result = model.transcribe(audio_path, **kwargs)
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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}
