from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from src.python_services.asr.processor import ASRProcessor
import os

app = FastAPI(title="ASR Service", version="0.1.0")

processor = None


@app.on_event("startup")
async def load_model():
    global processor
    model_size = os.getenv("WHISPER_MODEL", "base")
    force_lang = os.getenv("WHISPER_LANG", "")
    processor = ASRProcessor(model_size, force_lang)


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
    try:
        result = processor.transcribe(audio_path)
        segments = [
            TranscriptionSegment(**seg)
            for seg in result["segments"]
        ]
        return TranscriptionResponse(
            segments=segments,
            language=result["language"],
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-base"}
