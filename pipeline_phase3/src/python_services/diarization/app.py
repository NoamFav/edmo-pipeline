from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from src.python_services.diarization.processor import DiarizationProcessor
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Diarization Service", version="0.1.0")

processor = None


@app.on_event("startup")
async def load_model():
    global processor
    processor = DiarizationProcessor()


class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    segments: list[SpeakerSegment]
    num_speakers: int


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize_audio(audio_path: str = Body(..., description='Path to audio')):
    """Identify speakers in audio file."""
    try:
        result = processor.diarize(audio_path)
        segments = [SpeakerSegment(**seg) for seg in result["segments"]]
        return DiarizationResponse(
            segments=segments,
            num_speakers=result["num_speakers"]
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
