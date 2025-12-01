from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Diarization Service", version="0.1.0")


class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    segments: list[SpeakerSegment]
    num_speakers: int


@app.post("/diarize", response_model=DiarizationResponse)
async def diarize_audio(audio_path: str = Body(..., description="Path to audio")):
    """Identify speakers in audio file."""
    hf_token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    )

    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404, detail=f"Audio file not found: {audio_path}"
        )

    with ProgressHook() as hook:
        output = pipeline(audio_path, hook=hook)

    segments = []
    speakers = set()

    # Extract speaker segments from the diarization output
    for turn, speaker in output.speaker_diarization:
        segment = SpeakerSegment(
            start=turn.start, end=turn.end, speaker=f"speaker_{speaker}"
        )
        segments.append(segment)
        speakers.add(speaker)

    # Create the response object
    response = DiarizationResponse(segments=segments, num_speakers=len(speakers))

    return response


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
