from fastapi import FastAPI, Body
import numpy as np
from typing import Dict, List, Tuple

from src.python_services.nonverb_features.models import (
    SpeakerFeatures, ConversationMetrics,
    BasicMetricsResponse, SpeakerSegment,
    DiarizationResponse, SpeakerF0Stats,
    SpeakerSpectralStats, SpeakerTempoStats
)

from src.python_services.nonverb_features.processor import NonVerbalProcessor

app = FastAPI(title="Non-verbal Features Extraction Service", version="0.1.0")


@app.post("/basic_metrics", response_model=BasicMetricsResponse)
async def calculate_basic_metrics(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    conv_length: float = Body(
        ..., description="Total length of the conversation segment in seconds"
    ),
    percentiles: List[int] = Body(
        [10, 25, 75, 90],
        description=(
            "Percentiles calculated on the turn lengths distribution,"
            + "just leave the default"
        ),
    ),
):
    result = NonVerbalProcessor.calculate_basic_metrics(
        diarization.model_dump(),
        conv_length,
        percentiles
    )
    return BasicMetricsResponse(**result)


@app.post("/pitch_features", response_model=List[SpeakerF0Stats])
async def calculate_f0_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ..., description="Audio as a floating point time series (as if loaded with librosa.load())"
    ),
    sr: int = Body(
        16000, description="Sampling rate of the audio"
    )
):
    y_np = np.array(y)
    results = NonVerbalProcessor.calculate_pitch_features(
        diarization.model_dump(),
        y_np,
        sr
    )
    return [SpeakerF0Stats(**r) for r in results]


@app.post("/loudness_features", response_model=List[SpeakerSpectralStats])
async def calculate_loudness_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ..., description="Audio as a floating point time series (as if loaded with librosa.load())"
    ),
    sr: int = Body(
        16000, description="Sampling rate of the audio"
    )
):
    y_np = np.array(y)
    results = NonVerbalProcessor.calculate_loudness_features(
        diarization.model_dump(),
        y_np,
        sr
    )
    return [SpeakerSpectralStats(**r) for r in results]


@app.post("/tempo_features", response_model=List[SpeakerTempoStats])
async def calculate_rhythm_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ..., description="Audio as a floating point time series (as if loaded with librosa.load())"
    ),
    sr: int = Body(
        16000, description="Sampling rate of the audio"
    )
):
    y_np = np.array(y)
    results = NonVerbalProcessor.calculate_tempo_features(
        diarization.model_dump(),
        y_np,
        sr
    )
    return [SpeakerTempoStats(**r) for r in results]