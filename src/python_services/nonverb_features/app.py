from fastapi import FastAPI, Body
import numpy as np
from typing import Dict, List, Tuple

from src.python_services.nonverb_features.models import (
    SpeakerFeatures,
    ConversationMetrics,
    BasicMetricsResponse,
    SpeakerSegment,
    DiarizationResponse,
    SpeakerF0Stats,
    SpeakerSpectralStats,
    SpeakerTempoStats,
)

from src.python_services.nonverb_features.utils import (
    basic_metrics,
    extract_f0_curves,
    group_by_speakers,
    calculate_f0_statistics,
    calculate_spectrogram_features,
    calculate_tempo_features,
)

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
    return basic_metrics(diarization, conv_length, percentiles)


@app.post("/pitch_features", response_model=List[SpeakerF0Stats])
async def calculate_f0_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ...,
        description="Audio as a floating point time series (as if loaded with librosa.load())",
    ),
    sr: int = Body(16000, description="Sampling rate of the audio"),
):
    y_np = np.array(y)
    speaker_audio_segments = group_by_speakers(y_np, sr, diarization)
    speaker_f0_curves = extract_f0_curves(speaker_audio_segments, sr)
    f0_stats = calculate_f0_statistics(speaker_f0_curves)
    return f0_stats


@app.post("/loudness_features", response_model=List[SpeakerSpectralStats])
async def calculate_loudness_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ...,
        description="Audio as a floating point time series (as if loaded with librosa.load())",
    ),
    sr: int = Body(16000, description="Sampling rate of the audio"),
):
    y_np = np.array(y)
    speaker_audio_segments = group_by_speakers(y_np, sr, diarization)
    loudness_features = calculate_spectrogram_features(speaker_audio_segments)
    return loudness_features


@app.post("/tempo_features", response_model=List[SpeakerTempoStats])
async def calculate_rhythm_features(
    diarization: DiarizationResponse = Body(
        ..., description="Speaker diarization result"
    ),
    y: List[float] = Body(
        ...,
        description="Audio as a floating point time series (as if loaded with librosa.load())",
    ),
    sr: int = Body(16000, description="Sampling rate of the audio"),
):
    y_np = np.array(y)
    speaker_audio_segments = group_by_speakers(y_np, sr, diarization)
    tempo_features = calculate_tempo_features(speaker_audio_segments, sr)
    return tempo_features
