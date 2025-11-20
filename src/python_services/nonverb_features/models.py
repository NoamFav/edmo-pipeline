from pydantic import BaseModel, field_validator
from typing import Dict, List, Tuple
import math

class SpeakerFeatures(BaseModel):
    total_speaking_duration: float
    total_turns: int
    speech_ratio: float  # total_speaking_duration / conv_length

    mean_turn_duration: float
    median_turn_duration: float
    std_turn_duration: float
    min_turn_duration: float
    max_turn_duration: float
    percentiles: Dict[str, float]
    # --------------------------

    interruptions_made: int
    interruptions_received: int
    interrupted_by: Dict[str, int]


class ConversationMetrics(BaseModel):
    num_speakers: int
    total_speaking_time: float  # sum of speaking time of each speaker
    overlap_duration: float  # how long overalps lasted overall
    silence_duration: float  # how long silence lasted overall
    overlap_ratio: float  # overlap_duration / audio_length
    silence_ratio: float  # silence_duration / audio_length
    total_interruptions: int
    interruption_rate: float  # interruptions per minute


class BasicMetricsResponse(BaseModel):
    speakers: Dict[str, SpeakerFeatures]
    conversation: ConversationMetrics


class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    segments: list[SpeakerSegment]
    num_speakers: int
    

class SpeakerF0Stats(BaseModel):
    speaker: str
    mean_f0: float | None
    std_f0: float | None
    cv_f0: float | None
    skewness_f0: float | None
    kurtosis_f0: float | None
    min_f0: float | None
    max_f0: float | None
    range_f0: float | None
    normalized_range: float | None
    voiced_ratio: float | None
    total_frames: int
    voiced_frames: int
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_nan_to_none(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v


class SpeakerSpectralStats(BaseModel):
    speaker: str
    mean_rms: float | None
    std_rms: float | None
    num_segments: int
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_nan_to_none(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v


class SpeakerTempoStats(BaseModel):
    speaker: str
    mean_tempo: float | None
    std_tempo: float | None
    min_tempo: float | None
    max_tempo: float | None
    num_segments_analyzed: int
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_nan_to_none(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v