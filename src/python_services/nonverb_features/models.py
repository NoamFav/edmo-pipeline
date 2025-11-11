from pydantic import BaseModel
from typing import Dict, List, Tuple

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
    mean_f0: float
    std_f0: float
    cv_f0: float
    skewness_f0: float
    kurtosis_f0: float
    min_f0: float
    max_f0: float
    range_f0: float
    normalized_range: float
    voiced_ratio: float
    total_frames: int
    voiced_frames: int


class SpeakerSpectralStats(BaseModel):
    speaker: str
    mean_rms: float
    std_rms: float
    num_segments: int


class SpeakerTempoStats(BaseModel):
    speaker: str
    mean_tempo: float
    std_tempo: float
    min_tempo: float
    max_tempo: float
    num_segments_analyzed: int