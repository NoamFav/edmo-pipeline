from fastapi import FastAPI, Body
from pydantic import BaseModel
import numpy as np
from typing import Dict, List, Tuple

app = FastAPI(title="Non-verbal Features Extraction Service", version="0.1.0")


class SpeakerFeatures(BaseModel):
    total_speaking_duration: float
    total_turns: int
    speech_ratio: float  # total_speaking_duration / conv_length

    # A bunch of statistics on the turn lengths distribution
    # --------------------------
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


def basic_metrics(
    diarization_result: DiarizationResponse,
    audio_length: float,
    percentiles: List[int] = [10, 25, 75, 90],
) -> BasicMetricsResponse:
    """
    Comprehensive analysis of conversation dynamics from diarization results.

    Returns:
        BasicMetricsResponse (pydantic) with
        'speakers' and 'conversation' populated.
    """
    # Group segments by speaker and sort all segments chronologically
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}
    all_segments: List[Tuple[float, float, str]] = []

    for segment in diarization_result.segments:
        if segment.speaker not in speaker_segments:
            speaker_segments[segment.speaker] = []
        speaker_segments[segment.speaker].append((segment.start, segment.end))
        all_segments.append((segment.start, segment.end, segment.speaker))

    # Sort segments chronologically by start time
    all_segments.sort(key=lambda x: x[0])

    # Initialize interruption tracking (no comprehensions)
    interruptions: Dict[str, int] = {}
    for sp in speaker_segments.keys():
        interruptions[sp] = 0

    interrupted_by: Dict[str, Dict[str, int]] = {}
    for sp in speaker_segments.keys():
        inner: Dict[str, int] = {}
        for other in speaker_segments.keys():
            if other != sp:
                inner[other] = 0
        interrupted_by[sp] = inner

    # Detect interruptions
    active_speakers: Dict[str, float] = {}  # speaker -> end_time

    for start, end, speaker in all_segments:
        # Check if this speaker is interrupting anyone
        for active_speaker, active_end in list(active_speakers.items()):
            if active_speaker != speaker and start < active_end:
                interruptions[speaker] += 1
                # active_speaker is being interrupted by "speaker"
                interrupted_by[active_speaker][speaker] += 1

        # Update active speakers
        active_speakers[speaker] = end

        # Remove speakers who have finished before current start
        new_active: Dict[str, float] = {}
        for s, e in active_speakers.items():
            if e > start:
                new_active[s] = e
        active_speakers = new_active

    # Calculate features for each speaker
    speaker_features: Dict[str, SpeakerFeatures] = {}
    total_speaking_time = 0.0

    for speaker, segments in speaker_segments.items():
        # turn_durations = [end - start for start, end in segments]
        turn_durations: List[float] = []
        for start, end in segments:
            turn_durations.append(end - start)

        total_speaking_duration = sum(turn_durations)
        total_speaking_time += total_speaking_duration
        total_turns = len(turn_durations)
        if audio_length > 0:
            speech_ratio = total_speaking_duration / audio_length
        else:
            speech_ratio = 0.0

        arr = np.array(turn_durations, dtype=float)
        if arr.size:
            mean_turn_duration = float(np.mean(arr))
            median_turn_duration = float(np.median(arr))
            std_turn_duration = float(np.std(arr))
            min_turn_duration = float(np.min(arr))
            max_turn_duration = float(np.max(arr))
        else:
            mean_turn_duration = 0.0
            median_turn_duration = 0.0
            std_turn_duration = 0.0
            min_turn_duration = 0.0
            max_turn_duration = 0.0

        if arr.size:
            percentile_values_list = np.percentile(arr, percentiles).tolist()
        else:
            percentile_values_list = [0.0] * len(percentiles)

        percentile_dict: Dict[str, float] = {}
        for idx in range(len(percentiles)):
            p = percentiles[idx]
            v = percentile_values_list[idx]
            percentile_dict[f"percentile_{p}"] = float(v)

        # Build a plain dict for interrupted_by for this speaker
        ib_copy: Dict[str, int] = {}
        for k, v in interrupted_by[speaker].items():
            ib_copy[k] = int(v)

        speaker_features[speaker] = SpeakerFeatures(
            total_speaking_duration=float(total_speaking_duration),
            total_turns=int(total_turns),
            speech_ratio=float(speech_ratio),
            mean_turn_duration=mean_turn_duration,
            median_turn_duration=median_turn_duration,
            std_turn_duration=std_turn_duration,
            min_turn_duration=min_turn_duration,
            max_turn_duration=max_turn_duration,
            interruptions_made=int(interruptions[speaker]),
            interruptions_received=int(sum(interrupted_by[speaker].values())),
            interrupted_by=ib_copy,
            percentiles=percentile_dict,
        )

    # Conversation-level metrics
    # Build timeline and compute coverage
    timeline: List[Tuple[float, float, str]] = []
    for seg in diarization_result.segments:
        timeline.append((seg.start, seg.end, seg.speaker))

    timeline.sort(key=lambda x: x[0])

    total_coverage = 0.0
    last_end = 0.0
    for start, end, _ in timeline:
        if start > last_end:
            # Disjoint segment
            total_coverage += end - start
            last_end = end
        elif end > last_end:
            # Partial overlap
            total_coverage += end - last_end
            last_end = end

    overlap_duration = total_speaking_time - total_coverage
    silence_duration = max(0.0, audio_length - total_coverage)
    total_interruptions = 0
    for v in interruptions.values():
        total_interruptions += v

    if audio_length > 0:
        interruption_rate = total_interruptions / (audio_length / 60.0)
        overlap_ratio = float(overlap_duration / audio_length)
        silence_ratio = float(silence_duration / audio_length)
    else:
        interruption_rate = 0.0
        overlap_ratio = 0.0
        silence_ratio = 0.0

    conversation_metrics = ConversationMetrics(
        num_speakers=int(diarization_result.num_speakers),
        total_speaking_time=float(total_speaking_time),
        overlap_duration=float(overlap_duration),
        silence_duration=float(silence_duration),
        overlap_ratio=overlap_ratio,
        silence_ratio=silence_ratio,
        total_interruptions=int(total_interruptions),
        interruption_rate=float(interruption_rate),
    )

    return BasicMetricsResponse(
        speakers=speaker_features, conversation=conversation_metrics
    )
