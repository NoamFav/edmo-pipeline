import numpy as np
from typing import Dict, List, Tuple
import librosa
from scipy import stats as scipy_stats

from src.python_services.nonverb_features.models import (
    SpeakerFeatures, ConversationMetrics,
    BasicMetricsResponse, SpeakerSegment,
    DiarizationResponse, SpeakerF0Stats,
    SpeakerSpectralStats, SpeakerTempoStats
)


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


def group_by_speakers(y, sr, diarization_result: dict):
    # Dictionary to store audio segments for each speaker
    speaker_audio_segments = {}

    start = diarization_result["segments"][0]["start"]
    # Iterate through each segment in the diarization result
    for segment in diarization_result["segments"]:
        speaker = segment["speaker"]
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Convert time to sample indices
        start_sample = int((start_time - start) * sr)
        end_sample = int((end_time - start) * sr)
        
        # Extract the audio segment
        audio_segment = y[start_sample:end_sample]
        
        # Initialize speaker list if not exists
        if speaker not in speaker_audio_segments:
            speaker_audio_segments[speaker] = []
        
        # Append segment with metadata
        speaker_audio_segments[speaker].append({
            'audio': audio_segment,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        })
        
    return speaker_audio_segments


def extract_f0_curves(speaker_audio_segments, sr, fmin=70, fmax=500, frame_length=512, hop_length=160):
    """
    Extract F0 (fundamental frequency) curves for each speaker's audio segments.
    """
    speaker_f0_curves = {}
    
    for speaker, segments in speaker_audio_segments.items():
        speaker_f0_curves[speaker] = []
        
        for seg in segments:
            audio_segment = seg['audio']
            
            # Extract F0 using librosa's pyin algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_segment,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            # Create time array for the F0 curve
            times = librosa.frames_to_time(
                np.arange(len(f0)),
                sr=sr,
                hop_length=hop_length
            )
            
            # Store F0 curve with metadata
            speaker_f0_curves[speaker].append({
                'f0': f0,
                'voiced_flag': voiced_flag,
                'voiced_probs': voiced_probs,
                'times': times,
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration']
            })
    
    return speaker_f0_curves


def calculate_f0_statistics(speaker_f0_curves) -> List[SpeakerF0Stats]:
    """
    Calculate comprehensive F0 statistics for each speaker across all segments.

    Returns a list of SpeakerF0Stats instances (one per speaker).
    """
    results: List[SpeakerF0Stats] = []

    for speaker, segments in speaker_f0_curves.items():
        # Collect all F0 values across all segments
        all_f0_values = []
        total_frames = 0

        for seg in segments:
            f0 = seg['f0']
            # extend even if contains NaNs; we'll filter later
            if f0 is None:
                continue
            all_f0_values.extend(f0)
            total_frames += len(f0)

        # Convert to numpy array
        all_f0 = np.array(all_f0_values)

        # Separate voiced (non-NaN) frames
        voiced_f0 = all_f0[~np.isnan(all_f0)]
        voiced_frames = len(voiced_f0)

        # Calculate statistics
        if voiced_frames > 0:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            min_f0 = float(np.min(voiced_f0))
            max_f0 = float(np.max(voiced_f0))
            range_f0 = max_f0 - min_f0

            # Normalized statistics
            cv_f0 = float(std_f0 / mean_f0) if mean_f0 > 0 else 0.0
            normalized_range = float(range_f0 / mean_f0) if mean_f0 > 0 else 0.0

            # Higher-order moments
            skewness_f0 = float(scipy_stats.skew(voiced_f0))
            kurtosis_f0 = float(scipy_stats.kurtosis(voiced_f0))

            # Voiced ratio
            voiced_ratio = float(voiced_frames / total_frames) if total_frames > 0 else 0.0

            stats_model = SpeakerF0Stats(
                speaker=speaker,
                mean_f0=mean_f0,
                std_f0=std_f0,
                cv_f0=cv_f0,
                skewness_f0=skewness_f0,
                kurtosis_f0=kurtosis_f0,
                min_f0=min_f0,
                max_f0=max_f0,
                range_f0=range_f0,
                normalized_range=normalized_range,
                voiced_ratio=voiced_ratio,
                total_frames=int(total_frames),
                voiced_frames=int(voiced_frames)
            )
        else:
            # No voiced frames found — return zeros but keep counts
            stats_model = SpeakerF0Stats(
                speaker=speaker,
                mean_f0=0.0,
                std_f0=0.0,
                cv_f0=0.0,
                skewness_f0=0.0,
                kurtosis_f0=0.0,
                min_f0=0.0,
                max_f0=0.0,
                range_f0=0.0,
                normalized_range=0.0,
                voiced_ratio=0.0,
                total_frames=int(total_frames),
                voiced_frames=0
            )

        results.append(stats_model)

    return results


def calculate_spectrogram_features(speaker_audio_segments, n_fft=1024, hop_length=512) -> List[SpeakerSpectralStats]:
    """
    Calculate spectral features (RMS and std) from spectrograms for each speaker's segments.
    
    Returns a list of SpeakerSpectralStats instances (one per speaker).
    """
    results: List[SpeakerSpectralStats] = []
    
    for speaker, segments in speaker_audio_segments.items():
        rms_values = []
        
        for seg in segments:
            audio_segment = seg['audio']
            
            # Compute STFT (spectrogram)
            spectrogram = librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length)
            
            # Get magnitude spectrogram
            magnitude_spec = np.abs(spectrogram)
            
            # Calculate RMS energy for this segment
            # RMS is computed across frequency bins for each time frame, then averaged
            rms = librosa.feature.rms(S=magnitude_spec, hop_length=hop_length, frame_length=n_fft)[0]
            mean_rms = np.mean(rms)
            rms_values.append(mean_rms)
        
        # Create model instance with aggregated statistics
        spectral_stats = SpeakerSpectralStats(
            speaker=speaker,
            mean_rms=float(np.mean(rms_values)) if rms_values else 0.0,
            std_rms=float(np.std(rms_values)) if rms_values else 0.0,
            num_segments=len(rms_values)
        )
        results.append(spectral_stats)
    
    return results


def calculate_tempo_features(speaker_audio_segments, sr, hop_length=512) -> List[SpeakerTempoStats]:
    """
    Calculate tempo (BPM) features for each speaker's segments.
    
    Returns a list of SpeakerTempoStats instances (one per speaker).
    """
    results: List[SpeakerTempoStats] = []
    
    for speaker, segments in speaker_audio_segments.items():
        tempo_values = []
        
        for seg in segments:
            audio_segment = seg['audio']
            
            if len(audio_segment) < sr * 0.5:
                continue
            
            try:
                # Calculate onset strength envelope
                onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr, hop_length=hop_length)
                
                # Estimate tempo
                tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[0]
                tempo_values.append(float(tempo))
            except Exception as e:
                # Skip segments where tempo estimation fails
                print(f"  Warning: Could not estimate tempo for segment at {seg['start']:.2f}s: {e}")
                continue
        
        # Create model instance with aggregated statistics
        if tempo_values:
            tempo_stats = SpeakerTempoStats(
                speaker=speaker,
                mean_tempo=float(np.mean(tempo_values)),
                std_tempo=float(np.std(tempo_values)),
                min_tempo=float(np.min(tempo_values)),
                max_tempo=float(np.max(tempo_values)),
                num_segments_analyzed=len(tempo_values)
            )
        else:
            tempo_stats = SpeakerTempoStats(
                speaker=speaker,
                mean_tempo=0.0,
                std_tempo=0.0,
                min_tempo=0.0,
                max_tempo=0.0,
                num_segments_analyzed=0
            )
            print(f"  Warning: No valid tempo estimates for {speaker}")
        
        results.append(tempo_stats)
    
    return results