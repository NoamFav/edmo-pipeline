"""Non-verbal features processor - can be used independently or via FastAPI."""
from typing import Dict, List, Any
import numpy as np

from src.python_services.nonverb_features.models import (
    DiarizationResponse
)

from src.python_services.nonverb_features.utils import (
    basic_metrics as compute_basic_metrics,
    extract_f0_curves,
    group_by_speakers,
    calculate_f0_statistics,
    calculate_spectrogram_features,
    calculate_tempo_features as compute_tempo_features
)


class NonVerbalProcessor:
    """Processes audio for non-verbal feature extraction."""
    
    @staticmethod
    def calculate_basic_metrics(
        diarization: Dict[str, Any],
        conv_length: float,
        percentiles: List[int] = [10, 25, 75, 90]
    ) -> Dict[str, Any]:
        """Calculate basic conversation metrics.
        
        Args:
            diarization: Speaker diarization result (dict or DiarizationResponse)
            conv_length: Total length of the conversation segment in seconds
            percentiles: Percentiles to calculate on turn lengths distribution
            
        Returns:
            Dictionary with basic metrics
        """
        # Convert dict to DiarizationResponse if needed
        if isinstance(diarization, dict):
            diarization = DiarizationResponse(**diarization)
        
        result = compute_basic_metrics(diarization, conv_length, percentiles)
        
        # Convert to dict if needed
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        return result
    
    @staticmethod
    def calculate_pitch_features(
        diarization: Dict[str, Any],
        y: np.ndarray,
        sr: int = 16000
    ) -> List[Dict[str, Any]]:
        """Calculate pitch (F0) features.
        
        Args:
            diarization: Speaker diarization result
            y: Audio as a numpy array (floating point time series)
            sr: Sampling rate of the audio
            
        Returns:
            List of dictionaries with F0 statistics per speaker
        """
        speaker_audio_segments = group_by_speakers(y, sr, diarization)
        speaker_f0_curves = extract_f0_curves(speaker_audio_segments, sr)
        f0_stats = calculate_f0_statistics(speaker_f0_curves)
        
        # Convert to dicts if needed
        return [
            stat.model_dump() if hasattr(stat, 'model_dump') else stat
            for stat in f0_stats
        ]
    
    @staticmethod
    def calculate_loudness_features(
        diarization: Dict[str, Any],
        y: np.ndarray,
        sr: int = 16000
    ) -> List[Dict[str, Any]]:
        """Calculate loudness/spectral features.
        
        Args:
            diarization: Speaker diarization result
            y: Audio as a numpy array (floating point time series)
            sr: Sampling rate of the audio
            
        Returns:
            List of dictionaries with spectral statistics per speaker
        """
        
        speaker_audio_segments = group_by_speakers(y, sr, diarization)
        loudness_features = calculate_spectrogram_features(speaker_audio_segments)
        
        # Convert to dicts if needed
        return [
            feat.model_dump() if hasattr(feat, 'model_dump') else feat
            for feat in loudness_features
        ]
    
    @staticmethod
    def calculate_tempo_features(
        diarization: Dict[str, Any],
        y: np.ndarray,
        sr: int = 16000
    ) -> List[Dict[str, Any]]:
        """Calculate tempo/rhythm features.
        
        Args:
            diarization: Speaker diarization result
            y: Audio as a numpy array (floating point time series)
            sr: Sampling rate of the audio
            
        Returns:
            List of dictionaries with tempo statistics per speaker
        """
        speaker_audio_segments = group_by_speakers(y, sr, diarization)
        tempo_features = compute_tempo_features(speaker_audio_segments, sr)
        
        # Convert to dicts if needed
        return [
            feat.model_dump() if hasattr(feat, 'model_dump') else feat
            for feat in tempo_features
        ]


# Convenience functions for direct use
def calculate_basic_metrics(
    diarization: Dict[str, Any],
    conv_length: float,
    percentiles: List[int] = [10, 25, 75, 90]
) -> Dict[str, Any]:
    """Calculate basic conversation metrics."""
    return NonVerbalProcessor.calculate_basic_metrics(diarization, conv_length, percentiles)


def calculate_pitch_features(
    diarization: Dict[str, Any],
    y: np.ndarray,
    sr: int = 16000
) -> List[Dict[str, Any]]:
    """Calculate pitch (F0) features."""
    return NonVerbalProcessor.calculate_pitch_features(diarization, y, sr)


def calculate_loudness_features(
    diarization: Dict[str, Any],
    y: np.ndarray,
    sr: int = 16000
) -> List[Dict[str, Any]]:
    """Calculate loudness/spectral features."""
    return NonVerbalProcessor.calculate_loudness_features(diarization, y, sr)


def calculate_tempo_features(
    diarization: Dict[str, Any],
    y: np.ndarray,
    sr: int = 16000
) -> List[Dict[str, Any]]:
    """Calculate tempo/rhythm features."""
    return NonVerbalProcessor.calculate_tempo_features(diarization, y, sr)
