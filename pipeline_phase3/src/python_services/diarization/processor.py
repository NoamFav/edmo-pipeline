"""Diarization processor - can be used independently or via FastAPI."""
from typing import Dict, List, Optional
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
from dotenv import load_dotenv

from typing import Union


load_dotenv()


class DiarizationProcessor:
    """Processes audio for speaker diarization."""
    
    def __init__(self, hf_token: Optional[str] = None):
        """Initialize the diarization pipeline.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote models
                     If None, will try to read from HF_TOKEN environment variable
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be provided or set in environment")
        
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self.hf_token
        )
    
    def diarize(self, audio: Union[str, Dict]) -> Dict:
        """
        Identify speakers in audio.

        Args:
            audio: Either
                - path to audio file (str), or
                - dict with {'waveform': Tensor, 'sample_rate': int}

        Returns:
            Dictionary with speaker segments
        """

        # Case 1: path-based input
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"Audio file not found: {audio}")
            pipeline_input = audio

        # Case 2: in-memory audio
        elif isinstance(audio, dict):
            if "waveform" not in audio or "sample_rate" not in audio:
                raise ValueError("Audio dict must contain 'waveform' and 'sample_rate'")
            pipeline_input = audio

        else:
            raise TypeError("audio must be a file path or a dict with waveform + sample_rate")

        with ProgressHook() as hook:
            output = self.pipeline(pipeline_input, hook=hook)
    
   # def diarize(self, audio_path: str) -> Dict:
        """Identify speakers in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with:
                - segments: List of dicts with 'start', 'end', 'speaker'
                - num_speakers: Number of unique speakers detected
        """
        # if not os.path.exists(audio_path):
            # raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # with ProgressHook() as hook:
            # output = self.pipeline(audio_path, hook=hook)
        
        segments = []
        speakers = set()
        
        # Extract speaker segments from the diarization output
        for turn, speaker in output.speaker_diarization:
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": f"speaker_{speaker}"
            }
            segments.append(segment)
            speakers.add(speaker)
        
        return {
            "segments": segments,
            "num_speakers": len(speakers)
        }


# Global instance for use without FastAPI
_processor: Optional[DiarizationProcessor] = None


def get_processor() -> DiarizationProcessor:
    """Get or create the global diarization processor instance."""
    global _processor
    if _processor is None:
        _processor = DiarizationProcessor()
    return _processor


def diarize(audio_path: str) -> Dict:
    """Convenience function to diarize audio without creating a processor.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with segments and num_speakers
    """
    processor = get_processor()
    return processor.diarize(audio_path)
