"""ASR processor - can be used independently or via FastAPI."""
from typing import Dict, List, Optional
import whisper
import os


class ASRProcessor:
    """Processes audio for automatic speech recognition."""
    
    def __init__(self, model_size: str = "base", force_lang: str = ""):
        """Initialize the Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            force_lang: Force a specific language (e.g., "en", "nl"), empty for auto-detect
        """
        self.model = whisper.load_model(model_size)
        self.force_lang = force_lang
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, .m4a)
            
        Returns:
            Dictionary with:
                - segments: List of dicts with 'start', 'end', 'text'
                - language: Detected or forced language code
        """
        if not audio_path.endswith((".wav", ".mp3", ".m4a")):
            raise ValueError("Unsupported file format")
        
        kwargs = {"fp16": False}
        if self.force_lang:
            kwargs["language"] = self.force_lang
        
        result = self.model.transcribe(audio_path, **kwargs)
        
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in result["segments"]
        ]
        
        return {
            "segments": segments,
            "language": result["language"],
        }


# Global instance for use without FastAPI
_processor: Optional[ASRProcessor] = None


def get_processor() -> ASRProcessor:
    """Get or create the global ASR processor instance."""
    global _processor
    if _processor is None:
        model_size = os.getenv("WHISPER_MODEL", "base")
        force_lang = os.getenv("WHISPER_LANG", "")
        _processor = ASRProcessor(model_size, force_lang)
    return _processor


def transcribe(audio_path: str) -> Dict:
    """Convenience function to transcribe audio without creating a processor.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with segments and language
    """
    processor = get_processor()
    return processor.transcribe(audio_path)
