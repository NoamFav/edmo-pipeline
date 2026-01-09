"""Emotion detection processor - can be used independently or via FastAPI."""
from typing import List, Dict, Optional
from transformers import pipeline as hf_pipeline


class EmotionProcessor:
    """Processes text for emotion detection."""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """Initialize the emotion classifier.
        
        Args:
            model_name: HuggingFace model name for emotion classification
        """
        self.classifier = hf_pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
        )
    
    def detect_emotion(self, text: str, max_length: int = 400) -> Dict:
        """Detect emotions in text.
        
        Args:
            text: Input text to analyze
            max_length: Maximum chunk length (in characters) to avoid token limit
            
        Returns:
            Dictionary with:
                - emotions: List of dicts with 'label' and 'score'
                - dominant_emotion: The emotion with highest score
        """
        # Split text into chunks
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            # Rough estimate: 1 token ≈ 0.75 words
            if len(" ".join(current_chunk)) > max_length * 0.75:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Process each chunk
        all_emotions = {}
        for chunk in chunks:
            results = self.classifier(chunk)[0]
            for r in results:
                if r["label"] not in all_emotions:
                    all_emotions[r["label"]] = []
                all_emotions[r["label"]].append(r["score"])

        # Average scores across chunks
        emotions = [
            {"label": label, "score": sum(scores) / len(scores)}
            for label, scores in all_emotions.items()
        ]

        dominant = max(emotions, key=lambda x: x["score"])

        return {
            "emotions": emotions,
            "dominant_emotion": dominant["label"],
        }


# Global instance for use without FastAPI
_processor: Optional[EmotionProcessor] = None


def get_processor() -> EmotionProcessor:
    """Get or create the global emotion processor instance."""
    global _processor
    if _processor is None:
        _processor = EmotionProcessor()
    return _processor


def detect_emotion(text: str, processor: EmotionProcessor, max_length: int = 400) -> Dict:
    """Convenience function to detect emotions without creating a processor.
    
    Args:
        text: Input text to analyze
        max_length: Maximum chunk length
        
    Returns:
        Dictionary with emotions and dominant_emotion
    """
    return processor.detect_emotion(text, max_length)
