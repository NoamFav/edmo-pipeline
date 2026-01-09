"""NLP processor - can be used independently or via FastAPI."""
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from keybert import KeyBERT
import re


class NLPProcessor:
    """Processes text for various NLP tasks."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sentiment_model_name: str = "tabularisai/multilingual-sentiment-analysis",
    ):
        """Initialize NLP models.
        
        Args:
            embedding_model_name: HuggingFace model for embeddings and keywords
            sentiment_model_name: HuggingFace model for sentiment analysis
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.keyword_model = KeyBERT(embedding_model_name)
        self.sentiment_model = hf_pipeline(
            "text-classification",
            model=sentiment_model_name,
        )
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Convert text to lowercase and remove extra spaces or punctuation.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def extract_keywords(self, text: str, top_n: int = 5) -> Dict:
        """Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            Dictionary with 'keywords' and 'scores' lists
        """
        cleaned = self.preprocess_text(text)
        keywords_with_scores = self.keyword_model.extract_keywords(
            cleaned,
            top_n=top_n,
        )

        if keywords_with_scores:
            keywords, scores = map(list, zip(*keywords_with_scores))
        else:
            keywords, scores = [], []

        return {
            "keywords": list(keywords),
            "scores": list(scores),
        }
    
    def analyze_sentiment(self, text: str, max_length: int = 400) -> Dict:
        """Analyze sentiment of text.
        
        Args:
            text: Input text
            max_length: Maximum chunk length to avoid token limit
            
        Returns:
            Dictionary with 'label' and 'score'
        """
        cleaned = self.preprocess_text(text)
        
        # Split text into chunks
        words = cleaned.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            # Rough estimate: 1 token ≈ 0.75 words
            if len(' '.join(current_chunk)) > max_length * 0.75:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Handle empty text
        if not chunks:
            return {"label": "neutral", "score": 0.0}
        
        # Process each chunk
        all_sentiments = {}
        for chunk in chunks:
            result = self.sentiment_model(chunk, truncation=True, max_length=512)[0]
            label = result["label"]
            score = result["score"]
            
            if label not in all_sentiments:
                all_sentiments[label] = []
            all_sentiments[label].append(score)
        
        # Average scores across chunks and find dominant sentiment
        avg_sentiments = {
            label: sum(scores) / len(scores)
            for label, scores in all_sentiments.items()
        }
        
        dominant_label = max(avg_sentiments, key=avg_sentiments.get)
        
        return {
            "label": dominant_label,
            "score": avg_sentiments[dominant_label],
        }
    
    def embed_text(self, text: str) -> Dict:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'embedding' and 'embedding_dim'
        """
        cleaned = self.preprocess_text(text)
        embedding = self.embedding_model.encode(cleaned).tolist()
        return {
            "embedding": embedding,
            "embedding_dim": len(embedding),
        }


# Global instance for use without FastAPI
_processor: Optional[NLPProcessor] = None


def get_processor() -> NLPProcessor:
    """Get or create the global NLP processor instance."""
    global _processor
    if _processor is None:
        _processor = NLPProcessor()
    return _processor


def preprocess_text(text: str) -> str:
    """Convenience function to preprocess text."""
    return NLPProcessor.preprocess_text(text)


def extract_keywords(text: str, processor: NLPProcessor, top_n: int = 5) -> Dict:
    """Convenience function to extract keywords."""
    return processor.extract_keywords(text, top_n)


def analyze_sentiment(text: str, processor: NLPProcessor, max_length: int = 400) -> Dict:
    """Convenience function to analyze sentiment."""
    return processor.analyze_sentiment(text, max_length)


def embed_text(text: str, processor: NLPProcessor) -> Dict:
    """Convenience function to generate embeddings."""
    return processor.embed_text(text)
