from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
import re

app = FastAPI(title="NLP Service", version="0.1.0")

# Load embedding model on startup
embedding_model = None
keyword_model = None
sentiment_model = None


@app.on_event("startup")
async def load_model():
    global embedding_model, keyword_model, sentiment_model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    keyword_model = KeyBERT(
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    sentiment_model = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
    )


class TextRequest(BaseModel):
    text: str
    top_n: int = 5  # number of keywords to return


class PreprocessResponse(BaseModel):
    cleaned_text: str


class KeywordsResponse(BaseModel):
    keywords: list[str]
    scores: list[float]


class SentimentResponse(BaseModel):
    label: str
    score: float


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int


def simple_text_preprocessing(text: str) -> str:
    """Convert text to lowercase and remove extra spaces or punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_text(request: TextRequest):
    cleaned = simple_text_preprocessing(request.text)
    return PreprocessResponse(cleaned_text=cleaned)


@app.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: TextRequest):
    cleaned = simple_text_preprocessing(request.text)
    keywords_with_scores = keyword_model.extract_keywords(
        cleaned,
        top_n=request.top_n,
    )

    if keywords_with_scores:
        keywords, scores = map(list, zip(*keywords_with_scores))
    else:
        keywords, scores = [], []

    return KeywordsResponse(
        keywords=list(keywords),
        scores=list(scores),
    )


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    cleaned = simple_text_preprocessing(request.text)
    result = sentiment_model(cleaned)[0]
    return SentimentResponse(
        label=result["label"],
        score=result["score"],
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    cleaned = simple_text_preprocessing(request.text)
    embedding = embedding_model.encode(cleaned).tolist()
    return EmbeddingResponse(
        embedding=embedding,
        embedding_dim=len(embedding),
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
