from fastapi import FastAPI
from pydantic import BaseModel
from src.python_services.nlp.processor import NLPProcessor

app = FastAPI(title="NLP Service", version="0.1.0")

processor = None


@app.on_event("startup")
async def load_model():
    global processor
    processor = NLPProcessor()


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


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_text(request: TextRequest):
    cleaned = processor.preprocess_text(request.text)
    return PreprocessResponse(cleaned_text=cleaned)


@app.post("/keywords", response_model=KeywordsResponse)
async def extract_keywords(request: TextRequest):
    result = processor.extract_keywords(request.text, request.top_n)
    return KeywordsResponse(**result)


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    result = processor.analyze_sentiment(request.text)
    return SentimentResponse(**result)


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    result = processor.embed_text(request.text)
    return EmbeddingResponse(**result)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
