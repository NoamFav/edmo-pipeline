from fastapi import FastAPI
from pydantic import BaseModel
from src.python_services.emotion.processor import EmotionProcessor

app = FastAPI(title="Emotion Service", version="0.1.0")

processor = None


@app.on_event("startup")
async def load_model():
    global processor
    processor = EmotionProcessor()


class EmotionRequest(BaseModel):
    text: str


class EmotionScore(BaseModel):
    label: str
    score: float


class EmotionResponse(BaseModel):
    emotions: list[EmotionScore]
    dominant_emotion: str


@app.post("/detect", response_model=EmotionResponse)
async def detect_emotion(request: EmotionRequest):
    """Detect emotions in text."""
    result = processor.detect_emotion(request.text)
    
    # Convert dict to response model
    emotions = [EmotionScore(**e) for e in result["emotions"]]
    
    return EmotionResponse(
        emotions=emotions,
        dominant_emotion=result["dominant_emotion"],
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
