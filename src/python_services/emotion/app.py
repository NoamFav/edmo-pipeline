from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Emotion Service", version="0.1.0")

# Load emotion classifier on startup
classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    classifier = pipeline(
        "text-classification",
        model="joeddav/distilbert-base-uncased-go-emotions-student",
        top_k=None,
    )


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
    results = classifier(request.text)[0]
    emotions = [
        EmotionScore(
            label=r["label"],
            score=r["score"],
        )
        for r in results
    ]
    dominant = max(emotions, key=lambda x: x.score)

    return EmotionResponse(
        emotions=emotions,
        dominant_emotion=dominant.label,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
