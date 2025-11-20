from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Emotion Service", version="0.1.0")

classifier = None


@app.on_event("startup")
async def load_model():
    global classifier
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
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
    max_length = 400  # Model's token limit is 514, but it's safer to lower it

    # Split text into chunks
    words = request.text.split()
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
        results = classifier(chunk)[0]
        for r in results:
            if r["label"] not in all_emotions:
                all_emotions[r["label"]] = []
            all_emotions[r["label"]].append(r["score"])

    # Average scores across chunks
    emotions = [
        EmotionScore(
            label=label, score=sum(scores) / len(scores)
        )
        for label, scores in all_emotions.items()
    ]

    dominant = max(emotions, key=lambda x: x.score)

    return EmotionResponse(
        emotions=emotions,
        dominant_emotion=dominant.label,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
