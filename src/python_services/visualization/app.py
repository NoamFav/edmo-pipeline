from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Visualization Service", version="0.1.0")


class TimelineRequest(BaseModel):
    timestamps: list[float]
    clusters: list[int]
    robot_progress: list[float]


class RadarChartRequest(BaseModel):
    categories: list[str]
    values: list[float]
    student_name: str


@app.post("/generate-timeline")
async def generate_timeline(request: TimelineRequest):
    """Generate timeline visualization."""
    # TODO: Implement timeline generation
    return {"status": "generated", "path": "outputs/timeline.png"}


@app.post("/generate-radar")
async def generate_radar(request: RadarChartRequest):
    """Generate radar chart for student profile."""
    # TODO: Implement radar chart generation
    return {"status": "generated", "path": "outputs/radar.png"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
