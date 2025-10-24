from fastapi import FastAPI
from pydantic import BaseModel
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

matplotlib.use("Agg")

app = FastAPI(title="Visualization Service", version="0.1.0")


class TimelineRequest(BaseModel):
    timestamps: list[float]
    clusters: list[int]
    robot_progress: Optional[list[float]] = None
    output_dir: str = "outputs"


class RadarChartRequest(BaseModel):
    categories: list[str]
    values: list[float]
    student_name: str
    output_dir: str = "outputs"


def _ensure_dir(path: str):
    """Create folder if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)


@app.post("/generate-timeline")
async def generate_timeline(request: TimelineRequest):
    """Generate timeline visualization."""
    _ensure_dir(request.output_dir)
    path = os.path.join(request.output_dir, "timeline.png")

    _, ax = plt.subplots(figsize=(6, 1.5))
    ax.scatter(
        request.timestamps,
        request.clusters,
        c=request.clusters,
        cmap="tab10",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Session timeline")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return {"status": "generated", "path": path}


@app.post("/generate-radar")
async def generate_radar(request: RadarChartRequest):
    """Generate radar chart for student profile."""
    _ensure_dir(request.output_dir)
    path = os.path.join(request.output_dir, "radar.png")

    N = len(request.categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals = np.asarray(request.values, dtype=float)
    vals = np.concatenate([vals, vals[:1]])
    angles_closed = np.concatenate([angles, angles[:1]])

    _ = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_closed, vals)
    ax.fill(angles_closed, vals, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(request.categories)
    ax.set_title(request.student_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return {"status": "generated", "path": path}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
