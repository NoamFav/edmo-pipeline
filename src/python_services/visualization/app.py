import os
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI(title="Visualization Service", version="0.1.0")


class TimelineRequest(BaseModel):
    timestamps: List[float]
    clusters: List[int]
    robot_progress: Optional[List[float]] = None
    output_dir: Optional[str] = None  # <—


class RadarChartRequest(BaseModel):
    categories: List[str]
    values: List[float]
    student_name: str
    output_dir: Optional[str] = None  # <—


def _ensure_dir(d: Optional[str]) -> str:
    # default to sessionized folder under data/outputs if caller didn't send one
    if d and d.strip():
        os.makedirs(d, exist_ok=True)
        return d
    root = os.path.join("data", "outputs")
    os.makedirs(root, exist_ok=True)
    return root


@app.post("/generate-timeline")
async def generate_timeline(req: TimelineRequest):
    outdir = _ensure_dir(req.output_dir)
    path = os.path.join(outdir, "timeline.png")

    # simple stripe timeline: x=timestamps, y=cluster id
    ts = np.array(req.timestamps, dtype=float)
    cs = np.array(req.clusters, dtype=int)

    plt.figure(figsize=(10, 2.5))
    if ts.size > 0:
        # draw colored segments between timestamps
        for i, t0 in enumerate(ts):
            t1 = ts[i + 1] if i + 1 < ts.size else (t0 + 1.0)
            plt.fill_between([t0, t1], 0, 1, step="pre", alpha=0.7)
            plt.text(
                (t0 + t1) / 2.0,
                0.5,
                str(int(cs[i])),
                ha="center",
                va="center",
                fontsize=9,
            )
        plt.xlim(ts[0], ts[-1] if ts.size > 1 else ts[0] + 1.0)
    plt.yticks([])
    plt.xlabel("time (s)")
    plt.title("Cluster timeline")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return {"status": "generated", "path": path}


@app.post("/generate-radar")
async def generate_radar(req: RadarChartRequest):
    outdir = _ensure_dir(req.output_dir)
    path = os.path.join(outdir, "radar.png")

    cats = req.categories
    vals = np.array(req.values, dtype=float)
    if len(cats) == 0 or vals.size == 0:
        # create an empty placeholder so call doesn't fail
        open(path, "wb").close()
        return {"status": "generated", "path": path}

    # radar chart
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals = np.concatenate([vals, vals[:1]])
    angles = np.concatenate([angles, angles[:1]])

    plt.figure(figsize=(4, 4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, cats)
    ax.set_title(req.student_name)
    ax.set_rlabel_position(0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return {"status": "generated", "path": path}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
