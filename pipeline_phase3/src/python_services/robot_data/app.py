from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.python_services.robot_data.processor import RobotDataProcessor, compute_control_intervals

app = FastAPI(title="Session Analysis Service", version="1.0.0")


class ParseRequest(BaseModel):
    log_dir: str
    output_csv: str


class FeatureRequest(BaseModel):
    timeline_csv: str


class FeatureResponse(BaseModel):
    features: dict[str, float]


class ClusterRequest(BaseModel):
    features: list[list[float]]
    n_clusters: int = 3
    n_components: int = 2


class ClusterResponse(BaseModel):
    cluster_labels: list[int]
    pca_coords: list[list[float]]
    explained_variance: float


class PlotRequest(BaseModel):
    timeline_csv: str
    plot_type: str  # "control", "events", "temporal"


def plot_control_gantt(df: pd.DataFrame, png_path: Path):
    intervals = compute_control_intervals(df)
    if not intervals:
        return

    user_rows = {"A": 3, "B": 2, "C": 1, "D": 0}
    per_user = {u: [] for u in user_rows}
    for u, start, dur in intervals:
        if u in per_user and dur > 0:
            per_user[u].append((start, dur))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_title("Control Timeline")
    ax.set_xlabel("Time (s)")
    ax.set_yticks(list(user_rows.values()))
    ax.set_yticklabels(list(user_rows.keys()))
    ax.grid(True, linestyle=":", alpha=0.5)

    for u, row in user_rows.items():
        if per_user[u]:
            ax.broken_barh(per_user[u], (row - 0.35, 0.7))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_event_counts(df: pd.DataFrame, png_path: Path):
    users = ["A", "B", "C", "D"]
    counts = {u: int((df["target"] == u).sum()) for u in users}

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values())
    ax.set_title("Event Counts per User")
    ax.set_xlabel("User")
    ax.set_ylabel("Events")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_temporal_evolution(df: pd.DataFrame, png_path: Path):
    window_s = 10.0
    t_end = df["t_rel_s"].max()
    bins = np.arange(0, t_end + window_s, window_s)

    # Compute control fractions per window
    intervals = compute_control_intervals(df)
    control = {u: [] for u in ["A", "B", "C", "D"]}
    for u, start, dur in intervals:
        if u in control:
            control[u].append((start, start + dur))

    rows = []
    for i in range(len(bins) - 1):
        t0, t1 = float(bins[i]), float(bins[i + 1])
        window = df[(df["t_rel_s"] >= t0) & (df["t_rel_s"] < t1)]

        # Entropy
        actions = window["action"].value_counts().values
        if len(actions):
            p = actions / actions.sum()
            entropy = float(-(p * np.log2(p)).sum())
        else:
            entropy = 0.0

        row = {"t_mid": (t0 + t1) / 2, "entropy": entropy}
        for u in ["A", "B", "C", "D"]:
            u_time = sum(
                max(
                    0.0,
                    min(
                        t1,
                        end,
                    )
                    - max(
                        t0,
                        start,
                    ),
                )
                for start, end in control[u]
            )
            row[f"{u}_control_frac"] = u_time / window_s
        rows.append(row)

    dfw = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title("Temporal Evolution")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Control Fraction")

    colors = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
        "D": "#d62728",
    }
    for u in ["A", "B", "C", "D"]:
        ax1.plot(
            dfw["t_mid"],
            dfw[f"{u}_control_frac"],
            label=f"{u}",
            color=colors[u],
        )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        dfw["t_mid"],
        dfw["entropy"],
        color="black",
        lw=2,
        label="Entropy",
    )
    ax2.set_ylabel("Entropy (bits)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


@app.post("/parse_logs")
async def api_parse_logs(request: ParseRequest):
    """Parse log files and build timeline CSV."""
    try:
        return RobotDataProcessor.parse_logs(request.log_dir, request.output_csv)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_features", response_model=FeatureResponse)
async def api_extract_features(request: FeatureRequest):
    """Extract features from timeline CSV."""
    try:
        features = RobotDataProcessor.extract_features(request.timeline_csv)
        return FeatureResponse(features=features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster", response_model=ClusterResponse)
async def api_cluster(request: ClusterRequest):
    """Perform PCA + KMeans clustering."""
    X = np.asarray(request.features, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Invalid feature matrix")

    n_samples, n_features = X.shape
    n_comp = max(1, min(request.n_components, n_samples, n_features))
    k = max(1, min(request.n_clusters, n_samples))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    explained = float(np.sum(pca.explained_variance_ratio_))

    if k == 1:
        labels = [0] * n_samples
    else:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X_reduced).tolist()

    return ClusterResponse(
        cluster_labels=labels,
        pca_coords=X_reduced.tolist(),
        explained_variance=explained,
    )


@app.post("/plot")
async def api_plot(request: PlotRequest):
    """Generate visualization and return PNG file."""
    path = Path(request.timeline_csv)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.timeline_csv}"
        )

    output = path.parent / f"{request.plot_type}_{path.stem}.png"

    try:
        df = pd.read_csv(path).sort_values("t_rel_s").reset_index(drop=True)

        if request.plot_type == "control":
            plot_control_gantt(df, output)
        elif request.plot_type == "events":
            plot_event_counts(df, output)
        elif request.plot_type == "temporal":
            plot_temporal_evolution(df, output)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown plot type: {request.plot_type}",
            )

        return FileResponse(
            output,
            media_type="image/png",
            filename=output.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "session_analysis"}
