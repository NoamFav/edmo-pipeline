#!/usr/bin/env python
"""
Analysis Service — FastAPI endpoints for session analysis
Provides feature extraction, visualization, and clustering capabilities
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = FastAPI(title="Session Analysis Service", version="1.0.0")

# ───────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ───────────────────────────────────────────────────────────────────────────


class FeatureRequest(BaseModel):
    timeline_csv: str  # path to timeline.csv
    window_s: float = 10.0


class FeatureResponse(BaseModel):
    features: dict[str, float]
    participant: Optional[str] = None
    session: Optional[str] = None


class ClusterRequest(BaseModel):
    features: list[list[float]]
    n_clusters: int = 3
    n_components: int = 2


class ClusterResponse(BaseModel):
    cluster_labels: list[int]
    pca_coords: list[list[float]]
    explained_variance: float


class CorrelationRequest(BaseModel):
    features_csv: str  # path to features_all.csv
    feature_names: list[str]


class CorrelationResponse(BaseModel):
    correlation_matrix: list[list[float]]
    feature_names: list[str]


# ───────────────────────────────────────────────────────────────────────────
# Feature Extraction (simplified from robotdata1.py)
# ───────────────────────────────────────────────────────────────────────────


def extract_features(csv_path: Path, window_s: float = 10.0) -> dict:
    """Extract session-level features from timeline CSV."""
    df = pd.read_csv(csv_path).sort_values("t_rel_s").reset_index(drop=True)

    t0, tN = float(df["t_rel_s"].min()), float(df["t_rel_s"].max())
    total_time = max(0.0, tN - t0)

    feats = {
        "total_time_s": total_time,
        "num_events": int(len(df)),
    }

    # Inter-event timing
    times = df["t_rel_s"].astype(float).tolist()
    gaps = [b - a for a, b in zip(times, times[1:])]
    feats["inter_event_mean_s"] = float(np.mean(gaps)) if gaps else 0.0
    feats["inter_event_std_s"] = float(np.std(gaps)) if gaps else 0.0

    # Control intervals
    users = ["A", "B", "C", "D"]
    intervals = compute_control_intervals(df)
    control_time = {u: 0.0 for u in users}
    for u, start, dur in intervals:
        if u in control_time:
            control_time[u] += float(dur)

    for u in users:
        feats[f"{u}_control_frac"] = (
            (control_time[u] / total_time) if total_time > 0 else 0.0
        )

    # Action entropy
    action_counts = df["action"].value_counts().values
    if len(action_counts) > 0:
        p = action_counts / action_counts.sum()
        feats["action_entropy_bits"] = float(-(p * np.log2(p)).sum())
    else:
        feats["action_entropy_bits"] = 0.0

    # Control balance (std of control times)
    vals = np.array([control_time[u] for u in users])
    feats["control_balance_index"] = float(np.std(vals))

    return feats


def compute_control_intervals(
    df: pd.DataFrame,
) -> list[tuple[str, float, float]]:
    """Compute control intervals from control_start/control_end events."""
    intervals = []
    current_user, start_time = None, None
    d = (
        df[
            df["action"].isin(
                [
                    "control_start",
                    "control_end",
                ]
            )
        ].sort_values("t_rel_s"),
    )

    for _, row in d.iterrows():
        t, action, user = float(row["t_rel_s"]), row["action"], row["target"]

        if action == "control_start":
            if current_user and start_time and t > start_time:
                intervals.append((current_user, start_time, t - start_time))
            current_user, start_time = user, t
        elif action == "control_end" and current_user == user:
            if start_time and t > start_time:
                intervals.append((current_user, start_time, t - start_time))
            current_user, start_time = None, None

    if current_user and start_time:
        t_end = float(df["t_rel_s"].max())
        if t_end > start_time:
            intervals.append((current_user, start_time, t_end - start_time))

    return intervals


# ───────────────────────────────────────────────────────────────────────────
# Endpoints
# ───────────────────────────────────────────────────────────────────────────


@app.post("/extract_features", response_model=FeatureResponse)
async def api_extract_features(request: FeatureRequest):
    """Extract features from a timeline CSV."""
    path = Path(request.timeline_csv)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.timeline_csv}"
        )

    try:
        features = extract_features(path, window_s=request.window_s)
        participant = path.parent.parent.name if len(path.parts) >= 2 else None
        session = path.parent.name if len(path.parts) >= 1 else None

        return FeatureResponse(
            features=features, participant=participant, session=session
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cluster", response_model=ClusterResponse)
async def api_cluster(request: ClusterRequest):
    """Perform PCA + KMeans clustering on feature vectors."""
    X = np.asarray(request.features, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        raise HTTPException(status_code=400, detail="Invalid feature matrix")

    n_samples, n_features = X.shape
    n_comp = max(1, min(request.n_components, n_samples, n_features))
    k = max(1, min(request.n_clusters, n_samples))

    # PCA
    if n_comp == 1 or n_samples == 1:
        X_reduced = X[:, :1] if X.shape[1] > 0 else X
        explained = 1.0
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_comp, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        explained = float(np.sum(pca.explained_variance_ratio_))

    # KMeans
    if n_samples == 1 or k == 1:
        labels = [0] * n_samples
    else:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X_reduced).tolist()

    return ClusterResponse(
        cluster_labels=labels,
        pca_coords=X_reduced.tolist(),
        explained_variance=explained,
    )


@app.post("/correlation", response_model=CorrelationResponse)
async def api_correlation(request: CorrelationRequest):
    """Compute correlation matrix for selected features."""
    path = Path(request.features_csv)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.features_csv}"
        )

    try:
        df = pd.read_csv(path)
        present = [c for c in request.feature_names if c in df.columns]

        if not present:
            raise HTTPException(
                status_code=400, detail="No requested features found in CSV"
            )

        corr = df[present].corr(method="pearson")

        return CorrelationResponse(
            correlation_matrix=corr.values.tolist(),
            feature_names=corr.columns.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "analysis"}
