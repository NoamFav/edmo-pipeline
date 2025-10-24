from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dataclasses import dataclass

app = FastAPI(title="Session Analysis Service", version="1.0.0")


@dataclass
class Event:
    t_abs_s: float
    time_str: str
    file: str
    action: str
    target: Optional[str]
    value: Optional[float]
    message: str


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


LINE_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+\[.*?\]\s+(?P<msg>.*)$",
)
PATTERNS = [
    (
        re.compile(r"(?P<who>[ABCD]) is in control\."),
        lambda m: ("control_start", m.group("who"), None),
    ),
    (
        re.compile(r"(?P<who>[ABCD]) is no longer in control\."),
        lambda m: ("control_end", m.group("who"), None),
    ),
    (
        re.compile(r"(?P<who>[ABCD]) joined session"),
        lambda m: ("joined", m.group("who"), None),
    ),
    (
        re.compile(
            r"Frequency of all oscillators set to (?P<val>[-+]?\d*\.?\d+)",
        ),
        lambda m: ("set_frequency_all", "all", float(m.group("val"))),
    ),
]


def time_to_seconds(tstr: str) -> float:
    h, m, s = tstr.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_message(msg: str):
    for rx, maker in PATTERNS:
        m = rx.search(msg)
        if m:
            return maker(m)
    return ("other", None, None)


def parse_line(line: str, src_file: str) -> Optional[Event]:
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    tstr, msg = m.group("time"), m.group("msg")
    action, target, value = parse_message(msg)
    return Event(
        time_to_seconds(tstr),
        tstr,
        src_file,
        action,
        target,
        value,
        msg,
    )


def build_timeline(log_dir: Path) -> list[Event]:
    events = []
    for fname in [
        "session.log",
        "User0.log",
        "User1.log",
        "User2.log",
        "User3.log",
    ]:
        p = log_dir / fname
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ev = parse_line(line, fname)
                if ev:
                    events.append(ev)
    events.sort(key=lambda e: e.t_abs_s)
    return events


def export_csv(events: list[Event], out_csv: Path):
    if not events:
        raise ValueError("No events to export")
    t0 = events[0].t_abs_s
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_str",
                "t_rel_s",
                "file",
                "action",
                "target",
                "value",
                "message",
            ]
        )
        for ev in events:
            w.writerow(
                [
                    ev.time_str,
                    f"{ev.t_abs_s - t0:.6f}",
                    ev.file,
                    ev.action,
                    ev.target or "",
                    f"{ev.value}" if ev.value else "",
                    ev.message,
                ]
            )


def compute_control_intervals(
    df: pd.DataFrame,
) -> list[tuple[str, float, float]]:
    intervals = []
    current_user, start_time = None, None
    d = df[
        df["action"].isin(
            [
                "control_start",
                "control_end",
            ]
        )
    ].sort_values("t_rel_s")

    for _, row in d.iterrows():
        t, action, user = (
            float(row["t_rel_s"]),
            row["action"],
            row["target"],
        )
        if action == "control_start":
            if current_user and start_time and t > start_time:
                intervals.append((current_user, start_time, t - start_time))
            current_user, start_time = user, t
        elif (
            action == "control_end"
            and current_user == user
            and start_time
            and t > start_time
        ):
            intervals.append((current_user, start_time, t - start_time))
            current_user, start_time = None, None

    if current_user and start_time:
        t_end = float(df["t_rel_s"].max())
        if t_end > start_time:
            intervals.append(
                (
                    current_user,
                    start_time,
                    t_end - start_time,
                )
            )
    return intervals


def extract_features(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).sort_values("t_rel_s").reset_index(drop=True)
    t0, tN = float(df["t_rel_s"].min()), float(df["t_rel_s"].max())
    total_time = max(0.0, tN - t0)

    feats = {"total_time_s": total_time, "num_events": int(len(df))}

    # Inter-event timing
    times = df["t_rel_s"].astype(float).tolist()
    gaps = [b - a for a, b in zip(times, times[1:])]
    feats["inter_event_mean_s"] = float(np.mean(gaps)) if gaps else 0.0
    feats["inter_event_std_s"] = float(np.std(gaps)) if gaps else 0.0

    # Control fractions
    users = ["A", "B", "C", "D"]
    intervals = compute_control_intervals(df)
    control_time = {}

    for u in users:
        total = 0
        for user, start, dur in intervals:
            if user == u:
                total += dur
        control_time[u] = total

    for u in users:
        feats[f"{u}_control_frac"] = (
            (control_time[u] / total_time) if total_time > 0 else 0.0
        )
        feats[f"{u}_event_count"] = int((df["target"] == u).sum())
        feats[f"{u}_event_rate_per_s"] = (
            (feats[f"{u}_event_count"] / total_time) if total_time > 0 else 0.0
        )

    # Action entropy
    action_counts = df["action"].value_counts().values
    if len(action_counts) > 0:
        p = action_counts / action_counts.sum()
        feats["action_entropy_bits"] = float(-(p * np.log2(p + 1e-12)).sum())
    else:
        feats["action_entropy_bits"] = 0.0

    # Control balance
    vals = np.array([control_time[u] for u in users])
    feats["control_balance_index"] = float(np.std(vals))

    # Reaction time
    changes = df[
        df["action"].isin(
            [
                "set_frequency",
                "set_amplitude",
            ]
        )
    ].sort_values("t_rel_s")
    starts = df[df["action"] == "control_start"].sort_values("t_rel_s")
    rts = []
    for _, srow in starts.iterrows():
        u, t = srow["target"], float(srow["t_rel_s"])
        nxt = changes[(changes["target"] == u) & (changes["t_rel_s"] > t)]
        if len(nxt):
            rts.append(float(nxt.iloc[0]["t_rel_s"]) - t)
    feats["reaction_time_mean_s"] = float(np.mean(rts)) if rts else 0.0

    return feats


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
    log_dir = Path(request.log_dir)
    if not log_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Directory not found: {request.log_dir}"
        )

    try:
        events = build_timeline(log_dir)
        export_csv(events, Path(request.output_csv))
        return {
            "status": "success",
            "events": len(events),
            "output": request.output_csv,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_features", response_model=FeatureResponse)
async def api_extract_features(request: FeatureRequest):
    """Extract features from timeline CSV."""
    path = Path(request.timeline_csv)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {request.timeline_csv}"
        )

    try:
        features = extract_features(path)
        return FeatureResponse(features=features)
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
