#!/usr/bin/env python
# coding: utf-8
"""
edmo_utils.py
-------------
Central utility module for EDMO project.
Contains:
- Data loading and timeline utilities
- Control interval & event parsing helpers
- Communication and temporal feature extraction
- Plotting helpers (Gantt, event counts, temporal evolution)

Used by: robotdata1.py, visualisations1.py, feature_extraction1.py
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# ── Standard library imports
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Optional as Optional, List as List, Tuple as Tuple, Dict as Dict
from collections import defaultdict as defaultdict, Counter as Counter
import re as re, math as math, csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Optional scientific imports
try:
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import scipy.stats as stats
except Exception:
    sns = None
    StandardScaler = KMeans = PCA = None
    stats = None


# ───────────────────────────────────────────────────────────────────────────────
# Global constants
# ───────────────────────────────────────────────────────────────────────────────
USERS = ["A", "B", "C", "D"]
WINDOW_SIZE = 10.0  # seconds per temporal window


# ───────────────────────────────────────────────────────────────────────────────
# General utilities
# ───────────────────────────────────────────────────────────────────────────────
def load_timeline(path) -> pd.DataFrame:
    """
    Load a timeline CSV (or pass-through DataFrame) sorted by 't_rel_s'.
    Raises if column missing.
    """
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        df = pd.read_csv(Path(path))
    if "t_rel_s" not in df.columns:
        raise ValueError("timeline.csv missing required column 't_rel_s'")
    return df.sort_values("t_rel_s").reset_index(drop=True)


def _safe_mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else 0.0


def _safe_std(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.std(x)) if x else 0.0


def _entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = [c / total for c in counts if c > 0]
    return -sum(pi * math.log(pi + 1e-12, 2) for pi in p)


def _shannon_entropy(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _normalize_entropy(h, n):
    return float(h / math.log2(n)) if n > 1 and h > 0 else 0.0


def _safe_var(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.var(x)) if len(x) >= 1 else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# Control intervals
# ───────────────────────────────────────────────────────────────────────────────
def compute_control_intervals(df: pd.DataFrame):
    """
    Compute control intervals from control_start and control_end events.
    Returns [(user, start_time, duration), ...]
    """
    intervals = []
    current_user = None
    start_time = None
    d = df[df["action"].isin(["control_start", "control_end"])].sort_values("t_rel_s")

    for _, row in d.iterrows():
        t = float(row["t_rel_s"])
        a = row["action"]
        u = row["target"]

        if a == "control_start":
            if current_user is not None and start_time is not None:
                intervals.append((current_user, start_time, t - start_time))
            current_user, start_time = u, t

        elif a == "control_end":
            if current_user == u and start_time is not None:
                intervals.append((current_user, start_time, t - start_time))
                current_user, start_time = None, None

    if current_user is not None and start_time is not None:
        t_end = float(df["t_rel_s"].max())
        if t_end > start_time:
            intervals.append((current_user, start_time, t_end - start_time))

    return intervals


# ───────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────────────
def plot_control_gantt(df: pd.DataFrame, png_path: Path, title_suffix: str = ""):
    """Plot a Gantt-style control timeline per user."""
    intervals = compute_control_intervals(df)
    if not intervals:
        return

    user_rows = {"A": 3, "B": 2, "C": 1, "D": 0}
    per_user = {u: [] for u in user_rows}
    for u, start, dur in intervals:
        if u in per_user and dur > 0:
            per_user[u].append((start, dur))

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_title(f"Control Timeline {title_suffix}".strip())
    ax.set_xlabel("Time (s since start)")
    ax.set_yticks(list(user_rows.values()))
    ax.set_yticklabels(list(user_rows.keys()))
    ax.grid(True, linestyle=":", alpha=0.5)

    for u, row in user_rows.items():
        if per_user[u]:
            ax.broken_barh(per_user[u], (row - 0.35, 0.7))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_event_counts(df: pd.DataFrame, png_path: Path, title_suffix: str = ""):
    """Plot number of events per user (A/B/C/D)."""
    users = ["A", "B", "C", "D"]
    counts = {u: 0 for u in users}

    if "target" in df.columns:
        sub = df[df["target"].isin(users)]
        for u, c in sub["target"].value_counts().items():
            counts[u] = int(c)

    # safe unpacking even if counts is empty
    xs = list(counts.keys())
    ys = [counts.get(u, 0) for u in xs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys)
    ax.set_title(f"Event Counts {title_suffix}".strip())
    ax.set_xlabel("User")
    ax.set_ylabel("Events")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)



# ───────────────────────────────────────────────────────────────────────────────
# Temporal evolution (entropy + control fraction)
# ───────────────────────────────────────────────────────────────────────────────
def sliding_entropy(actions):
    """Shannon entropy of action types within a window."""
    if len(actions) == 0:
        return 0.0
    counts = pd.Series(actions).value_counts()
    probs = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()


def compute_temporal_features(csv_path: Path, window_s: float = WINDOW_SIZE):
    """
    Return DataFrame with rolling control and entropy per time window.
    Columns:
      - t_mid (center of window)
      - entropy (action entropy)
      - A_control_frac, B_control_frac, etc.
    """
    df = load_timeline(csv_path)
    t_end = df["t_rel_s"].max()
    bins = np.arange(0, t_end + window_s, window_s)
    rows = []

    control = defaultdict(list)
    cur_control, cur_start = None, 0
    for _, r in df.iterrows():
        t = r["t_rel_s"]
        if r["action"] == "control_start":
            if cur_control is not None:
                control[cur_control].append((cur_start, t))
            cur_control, cur_start = r["target"], t
        elif r["action"] == "control_end" and r["target"] == cur_control:
            control[cur_control].append((cur_start, t))
            cur_control = None
    if cur_control is not None:
        control[cur_control].append((cur_start, t_end))

    for i in range(len(bins) - 1):
        t0, t1 = bins[i], bins[i + 1]
        window = df[(df["t_rel_s"] >= t0) & (df["t_rel_s"] < t1)]
        entropy = sliding_entropy(window["action"].tolist())
        row = {"t_mid": (t0 + t1) / 2, "entropy": entropy}
        for u in USERS:
            u_time = 0.0
            for (start, end) in control.get(u, []):
                overlap = max(0, min(t1, end) - max(t0, start))
                u_time += overlap
            row[f"{u}_control_frac"] = u_time / window_s
        rows.append(row)

    return pd.DataFrame(rows)


def plot_temporal_evolution(csv_path: Path, out_path: Path):
    """Plot temporal evolution of control fractions and entropy."""
    df = compute_temporal_features(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    title = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
    ax1.set_title(f"Temporal Evolution — {title}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Control Fraction")

    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    for u in USERS:
        if f"{u}_control_frac" in df.columns:
            ax1.plot(df["t_mid"], df[f"{u}_control_frac"], label=f"{u} control", color=colors[u])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df["t_mid"], df["entropy"], color="black", lw=2, label="Action Entropy")
    ax2.set_ylabel("Entropy (bits)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Public API: export convenient names and key functions ──
__all__ = [
    # convenience re-exports
    "dataclass","Path","Optional","List","Tuple","Dict","defaultdict","Counter",
    "re","math","csv","np","pd","plt",
    # constants
    "USERS","WINDOW_SIZE",
    # core I/O / processing
    "build_timeline","export_csv","dedupe_timeline_csv","load_timeline",
    # analysis helpers & features
    "compute_control_intervals","compute_temporal_features","extract_session_features",
    # plotting
    "plot_control_gantt","plot_event_counts","plot_temporal_evolution",
    "id_to_letter_map","plot_session_timeline_with_changes",
    # optional analytics
    "sns","StandardScaler","KMeans","PCA","stats",
]
