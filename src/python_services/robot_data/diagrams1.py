#!/usr/bin/env python
# coding: utf-8
"""
Diagrams — minimal plotting utilities (self-contained).
If you already centralize helpers in `edmo_utils.py`, you can swap these
with imports from that module. This file intentionally stays standalone.
"""

from __future__ import annotations
from pathlib import Path

from edmo_utils import (
    pd, np, plt,                   
    plot_control_gantt as plot_control_timeline,
    plot_event_counts,
    
)



# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _load_df(df_or_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV path or return the DataFrame if already provided.
    Raises:
        ValueError: if input is neither a DataFrame nor an existing file path.
    """
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    p = Path(df_or_path)
    if p.exists() and p.is_file():
        return pd.read_csv(p)
    raise ValueError("Provide a DataFrame or an existing CSV file path.")


def _derive_controller_column(df: pd.DataFrame) -> pd.Series:
    """
    Derive a 'controller' Series using multiple cues:
      1) existing 'controller' column
      2) messages like 'X is in control'
      3) action/target rows indicating control grabs
    Forward-fills once a controller is first seen.
    """
    if "controller" in df.columns:
        return df["controller"].astype(str)

    ctrl = pd.Series([None] * len(df), index=df.index, dtype="object")

    # Heuristic 1: message text
    if "message" in df.columns:
        msg = df["message"].astype(str)
        mask = msg.str.contains(r"\b([ABCD])\b.*is in control", case=False, regex=True)
        ctrl.loc[mask] = msg[mask].str.extract(r"\b([ABCD])\b", expand=False)

    # Heuristic 2: action/target pairs
    if {"action", "target"}.issubset(df.columns):
        act = df["action"].astype(str).str.lower()
        tgt = df["target"].astype(str)
        grab = act.isin(["control_start", "take_control", "control", "grab_control"])
        ctrl.loc[grab] = tgt[grab].where(tgt[grab].isin(list("ABCD")))

    # Forward-fill in time order if possible
    if "t_rel_s" in df.columns and not df.empty:
        d = df.sort_values("t_rel_s")
        ctrl = ctrl.loc[d.index].ffill()
        return ctrl.reindex(df.index).ffill()

    return ctrl.ffill()


def _compute_segments(df: pd.DataFrame) -> List[Dict]:
    """
    Build contiguous controller segments from time-sorted rows.
    Returns: [{'who': 'A', 'start': s, 'end': e, 'dur': e-s}, ...]
    """
    if "t_rel_s" not in df.columns:
        raise ValueError("Expected 't_rel_s' in DataFrame.")
    if df.empty:
        return []

    d = df[["t_rel_s"]].copy()
    d["controller"] = _derive_controller_column(df)
    d = d.sort_values("t_rel_s").reset_index(drop=True)

    segs: List[Dict] = []
    current = d.loc[0, "controller"]
    seg_start = float(d.loc[0, "t_rel_s"])

    for i in range(1, len(d)):
        who = d.loc[i, "controller"]
        t = float(d.loc[i, "t_rel_s"])
        if who != current and pd.notna(who):
            if pd.notna(current):
                segs.append({"who": str(current), "start": seg_start, "end": t, "dur": t - seg_start})
            current, seg_start = who, t

    session_end = float(d["t_rel_s"].max())
    if pd.notna(current):
        segs.append({"who": str(current), "start": seg_start, "end": session_end, "dur": session_end - seg_start})

    return [s for s in segs if pd.notna(s["dur"]) and s["dur"] > 1e-9]


# ───────────────────────────────────────────────────────────────────────────────
# Plots
# ───────────────────────────────────────────────────────────────────────────────

def plot_control_timeline(df_or_path: Union[str, Path, pd.DataFrame], save_path: Union[str, Path, None] = None) -> None:
    """
    Plot the control timeline per participant.
    """
    df = _load_df(df_or_path)
    segs = _compute_segments(df)
    if not segs:
        print("No segments to plot.")
        return

    participants = sorted({s["who"] for s in segs if pd.notna(s["who"])})
    fig, ax = plt.subplots(figsize=(10, 2.8))

    for s in segs:
        y = participants.index(s["who"])
        ax.plot([s["start"], s["end"]], [y, y], linewidth=6)

    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants)
    ax.set_xlabel("Time (s)")
    ax.set_title("Control Timeline")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_event_counts(df_or_path: Union[str, Path, pd.DataFrame], save_path: Union[str, Path, None] = None) -> None:
    """
    Plot counts of derived 'controller' per participant (proxy for activity).
    """
    df = _load_df(df_or_path)
    ctrl = _derive_controller_column(df)
    counts = ctrl.dropna().astype(str).value_counts().sort_index()

    # Ensure A/B/C/D columns exist (even when zero)
    for u in ["A", "B", "C", "D"]:
        if u not in counts.index:
            counts.loc[u] = 0
    counts = counts.sort_index()

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Count")
    ax.set_title("Event Counts per Participant (derived)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_temporal_evolution(
    df_or_path: Union[str, Path, pd.DataFrame],
    window_s: float = 10.0,
    save_path: Union[str, Path, None] = None
) -> None:
    """
    Plot temporal evolution of control diversity using normalized entropy.
    Normalization: divide by log2(#participants seen in window).
    """
    df = _load_df(df_or_path).sort_values("t_rel_s").reset_index(drop=True)
    if df.empty or "t_rel_s" not in df.columns:
        print("No timeline to compute temporal evolution.")
        return

    ctrl = _derive_controller_column(df)
    t = df["t_rel_s"].astype(float).to_numpy()

    # Window centers: choose a reasonable number based on range, minimum 5 points
    t_min, t_max = t.min(), t.max()
    n_centers = max(5, int((t_max - t_min) / max(1.0, window_s / 2)))
    centers = np.linspace(t_min, t_max, n_centers)

    def win_entropy(c: float) -> float:
        mask = (t >= c - window_s / 2) & (t <= c + window_s / 2)
        vals = ctrl[mask].dropna().astype(str)
        if vals.empty:
            return np.nan
        p = vals.value_counts(normalize=True).to_numpy()
        m = len(vals.unique())
        # normalized entropy in [0,1]
        return float((-np.sum(p * np.log2(p))) / np.log2(m)) if m > 1 else 0.0

    H = np.array([win_entropy(c) for c in centers])

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(centers, H, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title(f"Control Diversity (window={window_s:.0f}s)")
    ax.grid(True, linestyle=":", linewidth=0.6)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# Example usage (optional)
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust this path to your local CSV:
    csv_path = Path(
        r"C:\Users\alexa\Desktop\Maastricht University\Project_3.1\20251003_141557\TestData\timeline3.csv"
    )

    if csv_path.exists():
        plot_control_timeline(csv_path)
        plot_event_counts(csv_path)
        plot_temporal_evolution(csv_path)
    else:
        print(f"⚠️ CSV not found: {csv_path}")
