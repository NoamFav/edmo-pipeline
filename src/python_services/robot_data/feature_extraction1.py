#!/usr/bin/env python
# coding: utf-8
"""
feature_extraction1.py — analysis & visuals over features_all.csv

Outputs:
- Correlation heatmap of selected features
- PCA scatter (2D) with session labels
- KMeans clusters on PCA space
- Bar of mean control fractions (A/B/C/D)
- Scatter: entropy vs control balance (colored by participant)

Notes:
- Robust to missing columns & NaNs (drops rows with all-NaN features).
- Uses only matplotlib (no seaborn).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats


# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────

CSV_PATH = Path("features_all.csv")
FEATURES = [
    "A_control_frac", "B_control_frac", "C_control_frac", "D_control_frac",
    "reaction_time_mean_s", "action_entropy_bits",
    "inter_event_burstiness", "A_event_rate_per_s", "B_event_rate_per_s",
]
CONTROL_FRAC_COLS = [c for c in ["A_control_frac", "B_control_frac", "C_control_frac", "D_control_frac"]]
LABEL_COLS = ["participant", "session"]  # expected meta columns
N_CLUSTERS = 3
RANDOM_STATE = 42


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def require_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return the subset of `cols` that actually exist in df (warn on missing)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns (skipped): {missing}")
    present = [c for c in cols if c in df.columns]
    if not present:
        raise ValueError("None of the requested columns are present.")
    return present


def zscore_or_none(X: pd.DataFrame) -> np.ndarray:
    """Standardize columns, skipping rows with all-NaN features."""
    # keep only rows that have at least one non-NaN among selected features
    valid_rows = X.notna().any(axis=1)
    if not valid_rows.any():
        raise ValueError("All rows are NaN across selected features.")
    X = X.loc[valid_rows]
    scaler = StandardScaler()
    return scaler.fit_transform(X), valid_rows


def safe_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation (Pearson) with NaNs handled (drops rows with any NaN in pair)."""
    return df.corr(method="pearson")


# ───────────────────────────────────────────────────────────────────────────────
# Plots (matplotlib only; no explicit colors)
# ───────────────────────────────────────────────────────────────────────────────

def plot_corr_heatmap(corr: pd.DataFrame, title: str = "Feature Correlations"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    # annotate cells
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def plot_pca_scatter(coords: np.ndarray, labels: pd.Series, title: str = "PCA projection of session features"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1])
    # label each point with session name (or index if missing)
    for i, txt in enumerate(labels.astype(str).fillna("").values):
        ax.text(coords[i, 0] + 0.02, coords[i, 1], txt, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.tight_layout()
    plt.show()


def plot_kmeans_on_pca(coords: np.ndarray, y: np.ndarray, title: str = "Session clusters (behavior patterns)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    plt.show()


def plot_mean_control_fracs(df: pd.DataFrame):
    cols = require_columns(df, CONTROL_FRAC_COLS)
    means = df[cols].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index, means.values)
    ax.set_title("Average Control Fraction per User")
    ax.set_ylabel("Fraction of Time in Control")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    plt.show()


def plot_entropy_vs_balance(df: pd.DataFrame):
    cols = require_columns(df, ["control_balance_index", "action_entropy_bits"])
    x = df[cols[0]].values
    y = df[cols[1]].values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y)
    ax.set_xlabel("Control Balance Index")
    ax.set_ylabel("Action Entropy (bits)")
    ax.set_title("Entropy vs Control Balance")
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.tight_layout()
    plt.show()


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)
    print("Loaded:", df.shape)

    # Ensure meta columns exist (create safe fallbacks)
    for c in LABEL_COLS:
        if c not in df.columns:
            df[c] = ""

    # 1) Feature correlation heatmap (on present features only)
    feats_present = require_columns(df, FEATURES)
    X = df[feats_present]
    corr = safe_corr(X)
    plot_corr_heatmap(corr, title="Feature Correlations")

    # 2) PCA on standardized features (drop rows that are all-NaN across feats)
    X_scaled, valid_mask = zscore_or_none(X)
    # Keep aligned labels for valid rows
    session_labels = df.loc[valid_mask, "session"]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    plot_pca_scatter(pca_coords, session_labels, title="PCA projection of session features")

    # 3) KMeans clusters on PCA space
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(pca_coords)
    df.loc[valid_mask, "cluster"] = labels
    print(df[["participant", "session", "cluster"]].head())
    plot_kmeans_on_pca(pca_coords, labels, title="Session clusters (behavior patterns)")

    # 4) Example t-test (guarded): reaction_time_mean_s per two participants (if they exist)
    if "reaction_time_mean_s" in df.columns and "participant" in df.columns:
        gA = df[df["participant"] == "Suzanne"]["reaction_time_mean_s"].dropna()
        gB = df[df["participant"] == "John"]["reaction_time_mean_s"].dropna()
        if len(gA) > 1 and len(gB) > 1:
            t_res = stats.ttest_ind(gA, gB, equal_var=False)
            print("Two-sample t-test (Suzanne vs John) on reaction_time_mean_s:", t_res)
        else:
            print("⚠️ Not enough data for t-test (need at least 2 values in each group).")

    # 5) Mean control fractions bar plot
    plot_mean_control_fracs(df)

    # 6) Entropy vs Control Balance scatter
    plot_entropy_vs_balance(df)


if __name__ == "__main__":
    main()
