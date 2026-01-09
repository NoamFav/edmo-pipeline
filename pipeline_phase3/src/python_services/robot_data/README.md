# Robot Control Analysis – README

This repository contains a small toolkit for parsing robot session logs, building a **timeline** of events, extracting **interaction features**, and creating **visualisations** for group analysis.

---

## 📦 Project layout

- `robotdata1.py` — Parse raw log files, build `timeline.csv`, deduplicate mirrored events, and generate per‑session plots (control Gantt, event counts). 
- `feature_extraction1.py` — Load aggregated `features_all.csv`, explore/visualise features (correlations, PCA, clustering, group stats). 
- `visualisations1.py` — Plotting helpers: control timeline, event counts, temporal evolution of control diversity/entropy. 
- `diagrams1.py` — Lightweight “diagram” notebook-as-script: control timeline, counts, temporal diversity; shares helpers with `visualisations1.py`. 
- `edmo_utils.py` — Shared utilities extracted from notebooks (controller derivation, segments, deduplication, communication features, plotting helpers). 

> The plotting/helper functions in `edmo_utils.py` mirror those found in `visualisations1.py`/`diagrams1.py` so you can import a single place across scripts.

---

## 🔧 Requirements

- **Python** 3.10+ (tested on Windows paths in examples)
- **Packages**
  - `pandas`, `numpy`, `matplotlib`
  - `scipy`, `scikit-learn`, `seaborn`
  - (optional) `nbformat`/`nbconvert` if you plan to convert notebooks

Install everything:
```bash
pip install pandas numpy matplotlib scipy scikit-learn seaborn
```

---

## 📂 Expected data layout

The scripts expect a structure like:
```
<PROJECT_ROOT>/
  20251003_141557/
    Sessions/
      20251003/
        Suzanne/
          142002/
            session.log
            User0.log
            User1.log
            oscillator0.log
            ...
```
- `robotdata1.py` walks participant/session folders under a given root and produces per‑session outputs. (See the bottom of the file for an example batch run.) 

---

## ▶️ Typical workflow

### 1) Build timelines from raw logs
Use `robotdata1.py` helpers:

- **Parse logs → events**: `build_timeline(log_dir, include_oscillators=True, osc_subsample=1.0, osc_emit_on_change_only=True)`  
  Parses `session.log`, `User*.log`, and optional `oscillator*.log`, returning a chronologically sorted list of events. 

- **Write CSV**: `export_csv(events, out_csv)` → writes `timeline.csv` with columns:
  `time_str, t_rel_s, delta_prev_s, file, action, target, value, message`. 

- **Deduplicate mirrors**: `dedupe_timeline_csv(timeline.csv, time_decimals=3)`  
  Buckets time, prefers `session.log` over `User*.log`, recomputes `delta_prev_s`. 

- **Batch example**: the script iterates over all sessions under a day folder, exporting timelines and saving plots. 

### 2) Compute per‑session features
`robotdata1.py` includes feature builders that rely on the timeline:

- **Control intervals**: `compute_control_intervals(df)` → `(user, start, duration)` tuples. 
- **Communication/interaction features**: `compute_comm_features(df)` → turn‑taking, responsiveness, balance (entropy), burstiness, synchrony, etc.

> Many helpers depend on robust controller derivation from logs (`_derive_controller_column`), which infers who is in control using messages and action/target pairs and forward-fills over time. 

### 3) Aggregate across sessions
- The batch at the bottom of `robotdata1.py` shows how to loop sessions, write each session’s `features.csv`, and append to a **combined** `features_all.csv`. 

### 4) Explore/visualise features
From `feature_extraction1.py`:
- Load: `df = pd.read_csv("features_all.csv")` and pick `features_of_interest`. 
- **Correlations**: heatmap of feature correlation matrix.   
- **PCA**: project sessions to 2D for pattern discovery.  
- **Clustering**: KMeans to group session behaviors. 
- **Group stats/plots**: e.g., average control fraction per user, reaction times per participant. 

---

## 📊 Built‑in visualisations

You can generate plots directly from a `timeline.csv` with either `visualisations1.py`, `diagrams1.py`, or `edmo_utils.py`:

- **Control timeline (Gantt‑like)**: `plot_control_timeline(timeline.csv, save_path="control_timeline.png")`  
  Shows contiguous control segments per participant. 

- **Event counts**: `plot_event_counts(timeline.csv, save_path="event_counts.png")`  
  Bar chart of who holds control/appears most. 

- **Temporal evolution / control diversity**: `plot_temporal_evolution(timeline.csv, window_s=10.0, save_path="temporal_diversity.png")`  
  Normalized entropy over time to gauge stability vs. variability. 

> Additional session‑level plots (Gantt, counts) also exist inside `robotdata1.py` (e.g., `plot_control_gantt`, `plot_event_counts`). 

---

## 🏃 How to run

### Option A — End‑to‑end (per day / all sessions)
Edit paths at the bottom of `robotdata1.py` to point to your day folder (e.g., `.../Sessions/20251003`), then run:
```bash
python robotdata1.py
```
This will:
- build/overwrite `timeline.csv` for each session,
- write `features.csv` per session,
- maintain/append `features_all.csv`,
- generate plots per session (timeline, counts, temporal evolution). 

### Option B — Ad‑hoc usage in a notebook or script
```python
from pathlib import Path
import pandas as pd
from robotdata1 import build_timeline, export_csv, dedupe_timeline_csv
from edmo_utils import plot_control_timeline, plot_temporal_evolution

session_dir = Path(".../Sessions/20251003/Suzanne/142002")
events = build_timeline(session_dir, include_oscillators=True)
export_csv(events, session_dir/"timeline.csv")
dedupe_timeline_csv(session_dir/"timeline.csv")

plot_control_timeline(session_dir/"timeline.csv", save_path=session_dir/"control_timeline.png")
plot_temporal_evolution(session_dir/"timeline.csv", window_s=10.0, save_path=session_dir/"temporal_evolution.png")
```

### Option C — Feature exploration
```bash
python feature_extraction1.py
```
This expects an existing `features_all.csv` in the working directory and opens several analysis plots. 

---

## 🧪 Tips & gotchas

- **Future imports on Windows:** ensure `from __future__ import annotations` is at the very top of your files (after shebang/encoding and optional docstring) to avoid syntax errors.  
- **Controller derivation:** If timelines lack explicit `control_start/control_end` messages, the helpers infer control from messages and actions and then forward‑fill — verify results by plotting a control timeline. 
- **Deduping mirrored events:** Use `dedupe_timeline_csv` after generating a timeline to resolve mirrored entries from different logs. 
- **Oscillator noise:** When including `oscillator*.log`, use `osc_subsample` and `osc_emit_on_change_only=True` to reduce spammy events.

---

## 📚 Key API summary

- `build_timeline(log_dir, include_oscillators=True, osc_subsample=1.0, osc_emit_on_change_only=True) -> List[Event]` 
- `export_csv(events, out_csv)` → write `timeline.csv`
- `dedupe_timeline_csv(path_in, path_out=None, time_decimals=3)` → clean timeline CSV 
- `compute_control_intervals(df)` → intervals for plotting/metrics  
- `compute_comm_features(df)` → interaction metrics for feedback/analysis 
- `plot_control_timeline(...)`, `plot_event_counts(...)`, `plot_temporal_evolution(...)` → quick visuals from CSV or DataFrame. 

---

## 📄 License / attribution

Internal student project materials for analysis and demonstration purposes. If re‑using the helpers in another project, please copy with attribution.
