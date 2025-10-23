#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

from edmo_utils import (
    Path, pd, plt,
    load_timeline,
    plot_control_gantt,
    plot_event_counts,
    plot_temporal_evolution,
)


"""
visualisations1.py
------------------
Thin orchestration layer for plots. No helper logic is duplicated here.

Outputs per session (when timeline.csv exists):
  • timeline_control.png      — Gantt-like control bars
  • timeline_event_counts.png — event counts per user
  • temporal_evolution.png    — per-user control fractions + entropy
"""


# ───────────────────────────────────────────────────────────────────────────────
# Per-session plotting
# ───────────────────────────────────────────────────────────────────────────────

def plot_suite_for_session(
    session_dir: Path,
    csv_name: str = "timeline.csv",
    out_control: str = "timeline_control.png",
    out_counts: str = "timeline_event_counts.png",
    out_temporal: str = "temporal_evolution.png",
) -> bool:
    """
    Generate the 3 standard plots for a single session directory.

    Returns:
        True if plots were created, False if timeline was missing.
    """
    csv_path = session_dir / csv_name
    if not csv_path.exists():
        print(f"⏭️  No {csv_name} in {session_dir} — skipping")
        return False

    # Control Gantt + counts use the loaded DataFrame
    df = load_timeline(csv_path)
    tag = f"— {session_dir.parent.name}/{session_dir.name}"

    plot_control_gantt(df, session_dir / out_control, title_suffix=tag)
    plot_event_counts(df, session_dir / out_counts, title_suffix=tag)

    # Temporal evolution reads the CSV directly
    plot_temporal_evolution(csv_path, session_dir / out_temporal)

    print(f"🖼️  {session_dir}: {out_control}, {out_counts}, {out_temporal}")
    return True


# ───────────────────────────────────────────────────────────────────────────────
# Batch/day plotting
# ───────────────────────────────────────────────────────────────────────────────

def plot_suite_for_day(day_root: Path, csv_name: str = "timeline.csv") -> pd.DataFrame:
    """
    Recurse <day_root>/<participant>/<session>/timeline.csv and generate plots.

    Returns:
        A DataFrame summary of which sessions were processed.
    """
    rows = []
    if not day_root.exists():
        print(f"⚠️  Day root not found: {day_root}")
        return pd.DataFrame(rows)

    for participant_dir in sorted(p for p in day_root.iterdir() if p.is_dir()):
        for session_dir in sorted(s for s in participant_dir.iterdir() if s.is_dir()):
            ok = plot_suite_for_session(session_dir, csv_name=csv_name)
            rows.append({
                "participant": participant_dir.name,
                "session": session_dir.name,
                "plotted": bool(ok),
            })
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────────────
# CLI entry (optional)
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust to your local structure as needed.
    day_root = Path("20251003_141557/Sessions/20251003")

    if day_root.exists():
        summary = plot_suite_for_day(day_root)
        if not summary.empty:
            out = Path("visualisations_summary.csv")
            summary.to_csv(out, index=False)
            print(f"🧾 Wrote summary: {out.resolve()}")
    else:
        # Or run a single session explicitly:
        one = Path("20251003_141557/Sessions/20251003/Suzanne/142002")
        _ = plot_suite_for_session(one)
