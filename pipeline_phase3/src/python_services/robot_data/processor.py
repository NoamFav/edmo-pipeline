"""Robot data processor - can be used independently or via FastAPI."""
from typing import Dict, List, Optional
from pathlib import Path
import re
import csv
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Event:
    t_abs_s: float
    time_str: str
    file: str
    action: str
    target: Optional[str]
    value: Optional[float]
    message: str


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
    """Convert HH:MM:SS.mmm to seconds."""
    h, m, s = tstr.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_message(msg: str):
    """Parse log message and extract action, target, value."""
    for rx, maker in PATTERNS:
        m = rx.search(msg)
        if m:
            return maker(m)
    return ("other", None, None)


def parse_line(line: str, src_file: str) -> Optional[Event]:
    """Parse a single log line."""
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


def build_timeline(log_dir: Path) -> List[Event]:
    """Build timeline from all log files in directory."""
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


def export_csv(events: List[Event], out_csv: Path):
    """Export timeline to CSV file."""
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


def compute_control_intervals(df: pd.DataFrame) -> List[tuple]:
    """Compute control intervals from timeline DataFrame."""
    intervals = []
    current_user, start_time = None, None
    d = df[
        df["action"].isin(["control_start", "control_end"])
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
            intervals.append((current_user, start_time, t_end - start_time))
    return intervals


class RobotDataProcessor:
    """Processes robot session logs."""
    
    @staticmethod
    def parse_logs(log_dir: str, output_csv: str) -> Dict:
        """Parse log files and build timeline CSV.
        
        Args:
            log_dir: Directory containing log files
            output_csv: Path to output CSV file
            
        Returns:
            Dictionary with status, event count, and output path
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            raise FileNotFoundError(f"Directory not found: {log_dir}")
        
        events = build_timeline(log_path)
        export_csv(events, Path(output_csv))
        
        return {
            "status": "success",
            "events": len(events),
            "output": output_csv,
        }
    
    @staticmethod
    def extract_features(timeline_csv: str) -> Dict:
        """Extract features from timeline CSV.
        
        Args:
            timeline_csv: Path to timeline CSV file
            
        Returns:
            Dictionary with extracted features
        """
        path = Path(timeline_csv)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {timeline_csv}")
        
        df = pd.read_csv(path).sort_values("t_rel_s").reset_index(drop=True)
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
            df["action"].isin(["set_frequency", "set_amplitude"])
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


# Convenience functions
def parse_logs(log_dir: str, output_csv: str) -> Dict:
    """Parse log files and build timeline CSV."""
    return RobotDataProcessor.parse_logs(log_dir, output_csv)


def extract_features(timeline_csv: str) -> Dict:
    """Extract features from timeline CSV."""
    return RobotDataProcessor.extract_features(timeline_csv)
