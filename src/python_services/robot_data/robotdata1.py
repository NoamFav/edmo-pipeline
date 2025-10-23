#!/usr/bin/env python
# coding: utf-8

# # Robotdata
# 
# > Auto-cleaned version. Imports consolidated; functions available in `edmo_utils.py`.
# 

from __future__ import annotations
from edmo_utils import (
    dataclass, Path, Optional, List, Tuple, Dict, defaultdict,
    re, math, csv, np, pd, plt,
    USERS, WINDOW_SIZE,
    # functions available from edmo_utils
    load_timeline,
    compute_control_intervals,
    plot_control_gantt, plot_event_counts,
    compute_temporal_features, plot_temporal_evolution,
)



# Function to convert time to seconds to build timeline
def time_to_seconds(tstr : str) -> float:
    """
    Convert a time string in the format "HH:MM:SS.sss" to total seconds.
    Args:
        tstr (str): Time string in the format "HH:MM:SS.sss".
    Returns:
        float: Total time in seconds.
    """

    h,m,s = tstr.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class Event:
    """
    A single log event.
    """
    t_abs_s: float           # absolute seconds from 00:00:00.000
    time_str: str            # timestamp string
    file: str                # which log file it came from
    action: str              # e.g. control_start, set_frequency, etc.
    target: Optional[str]    # user or oscillator
    value: Optional[float]   # numeric value if present
    message: str             # full log line


# Lines look like: "HH:MM:SS.micro [TAG] message..."
LINE_RE = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+\[.*?\]\s+(?P<msg>.*)$")

# High-level event patterns we care about
PATTERNS: List[Tuple[re.Pattern, callable]] = [
    # Session lifecycle
    (re.compile(r"Session for (?P<user>.+) started\."), 
     lambda m: ("session_start", m.group("user"), None)),
    (re.compile(r"(?P<who>[ABCD]) joined session, assigned id (?P<id>\d+)\."), 
     lambda m: ("joined", m.group("who"), float(m.group("id")))),
    (re.compile(r"(?P<who>[ABCD]) left session, id (?P<id>\d+) returned to pool\."), 
     lambda m: ("left", m.group("who"), float(m.group("id")))),

    # Control possession
    (re.compile(r"(?P<who>[ABCD]) is in control\."), 
     lambda m: ("control_start", m.group("who"), None)),
    (re.compile(r"(?P<who>[ABCD]) is no longer in control\."), 
     lambda m: ("control_end", m.group("who"), None)),

    # Global oscillator changes
    (re.compile(r"Frequency of all oscillators set to (?P<val>[-+]?\d*\.?\d+)\s*by user\."), 
     lambda m: ("set_frequency_all", "all", float(m.group("val")))),

    # Per-user/osc parameter changes (examples)
    (re.compile(r"User\s+(?P<uid>\d+)\s+sets\s+frequency\s+to\s+(?P<val>[-+]?\d*\.?\d+)"), 
     lambda m: ("set_frequency", f"user{m.group('uid')}", float(m.group("val")))),
    (re.compile(r"User\s+sets\s+amplitude\s+to\s+(?P<val>[-+]?\d*\.?\d+)"), 
     lambda m: ("set_amplitude", None, float(m.group("val")))),
    (re.compile(r"Amplitude of oscillator\s+(?P<osc>\d+)\s+set to\s+(?P<val>[-+]?\d*\.?\d+)\s+by user\."), 
     lambda m: ("set_amplitude_osc", m.group("osc"), float(m.group("val")))),
    (re.compile(r"User\s+sets\s+offset\s+to\s+(?P<val>[-+]?\d*\.?\d+)"), 
     lambda m: ("set_offset", None, float(m.group("val")))),
]

def parse_message(msg: str) -> Tuple[str, Optional[str], Optional[float]]:
    """
    Parse a log message and extract the event type, target, and value.
    Args:
        msg (str): The log message to parse.
    Returns:
        Tuple[str, Optional[str], Optional[float]]: A tuple containing the event type,"""

    for rx, maker in PATTERNS:
        m = rx.search(msg)
        if m:
            return maker(m)
    return ("other", None, None)

def parse_session_or_user_line(line: str, src_file: str) -> Optional[Event]:
    """
    Parse a log line for session or user events.
    Args:
        line (str): The log line to parse.
        src_file (str): The source file name.
        Returns:
            Optional[Event]: An Event object if the line matches, else None.

    """

    m = LINE_RE.match(line.strip())
    if not m:
        return None
    tstr = m.group("time")
    msg  = m.group("msg")
    t_abs = time_to_seconds(tstr)
    action, target, value = parse_message(msg)
    return Event(t_abs, tstr, src_file, action, target, value, msg)

OSC_LINE_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+\[.*?\]\s+"
    r"(?:oscillator|osc)\s*(?P<idx>\d+)\s*:\s*"
    r"(?:(?:freq|f)=(?P<freq>[-+]?\d*\.?\d+))?[, ]*"
    r"(?:(?:amp|A)=(?P<amp>[-+]?\d*\.?\d+))?[, ]*"
    r"(?:(?:offs|off|O)=(?P<off>[-+]?\d*\.?\d+))?",
    re.IGNORECASE
)

def parse_oscillator_line(line: str) -> Optional[Tuple[float, str, Dict[str, float], str]]:
    """
    Parse a log line for oscillator parameter changes.
    Args:
        line (str): The log line to parse.
    Returns:
        Optional[Tuple[float, str, Dict[str, float], str]]: A tuple
            containing absolute time, oscillator index, parameter values, and the original line,
            or None if the line does not match.
    """

    m = OSC_LINE_RE.match(line.strip())
    if not m:
        return None
    tstr = m.group("time")
    t_abs = time_to_seconds(tstr)
    idx = m.group("idx")
    vals: Dict[str, float] = {}
    for key in ("freq", "amp", "off"):
        v = m.group(key)
        if v is not None:
            vals[key] = float(v)
    return (t_abs, idx, vals, line.strip())


#====== BUILD TIMELINE FROM LOG FILES ======#
def build_timeline(
    log_dir: Path,
    files: Optional[List[str]] = None,
    include_oscillators: bool = False,
    osc_subsample: float = 0.0,             # seconds; emit at most one event per osc within this interval
    osc_emit_on_change_only: bool = True    # only emit when value changes
) -> List[Event]:
    """
    Build a timeline of events from log files in a directory.
    Args:
        log_dir (Path): Directory containing log files.
        files (Optional[List[str]]): List of log file names to process. If None, defaults to session.log and User*.log.
        include_oscillators (bool): Whether to include oscillator logs.
        osc_subsample (float): Minimum time interval between emitted oscillator events.
        osc_emit_on_change_only (bool): Whether to emit oscillator events only on value changes.
    Returns:
        List[Event]: A sorted list of Event objects.
        
    """
    if files is None:
        files = ["session.log", "User0.log", "User1.log", "User2.log", "User3.log"]

    events: List[Event] = []

    # 1) Session + User logs
    for fname in files:
        p = (log_dir / fname)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ev = parse_session_or_user_line(line, fname)
                if ev is not None:
                    events.append(ev)

    if include_oscillators:
        osc_files = [f"oscillator{i}.log" for i in range(4)]
        last_emit_time: Dict[str, float] = {}          # per osc idx
        last_vals: Dict[Tuple[str, str], float] = {}   # (idx, key) -> last value

        for fname in osc_files:
            p = (log_dir / fname)
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed = parse_oscillator_line(line)
                    if not parsed:
                        continue
                    t_abs, idx, vals, raw = parsed

                    # Apply subsampling (emit at most 1 event per osc per window)
                    if osc_subsample and t_abs < last_emit_time.get(idx, -1e12) + osc_subsample:
                        continue

                    # Emit only on changes (per (osc, key))
                    to_emit = []
                    for k, v in vals.items():
                        key = (idx, k)
                        if (not osc_emit_on_change_only) or (last_vals.get(key) != v):
                            to_emit.append((k, v))
                            last_vals[key] = v

                    if not to_emit:
                        continue

                    for k, v in to_emit:
                        events.append(Event(
                            t_abs_s = t_abs,
                            time_str = line.split()[0],  # the HH:MM:SS.micro part
                            file = fname,
                            action = f"osc_{k}",
                            target = str(idx),
                            value = float(v),
                            message = raw
                        ))
                    last_emit_time[idx] = t_abs

    # 3) Sort by time and return

    events.sort(key=lambda e: e.t_abs_s)
    return events


def export_csv(events: List[Event], out_csv: Path) -> None:
    """
    Export a list of Event objects to a CSV file.
    Args:
        events (List[Event]): List of Event objects to export.
        out_csv (Path): Path to the output CSV file.
    Returns:
        None
    """

    if not events:
        raise RuntimeError("No events to export.")
    
    t0 = events[0].t_abs_s

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_str", "t_rel_s", "delta_prev_s", "file", "action", "target", "value", "message"])
        prev = None
        for ev in events:
            delta = 0.0 if prev is None else (ev.t_abs_s - prev)
            w.writerow([
                ev.time_str or "",
                f"{ev.t_abs_s - t0:.6f}",
                f"{delta:.6f}",
                ev.file,
                ev.action,
                ev.target or "",
                f"{ev.value}" if ev.value is not None else "",
                ev.message
            ])
            prev = ev.t_abs_s



def dedupe_timeline_csv(path_in: Path, path_out: Path = None, time_decimals: int = 3):
    """
    Remove near-duplicate events that are mirrored across multiple files.
    - Buckets time to `time_decimals` (e.g., 3 => milliseconds)
    - Prefers session.log over User*.log.
    Args:
        path_in (Path): Input CSV file path.
        path_out (Path, optional): Output CSV file path. If None, overwrites input.
        time_decimals (int): Number of decimal places to round time for bucketing.
    Returns:
        pd.DataFrame: The deduplicated DataFrame.
    """
    df = pd.read_csv(path_in)

    # normalize message text
    df["msg_norm"] = df["message"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    # bucket time (relative) to control near-equal timestamps
    if "t_rel_s" not in df.columns:
        raise ValueError("CSV missing 't_rel_s' column.")
    df["t_bucket"] = df["t_rel_s"].round(time_decimals)

    # source priority: prefer session.log, then User0, User1, ...
    src_priority = {"session.log": 0}
    # default any other file to rank 1 (or higher)
    df["src_rank"] = df["file"].map(src_priority).fillna(1)

    # sort so the one we KEEP comes first
    df = df.sort_values(
        by=["t_bucket", "action", "target", "value", "src_rank", "file", "t_rel_s"],
        ascending=[True, True, True, True, True, True, True]
    )

    # drop duplicates across the identifying fields
    df = df.drop_duplicates(
        subset=["t_bucket", "action", "target", "value", "msg_norm"],
        keep="first"
    ).copy()

    # recompute delta_prev_s using actual t_rel_s (not t_bucket)
    df = df.sort_values("t_rel_s").reset_index(drop=True)
    df["delta_prev_s"] = df["t_rel_s"].diff().fillna(0.0)

    # clean up helper cols
    df = df.drop(columns=["msg_norm", "t_bucket", "src_rank"])

    # write out
    if path_out is None:
        path_out = path_in  # overwrite
    df.to_csv(path_out, index=False)
    return df


def load_timeline(csv_path : Path) -> pd.DataFrame:
    """
    Load a timeline CSV and sort by relative time.
    Args:
        csv_path (Path): Path to the timeline CSV file.
    Returns:
        pd.DataFrame: DataFrame sorted by 't_rel_s' if present.
    """

    df = pd.read_csv(csv_path)

    if "t_rel_s" in df.columns:
        df = df.sort_values("t_rel_s").reset_index(drop = True)
        return df


def compute_control_intervals(df: pd.DataFrame):
    """
    Compute control intervals from control_start and control_end events.
    Args:
        df (pd.DataFrame): Timeline DataFrame.
    Returns:
        [(user, start, duration), ...]
    """

    intervals: List[Tuple[str, float, float]] = []
    current_user: Optional[str] = None
    start_t: Optional[float] = None

    # filter only control events
    d = df[df["action"].isin(["control_start", "control_end"])].sort_values("t_rel_s")

    for _, r in d.iterrows():
        t = float(r["t_rel_s"])
        a = r["action"]
        u = str(r["target"])
        if a == "control_start":
            if current_user is not None and start_t is not None and t > start_t:
                intervals.append((current_user, start_t, t - start_t))
            current_user, start_t = u, t
        elif a == "control_end":
            if current_user == u and start_t is not None and t > start_t:
                intervals.append((current_user, start_t, t - start_t))
                current_user, start_t = None, None

    if current_user is not None and start_t is not None:
        t_end = float(df["t_rel_s"].max())
        if t_end > start_t:
            intervals.append((current_user, start_t, t_end - start_t))
    return intervals


def plot_control_gantt(df: pd.DataFrame, png_path: Path, title_suffix=""):
    """
    Plot a Gantt chart of control intervals.
    Args:
        df (pd.DataFrame): Timeline DataFrame.
        png_path (Path): Path to save the PNG plot.
        title_suffix (str): Optional suffix for the plot title.
    Returns:
        None
    """

    intervals = compute_control_intervals(df)
    if not intervals:
        return

    # map users to y-rows (A,B,C,D)
    user_rows = {"A": 3, "B": 2, "C": 1, "D": 0}
    # collect segments per user for broken_barh
    per_user = {u: [] for u in user_rows}
    for u, start, duration in intervals:
        if u in per_user and duration > 0:
            per_user[u].append((start, duration))

    fig = plt.figure(figsize=(12, 3.5))
    ax = plt.gca()
    ax.set_title(f"Control Timeline {title_suffix}".strip())
    ax.set_xlabel("Time (s since session start)")
    ax.set_yticks(list(user_rows.values()))
    ax.set_yticklabels(list(user_rows.keys()))
    ax.grid(True, linestyle=":", alpha=0.5)

    # draw one broken_barh per user; no explicit colors (let matplotlib choose)
    for u, row in user_rows.items():
        if per_user[u]:
            ax.broken_barh(per_user[u], (row - 0.35, 0.7))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_event_counts(df: pd.DataFrame, png_path: Path, title_suffix=""):
    """
    Plot event counts by user.

    Args:
        df (pd.DataFrame): Timeline DataFrame.
        png_path (Path): Path to save the PNG plot.
        title_suffix (str): Optional suffix for the plot title.
    Returns:
        None
    """

    # count events by user (A/B/C/D)
    users = ["A", "B", "C", "D"]
    counts = {u: 0 for u in users}
    sub = df[df["target"].isin(users)]
    for u, c in sub["target"].value_counts().items():
        counts[u] = int(c)

    xs = list(counts.keys())
    ys = [counts[u] for u in xs]

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_title(f"Event Counts {title_suffix}".strip())
    ax.set_xlabel("User")
    ax.set_ylabel("Events")
    ax.bar(xs, ys)  # no explicit colors
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

# ============== Temporal features & plots ==============

USERS = ["A", "B", "C", "D"]
WINDOW_SIZE = 10.0  # seconds per window

def sliding_entropy(actions: List[str]) -> float:
    """Shannon entropy (bits) of action types in a window."""
    if len(actions) == 0:
        return 0.0
    counts = pd.Series(actions).value_counts()
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def compute_temporal_features(csv_path: Path, window_s: float = WINDOW_SIZE) -> pd.DataFrame:
    """
    Return dataframe with rolling control fractions and entropy per window.
    """
    df = pd.read_csv(csv_path).sort_values("t_rel_s").reset_index(drop=True)
    t_end = df["t_rel_s"].max()
    bins = np.arange(0, t_end + window_s, window_s)
    rows = []

    # derive control intervals
    control = defaultdict(list)
    cur_control = None
    cur_start = 0.0
    for _, r in df.iterrows():
        t = float(r["t_rel_s"])
        if r["action"] == "control_start":
            if cur_control is not None:
                control[cur_control].append((cur_start, t))
            cur_control = str(r["target"])
            cur_start = t
        elif r["action"] == "control_end" and str(r["target"]) == cur_control:
            control[cur_control].append((cur_start, t))
            cur_control = None
    if cur_control is not None:
        control[cur_control].append((cur_start, float(t_end)))

    # per-window summaries
    for i in range(len(bins) - 1):
        t0, t1 = float(bins[i]), float(bins[i + 1])
        window = df[(df["t_rel_s"] >= t0) & (df["t_rel_s"] < t1)]
        actions = window["action"].tolist()
        entropy = sliding_entropy(actions)

        row = {"t_mid": (t0 + t1) / 2, "entropy": entropy}
        for u in USERS:
            u_time = 0.0
            for (start, end) in control.get(u, []):
                overlap = max(0.0, min(t1, end) - max(t0, start))
                u_time += overlap
            row[f"{u}_control_frac"] = u_time / window_s
        rows.append(row)

    return pd.DataFrame(rows)


def plot_temporal_evolution(csv_path: Path, out_path: Path) -> None:
    """Plot control fractions (per user) + entropy over time."""
    dfw = compute_temporal_features(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    title = f"{csv_path.parent.parent.name}/{csv_path.parent.name}"
    ax1.set_title(f"Temporal Evolution — {title}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Control Fraction")

    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    for u in USERS:
        ax1.plot(dfw["t_mid"], dfw[f"{u}_control_frac"], label=f"{u} control", color=colors[u])
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(dfw["t_mid"], dfw["entropy"], color="black", lw=2, label="Action Entropy")
    ax2.set_ylabel("Entropy (bits)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============== Higher-level features ==============

def _shannon_entropy(p: np.ndarray) -> float:
    p = np.array(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _normalize_entropy(h: float, n: int) -> float:
    return float(h / math.log2(n)) if n > 1 and h > 0 else 0.0

def _safe_var(x: List[float]) -> float:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.var(x)) if len(x) >= 1 else 0.0

def _safe_mean(x: List[float]) -> float:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else 0.0

def _safe_std(x: List[float]) -> float:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.std(x)) if x else 0.0

def _entropy_from_counts(counts: List[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = [c / total for c in counts if c > 0]
    # avoid log(0)
    return -sum(pi * math.log(pi + 1e-12, 2) for pi in p)

def _burstiness(intervals: List[float]) -> float:
    """Goh & Barabási (sigma - mu)/(sigma + mu) ∈ [-1,1]."""
    if not intervals:
        return 0.0
    mu = float(np.mean(intervals))
    sigma = float(np.std(intervals))
    return 0.0 if (sigma + mu) == 0 else float((sigma - mu) / (sigma + mu))

def _id_to_letter_map(df: pd.DataFrame) -> dict:
    """Map numeric user ids to letters using 'joined' rows (value=id, target=letter)."""
    m = {}
    j = df[df["action"] == "joined"]
    for _, r in j.iterrows():
        val = r.get("value")
        if pd.notna(val):
            m[int(float(val))] = str(r["target"])
    return m

def _normalize_event_user(row: pd.Series, id2letter: dict) -> Optional[str]:
    """Normalize event target to A/B/C/D if possible."""
    t = str(row.get("target"))
    if t in USERS:
        return t
    m = re.match(r"user(\d+)$", t, flags=re.IGNORECASE)
    if m and int(m.group(1)) in id2letter:
        return id2letter[int(m.group(1))]
    f = str(row.get("file", ""))
    mf = re.match(r"User(\d+)\.log$", f)
    if mf and int(mf.group(1)) in id2letter:
        return id2letter[int(mf.group(1))]
    return None


def extract_session_features(csv_path: Path) -> dict:
    """Compute session-level features from timeline.csv."""
    df = load_timeline(csv_path)
    df = df.sort_values("t_rel_s").reset_index(drop=True)

    # basic time info
    t0 = float(df["t_rel_s"].min())
    tN = float(df["t_rel_s"].max())
    total_time = max(0.0, tN - t0)

    feats: Dict[str, float] = {
        "total_time_s": total_time,
        "num_events": int(len(df)),
    }

    # inter-event timing
    times = df["t_rel_s"].astype(float).tolist()
    gaps = [b - a for a, b in zip(times, times[1:])]
    feats["inter_event_mean_s"] = _safe_mean(gaps)
    feats["inter_event_std_s"] = _safe_std(gaps)
    feats["inter_event_burstiness"] = _burstiness(gaps)

    # per-user event counts/rates
    df_users = df[df["target"].isin(USERS)]
    cnt = df_users["target"].value_counts().to_dict()
    for u in USERS:
        c = int(cnt.get(u, 0))
        feats[f"{u}_event_count"] = c
        feats[f"{u}_event_rate_per_s"] = (c / total_time) if total_time > 0 else 0.0

    # control time fractions
    intervals = compute_control_intervals(df)
    control_time = {u: 0.0 for u in USERS}
    for u, start, dur in intervals:
        if u in control_time:
            control_time[u] += float(dur)
    for u in USERS:
        feats[f"{u}_control_time_s"] = control_time[u]
        feats[f"{u}_control_frac"] = (control_time[u] / total_time) if total_time > 0 else 0.0

    # inequality measures
    vals = np.array([control_time[u] for u in USERS], dtype=float)
    mean_val = vals.mean() if vals.size else 0.0
    feats["control_balance_index"] = float(np.std(vals)) if vals.sum() > 0 else 0.0
    feats["control_balance_normstd"] = float(np.std(vals) / mean_val) if mean_val > 0 else 0.0

    def gini(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0 or np.all(x == 0):
            return 0.0
        n = x.size
        sorted_x = np.sort(x)
        cumx = np.cumsum(sorted_x)
        return float((2.0 * np.sum((np.arange(1, n + 1) * sorted_x))) / (n * cumx[-1]) - (n + 1) / n)

    feats["control_gini"] = gini(vals)

    # action-type entropy + dominance
    action_counts = df["action"].value_counts().to_dict()
    feats["action_entropy_bits"] = _entropy_from_counts(list(action_counts.values()))
    top = max(action_counts.values()) if action_counts else 0
    feats["top_action_frac"] = (top / sum(action_counts.values())) if action_counts else 0.0

    # parameter-change features
    id2letter = _id_to_letter_map(df)
    param_mask = df["action"].isin(["set_frequency", "set_amplitude", "set_offset", "set_amplitude_osc", "set_frequency_all"])
    D = df[param_mask].copy()
    D["actor"] = D.apply(lambda r: _normalize_event_user(r, id2letter), axis=1)

    def per_user_stats(name: str, mask: pd.Series):
        sub = D[mask & D["actor"].isin(USERS)].copy()
        for u in USERS:
            feats[f"{u}_{name}_count"] = int((sub["actor"] == u).sum())
            ser = sub[sub["actor"] == u][["t_rel_s", "value"]].dropna().sort_values("t_rel_s")["value"].astype(float)
            deltas = ser.diff().abs().dropna().tolist()
            feats[f"{u}_{name}_mean_abs_delta"] = _safe_mean(deltas)
            feats[f"{u}_{name}_std_abs_delta"] = _safe_std(deltas)

    per_user_stats("freq", D["action"].isin(["set_frequency"]))
    per_user_stats("amp",  D["action"].isin(["set_amplitude", "set_amplitude_osc"]))
    per_user_stats("off",  D["action"].isin(["set_offset"]))

    g = D[D["action"] == "set_frequency_all"]
    feats["global_freq_change_count"] = int(len(g))
    feats["global_freq_mean"] = _safe_mean(g["value"].dropna().astype(float).tolist())

    # reaction time: control_start -> first change by same user
    changes = D[D["actor"].isin(USERS)].sort_values("t_rel_s")
    starts = df[df["action"] == "control_start"].sort_values("t_rel_s")
    rts = []
    for _, srow in starts.iterrows():
        u = srow["target"]
        t = float(srow["t_rel_s"])
        nxt = changes[(changes["actor"] == u) & (changes["t_rel_s"] > t)]
        if len(nxt):
            rts.append(float(nxt.iloc[0]["t_rel_s"]) - t)
    feats["reaction_time_mean_s"] = _safe_mean(rts)
    feats["reaction_time_std_s"] = _safe_std(rts)

    # transition counts/probabilities
    seq = df[df["target"].isin(USERS)][["t_rel_s", "target"]].sort_values("t_rel_s")["target"].tolist()
    trans = {(a, b): 0 for a in USERS for b in USERS}
    for a, b in zip(seq, seq[1:]):
        trans[(a, b)] += 1
    for a in USERS:
        row_total = sum(trans[(a, b)] for b in USERS)
        for b in USERS:
            feats[f"turn_{a}_to_{b}_count"] = trans[(a, b)]
            feats[f"turn_{a}_to_{b}_prob"] = (trans[(a, b)] / row_total) if row_total > 0 else 0.0

    # entropies
    total_next_counts = [sum(trans[(a, b)] for a in USERS) for b in USERS]
    feats["transition_next_entropy_bits"] = _entropy_from_counts(total_next_counts)
    cond_H = 0.0
    total_all = sum(sum(trans[(x, b)] for b in USERS) for x in USERS)
    for a in USERS:
        row_total = sum(trans[(a, b)] for b in USERS)
        p_c = row_total / total_all if total_all > 0 else 0.0
        if row_total > 0:
            H_next_given_a = _entropy_from_counts([trans[(a, b)] for b in USERS])
            cond_H += p_c * H_next_given_a
    feats["transition_conditional_entropy_bits"] = float(cond_H)

    # oscillator stability (if present)
    osc_df = df[df["action"].str.startswith("osc_", na=False)].copy()
    osc_df["value"] = pd.to_numeric(osc_df["value"], errors="coerce")
    for idx in sorted(osc_df["target"].dropna().unique(), key=lambda x: str(x)):
        sub = osc_df[osc_df["target"] == idx].sort_values("t_rel_s")
        vals = sub["value"].dropna().astype(float).tolist()
        feats[f"osc_{idx}_n_changes"] = int(len(vals))
        feats[f"osc_{idx}_n_unique"] = int(len(set(vals)))
        feats[f"osc_{idx}_var"] = float(np.var(vals)) if vals else 0.0
        deltas = np.abs(np.diff(vals)) if len(vals) >= 2 else np.array([])
        feats[f"osc_{idx}_median_abs_delta"] = float(np.median(deltas)) if deltas.size else 0.0

    # temporal features (autocorr + mean entropy)
    try:
        tf = compute_temporal_features(csv_path, window_s=10.0)
        for u in USERS:
            if f"{u}_control_frac" in tf.columns and len(tf) >= 3:
                s = tf[f"{u}_control_frac"].astype(float)
                ac = float(s.autocorr(lag=1)) if s.autocorr(lag=1) is not None else 0.0
                feats[f"{u}_control_autocorr_lag1"] = ac
            else:
                feats[f"{u}_control_autocorr_lag1"] = 0.0
        feats["temporal_mean_entropy_bits"] = float(tf["entropy"].mean()) if "entropy" in tf.columns else 0.0
    except Exception:
        for u in USERS:
            feats[f"{u}_control_autocorr_lag1"] = 0.0
        feats["temporal_mean_entropy_bits"] = 0.0

    # conflict time: overlapping control
    evts = []
    for _, start, dur in intervals:
        evts.append((start, +1))
        evts.append((start + dur, -1))
    evts.sort()
    active = 0
    last_t = None
    total_conflict_time = 0.0
    max_concurrent = 0
    for t, delta in evts:
        if last_t is not None and t > last_t and active > 1:
            total_conflict_time += (t - last_t)
        active += delta
        max_concurrent = max(max_concurrent, active)
        last_t = t
    feats["total_conflict_time_s"] = float(total_conflict_time)
    feats["conflict_fraction"] = (total_conflict_time / total_time) if total_time > 0 else 0.0
    feats["max_concurrent_controllers"] = int(max_concurrent)

    return feats


# ============== Pretty timeline with annotations ==============

def id_to_letter_map(df: pd.DataFrame) -> dict:
    """Public version of id map (wrapper)."""
    return _id_to_letter_map(df)


def plot_session_timeline_with_changes(csv_path: Path, png_path: Path) -> None:
    """
    Timeline with control bars + parameter-change annotations.
    """
    df = load_timeline(csv_path)
    id2letter = id_to_letter_map(df)

    user_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
    user_rows = {"A": 3, "B": 2, "C": 1, "D": 0}

    fig = plt.figure(figsize=(13, 5))
    ax = plt.gca()
    ax.set_title(f"Timeline — {csv_path.parent.parent.name}/{csv_path.parent.name}")
    ax.set_xlabel("Time (s since session start)")
    ax.set_yticks(list(user_rows.values()))
    ax.set_yticklabels(list(user_rows.keys()))
    ax.grid(True, linestyle=":", alpha=0.5)

    xmin, xmax = df["t_rel_s"].min(), df["t_rel_s"].max()

    # background guides
    for u in user_rows:
        ax.plot([xmin, xmax], [user_rows[u], user_rows[u]], "--", color="#cccccc", lw=0.7)

    # shaded control intervals
    for u, start, dur in compute_control_intervals(df):
        if u in user_rows:
            ax.broken_barh(
                [(start, dur)],
                (user_rows[u] - 0.35, 0.70),
                facecolors=user_colors.get(u, "#888888"),
                alpha=0.18
            )

    # change events
    def normalize_event_user(row):
        t = str(row.get("target"))
        if t in user_rows:
            return t
        m = re.match(r"user(\d+)$", t or "", flags=re.IGNORECASE)
        if m and int(m.group(1)) in id2letter:
            return id2letter[int(m.group(1))]
        f = str(row.get("file", ""))
        mf = re.match(r"User(\d+)\.log$", f)
        if mf and int(mf.group(1)) in id2letter:
            return id2letter[int(mf.group(1))]
        return None

    change_mask = df["action"].isin([
        "set_frequency", "set_frequency_all", "set_amplitude", "set_amplitude_osc", "set_offset"
    ])
    changes = df[change_mask].copy()

    def label_for(r):
        a = r["action"]; v = r["value"]
        if a == "set_frequency":      return f"f{v:g}"
        if a == "set_frequency_all":  return f"ALL freq={v:g}"
        if a == "set_amplitude":      return f"a{v:g}"
        if a == "set_amplitude_osc":  return f"amp[{r.get('target','?')}]={v:g}"
        if a == "set_offset":         return f"o{v:g}"
        return a

    for _, r in changes.iterrows():
        u = normalize_event_user(r)
        if u not in user_rows:
            continue
        y = user_rows[u]
        x = float(r["t_rel_s"])
        c = user_colors.get(u, "#666666")
        ax.scatter(x, y, s=45, color=c, edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(
            label_for(r),
            xy=(x, y),
            xytext=(4, 10), textcoords="offset points",
            fontsize=9, color=c,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=c, alpha=0.7)
        )

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)


# ============== Batch runner (day folder) ==============

if __name__ == "__main__":
    # Example paths — adjust these as needed
    # Single session preview
    one_csv = Path("20251003_141557/Sessions/20251003/Suzanne/142002/timeline.csv")
    if one_csv.exists():
        plot_session_timeline_with_changes(one_csv, one_csv.with_name("timeline_changes.png"))

    # All sessions in a day
    root_dir = Path("20251003_141557/Sessions/20251003")
    combined: List[dict] = []
    force = True  # overwrite timelines if re-building

    if root_dir.exists():
        for participant_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(s for s in participant_dir.iterdir() if s.is_dir()):
                csv_path = session_dir / "timeline.csv"

                # (Optional) rebuild timeline from logs — set force=True to regenerate
                out_csv = csv_path
                if (not out_csv.exists()) or force:
                    try:
                        print(f"🧭 Building: {participant_dir.name}/{session_dir.name}")
                        events = build_timeline(
                            log_dir=session_dir,
                            include_oscillators=True,
                            osc_subsample=1.0,
                            osc_emit_on_change_only=True
                        )
                        export_csv(events, out_csv)
                        print(f"✅ Saved {out_csv} ({len(events)} events)")
                    except Exception as e:
                        print(f"⚠️ Error in {participant_dir.name}/{session_dir.name}: {e}")
                        continue

                if not csv_path.exists():
                    print(f"⏭️  No timeline.csv in {participant_dir.name}/{session_dir.name} — skipping")
                    continue

                # dedupe
                dedupe_timeline_csv(csv_path, csv_path, time_decimals=3)
                print(f"✅ Deduped {participant_dir.name}/{session_dir.name}")

                # features
                feats = extract_session_features(csv_path)
                feats["participant"] = participant_dir.name
                feats["session"] = session_dir.name
                pd.DataFrame([feats]).to_csv(session_dir / "features.csv", index=False)
                combined.append(feats)
                print(f"✅ {participant_dir.name}/{session_dir.name}: features.csv written")

                # plots
                plot_temporal_evolution(csv_path, session_dir / "temporal_evolution.png")
                df = load_timeline(csv_path)
                tag = f"— {participant_dir.name}/{session_dir.name}"
                plot_control_gantt(df, session_dir / "timeline_control.png", title_suffix=tag)
                plot_event_counts(df, session_dir / "timeline_event_counts.png", title_suffix=tag)

        # combined features
        if combined:
            out = pd.DataFrame(combined)
            out.to_csv("features_all.csv", index=False)
            print(f"🧾 Combined features -> features_all.csv ({len(out)} sessions)")