import json

time_features = [
    "robot_speed_features_window_start",
    "robot_speed_features_window_index",
    "robot_speed_features_window_end",
    "audio_features_window_start",
    "audio_features_window_end",
    "audio_features_window_index",
]

class Datapoint:
    dimension_labels: list[str]
    dimension_values: list[float]

    def __init__(self, labels, values):
        self.dimension_labels = labels
        self.dimension_values = values

def get_by_path(obj, path):
    try:
        parts = path.replace("]", "").split(".")
        for p in parts:
            if "[" in p:
                key, idx = p.split("[")
                obj = obj[key][int(idx)]
            else:
                obj = obj[p]
        return obj
    except (KeyError, IndexError, TypeError):
        return None  # return None instead of raising

def extract_datapoints_except_last(filename, feature_paths, feature_labels=None):
    # load the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)
    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # Intersect window indices and remove last window
    common_indices = sorted(set(audio_windows.keys()) & set(robot_windows.keys()))
    if common_indices:
        common_indices = common_indices[:-1]  # remove last window

    datapoints = []

    for idx in common_indices:
        window_data = {
            "audio_features": audio_windows[idx],
            "robot_speed_features": robot_windows[idx]
        }

        values = []
        all_features_present = True

        for path in feature_paths:
            val = get_by_path(window_data, path)
            if isinstance(val, (int, float)):
                values.append(val)
            else:
                all_features_present = False
                break  # skip this window entirely

        if all_features_present:
            labels_to_use = feature_labels if feature_labels is not None else feature_paths
            datapoints.append(Datapoint(labels_to_use, values))

    return datapoints

def collect_numeric_paths(obj, prefix=""):
    paths = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.extend(collect_numeric_paths(v, new_prefix))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            paths.extend(collect_numeric_paths(v, new_prefix))

    elif isinstance(obj, (int, float)):
        paths.append(prefix)

    return paths

def find_complete_feature_paths(data):
    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # intersect indices and ignore last window
    common_indices = sorted(set(audio_windows) & set(robot_windows))
    if common_indices:
        common_indices = common_indices[:-1]

    if not common_indices:
        return []

    # collect candidate paths from first valid window
    first_idx = common_indices[0]
    first_window = {
        "audio_features": audio_windows[first_idx],
        "robot_speed_features": robot_windows[first_idx],
    }

    candidate_paths = collect_numeric_paths(first_window)
    valid_paths = []

    for path in candidate_paths:
        valid = True
        for idx in common_indices:
            window_data = {
                "audio_features": audio_windows[idx],
                "robot_speed_features": robot_windows[idx],
            }
            val = get_by_path(window_data, path)
            if not isinstance(val, (int, float)):
                valid = False
                break
        if valid:
            valid_paths.append(path)

    return valid_paths

def extract_all_complete_features_except_last(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    feature_paths = find_complete_feature_paths(data)

    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # intersect indices and ignore last window
    common_indices = sorted(set(audio_windows) & set(robot_windows))
    if common_indices:
        common_indices = common_indices[:-1]

    datapoints = []

    for idx in common_indices:
        window_data = {
            "audio_features": audio_windows[idx],
            "robot_speed_features": robot_windows[idx],
        }

        values = [get_by_path(window_data, p) for p in feature_paths]
        datapoints.append(Datapoint(feature_paths, values))

    return datapoints

def path_to_feature_label(path: str) -> str:
    """
    Convert a JSON path like:
    audio_features.sentiment.scores[0]
    → sentiment_scores_0
    """
    label = path.replace(".", "_").replace("[", "_").replace("]", "")
    return label

def remove_features_from_datapoints(datapoints, labels_to_remove):
    """
    Removes dimensions whose labels are in labels_to_remove
    from each Datapoint object (in-place safe via reconstruction).
    """
    cleaned_datapoints = []

    for dp in datapoints:
        keep_indices = [
            i for i, label in enumerate(dp.dimension_labels)
            if label not in labels_to_remove
        ]

        new_labels = [dp.dimension_labels[i] for i in keep_indices]
        new_values = [dp.dimension_values[i] for i in keep_indices]

        cleaned_datapoints.append(
            Datapoint(new_labels, new_values)
        )

    return cleaned_datapoints

def full_extraction(files, ignore_time_features=True):
    # --- Step 1: discover common complete features across all files ---
    feature_sets = []

    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
        feature_sets.append(
            set(find_complete_feature_paths(data))
        )

    common_feature_paths = sorted(set.intersection(*feature_sets))

    # --- Step 2: generate readable labels ---
    feature_labels = [path_to_feature_label(p) for p in common_feature_paths]

    # --- Step 3: extract datapoints using shared features ---
    all_datapoints = []
    file_labels = []

    for i, f in enumerate(files):
        dps = extract_datapoints_except_last(
            f,
            common_feature_paths,
            feature_labels=feature_labels
        )
        all_datapoints.extend(dps)
        file_labels.extend([f"experiment_{i + 1}"] * len(dps))

    # --- Step 4: optionally remove time features AFTER extraction ---
    if ignore_time_features:
        all_datapoints = remove_features_from_datapoints(
            all_datapoints,
            time_features
        )

        # update feature_labels to match datapoints
        feature_labels = all_datapoints[0].dimension_labels if all_datapoints else []
    return feature_labels, all_datapoints
