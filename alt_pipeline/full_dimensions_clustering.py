import json
import numpy as np
import json_extraction  # your module
from sklearn.preprocessing import StandardScaler
import dim_red_clustering_functions

files = [
    "data/111455_features.json",
    "data/114654_features.json",
    "data/133150_features.json",
    "data/140252_features.json"
]
# --- Step 1: discover common complete features across all files ---
feature_sets = []

for f in files:
    with open(f, "r") as fh:
        data = json.load(fh)
    feature_sets.append(
        set(json_extraction.find_complete_feature_paths(data))
    )

common_feature_paths = sorted(set.intersection(*feature_sets))
print(f"Number of shared features: {len(common_feature_paths)}")

# --- Step 2: generate readable labels ---
feature_labels = [json_extraction.path_to_feature_label(p) for p in common_feature_paths]

# --- Step 3: extract datapoints using shared features ---
all_datapoints = []
file_labels = []

for i, f in enumerate(files):
    dps = json_extraction.extract_datapoints_except_last(
        f,
        common_feature_paths,
        feature_labels=feature_labels
    )
    all_datapoints.extend(dps)
    file_labels.extend([f"experiment_{i+1}"] * len(dps))

print(f"Total datapoints: {len(all_datapoints)}")

# --- Step 4: convert to NumPy array ---
X = np.array([dp.dimension_values for dp in all_datapoints])

print("Feature labels:")
for lbl in feature_labels:
    print(" -", lbl)

print(f"X shape: {X.shape}")

# X: full-dimensional feature matrix
X_scaled = StandardScaler().fit_transform(X)

(
    silhouette_scores,
    best_score,
    best_k,
    cluster_labels,
    u,
    cntr,
    fpc
) = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
    X_scaled,
    k_range=range(2, 11),
    m=2.0,
    random_state=42
)
print(best_k)
print(best_score)
print(silhouette_scores)

