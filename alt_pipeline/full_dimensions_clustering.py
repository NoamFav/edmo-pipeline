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
feature_labels, datapoints = json_extraction.full_extraction(files)
X = np.array([dp.dimension_values for dp in datapoints])
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
    k_range=range(2, 15),
    m=2.0,
    random_state=42
)
print(best_k)
print(best_score)
print(silhouette_scores)

