import numpy as np
import plotting_pca_clustering
import dim_red_clustering_functions
import json_extraction

""""
# set parameters
selected_features = [
    "audio_features.nonverbal.basic_metrics.conversation.overlap_duration",
    "audio_features.nonverbal.basic_metrics.conversation.num_speakers",
    "audio_features.nlp.sentiment.score",
    "audio_features.nonverbal.basic_metrics.conversation.total_speaking_time",
    "audio_features.emotion.emotions[0].score",
    "audio_features.emotion.emotions[4].score",
    "robot_speed_features.avg_speed_cm_s"
]
"""

files = [
    "data/111455_features.json",
    "data/114654_features.json",
    "data/133150_features.json",
    "data/140252_features.json"
]

feature_labels, datapoints = json_extraction.full_extraction(files)
X = np.array([dp.dimension_values for dp in datapoints])
X_scaled = StandardScaler().fit_transform(X)
print(feature_labels)
print([l for l in datapoints[0].dimension_labels if "window" in l])

for i, f in enumerate(files):
    dps = json_extraction.extract_datapoints_except_last(f, selected_features, features_labels)
    all_datapoints.extend(dps)
    file_labels.extend([f"experiment_{i+1}"] * len(dps))  # same label for all windows of this file
print(f"Total datapoints: {len(all_datapoints)}")
X = np.array([dp.dimension_values for dp in all_datapoints])

# --- Create PCA object with 2 components ---
dim_red = dim_red_clustering_functions.create_dim_red_method("PCA", n_dimensions=2)
# --- Fit PCA ---
X_reduced = dim_red.fit(X_scaled)
# --- Apply fuzzy C-means ---
n_clusters = 3
silhouette_scores_pca ,best_score_pca, best_k_pca, cluster_labels, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(X_reduced)
print("silhette score: " + str(best_score_pca))
print("best k: " + str(best_k_pca))
print("all scores pca: " + str(silhouette_scores_pca))

# here we do pls
X = np.array([dp.dimension_values[:-1] for dp in datapoints])
# Y: last feature
Y = np.array([dp.dimension_values[-1] for dp in datapoints]).reshape(-1, 1)  # column vector
# --- PLS ---
dim_red_pls = dim_red_clustering_functions.PLS(n_dimensions=2)
X_pls = dim_red_pls.fit(X, Y)  # X_pls is like PCA-reduced scores
# Fuzzy C-mean clustering on PLS
#cluster_labels_pls, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans(X_pls, n_clusters=3)
silhouette_scores_pls ,best_score_pls, best_k_pls, cluster_labels_pls, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(X_pls)

#generate plots
plotting_pca_clustering.plot_clusters(X_pls, cluster_labels_pls, output_dir="output/pls_plots")
plotting_pca_clustering.plot_clusters(X_reduced, cluster_labels)
plotting_pca_clustering.plot_pca_results(X_reduced, dim_red, feature_labels, files)
features_labels_pls = feature_labels[:-1]
plotting_pca_clustering.plot_pls_results(
    X_scores=X_pls,
    Y= Y,
    dim_red=dim_red_pls,
    features_labels=features_labels_pls,
    file_labels=files,   # pass as keyword
    output_dir="output/pls_plots"
)
