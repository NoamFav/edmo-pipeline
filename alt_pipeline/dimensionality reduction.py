import json
import numpy as np
import sklearn.decomposition as skd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
from fcmeans import FCM
from sklearn.cross_decomposition import PLSRegression
import plotting_pca_clustering

selected_features = [
    "audio_features.nonverbal.basic_metrics.conversation.overlap_duration",
    "audio_features.nonverbal.basic_metrics.conversation.num_speakers",
    "audio_features.nlp.sentiment.score",
    "audio_features.nonverbal.basic_metrics.conversation.total_speaking_time",
    "audio_features.emotion.emotions[0].score",
    "audio_features.emotion.emotions[4].score",
    "robot_speed_features.avg_speed_cm_s"
]

features_labels = [
    "overlap_duration",
    "number_of_speakers",
    "sentiment_score",
    "total_speaking_time",
    "neutral_emotion_score",
    "anger_emotion_score",
    "robot_speed"
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

class DimensionalityReductionMethod(ABC):
    # abstract class for the dim red method choice
    n_dimensions: int

    @abstractmethod
    def __init__(self, n_dimensions: int):
        self.n_dimensions = n_dimensions  # required for all shapes

    @abstractmethod
    def fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def dimension_explained_variance(self):
        pass

    @abstractmethod
    def total_explained_variance(self):
        pass

    @abstractmethod
    def components(self):
        pass

class PCA(DimensionalityReductionMethod):
    pca: skd.PCA

    def __init__(self, n_dimensions: int):
        self.pca = skd.PCA(n_components=n_dimensions)

    def fit(self, data: np.ndarray):
        x_reduced = self.pca.fit_transform(data)
        return x_reduced

    def dimension_explained_variance(self):
        return self.pca.explained_variance_ratio_

    def total_explained_variance(self):
        explained = float(np.sum(self.pca.explained_variance_ratio_))
        return explained

    def components(self):
        return self.pca.components_

class SPCA(DimensionalityReductionMethod):
    spca: skd.SparsePCA
    original_data: np.ndarray
    transformed_data: np.ndarray

    def __init__(self, n_dimensions: int):
        self.spca = skd.SparsePCA(n_components=n_dimensions)
        return

    def fit(self, data: np.ndarray):
        self.original_data = data
        x_reduced = self.spca.fit_transform(data)
        self.transformed_data = x_reduced
        return x_reduced

    def dimension_explained_variance(self):
        component_var = []
        for i in range(self.spca.n_components):
            # Project X onto the i-th component
            x_i = np.dot(
                self.transformed_data[
                    :,
                    i : i + 1,
                ],
                self.spca.components_[
                    i : i + 1,
                    :,
                ],
            )

            # Compute how much total variance this component explains
            var_i = np.var(x_i, axis=0).sum()
            component_var.append(var_i)
        component_var = np.array(component_var)
        component_var = np.array(component_var)
        total_var = np.var(self.original_data, axis=0).sum()
        explained_var_ratio = component_var / total_var
        return explained_var_ratio

    def total_explained_variance(self):
        # Reconstruct the data
        x_reconstructed = np.dot(self.transformed_data, self.spca.components_)

        # Compute total variance in original data
        original_var = np.var(self.original_data, axis=0).sum()

        # Compute variance of reconstructed data
        recon_var = np.var(x_reconstructed, axis=0).sum()

        explained_variance_ratio = recon_var / original_var
        print(
            "Approximate total explained variance ratio:",
            explained_variance_ratio,
        )
        return explained_variance_ratio

    def components(self):
        return self.spca.components_

class PLS(DimensionalityReductionMethod):
    pls: PLSRegression
    X_original: np.ndarray
    Y_original: np.ndarray
    X_scores: np.ndarray  # equivalent to PCA-reduced data

    def __init__(self, n_dimensions: int):
        """
        n_dimensions here corresponds to the number of PLS components.
        """
        self.y_loadings_ = None
        self.x_loadings_ = None
        self.pls = PLSRegression(n_components=n_dimensions)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit PLS model.

        Parameters:
        -----------
        X : np.ndarray
            Predictor variables (samples x features)
        Y : np.ndarray
            Target variables (samples x outputs, e.g., robot speed features)

        Returns:
        --------
        X_scores : np.ndarray
            The PLS scores (latent components) of X, shape: (samples x n_components)
        """
        X_scores, Y_scores = self.pls.fit_transform(X, Y)
        self.X_original = X
        self.Y_original = Y
        self.X_scores = X_scores  # PCA-like reduced scores
        self.x_loadings_ = self.pls.x_loadings_  # store the actual loadings (n_features x n_components)
        self.y_loadings_ = self.pls.y_loadings_  # latent representation of X
        return self.X_scores

    def dimension_explained_variance(self):
        """
        Returns the proportion of variance in Y explained by each PLS component.
        """
        # Variance explained in Y by each component
        y_var = np.var(self.pls.y_scores_ @ self.pls.y_loadings_.T, axis=0).sum()
        total_var = np.var(self.Y_original, axis=0).sum()
        explained_ratio = np.array([y_var / total_var] * self.pls.n_components)
        return explained_ratio

    def total_explained_variance(self):
        """
        Returns the total variance in Y explained by the model.
        """
        y_pred = self.pls.predict(self.X_original)
        recon_var = np.var(y_pred, axis=0).sum()
        total_var = np.var(self.Y_original, axis=0).sum()
        return recon_var / total_var

    def components(self):
        """
        Returns the PLS X loadings (equivalent to PCA components).
        """
        return self.pls.x_loadings_

def create_dim_red_method(
    kind: str, n_dimensions: int = 2
) -> DimensionalityReductionMethod:
    # Factory for the dim red method choice, can return PCA or SPCA
    DIMENSIONALITY_REDUCTION_CLASSES = {
        "PCA": PCA,
        "SparsePCA": SPCA,
    }
    try:
        return DIMENSIONALITY_REDUCTION_CLASSES[kind](n_dimensions)
    except KeyError:
        raise ValueError(f"Unknown Dimensionality Reduction Method: {kind}")

def datapoints_to_matrix(datapoints: list[Datapoint]):
    return np.array([dp.dimension_values for dp in datapoints])

def plot_pca_results(
    X_reduced,
    dim_red,
    features_labels,
    file_labels=None,  # <--- new optional argument
    output_dir="output/pca_plots"
):
    """
    Plots PCA results: scatter plot of reduced data, component loadings,
    and explained variance. Optionally colors scatter plot by file/source.

    Parameters:
    -----------
    X_reduced : np.ndarray
        PCA-reduced data (samples x components)
    dim_red : object
        PCA object with methods:
            - components(): returns PCA component loadings
            - dimension_explained_variance(): returns explained variance ratio
    features_labels : list
        List of feature names corresponding to original data
    file_labels : list, optional
        List of strings same length as X_reduced indicating origin of each datapoint
    output_dir : str
        Directory to save plots
    """

    os.makedirs(output_dir, exist_ok=True)

    # === 1. Scatter Plot of PCA-Reduced Data ===
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (PC1 vs PC2)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter.png"), dpi=300)
    plt.close()

    # === 1. Scatter Plot of PCA-Reduced Data ===
    plt.figure(figsize=(8, 6))
    if file_labels is not None:
        unique_labels = list(sorted(set(file_labels)))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            idxs = [j for j, f in enumerate(file_labels) if f == lbl]
            plt.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], label=lbl, color=cmap(i))
        plt.legend()
        plt.title("PCA Projection (PC1 vs PC2) by File")
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
        plt.title("PCA Projection (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_scatter_file_colored.png"), dpi=300)
    plt.close()

    # === 2. PCA Component Loadings ===
    components = dim_red.components()
    for i, comp in enumerate(components):
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(comp)), comp)
        plt.xticks(range(len(comp)), features_labels, rotation=45, ha="right")
        plt.title(f"PCA Component {i + 1} Loadings")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"component_{i + 1}_loadings.png"), dpi=300)
        plt.close()

    # === 3. Explained Variance per Component ===
    explained = dim_red.dimension_explained_variance()
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained) + 1), explained)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance Per Component")
    plt.xticks(range(1, len(explained) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "explained_variance_bar.png"), dpi=300)
    plt.close()

    print(f"PCA plots saved in {output_dir}")

def plot_pls_results(X_scores, dim_red, features_labels, Y=None, file_labels=None, output_dir="output/pls_plots"):
    """
    Plots PLS results: scatter of X_scores, component loadings, explained variance,
    and optionally colors scatter by Y values.

    Parameters
    ----------
    X_scores : np.ndarray
        PLS-reduced scores (samples x n_components)
    dim_red : PLS wrapper
        Your PLS object after fitting (must have X_scores, x_loadings_ attributes)
    features_labels : list[str]
        Labels of original X features
    Y : np.ndarray or None
        Optional target variable for coloring scatter points
    file_labels : list[str] or None
        Labels for coloring points by file or experiment
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    n_components = X_scores.shape[1]

    # --- 1. Scatter of PLS scores (PC1 vs PC2) ---
    plt.figure(figsize=(8, 6))
    if file_labels is not None:
        unique_labels = sorted(set(file_labels))
        cmap = plt.get_cmap("tab10", len(unique_labels))
        for i, ul in enumerate(unique_labels):
            idxs = [j for j, f in enumerate(file_labels) if f == ul]
            plt.scatter(X_scores[idxs, 0], X_scores[idxs, 1], label=ul, color=cmap(i))
        plt.legend()
    else:
        plt.scatter(X_scores[:, 0], X_scores[:, 1])
    plt.xlabel("PLS1")
    plt.ylabel("PLS2")
    plt.title("PLS Scores (X_scores)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pls_scores.png"), dpi=300)
    plt.close()

    # --- 2. Loadings ---
    loadings = dim_red.x_loadings_  # n_features x n_components
    for i in range(n_components):
        plt.figure(figsize=(10, 5))
        plt.bar(range(loadings.shape[0]), loadings[:, i])
        plt.xticks(range(loadings.shape[0]), features_labels, rotation=45, ha="right")
        plt.title(f"PLS Component {i+1} Loadings")
        plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pls_component_{i+1}_loadings.png"), dpi=300)
        plt.close()

    # --- 3. Explained variance of components (approximate) ---
    explained = dim_red.dimension_explained_variance()
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), explained)
    plt.xlabel("PLS Component")
    plt.ylabel("Explained Variance Ratio (approx.)")
    plt.title("PLS Explained Variance")
    plt.xticks(range(1, n_components + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pls_explained_variance.png"), dpi=300)
    plt.close()

    print(f"PLS plots saved in {output_dir}")

    # --- 4. Scatter colored by Y values only ---
    if Y is not None:
        Y = np.array(Y, dtype=float).flatten()  # ensure numeric
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_scores[:, 0], X_scores[:, 1], c=Y, cmap="viridis", s=50)
        plt.colorbar(scatter, label="Y Value")
        plt.xlabel("PLS Component 1")
        plt.ylabel("PLS Component 2")
        plt.title("PLS Scatter Plot Colored by Y")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pls_scatter_colored_by_y.png"), dpi=300)
        plt.close()
        print(f"PLS scatter plot (colored by Y) saved to {output_dir}")
    else:
        print("No numeric Y provided, skipping PLS scatter colored by Y.")

# --- Fuzzy C-Means Clustering on PCA-reduced data ---
def perform_fuzzy_cmeans(X_reduced, n_clusters=3, m=2.0, max_iter=1000, error=0.005, random_state=None):
    """
    Perform fuzzy C-means clustering on PCA-reduced data using fcmeans.FCM.

    Parameters:
    -----------
    X_reduced : np.ndarray
        Data after PCA (n_samples x n_features)
    n_clusters : int
        Number of clusters
    m : float
        Fuzziness parameter (>1)
    max_iter : int
        Maximum number of iterations
    error : float
        Convergence tolerance
    random_state : int or None
        Random seed for reproducibility

    Returns:
    --------
    cluster_labels : np.ndarray
        Hard cluster assignments for each sample
    u : np.ndarray
        Fuzzy membership matrix (n_samples x n_clusters)
    cntr : np.ndarray
        Cluster centers (n_clusters x n_features)
    fpc : float
        Fuzzy partition coefficient (quality of clustering)
    """
    fcm = FCM(n_clusters=n_clusters, m=m, max_iter=max_iter, error=error, random_state=random_state)

    # Fit FCM to data
    fcm.fit(X_reduced)

    # Hard cluster assignments
    cluster_labels = fcm.predict(X_reduced)  # shape: (n_samples,)

    # Fuzzy membership matrix
    u = fcm.u  # shape: (n_samples, n_clusters)

    # Cluster centers
    cntr = fcm.centers  # shape: (n_clusters, n_features)

    # Fuzzy partition coefficient (sum of squared memberships)
    fpc = np.sum(u ** 2) / X_reduced.shape[0]

    return cluster_labels, u, cntr, fpc

def plot_clusters(X_reduced, cluster_labels, output_dir="output/pca_plots", title="Scatter with Clusters"):
    """
    Creates a scatter plot of reduced data colored by cluster membership.

    Parameters:
    -----------
    X_reduced : np.ndarray
        PCA-reduced data (n_samples x 2)
    cluster_labels : np.ndarray or list
        Cluster assignment for each sample (hard labels)
    output_dir : str
        Directory to save the plot
    title : str
        Title of the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_clusters = np.unique(cluster_labels)
    cmap = plt.colormaps["tab10"]  # categorical colormap

    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(unique_clusters):
        idxs = np.where(cluster_labels == cluster)[0]
        plt.scatter(X_reduced[idxs, 0], X_reduced[idxs, 1], label=f"Cluster {cluster+1}", color=cmap(i))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_clusters.png"), dpi=300)
    plt.close()

    print(f"scatter plot with clusters saved in {output_dir}")

# --- Extract datapoints with all features present ---
files = [
    "data/111455_features.json",
    "data/114654_features.json",
    "data/133150_features.json",
    "data/140252_features.json"
]

all_datapoints = []
file_labels = []  # for coloring

for i, f in enumerate(files):
    dps = extract_datapoints_except_last(f, selected_features, features_labels)
    all_datapoints.extend(dps)
    file_labels.extend([f"experiment_{i+1}"] * len(dps))  # same label for all windows of this file
print(f"Total datapoints: {len(all_datapoints)}")

X = np.array([dp.dimension_values for dp in all_datapoints])

# --- Create PCA object with 2 components ---
dim_red = create_dim_red_method("PCA", n_dimensions=2)
# --- Fit PCA ---
X_reduced = dim_red.fit(X)
# --- Plot PCA results ---
plot_pca_results(X_reduced, dim_red, features_labels, file_labels)
# --- Apply fuzzy C-means ---
n_clusters = 3
cluster_labels, u, cntr, fpc = perform_fuzzy_cmeans(X_reduced, n_clusters=n_clusters)
plot_clusters(X_reduced, cluster_labels)

# here we do pls
X = np.array([dp.dimension_values[:-1] for dp in all_datapoints])
# Y: last feature
Y = np.array([dp.dimension_values[-1] for dp in all_datapoints]).reshape(-1, 1)  # column vector
# --- PLS ---
dim_red_pls = PLS(n_dimensions=2)
X_pls = dim_red_pls.fit(X, Y)  # X_pls is like PCA-reduced scores

# Plot PLS results
features_labels_pls = features_labels[:-1]
plot_pls_results(
    X_scores=X_pls,
    Y= Y,
    dim_red=dim_red_pls,
    features_labels=features_labels_pls,
    file_labels=file_labels,   # pass as keyword
    output_dir="output/pls_plots"
)

# Fuzzy C-means on PLS
cluster_labels_pls, u, cntr, fpc = perform_fuzzy_cmeans(X_pls, n_clusters=3)
plot_clusters(X_pls, cluster_labels_pls, output_dir="output/pls_plots")
