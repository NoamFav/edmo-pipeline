import json
import numpy as np
import sklearn.decomposition as skd
from abc import ABC, abstractmethod
from fcmeans import FCM
from sklearn.cross_decomposition import PLSRegression

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
