import warnings

import numpy as np
import pandas as pd
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

def Silhouette(prox_matrix,
               proximity_type=("dissimilarity", "similarity"),
               method=("medoid", "pac"),
               average=("crisp", "fuzzy", "median"),
               prob_matrix=None,
               a=2,
               sort=False,
               print_summary=False,
               clust_fun=None,
               **kwargs):

    # --- Validate prox_matrix and clust_fun ---
    if clust_fun is None:
        if not isinstance(prox_matrix, np.ndarray) or not np.issubdtype(prox_matrix.dtype, np.number):
            raise ValueError("When clust_fun is None, prox_matrix must be a numeric matrix.")
        if prox_matrix.shape[1] < 2:
            raise ValueError("prox_matrix must have at least two columns (clusters).")
    else:
        if not isinstance(prox_matrix, str):
            raise ValueError("When clust_fun is not None, prox_matrix must be a string naming a matrix component.")
        if not callable(clust_fun):
            raise ValueError("clust_fun must be a callable.")

        try:
            clust_out = clust_fun(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error in clustering function: {e}")

        if prox_matrix not in clust_out:
            raise ValueError(f"Component '{prox_matrix}' not found in clustering output.")

        prox_matrix = clust_out[prox_matrix]

        if prob_matrix is not None:
            if prob_matrix not in clust_out:
                raise ValueError(f"Component '{prob_matrix}' not found in clustering output.")
            prob_matrix = clust_out[prob_matrix]

        if not isinstance(prox_matrix, np.ndarray) or not np.issubdtype(prox_matrix.dtype, np.number):
            raise ValueError("Extracted prox_matrix must be a numeric matrix.")
        if prox_matrix.shape[1] < 2:
            raise ValueError("Extracted prox_matrix must have at least two columns (clusters).")

    # --- Validate prob_matrix ---
    if prob_matrix is not None:
        if not isinstance(prob_matrix, np.ndarray) or not np.issubdtype(prob_matrix.dtype, np.number):
            raise ValueError("Extracted prob_matrix must be a numeric matrix.")
        if prob_matrix.shape[1] < 2:
            raise ValueError("Extracted prob_matrix must have at least two columns (clusters).")
        if not np.allclose(prob_matrix.sum(axis=1), 1.0, atol=np.sqrt(np.finfo(float).eps)):
            raise ValueError("Each row of prob_matrix must sum to 1.")

    # --- match.arg equivalents ---
    proximity_type = proximity_type[0] if isinstance(proximity_type, tuple) else proximity_type
    method = method[0] if isinstance(method, tuple) else method
    average = average[0] if isinstance(average, tuple) else average

    if average == "fuzzy" and prob_matrix is None:
        warnings.warn("average = 'fuzzy' requires prob_matrix; falling back to 'crisp'.")
        average = "crisp"

    if not isinstance(a, (int, float)) or a <= 0:
        raise ValueError("a must be a positive numeric value.")

    # --- Helpers ---
    def maxn(x, n):
        return np.argsort(-x)[n-1]

    def minn(x, n):
        return np.argsort(x)[n-1]

    # --- Determine cluster & neighbor ---
    if prob_matrix is None:
        if proximity_type == "similarity":
            cluster = np.apply_along_axis(maxn, 1, prox_matrix, 1)
            neighbor = np.apply_along_axis(maxn, 1, prox_matrix, 2)
        else:
            cluster = np.apply_along_axis(minn, 1, prox_matrix, 1)
            neighbor = np.apply_along_axis(minn, 1, prox_matrix, 2)
    else:
        cluster = np.apply_along_axis(maxn, 1, prob_matrix, 1)
        neighbor = np.apply_along_axis(maxn, 1, prob_matrix, 2)

    n = prox_matrix.shape[0]
    sil_width = np.zeros(n)
    weight = np.zeros(n)

    # --- Silhouette computation ---
    for i in range(n):
        ci, ni = cluster[i], neighbor[i]

        if proximity_type == "similarity":
            num = prox_matrix[i, ci] - prox_matrix[i, ni]
        else:
            num = prox_matrix[i, ni] - prox_matrix[i, ci]

        if method == "pac":
            den = prox_matrix[i, ci] + prox_matrix[i, ni]
        else:
            den = max(prox_matrix[i, ci], prox_matrix[i, ni])

        sil_width[i] = num / den if den != 0 else 0.0

        if prob_matrix is not None:
            weight[i] = (prob_matrix[i, ci] - prob_matrix[i, ni]) ** a

    # --- Build output ---
    data = {
        "cluster": cluster + 1,    # R is 1-based
        "neighbor": neighbor + 1,
        "sil_width": sil_width
    }

    if prob_matrix is not None:
        data["weight"] = weight

    widths = pd.DataFrame(data)

    # --- Sorting ---
    if sort:
        widths["_row"] = widths.index
        widths = widths.sort_values(["cluster", "sil_width"], ascending=[True, False])
        widths = widths.set_index("_row").sort_index()

    # --- Attach attributes ---
    widths.attrs["proximity_type"] = proximity_type
    widths.attrs["method"] = method
    widths.attrs["average"] = average

    # --- Print summary ---
    if print_summary:
        if average == "crisp":
            clus_avg = widths.groupby("cluster")["sil_width"].mean()
            avg_width = widths["sil_width"].mean()
        elif average == "fuzzy":
            sw = widths["weight"] * widths["sil_width"]
            clus_avg = widths.groupby("cluster").apply(
                lambda df: sw[df.index].sum() / df["weight"].sum()
            )
            avg_width = sw.sum() / widths["weight"].sum()
        else:
            clus_avg = widths.groupby("cluster")["sil_width"].median()
            avg_width = widths["sil_width"].median()

        header = f"{average.capitalize()} {proximity_type} {method} silhouette: {avg_width:.4f}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        print(pd.DataFrame({
            "cluster": clus_avg.index,
            "size": widths.groupby("cluster").size(),
            "avg.sil.width": clus_avg.round(4)
        }))
        print("\nAvailable attributes:", ", ".join(widths.attrs.keys()))

    return widths

def softSilhouette(prob_matrix,
                   prob_type=("pp", "nlpp", "pd"),
                   method=("pac", "medoid"),
                   average=("crisp", "fuzzy", "median"),
                   a=2,
                   sort=False,
                   print_summary=False,
                   clust_fun=None,
                   **kwargs):

    # --- Validate prob_matrix and clust_fun ---
    if clust_fun is None:
        if not isinstance(prob_matrix, np.ndarray) or not np.issubdtype(prob_matrix.dtype, np.number):
            raise ValueError("When clust_fun is None, prob_matrix must be a numeric matrix.")
        if prob_matrix.shape[1] < 2:
            raise ValueError("prob_matrix must have at least two columns (clusters).")
    else:
        if not isinstance(prob_matrix, str):
            raise ValueError("When clust_fun is not None, prob_matrix must be a string naming a matrix component.")
        if not callable(clust_fun):
            raise ValueError("clust_fun must be a callable.")

        try:
            clust_out = clust_fun(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error in clustering function: {e}")

        if prob_matrix not in clust_out:
            raise ValueError(f"Component '{prob_matrix}' not found in clustering output.")

        prob_matrix = clust_out[prob_matrix]

        if not isinstance(prob_matrix, np.ndarray) or not np.issubdtype(prob_matrix.dtype, np.number):
            raise ValueError("Extracted prob_matrix must be a numeric matrix.")
        if prob_matrix.shape[1] < 2:
            raise ValueError("Extracted prob_matrix must have at least two columns (clusters).")

    # --- Row sums must be 1 ---
    if not np.allclose(prob_matrix.sum(axis=1), 1.0, atol=np.sqrt(np.finfo(float).eps)):
        raise ValueError("Each row of prob_matrix must sum to 1")

    if not isinstance(a, (int, float)) or a <= 0:
        raise ValueError("a must be a positive numeric value.")

    # --- match.arg equivalent ---
    prob_type = prob_type[0] if isinstance(prob_type, tuple) else prob_type
    method = method[0] if isinstance(method, tuple) else method
    average = average[0] if isinstance(average, tuple) else average

    # --- Probability type logic ---
    if prob_type == "pp":
        proximity_type = "similarity"
        prox_matrix = prob_matrix

    elif prob_type == "nlpp":
        proximity_type = "dissimilarity"
        prox_matrix = -np.log(prob_matrix)

    elif prob_type == "pd":
        proximity_type = "similarity"
        pm_den = np.tile(prob_matrix.sum(axis=0) / prob_matrix.shape[0],
                          (prob_matrix.shape[0], 1))
        if np.any(pm_den == 0):
            raise ValueError("Column sums in prob_matrix must be non-zero for prob_type = 'pd'.")
        prox_matrix = prob_matrix / pm_den

    else:
        raise ValueError("Unknown prob_type")

    # --- Prepare Silhouette arguments ---
    sil_args = dict(
        prox_matrix=prox_matrix,
        proximity_type=proximity_type,
        method=method,
        average=average,
        a=a,
        sort=sort,
        print_summary=print_summary
    )

    if average == "fuzzy":
        sil_args["prob_matrix"] = prob_matrix
    else:
        sil_args["prob_matrix"] = None

    # --- Call Silhouette ---
    result = Silhouette(**sil_args)
    return result

def perform_fuzzy_cmeans_auto_k(
    X_reduced,
    k_range=range(2, 11),
    m=2.0,
    max_iter=1000,
    error=0.005,
    random_state=None
):
    """
    Perform fuzzy C-means clustering with automatic selection of K
    using soft silhouette.

    Returns:
    --------
    best_silhouette : float
        Best average soft silhouette score
    best_k : int
        Optimal number of clusters
    cluster_labels : np.ndarray
        Hard cluster assignments
    u : np.ndarray
        Fuzzy membership matrix
    cntr : np.ndarray
        Cluster centers
    fpc : float
        Fuzzy partition coefficient
    """
    silhouette_scores = {}
    best_score = -np.inf
    best_model = None
    best_k = None

    for k in k_range:
        fcm = FCM(
            n_clusters=k,
            m=m,
            max_iter=max_iter,
            error=error,
            random_state=random_state
        )

        fcm.fit(X_reduced)

        # --- Soft silhouette ---
        sil_df = softSilhouette(
            prob_matrix=fcm.u,
            prob_type="pp",
            method="pac",
            average="fuzzy",
            print_summary=False
        )

        sil_score = sil_df["sil_width"].mean()
        silhouette_scores[k] = sil_score

        if sil_score > best_score:
            best_score = sil_score
            best_model = fcm
            best_k = k

    # --- Extract final outputs (same as perform_fuzzy_cmeans) ---
    cluster_labels = best_model.predict(X_reduced)
    u = best_model.u
    cntr = best_model.centers
    fpc = np.sum(u ** 2) / X_reduced.shape[0]

    return silhouette_scores, best_score, best_k, cluster_labels, u, cntr, fpc
