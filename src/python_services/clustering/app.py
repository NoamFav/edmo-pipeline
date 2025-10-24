from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import sklearn.decomposition as skd
from skfuzzy import cluster as fcluster  # explicit submodule import
from abc import ABC, abstractmethod

app = FastAPI(title="Clustering Service", version="0.1.0")


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


class ClusterRequest(BaseModel):
    features: list[list[float]]
    n_clusters: int = 5
    n_components: int = 3
    dim_red_method: str = "PCA"


class ClusterResponse(BaseModel):
    cluster_labels: list[int]
    membership_matrix: list[list[float]]
    reduced_features: list[list[float]]
    explained_variance: float
    explained_variance_per_dimension: list[float]
    dimension_components: list[list[float]]
    reduction_used: str | None = None


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_features(request: ClusterRequest):
    X = np.asarray(request.features, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return ClusterResponse(
            cluster_labels=[],
            membership_matrix=[],
            reduced_features=[],
            explained_variance=0.0,
            dimension_components=[],
            explained_variance_per_dimension=[],
        )

    n_samples, n_features = X.shape

    n_comp_max = max(1, min(n_samples, n_features))
    n_comp = max(1, min(request.n_components, n_comp_max))
    if n_comp == 1 and (n_samples == 1 or n_features == 1):
        X_reduced = X.reshape(n_samples, -1)[:, :1]
        explained = 1.0
        explained_per_dimension: list[float] = [1.0]
        dimension_components: np.ndarray = np.array([])
    else:
        dim_red = create_dim_red_method(request.dim_red_method)
        X_reduced = dim_red.fit(X)
        explained = dim_red.total_explained_variance()
        explained_per_dimension = dim_red.dimension_explained_variance()
        dimension_components = dim_red.components()

    k = max(1, min(request.n_clusters, n_samples))

    # Single-sample or single-cluster
    if n_samples == 1 or k == 1:
        labels = [0] * n_samples
        membership = np.zeros((n_samples, k), dtype=float)
        membership[:, 0] = 1.0
        return ClusterResponse(
            cluster_labels=labels,
            membership_matrix=membership.tolist(),
            reduced_features=X_reduced.tolist(),
            explained_variance=explained,
            explained_variance_per_dimension=explained_per_dimension,
            dimension_components=dimension_components.tolist(),
        )

    data = X_reduced.T
    cntr, u, u0, d, jm, p, fpc = fcluster.cmeans(
        data=data,
        c=k,
        m=2.0,
        error=0.005,
        maxiter=150,
        init=None,
        seed=0,
    )

    labels = np.argmax(u, axis=0).astype(int).tolist()
    return ClusterResponse(
        cluster_labels=labels,
        membership_matrix=u.T.tolist(),
        reduced_features=X_reduced.tolist(),
        explained_variance=explained,
        explained_variance_per_dimension=explained_per_dimension,
        dimension_components=dimension_components.tolist(),
        reduction_used=request.dim_red_method,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
