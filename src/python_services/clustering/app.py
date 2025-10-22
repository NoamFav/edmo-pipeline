from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.decomposition import PCA
from skfuzzy import cluster as fcluster  # explicit submodule import

app = FastAPI(title="Clustering Service", version="0.1.0")


class ClusterRequest(BaseModel):
    features: list[list[float]]
    n_clusters: int = 5
    n_components: int = 3


class ClusterResponse(BaseModel):
    cluster_labels: list[int]
    membership_matrix: list[list[float]]
    reduced_features: list[list[float]]
    explained_variance: float


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_features(request: ClusterRequest):
    X = np.asarray(request.features, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return ClusterResponse(
            cluster_labels=[],
            membership_matrix=[],
            reduced_features=[],
            explained_variance=0.0,
        )

    n_samples, n_features = X.shape

    # ---------- PCA (clamped) ----------
    # must be 1..min(n_samples, n_features)
    n_comp_max = max(1, min(n_samples, n_features))
    n_comp = max(1, min(request.n_components, n_comp_max))

    if n_comp == 1 and (n_samples == 1 or n_features == 1):
        # trivial 1D case: skip PCA math, treat as identity
        X_reduced = X.reshape(n_samples, -1)[:, :1]  # shape (n, 1)
        explained = 1.0
    else:
        pca = PCA(n_components=n_comp, random_state=0)
        X_reduced = pca.fit_transform(X)
        explained = float(np.sum(pca.explained_variance_ratio_))

    # ---------- Clusters (clamped) ----------
    # you cannot have more clusters than samples
    k = max(1, min(request.n_clusters, n_samples))

    # Single-sample or single-cluster: degenerate but valid
    if n_samples == 1 or k == 1:
        labels = [0] * n_samples
        membership = np.zeros((n_samples, k), dtype=float)
        membership[:, 0] = 1.0
        return ClusterResponse(
            cluster_labels=labels,
            membership_matrix=membership.tolist(),
            reduced_features=X_reduced.tolist(),
            explained_variance=explained,
        )

    # scikit-fuzzy wants data as (features, samples)
    data = X_reduced.T  # (d, n)
    cntr, u, u0, d, jm, p, fpc = fcluster.cmeans(
        data=data,
        c=k,
        m=2.0,
        error=0.005,
        maxiter=150,
        init=None,
        seed=0,
    )

    labels = np.argmax(u, axis=0).astype(int).tolist()  # (n,)
    return ClusterResponse(
        cluster_labels=labels,
        membership_matrix=u.T.tolist(),  # (n, k)
        reduced_features=X_reduced.tolist(),  # (n, n_comp)
        explained_variance=explained,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
