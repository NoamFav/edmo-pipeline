from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.decomposition import PCA
import skfuzzy as fuzz

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
    """Perform fuzzy c-means clustering on features."""
    X = np.array(request.features)

    # Dimensionality reduction
    pca = PCA(n_components=request.n_components)
    X_reduced = pca.fit_transform(X)

    # Fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_reduced.T, request.n_clusters, 2, error=0.005, maxiter=150
    )

    # Get hard cluster assignments
    cluster_labels = np.argmax(u, axis=0).tolist()

    return ClusterResponse(
        cluster_labels=cluster_labels,
        membership_matrix=u.T.tolist(),
        reduced_features=X_reduced.tolist(),
        explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
