import json
import numpy as np
import sklearn.decomposition as skd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os

selected_features = [
    "emotion.emotions[0].score",
    "emotion.emotions[1].score",
    "nlp.sentiment.score",
    "nonverbal.basic_metrics.conversation.total_speaking_time",
    "nonverbal.pitch[0].mean_f0"
]

features_labels = [
    "a",
    "b",
    "c",
    "total_speaking_time",
    "e"
]

class Datapoint:
    dimension_labels: list[str]
    dimension_values: list[float]

    def __init__(self, labels, values):
        self.dimension_labels = labels
        self.dimension_values = values

def get_by_path(obj, path):
    """Fetches value from nested dict/list using a dot + [i] notation."""
    parts = path.replace("]", "").split(".")

    for p in parts:
        if "[" in p:  # list access
            key, idx = p.split("[")
            obj = obj[key][int(idx)]
        else:
            obj = obj[p]
    return obj

def extract_datapoints(filename:str, feature_paths:list[str]):
    datapoints = []
    with open(filename, 'r') as file:
        data = json.load(file)
        for item in data["features"]:
            try:
                labels = []
                values = []
                for path in feature_paths:
                    val = get_by_path(item, path)
                    if isinstance(val, (int, float)):
                        labels.append(path)
                        values.append(val)
                datapoints.append(Datapoint(labels, values))
            except Exception:
                print("e")
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

def plot_pca_results(X_reduced, dim_red, features_labels, output_dir="output/pca_plots"):
    """
    Plots PCA results: scatter plot of reduced data, component loadings, and explained variance.

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
    output_dir : str
        Directory to save plots
    """

    # Ensure output directory exists
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

datapoints = extract_datapoints("data/test_audio_features.json", selected_features)
dim_red = create_dim_red_method("PCA", n_dimensions=2)
datapoints_np_array = extract_datapoints("data/test_audio_features.json", selected_features)
X = datapoints_to_matrix(datapoints_np_array)

# --- Fit PCA ---
X_reduced = dim_red.fit(X)

plot_pca_results(X_reduced, dim_red, features_labels)


# --- Output prints ---
print("Reduced coordinates (each row is a datapoint):")
print(X_reduced)

print("\nTotal explained variance:")
print(dim_red.total_explained_variance())

print("\nExplained variance per dimension:")
print(dim_red.dimension_explained_variance())

components = dim_red.components()
print("\nPrincipal components (new axes):")
print(components)

# Human-readable mapping
print("\nPrincipal component interpretation:")
for i, comp in enumerate(components):
    print(f"PC{i+1}:")
    for label, weight in zip(selected_features, comp):
        print(f"  {label}: {weight}")
    print()


