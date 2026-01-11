import numpy as np
import matplotlib.pyplot as plt
import os



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
