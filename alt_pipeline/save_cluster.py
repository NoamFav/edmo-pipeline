import pickle
import dim_red_clustering_functions
import full_dimensions_clustering
import numpy as np
              
def save_cluster(clustered_data, filename):
    """
    Save clustered data to a pickle file.
    Imports:
    - clustered_data: An instance of ClusteredDatta containing clustering results.
    - filename: The path to the file where data will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(clustered_data, f)

def load_cluster(filename):
    """Load clustered data from a pickle file.
    Imports:
    - filename: The path to the file from which data will be loaded.
    Returns:
    - clustered_data: An instance of ClusteredDatta containing clustering results.
    """

    with open(filename, 'rb') as f:
        clustered_data = pickle.load(f)
    return clustered_data

def test_save():
    # Create some sample clustered data
    XScaled = full_dimensions_clustering.X_scaled
    clustered_data = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
        XScaled,
        k_range=range(2, 11),
        m=2.0,
        random_state=42
    )
    save_cluster(clustered_data, "./alt_pipeline/test_cluster.pkl")
    print("Cluster Saved")
    print("Checking if saved matched clustered data...")
    loaded_data = load_cluster("./alt_pipeline/test_cluster.pkl")
    assert len(clustered_data) == len(loaded_data), "Data length mismatch!"

    """Check if the loaded data matches the original data.
    Raises an AssertionError if there is a mismatch.
    for loop over each element in the clustered_data and loaded_data and compare them individually.
    """
    for i, (orig, loaded) in enumerate(zip(clustered_data, loaded_data)):
        if isinstance(orig, np.ndarray):
            if not np.array_equal(orig, loaded):
                raise AssertionError(f"Data mismatch at index {i}!")
        else:
            if orig != loaded:
                raise AssertionError(f"Data mismatch at index {i}!")
    print("Data matches!")

def test_load():
    # Assuming test_save has already created the file
    loaded_data = load_cluster("test_cluster.pkl")
    print("Cluster Loaded")


test_save()
test_load()