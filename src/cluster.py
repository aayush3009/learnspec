import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import umap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


### Benchmarking various clustering algorithms
def benchmark_clustering(X, visualize=False, name="Latent space"):
    results = {}

    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)
    results['KMeans'] = labels_kmeans

    # GMM
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels_gmm = gmm.fit_predict(X)
    results['GMM'] = labels_gmm

    # DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X)
    results['DBSCAN'] = labels_dbscan

    # HDBSCAN (optional)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels_hdb = clusterer.fit_predict(X)
    results['HDBSCAN'] = labels_hdb

    print(f"\n Clustering results for {name}:")
    for method, labels in results.items():
        mask = labels != -1  # Ignore noise for DBSCAN/HDBSCAN
        if len(set(labels[mask])) < 2:
            print(f"{method}: Too few clusters")
            continue
        silhouette = silhouette_score(X[mask], labels[mask])
        db_score = davies_bouldin_score(X[mask], labels[mask])
        print(f"{method}: Silhouette = {silhouette:.3f}, DB = {db_score:.3f}")

    # Optionally visualize
    if visualize:    
        plt.close()
        for method, labels in results.items():
            plt.figure(figsize=(5, 4))
            plt.title(f"{name} - {method}")
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=5)
            plt.colorbar(label='Cluster ID')
            plt.tight_layout()
            plt.show()

    return results

def cluster_latent_space_hdbscan(Z, min_cluster_size=20, min_samples=2):
    """
    Cluster the latent space using HDBSCAN.
    
    Args:
        Z (numpy.ndarray): Latent space representation
        min_cluster_size (int): Minimum size of clusters
        min_samples (int): Minimum number of samples in a neighborhood for a point to be considered a core point
    
    Returns:
        numpy.ndarray: Cluster labels
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(Z)
    return labels

def cluster_latent_space_gmm(Z, n_components=5, covariance_type='full', random_state=42):
    """
    Cluster the latent space using Gaussian Mixture Model (GMM).
    
    Args:
        Z (numpy.ndarray): Latent space representation
        n_components (int): Number of mixture components
    
    Returns:
        numpy.ndarray: Cluster labels
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    labels = gmm.fit_predict(Z)
    return labels

def umap_latent_space(Z, n_neighbors=5, min_dist=0.0, metric='euclidean'):
    """
    Reduce the dimensionality of the latent space using UMAP.
    
    Args:
        Z (numpy.ndarray): Latent space representation
        n_neighbors (int): Size of local neighborhood
        min_dist (float): Minimum distance between points in the embedding space
        metric (str): Metric to use for distance computation
    
    Returns:
        numpy.ndarray: UMAP reduced representation
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    Z_umap = reducer.fit_transform(Z)
    return Z_umap

def plot_cluster_spectra(cluster_label, cluster_spectra, wavelength):
    plt.close()
    plt.figure(figsize=(12, 6))
    for i, spectrum in enumerate(cluster_spectra):
        plt.plot(wavelength, spectrum, alpha=0.5)
    plt.title(f"Spectra in Cluster {cluster_label}")
    plt.xlabel("Wavelength (A)")
    plt.ylabel("Flux (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()