import argparse
import os
import numpy as np
import tensorflow as tf

### Function to generate random synthetic spectra
def generate_random_synthetic_spectra(decoder, n_samples=10, latent_dim=16):
    """Generate completely random synthetic spectra."""
    # Sample from standard normal distribution (the VAE's prior)
    random_latent = np.random.normal(0, 1, size=(n_samples, latent_dim))
    
    # Generate spectra using the decoder
    synthetic_spectra = decoder.predict(random_latent)
    
    return synthetic_spectra, random_latent

### Function to interpolate between two real spectra
def interpolate_spectra(encoder, decoder, data, indices, n_steps=10):
    """Generate spectra interpolating between two real spectra."""
    # Get latent representations of the two spectra
    z_mean_1, _, _ = encoder.predict(data[indices[0]:indices[0]+1])
    z_mean_2, _, _ = encoder.predict(data[indices[1]:indices[1]+1])
    
    # Create interpolation steps
    interpolated_latent = []
    for alpha in np.linspace(0, 1, n_steps):
        z_interp = (1 - alpha) * z_mean_1 + alpha * z_mean_2
        interpolated_latent.append(z_interp)
    
    interpolated_latent = np.vstack(interpolated_latent)
    
    # Generate interpolated spectra
    interpolated_spectra = decoder.predict(interpolated_latent)
    
    return interpolated_spectra, interpolated_latent

### Generate cluster-like spectra
def generate_cluster_like_spectra(z_mean, cluster_labels, target_cluster, 
                                decoder, n_samples=10):
    """Generate synthetic spectra similar to a specific cluster."""
    # Get latent vectors for the target cluster
    cluster_latent = z_mean[cluster_labels == target_cluster]
    
    # Calculate cluster statistics
    cluster_mean = np.mean(cluster_latent, axis=0)
    cluster_cov = np.cov(cluster_latent.T)
    
    # Sample from the cluster's distribution
    synthetic_latent = np.random.multivariate_normal(
        cluster_mean, cluster_cov, size=n_samples
    )
    
    # Generate spectra
    synthetic_spectra = decoder.predict(synthetic_latent)
    
    return synthetic_spectra, synthetic_latent


### Generate comprehensive synthetic spectra dataset
def create_comprehensive_synthetic_dataset(encoder, decoder, data, z_mean, 
                                         cluster_labels, redshifts, n_total=1000):
    """Create a comprehensive synthetic dataset using multiple generation methods."""
    synthetic_spectra = []
    generation_methods = []
    
    # 40% random sampling
    n_random = int(0.4 * n_total)
    random_synth, _ = generate_random_synthetic_spectra(decoder, n_random, latent_dim)
    synthetic_spectra.extend(random_synth)
    generation_methods.extend(['random'] * n_random)
    
    # 30% cluster-based
    n_cluster = int(0.3 * n_total)
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    for cluster in unique_clusters:
        n_per_cluster = n_cluster // len(unique_clusters)
        if n_per_cluster > 0:
            cluster_synth, _ = generate_cluster_like_spectra(
                z_mean, cluster_labels, cluster, decoder, n_per_cluster
            )
            synthetic_spectra.extend(cluster_synth)
            generation_methods.extend([f'cluster_{cluster}'] * n_per_cluster)
    
    # 30% interpolation
    n_interp = n_total - len(synthetic_spectra)
    for _ in range(n_interp):
        idx1, idx2 = np.random.choice(len(data), 2, replace=False)
        interp_spectra, _ = interpolate_spectra(encoder, decoder, data, [idx1, idx2], n_steps=1)
        synthetic_spectra.extend(interp_spectra)
        generation_methods.extend(['interpolation'])
    
    return np.array(synthetic_spectra[:n_total]), generation_methods[:n_total]


### Comparisons between synthetic and real spectra
def compare_synthetic_to_nearest(synthetic_spectra, training_data, wavelength):
    """Compare synthetic spectra to their nearest neighbors in the training set."""
    from sklearn.neighbors import NearestNeighbors
    
    # Reshape training data for nearest neighbor search
    training_data_reshaped = training_data.reshape(training_data.shape[0], -1)
    
    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=1).fit(training_data_reshaped)
    
    # Reshape synthetic spectra for comparison
    synthetic_reshaped = synthetic_spectra.reshape(synthetic_spectra.shape[0], -1)
    
    # Find nearest neighbors for each synthetic spectrum
    distances, indices = nbrs.kneighbors(synthetic_reshaped)
    
    return indices, distances

