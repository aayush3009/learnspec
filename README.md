# LearnSpec: Spectral Analysis with VAE

A machine learning project for analyzing astronomical spectra from the JADES survey using Variational Autoencoders (VAE) and unsupervised clustering techniques.

## Overview

LearnSpec is a deep learning pipeline designed for analyzing and classifying astronomical spectra from JWST using the publicly available DAWN JWST Archive (DJA) datasets. The project uses Variational Autoencoders (VAEs) for dimensionality reduction and feature learning, followed by clustering techniques like Gaussian Mixture Models to identify patterns in spectral data.

## Features

- Spectral data preprocessing and normalization
- Custom VAE implementation for spectral analysis and compression
- Support for high-redshift galaxies (z > 4)
- Dimensionality reduction using UMAP
- Multiple clustering methods (GMM, HDBSCAN)
- Visualization tools for spectral data and latent space
- Support for multiple JWST survey datasets

## Requirements

See `requirements.txt` for detailed dependencies.

## Project Structure

```
learnspec/
├── data/                  # Preprocessed data files (excluded from git)
│   ├── resampled_data*.npy
│   ├── wavelength*.npy
│   └── redshifts*.npy  
├── functions/             # Utility functions
│   ├── directories.py     # Directory/path handling
│   ├── readspec.py        # Spectrum reading functions
│   └── stacking.py        # Spectrum stacking utilities
├── models/                # Saved models directory
│   ├── *_encoder.keras    # Saved encoder models
│   └── *_decoder.keras    # Saved decoder models
├── notebooks/             # Jupyter notebooks
│   └── vae_clustering.ipynb  # Main analysis notebook
├── src/                   # Core source code
│   ├── train.py           # VAE model training functions
│   ├── predict.py         # Model prediction utilities
│   └── cluster.py         # Clustering algorithms
└── Dockerfile             # Docker configuration
```

## Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/aayush3009/learnspec.git
cd learnspec

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Docker Installation

```bash
# Build and run the Docker container
docker-compose up --build
```

## Usage

The main analysis pipeline is implemented in the Jupyter notebook `notebooks/vae_clustering.ipynb`. The notebook includes:

1. Data loading and preprocessing
2. VAE model creation and training
3. Latent space analysis with UMAP
4. Clustering of spectra
5. Visualization of cluster properties

### Example: Creating and training a VAE model

```python
from learnspec.src import train

# Load preprocessed data
resampled_data, wavelength, redshifts, speclist = train.load_data()

# Create and train VAE model
input_dim = train.get_input_dim(resampled_data)
latent_dim = 16
vae_model = train.create_vae_model(input_dim, latent_dim)
history = train.train_vae_model(vae_model, resampled_data, epochs=500, batch_size=64)
```

### Example: Clustering spectra in latent space

```python
from learnspec.src import predict, cluster

# Extract latent space representation
z_mean, z_log_var, z = predict.evaluate_latent_space(vae_model.encoder, resampled_data)

# Perform UMAP dimensionality reduction
z_2d = cluster.umap_latent_space(z_mean, n_neighbors=5, min_dist=0.0)

# Cluster using Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
cluster_labels = gmm.fit_predict(z_2d)
```

## Model Saving and Loading

```python
# Save encoder and decoder separately (recommended)
vae_model.encoder.save("models/vae_encoder.keras")
vae_model.decoder.save("models/vae_decoder.keras")

# Load encoder and decoder
from tensorflow import keras
encoder = keras.models.load_model("models/vae_encoder.keras")
decoder = keras.models.load_model("models/vae_decoder.keras")

# Alternatively, save and load weights
vae_model.save_weights("models/vae_model.weights.h5")
vae_model.load_weights("models/vae_model.weights.h5")
```

## Author

Aayush Saxena

aayush.saxena@physics.ox.ac.uk