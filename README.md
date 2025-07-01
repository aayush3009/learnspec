# LearnSpec: Spectral Analysis with VAE

A machine learning project for analyzing astronomical spectra from JWST using Variational Autoencoders (VAE) and clustering techniques.

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
│   ├── load.py            # Data loading functions
    ├── train.py           # VAE model training functions
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
# Build Docker image
docker compose build

# Run container
docker compose run learnspec
```

## Usage

### Training a VAE Model

```python
python -m learnspec.src.train \
    --input-data /path/to/data.npy \
    --validation-split 0.2 \
    --latent-dim 16 \
    --epochs 1000 \
    --batch-size 32 \
    --save-models \
    --plot-training
```

### Making Predictions

```python
python -m learnspec.src.predict \
    --encoder-path models/encoder.keras \
    --decoder-path models/decoder.keras \
    --input-data data/test_data.npy
```

### Clustering spectra in latent space

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


## Docker Support

### Building the container

```bash
cd /path/to/learnspec
docker compose build
```


### Running the container

```bash
docker compose run learnspec
```


### Updating the Container

After code changes:

```bash
docker compose build --no-cache
```

## Author

Aayush Saxena (aayush.saxena@physics.ox.ac.uk)