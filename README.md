# LearnSpec: Spectral Analysis with VAE

A machine learning project for analyzing astronomical spectra from the JADES survey using Variational Autoencoders (VAE) and clustering techniques.

## Overview

This project implements a deep learning pipeline to analyze and classify astronomical spectra from the JADES (JWST Advanced Deep Extragalactic Survey) dataset. It uses a Variational Autoencoder (VAE) for dimensionality reduction and feature learning, followed by UMAP and HDBSCAN for clustering analysis.

## Features

- Spectral data preprocessing and normalization
- Custom VAE implementation for spectral analysis
- Dimensionality reduction using UMAP
- Clustering analysis using HDBSCAN
- Visualization tools for spectral data and latent space
- Support for multiple JADES survey datasets

## Requirements

```txt
numpy
matplotlib
tensorflow
astropy
pandas
seaborn
umap-learn
hdbscan
specutils
```

## Project Structure

```
learnspec/
├── data/
│   ├── redshift_z4_16.npy
│   ├── resampled_data_z4_16.npy
│   └── wavelength_z4_16.npy
├── functions/
│   ├── directories.py
│   ├── readspec.py
│   └── stacking.py
├── notebooks/
│   └── vae_clustering.ipynb
└── src/
```

## Usage

The main analysis pipeline is implemented in the Jupyter notebook `notebooks/vae_clustering.ipynb`. The notebook includes:

1. Data loading and preprocessing
2. VAE model implementation and training
3. Latent space visualization
4. Clustering analysis
5. Cluster visualization and interpretation

## VAE Architecture

The VAE consists of:

- Encoder: Dense layers (512 → 256 → latent_dim)
- Latent space: 16 dimensions
- Decoder: Dense layers (latent_dim → 256 → 512 → input_dim)
- Custom loss function handling masked values and KL divergence

## Contact

Aayush Saxena
aayush.saxena@physics.ox.ac.uk