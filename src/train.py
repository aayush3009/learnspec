"""
train.py - Training and utility functions for the LearnSpec VAE model

This module provides functions for preprocessing spectral data,
building and training VAE models for spectroscopic analysis.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import tensorflow as tf
from tensorflow.keras import layers, Model

### Load in the pre-processed data
def load_data(scaler=True, scale_factor=1., simple_scaling=False, simple_scaler=MinMaxScaler, remove_nan=True, piecewise_smoothing=True, simple_smoothing=False, save_transformed_data=False):
    ### Load resampled spectra
    original_resampled_data = np.load("/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/dja_z4-16/dja_resampled_spectra_z4-16.npy")
    ### Create a copy of the original resampled data
    resampled_data = np.copy(original_resampled_data)

    ### Load the wavelength grid
    wavelength = np.load("/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/dja_z4-16/dja_resampled_wavelength_z4-16.npy")
    ### Load the redshift information
    redshifts = np.load("/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/dja_z4-16/dja_redshift_info_z4-16.npy")

    ### Load the source IDs
    speclist = np.load("/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/dja_z4-16/dja_source_id_info_z4-16.npy", allow_pickle=True)

    ### Let's remove NaN values from the spectra by setting them to 0
    if remove_nan:
        for i in range(len(resampled_data)):
            resampled_data[i][np.isnan(resampled_data[i])] = 0.0
            resampled_data[i][np.isinf(resampled_data[i])] = 0.0

    ### Piecewise smoothing based on wavelength
    if piecewise_smoothing:
        ### Find nearest wavelength index
        def find_nearest_index(wavelengths, target_wavelength):
            return np.abs(wavelengths - target_wavelength).argmin()

        def piecewise_smooth_spectrum(spectrum, wavelengths):
            smoothed = np.zeros_like(spectrum)
            
            # Define regions and corresponding sigmas
            idx3000 = find_nearest_index(wavelengths, 3000)
            idx6000 = find_nearest_index(wavelengths, 6000)

            smoothed[:idx3000] = gaussian_filter1d(spectrum[:idx3000], sigma=2.0)
            smoothed[idx3000:idx6000] = gaussian_filter1d(spectrum[idx3000:idx6000], sigma=1.5)
            smoothed[idx6000:] = gaussian_filter1d(spectrum[idx6000:], sigma=1.0)

            return smoothed
        # Apply the piecewise smoothing function to each spectrum
        resampled_data = np.array([piecewise_smooth_spectrum(spec, wavelength) for spec in resampled_data])

    if simple_smoothing:
        resampled_data = gaussian_filter1d(resampled_data, sigma=1.5, axis=1)

    ### Rescale the data using the arcsinh transformation. Other transformations can be used as well.
    if scaler:
        arcsinh_transformer = FunctionTransformer(np.arcsinh, validate=True)
        resampled_data = arcsinh_transformer.fit_transform(resampled_data)/scale_factor

    if simple_scaling:
        scaler_function = simple_scaler()
        resampled_data = scaler_function.fit_transform(resampled_data)

    if save_transformed_data:
        np.save("/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/dja_z4-16/dja_resampled_spectra_z4-16_transformed.npy", resampled_data)

    return resampled_data, wavelength, redshifts, speclist


### Identify spectra that may have negative values and return their indices
def detect_negative_spikes(spectra, threshold=-2.0, prominence=0.5):
    """Flag spectra with strong negative fluxes in arcsinh-transformed data."""
    flagged = []
    for i, spec in enumerate(spectra):
        # Focus only on values that deviate significantly below median
        median = np.median(spec)
        min_flux = np.min(spec)
        if (min_flux - median) < threshold and abs(min_flux - median) > prominence:
            flagged.append(i)
    return flagged


### Create a VAE model to be trained on the pre-processed data
def get_input_dim(data):
    """
    Get the input dimension from data.
    
    Args:
        data (numpy.ndarray): Input data array
        
    Returns:
        int: Input dimension (number of features)
    """
    input_dim = len(data[0])  # Change this to your spectral length
    return input_dim


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    
    Args:
        args (list): Mean and log variance of the latent distribution
        
    Returns:
        tf.Tensor: Sampled latent vector
    """
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# === Custom VAE Model ===
class VAE(Model):
    """
    Variational Autoencoder (VAE) model for spectral analysis.
    
    This class implements a custom VAE with tracking of loss components.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # === Masking regions where data == 0 (i.e., formerly NaN) ===
            mask = tf.cast(tf.not_equal(data, 0.0), tf.float32)
            diff = mask * (data - reconstruction)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))

            # KL divergence (safe)
            z_log_var_clipped = tf.clip_by_value(z_log_var, -10.0, 10.0)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var_clipped - tf.square(z_mean) - tf.exp(z_log_var_clipped), axis=1
            ))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    
def create_vae_model(input_dim, latent_dim=16, learning_rate=1e-4):
    """
    Creates a Variational Autoencoder model with specified dimensions
    
    Args:
        input_dim (int): Dimension of input data
        latent_dim (int): Dimension of latent space
        learning_rate (float): Learning rate for Adam optimizer
        
    Returns:
        vae: Compiled VAE model
    """
    # === Encoder ===
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # === Decoder ===
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(latent_inputs)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='linear')(x)

    decoder = Model(latent_inputs, outputs, name="decoder")

    # Create and compile VAE
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    return vae

### Function to train the VAE model
def train_vae_model(model, data, epochs=100, batch_size=64, shuffle=True):
    """
    Trains the VAE model on the provided data
    
    Args:
        vae: Compiled VAE model
        data (np.ndarray): Training data
        epochs (int): Number of epochs to train
        batch_size (int): Size of each training batch
        shuffle (bool): Whether to shuffle the data before training
        
    Returns:
        history: Training history
    """
    # Train the VAE model
    history = model.fit(
        data, 
        epochs=epochs, 
        batch_size=batch_size, 
        shuffle=shuffle)
    
    return history
