"""
vae.py - Variational Autoencoder (VAE) implementation for spectral analysis.

This module contains the VAE class and related functions.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

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

        # Define metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # Add validation metrics
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="val_reconstruction_loss")
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")

    def reset_metrics(self):
        """Reset all metrics trackers."""
        metrics = [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker
        ]
        for metric in metrics:
            metric.reset_state()


    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.reconstruction_loss_tracker, 
            self.kl_loss_tracker,
            self.val_total_loss_tracker, 
            self.val_reconstruction_loss_tracker, 
            self.val_kl_loss_tracker
        ]
    
    def call(self, data):
        """
        Forward pass through the VAE.
        
        Args:
            data (tf.Tensor): Input data
            
        Returns:
            tf.Tensor: Reconstructed output
        """
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

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
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

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

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result()
        }
    

    @classmethod
    def from_config(cls, config):
        """
        Create a VAE instance from its configuration.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            VAE: An instance of the VAE model
        """
        encoder = Model.from_config(config['encoder'])
        decoder = Model.from_config(config['decoder'])
        return cls(encoder=encoder, decoder=decoder, **config)

    
    
def create_vae_model(input_dim, latent_dim=16, initial_learning_rate=1e-4, l2_reg=0.001):
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

    # First layer - compress to 512
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Second layer - compress to 256
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Third layer - compress to 128
    x = layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Fourth layer - compress to 64
    x = layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)  # Low dropout rate in the last Encoder layer

    # Latent space
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # === Decoder ===
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    # First layer - expand to 64
    x = layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)  # Low dropout rate in the first Decoder layer

    # Second layer - expand to 128
    x = layers.Dense(
        128, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Third layer - expand to 256
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Fourth layer - expand to 512
    x = layers.Dense(
        512, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add Dropout for regularization

    # Output layer
    outputs = layers.Dense(input_dim, activation='linear')(x)
    decoder = Model(latent_inputs, outputs, name="decoder")

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500,
        decay_rate=0.95,
        staircase=True
    )

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=2.0
    )

    # Create and compile VAE
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizer)
    
    return vae