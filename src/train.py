"""
train.py - Training and utility functions for the LearnSpec VAE model

This module provides functions for preprocessing spectral data,
building and training VAE models for spectroscopic analysis.
"""
import numpy as np
import argparse

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
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker, 
            self.reconstruction_loss_tracker, 
            self.kl_loss_tracker]
    
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


# def main():
#     """
#     Main function to allow command-line model training.
    
#     Example usage:
#     python -m learnspec.src.train --datadir ../data --data data/resampled_data.npy --wavelength data/wavelength.npy --redshifts data/redshifts.npy --speclist data/speclist.npy 
#     --scaler True --scale_factor 1.0 --latent-dim 16 --learning-rate 1e-4 --epochs 500 --batch-size 64 --save-dir ../models --model-name vae_model --plot-history
#     """

#     parser = argparse.ArgumentParser(description='Train a VAE model on spectroscopic data')
    
#     # Model options                    
#     model_group = parser.add_argument_group('Model Options')
#     model_group.add_argument('--latent-dim', type=int, default=16, 
#                         help='Dimensionality of the latent space')
#     model_group.add_argument('--learning-rate', type=float, default=1e-4, 
#                         help='Learning rate for Adam optimizer')

                        
#     # Training options
#     training_group = parser.add_argument_group('Training Options')
#     training_group.add_argument('--epochs', type=int, default=500, 
#                         help='Number of training epochs')
#     training_group.add_argument('--batch-size', type=int, default=64, 
#                         help='Batch size for training')
#     training_group.add_argument('--model-name', type=str, default='vae_model',
#                         help='Base name for saved model files')    
    
   
#     # Output options
#     output_group = parser.add_argument_group('Output Options')
#     model_group.add_argument('--scaler', action='store_true', default=False
#                         help='Apply arcsinh transformation to the data')
#     output_group.add_argument('--save-dir', type=str, default='../models', default=None,
#                         help='Directory to save models')
#     output_group.add_argument('--plot-history', action='store_true', default=False
#                         help='Plot training history and save to file')
    
#     args = parser.parse_args()

#     # Use dataset name as model name if not provided
#     if args.model_name is None:
#         args.model_name = args.dataset
    
#     # Print training configuration
#     print(f"Training VAE model with configuration:")
#     print(f"  - Latent dimensions: {args.latent_dim}")
#     print(f"  - Learning rate: {args.learning_rate}")
#     print(f"  - Epochs: {args.epochs}")
#     print(f"  - Batch size: {args.batch_size}")
    
#     # Create and train model
#     try:
#         print("Creating VAE model...")
#         vae_model = create_vae_model(
#             input_dim=input_dim, 
#             latent_dim=args.latent_dim,
#             learning_rate=args.learning_rate
#         )
        
#         print(f"Starting training for {args.epochs} epochs...")
#         history = train_vae_model(
#             vae_model, 
#             resampled_data, 
#             epochs=args.epochs, 
#             batch_size=args.batch_size,
#             validation_split=args.validation_split
#         )
        
#     except Exception as e:
#         print(f"Error during model creation or training: {e}")
#         return
    
#     # Save models
#     if args.save_models:
#         try:
#             os.makedirs(args.save_dir, exist_ok=True)
            
#             # Create model file names
#             base_name = f"{args.model_name}_dim{args.latent_dim}"
#             encoder_path = os.path.join(args.save_dir, f"{base_name}_encoder.keras")
#             decoder_path = os.path.join(args.save_dir, f"{base_name}_decoder.keras")
#             weights_path = os.path.join(args.save_dir, f"{base_name}_weights.h5")
            
#             print(f"Saving encoder to {encoder_path}")
#             vae_model.encoder.save(encoder_path)
            
#             print(f"Saving decoder to {decoder_path}")
#             vae_model.decoder.save(decoder_path)
            
#             print(f"Saving weights to {weights_path}")
#             vae_model.save_weights(weights_path)
            
#             print("Training complete and models saved!")
        
#         except Exception as e:
#             print(f"Error saving models: {e}")
#             if not args.plot_history:
#                 return vae_model, history

#     else:
#         print("Skipping model saving (use --save-models to save)")
            
#     # Print final loss values
#     print(f"Final loss: {history.history['loss'][-1]:.4f}")
#     print(f"Final reconstruction loss: {history.history['reconstruction_loss'][-1]:.4f}")
#     print(f"Final KL loss: {history.history['kl_loss'][-1]:.4f}")
            
#     # Plot training history if requested
#     if args.plot_history:
#         try:
#             import matplotlib.pyplot as plt
#             plt.figure(figsize=(10, 6))
#             plt.plot(history.history['loss'], label='Total Loss')
#             plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
#             plt.plot(history.history['kl_loss'], label='KL Loss')
#             plt.xlabel('Epochs')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.title('VAE Training Loss')
            
#             history_plot_path = os.path.join(args.save_dir, f"{base_name}_training_history.png")
#             plt.savefig(history_plot_path, dpi=300)
#             print(f"Training history plot saved to {history_plot_path}")
#         except Exception as e:
#             print(f"Error creating training history plot: {e}")

#     else:
#         print("Skipping training history plot (use --plot-history to plot)")
    
#     # Return the trained model and history
#     print("Training complete!")           
#     return vae_model, history


# if __name__ == "__main__":
#     main()