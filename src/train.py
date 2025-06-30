"""
train.py - Training and utility functions for the LearnSpec VAE model

This module provides functions for preprocessing spectral data,
building and training VAE models for spectroscopic analysis.
"""
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
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

    
    
def create_vae_model(input_dim, latent_dim=16, initial_learning_rate=1e-4):
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

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    )

    # Create and compile VAE
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss=None
                )
    
    return vae


class ConvergenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=50, min_delta=1e-4, min_epochs=200):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        # self.best_loss = float('inf')
        self.best_val_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.converged = False
        # self.history = []
    
    def on_epoch_end(self, epoch, logs=None):
        # current_loss = logs.get('reconstruction_loss')
        # self.history.append(current_loss)

        # Monitor validation loss instead of training loss
        val_loss = logs.get('val_reconstruction_loss')
        
        if epoch < self.min_epochs:
            return
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.converged = True
                print(f"\nConvergence detected at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.6f}")


### Function to train the VAE model
def train_vae_model(model, data, validation_split=0.15, max_epochs=1000, batch_size=64, shuffle=True):
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

    # Split data into training and validation sets
    val_size = int(len(data) * validation_split)
    indices = np.random.permutation(len(data))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]

    # Build model by calling it on a single batch
    sample_batch = train_data[:batch_size]
    model(sample_batch)

    # Reset metrics before training
    model.reset_metrics()

    # Callbacks for monitoring training
    callbacks = [
        # Early stopping based on validation loss
        tf.keras.callbacks.EarlyStopping(
            monitor='val_reconstruction_loss',
            mode='min',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        
        # # Learning rate reduction on plateau
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_reconstruction_loss',
        #     factor=0.5,
        #     patience=30,
        #     min_lr=1e-6,
        #     verbose=1
        # ),

        # TensorBoard callback for logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.weights.h5',
            monitor='val_reconstruction_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    print("\nTraining Configuration:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Maximum epochs: {max_epochs}")

    # Train the VAE model
    history = model.fit(
        train_data,
        validation_data=(val_data, None),
        epochs=max_epochs,
        batch_size=batch_size, 
        shuffle=shuffle,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    model.load_weights('best_model.weights.h5')

    # Print final training metrics
    print("\nFinal Training Metrics:")
    print(f"Best validation reconstruction loss: {min(history.history['val_reconstruction_loss']):.6f}")
    print(f"Best validation KL loss: {min(history.history['val_kl_loss']):.6f}")
    
    return history


def main():
    """
    Main function to handle VAE model training pipeline.
    Handles model creation, training, and saving.

    Example usage:
    python train.py --input-data /path/to/data.npy --latent-dim 16 --epochs 500 --save
    """

    parser = argparse.ArgumentParser(description='Train VAE model on spectroscopic data')
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--input-data', type=str, required=True,
                           help='Path to input data array (.npy file)')
    
    # Model configuration
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--latent-dim', type=int, default=16,
                           help='Dimension of latent space')
    model_group.add_argument('--learning-rate', type=float, default=1e-4,
                           help='Learning rate for Adam optimizer')
    
    # Training parameters
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--epochs', type=int, default=500,
                           help='Number of training epochs')
    train_group.add_argument('--batch-size', type=int, default=64,
                           help='Training batch size')
    train_group.add_argument('--validation-split', type=float, default=0.1,
                           help='Fraction of data for validation')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--save-models', action='store_true',
                           help='Save trained models to disk')
    output_group.add_argument('--save-dir', type=str, default='../models',
                           help='Directory to save model files')
    output_group.add_argument('--model-name', type=str, default='vae_model',
                           help='Base name for saved model files')
    output_group.add_argument('--plot-training', action='store_true',
                           help='Plot training history')
    
    args = parser.parse_args()
    
    # Load input data
    try:
        print("Loading input data...")
        data = np.load(args.input_data)
        input_dim = get_input_dim(data)
        print(f"Loaded data with shape: {data.shape}, input dimension: {input_dim}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create and train VAE model
    try:
        print("\nCreating VAE model...")
        vae_model = create_vae_model(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate
        )
        
        print("\nStarting model training...")
        history = train_vae_model(
            model=vae_model,
            data=data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        
        print("\nTraining completed successfully")
        
    except Exception as e:
        print(f"Error during model creation/training: {e}")
        return None
    
    # Save models if requested
    if args.save_models:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            base_name = f"{args.model_name}_dim{args.latent_dim}"
            
            # Save encoder
            encoder_path = os.path.join(args.save_dir, f"{base_name}_encoder.keras")
            print(f"Saving encoder to: {encoder_path}")
            vae_model.encoder.save(encoder_path)
            
            # Save decoder
            decoder_path = os.path.join(args.save_dir, f"{base_name}_decoder.keras")
            print(f"Saving decoder to: {decoder_path}")
            vae_model.decoder.save(decoder_path)
            
            # Save weights
            weights_path = os.path.join(args.save_dir, f"{base_name}_weights.h5")
            print(f"Saving weights to: {weights_path}")
            vae_model.save_weights(weights_path)
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    # Plot training history if requested
    if args.plot_training:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Total Loss')
            plt.plot(history.history['reconstruction_loss'], 
                    label='Reconstruction Loss')
            plt.plot(history.history['kl_loss'], label='KL Loss')
            
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'VAE Training History')
            
            if args.save_models:
                plot_path = os.path.join(args.save_dir, 
                                       f"{args.model_name}_training_history.png")
                plt.savefig(plot_path, dpi=300)
                print(f"Saved training history plot to: {plot_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    return vae_model, history

if __name__ == "__main__":
    main()