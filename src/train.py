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
# from tensorflow.keras import layers, Model

class ConvergenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=50, min_delta=2.0, min_epochs=100):
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

        if val_loss is None:
            # Fallback to val_loss if val_reconstruction_loss not available
            val_loss = logs.get('val_loss')

        if val_loss is None:
            print("Warning: No validation loss found to monitor")
            return
        
        # Print debugging information occasionally
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}")
            print(f"Current val loss: {val_loss:.6f}")
            print(f"Best val loss: {self.best_val_loss:.6f}")
            print(f"Wait counter: {self.wait}")
        
        # Don't check convergence until minimum epochs reached
        if epoch < self.min_epochs:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            return
            
        if val_loss < (self.best_val_loss - self.min_delta):
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
        model: Compiled VAE model
        data (np.ndarray): Training data
        validation_split: Fraction of training data used for validation
        max_epochs (int): Maximum number of epochs to train
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

    # Create convergence callback instance
    convergence = ConvergenceCallback(
        patience=50,
        min_delta=2,
        min_epochs=150
    )

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
        ),

        # Add the convergence callback here
        convergence
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

    # Check convergence status after training
    if convergence.converged:
        print(f"\n✓ Model converged after {convergence.stopped_epoch + 1} epochs")
        print(f"Best validation loss: {convergence.best_val_loss:.6f}")
    else:
        print(f"\n⚠ Model reached maximum epochs ({max_epochs}) without convergence")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
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
    python train.py --input-data /path/to/data.npy --validation-split 0.2 --epochs 1000 --save-models --plot-training
    """

    parser = argparse.ArgumentParser(description='Train VAE model on spectroscopic data')
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--input-data', type=str, required=True,
                           help='Path to input data array (.npy file)')
    data_group.add_argument('--validation-split', type=float, default=0.2,
                           help='Fraction of data to use for validation (default: 0.2)')
    
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
    train_group.add_argument('--early-stopping-patience', type=int, default=50,
                           help='Patience for early stopping')
    
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
            validation_split=args.validation_split,
            max_epochs=args.epochs,
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

            # Save training history
            history_path = os.path.join(args.save_dir, f"{base_name}_history.npz")
            np.savez(history_path, 
                    loss=history.history['loss'],
                    val_loss=history.history['val_loss'],
                    reconstruction_loss=history.history['reconstruction_loss'],
                    val_reconstruction_loss=history.history['val_reconstruction_loss'],
                    kl_loss=history.history['kl_loss'],
                    val_kl_loss=history.history['val_kl_loss'])
            print(f"Saved training history to: {history_path}")
            
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