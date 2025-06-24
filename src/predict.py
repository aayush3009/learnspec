import argparse
import os
import numpy as np
import tensorflow as tf


def evaluate_latent_space(encoder, data):
    """
    Get the latent space representation of input data.
    
    Args:
        encoder: Trained encoder model
        data (numpy.ndarray): Input data
        
    Returns:
        tuple: (z_mean, z_log_var, z) latent space representations
    """
    z_mean, z_log_var, z = encoder.predict(data)
    return z_mean, z_log_var, z

def reconstruct_spectra(decoder, latent_vectors):
    """
    Reconstruct spectra from latent space vectors.
    
    Args:
        decoder: Trained decoder model
        latent_vectors (numpy.ndarray): Latent space vectors
        
    Returns:
        numpy.ndarray: Reconstructed spectra
    """
    return decoder.predict(latent_vectors)

def calculate_reconstruction_error(input, reconstructed):
    """
    Calculate the reconstruction error between original and reconstructed spectra.
    
    Args:
        original (numpy.ndarray): Original spectra
        reconstructed (numpy.ndarray): Reconstructed spectra
        
    Returns:
        numpy.ndarray: Reconstruction errors
    """
    return np.mean(np.abs(input - reconstructed), axis=1)

def main():
    """
    Main function to handle VAE model prediction pipeline.
    Loads trained models and performs inference on input data.
    Example usage:
    python predict.py \
    --encoder-path models/encoder.keras \
    --decoder-path models/decoder.keras \
    --input-data data/test_data.npy \
    --plot-results
    """

    
    parser = argparse.ArgumentParser(description='Run inference with trained VAE model')
    
    # Model loading options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--encoder-path', type=str, required=True,
                           help='Path to saved encoder model')
    model_group.add_argument('--decoder-path', type=str, required=True,
                           help='Path to saved decoder model')
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--input-data', type=str, required=True,
                           help='Path to input data for prediction (.npy file)')
    data_group.add_argument('--batch-size', type=int, default=32,
                           help='Batch size for prediction')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--save-dir', type=str, default='../results',
                           help='Directory to save prediction results')
    output_group.add_argument('--output-prefix', type=str, default='vae_predictions',
                           help='Prefix for output files')
    output_group.add_argument('--plot-results', action='store_true',
                           help='Plot latent space and reconstructions')
    output_group.add_argument('--plot-samples', type=int, default=5,
                           help='Number of random samples to plot')
    
    args = parser.parse_args()
    
    # Load models
    try:
        print("Loading trained models...")
        encoder = tf.keras.models.load_model(args.encoder_path)
        decoder = tf.keras.models.load_model(args.decoder_path)
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    # Load input data
    try:
        print("Loading input data...")
        data = np.load(args.input_data)
        print(f"Loaded data with shape: {data.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get latent space representation
    try:
        print("Evaluating latent space...")
        z_mean, z_log_var, z = evaluate_latent_space(encoder, data)
        
        # Save latent representations
        latent_path = os.path.join(args.save_dir, f"{args.output_prefix}_latent.npz")
        np.savez(latent_path, 
                 z_mean=z_mean, 
                 z_log_var=z_log_var, 
                 z=z)
        print(f"Saved latent representations to {latent_path}")
        
    except Exception as e:
        print(f"Error during latent space evaluation: {e}")
        return None
    
    # Reconstruct input data
    try:
        print("Reconstructing input data...")
        reconstructed_data = reconstruct_spectra(decoder, z_mean)
        
        # Save reconstructions
        recon_path = os.path.join(args.save_dir, f"{args.output_prefix}_reconstructed.npy")
        np.save(recon_path, reconstructed_data)
        print(f"Saved reconstructions to {recon_path}")
        
        # Calculate reconstruction errors
        errors = calculate_reconstruction_error(data, reconstructed_data)
        error_path = os.path.join(args.save_dir, f"{args.output_prefix}_errors.npy")
        np.save(error_path, errors)
        print(f"Saved reconstruction errors to {error_path}")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return None
    
    # Plot results if requested
    if args.plot_results:
        try:
            import matplotlib.pyplot as plt
            
            # Plot latent space (first two dimensions)
            plt.figure(figsize=(10, 8))
            plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title('Latent Space Representation')
            latent_plot_path = os.path.join(args.save_dir, f"{args.output_prefix}_latent_space.png")
            plt.savefig(latent_plot_path, dpi=300)
            plt.close()
            
            # Plot random sample reconstructions
            n_samples = min(args.plot_samples, len(data))
            sample_idx = np.random.choice(len(data), n_samples, replace=False)
            
            plt.figure(figsize=(15, 3*n_samples))
            for i, idx in enumerate(sample_idx):
                plt.subplot(n_samples, 1, i+1)
                plt.plot(data[idx], label='Original', alpha=0.8)
                plt.plot(reconstructed_data[idx], label='Reconstructed', alpha=0.8)
                plt.title(f'Sample {idx} (Error: {errors[idx]:.4f})')
                plt.legend()
            
            plt.tight_layout()
            recon_plot_path = os.path.join(args.save_dir, f"{args.output_prefix}_reconstructions.png")
            plt.savefig(recon_plot_path, dpi=300)
            plt.close()
            
            print(f"Saved plots to {args.save_dir}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    return z_mean, z_log_var, z, reconstructed_data, errors

if __name__ == "__main__":
    main()