import numpy as np

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

def calculate_reconstruction_error(original, reconstructed):
    """
    Calculate the reconstruction error between original and reconstructed spectra.
    
    Args:
        original (numpy.ndarray): Original spectra
        reconstructed (numpy.ndarray): Reconstructed spectra
        
    Returns:
        numpy.ndarray: Reconstruction errors
    """
    return np.mean(np.abs(original - reconstructed), axis=1)