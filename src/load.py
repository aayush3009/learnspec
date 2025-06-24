import numpy as np
import argparse
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

### Load in the pre-processed data
def load_data(data_dir="/Users/aayushsaxena/Desktop/Oxford/scripts/learnspec/data/", 
              data_path="dja_z4-16/dja_resampled_spectra_z4-16.npy", 
              wavelength_path="dja_z4-16/dja_resampled_wavelength_z4-16.npy", 
              redshift_path="dja_z4-16/dja_redshift_info_z4-16.npy", 
              speclist_path="dja_z4-16/dja_source_id_info_z4-16.npy"):
    ### Load resampled spectra
    resampled_data = np.load(data_dir+data_path)

    ### Load the wavelength grid
    wavelength = np.load(data_dir+wavelength_path)
    ### Load the redshift information
    redshifts = np.load(data_dir+redshift_path)

    ### Load the source IDs
    speclist = np.load(data_dir+speclist_path, allow_pickle=True)

    return resampled_data, wavelength, redshifts, speclist


def process_data(resampled_data, scaler=True, scale_factor=1., simple_scaling=False, simple_scaler=MinMaxScaler, remove_nan=True, piecewise_smoothing=True, simple_smoothing=False, save_transformed_data=False, output_data_path=None):    
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
        np.save(str(output_data_path), resampled_data)

    return resampled_data


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


def main():
    """
    Main function to allow command-line data loading.
    
    Example usage:
    python -m learnspec.src.load --datadir ../data --data data/resampled_data.npy --wavelength data/wavelength.npy --redshifts data/redshifts.npy --speclist data/speclist.npy 
    --scaler True --scale_factor 1.0 --simple_scaling False --simple_smoothing False --remove_nan True --piecewise_smoothing True --save_transformed_data False
    --output_data_path data/processed_data.npy
    """

    parser = argparse.ArgumentParser(description='Load spectroscopic data and preprocess it for training a model.')
    # Data input options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--datadir', type=str, default='../data',
                        help='Directory containing input data files')
    data_group.add_argument('--data', type=str, required=True,
                        help='Path to resampled spectral data (.npy file)')
    data_group.add_argument('--wavelength', type=str, required=True,
                        help='Path to wavelength array (.npy file)')
    data_group.add_argument('--redshifts', type=str, required=True,
                        help='Path to redshifts array (.npy file)')
    data_group.add_argument('--speclist', type=str, required=True,
                        help='Path to speclist array (.npy file)')
    
    # Data processing options
    processing_group = parser.add_argument_group('Data Processing Options')
    processing_group.add_argument('--scaler', type=bool, default=True, 
                        help='Apply arcsinh scaling to data')
    processing_group.add_argument('--scale_factor', type=float, default=1.0, 
                        help='Scale factor for arcsinh transformation')
    processing_group.add_argument('--simple_scaling', type=bool, default=False, 
                        help='Apply simple scaling (MinMaxScaler) to data')
    processing_group.add_argument('--simple_smoothing', type=bool, default=False,
                        help='Apply simple Gaussian smoothing to data')
    processing_group.add_argument('--remove_nan', type=bool, default=True, 
                        help='Remove NaN values from data')
    processing_group.add_argument('--piecewise_smoothing', type=bool, default=True, 
                        help='Apply piecewise smoothing to data')
    processing_group.add_argument('--save_transformed_data', type=bool, default=False, 
                        help='Save transformed data to file')
    processing_group.add_argument('--output_data_path', type=str, default=None,
                        help='Path to save the processed data (if save_transformed_data is True)')
    
    args = parser.parse_args()

    # Use dataset name as model name if not provided
    if args.model_name is None:
        args.model_name = args.dataset

    
    # Load data
    try:
        print(f"Loading dataset '{args.dataset}' from {args.data_dir}...")
        resampled_data, wavelength, redshifts, speclist = load_data(
            data_dir=args.data_dir,
            data_path=args.data,
            wavelength_path=args.wavelength,
            redshift_path=args.redshifts,
            speclist_path=args.speclist
        )
        
        input_dim = get_input_dim(resampled_data)
        print(f"Data loaded: {len(resampled_data)} spectra with {input_dim} features")
        print(f"Redshift range: {np.min(redshifts):.2f} to {np.max(redshifts):.2f}")
        
        # Check for problematic spectra
        neg_indices = detect_negative_spikes(resampled_data)
        if len(neg_indices) > 0:
            print(f"WARNING: {len(neg_indices)} spectra have significant negative values")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Process data
    try:
        print("Processing data...")
        resampled_data = process_data(
            resampled_data, 
            scaler=args.scaler, 
            scale_factor=args.scale_factor,
            simple_scaling=args.simple_scaling,
            simple_smoothing=False,  # Set to True if you want to apply Gaussian smoothing 
            remove_nan=True,  # Set to False if you want to keep NaN values
            piecewise_smoothing=True,  # Set to False if you don't want piecewise smoothing
            save_transformed_data=False,  # Set to True if you want to save the transformed data
            output_data_path=args.save_dir + '/processed_data.npy'  # Path to save the processed data
        )
        print(f"Data processed: {resampled_data.shape[0]} spectra with {resampled_data.shape[1]} features")
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Save processed data if required
    if args.save_transformed_data and args.output_data_path:
        try:
            np.save(args.output_data_path, resampled_data)
            print(f"Processed data saved to {args.output_data_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

    print("Data loading and processing completed successfully.")
    
    return resampled_data, wavelength, redshifts, speclist

if __name__ == "__main__":
    main()