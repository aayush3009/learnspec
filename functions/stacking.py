import numpy as np

import specutils as spec
from specutils.spectra import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from specutils.analysis import snr

from measure_spec import spec_analysis as ms

from astropy.io import fits
from astropy.table import Table

from astropy.stats import sigma_clipped_stats
from astropy.nddata import StdDevUncertainty
import astropy.units as u

import matplotlib.pyplot as plt

import mpdaf as mp
import astropy.units as u

from joblib import Parallel, delayed

def resample_spec(spectrum, wavelength_grid):
    fluxcon = FluxConservingResampler()
    resampled_spec = fluxcon(spectrum, wavelength_grid)
    
    return(resampled_spec)


def renormalize_spec(speclist, redshifts):    
    ### Define an empty array to append the de-redshifted spectra
	rest_spectra = []

	for i in range(len(speclist)):
		input_spectrum = speclist[i]
		z = redshifts[i]

		### Read spectrum
		spectrum = ms.read_spectrum(input_spectrum, z)
		### de-redshift
		rest_spectrum = ms.deredshift_spec(spectrum, z)
		### Normalize spectrum based on flux at 1500A
		spec_1500 = rest_spectrum[1450*u.AA:1550*u.AA].data
		flux_1500 = np.nanmean(spec_1500)
		rest_spectrum_normed = rest_spectrum/flux_1500
		rest_spectra.append(rest_spectrum_normed)

	return(rest_spectra)

def renormalize_spec(input_spectrum, z):    	
	### Read spectrum
	spectrum = ms.read_spectrum(input_spectrum, z)
	### de-redshift
	rest_spectrum = ms.deredshift_spec(spectrum, z)
	### Normalize spectrum based on flux at 1500A
	spec_1500 = rest_spectrum[1450*u.AA:1550*u.AA].data
	flux_1500 = np.nanmean(spec_1500)
	rest_spectrum_normed = rest_spectrum/flux_1500

	return(rest_spectrum_normed)

def renormalize_spec_gratings(speclist, redshifts, grating='g140m'):
	### Define an empty array to append the de-redshifted spectra
	rest_spectra = []
	weights = []

	for i in range(len(speclist)):
		filename = speclist[i]
		z = redshifts[i]
		
		### Read spectrum
		spectrum = ms.read_spectrum(filename, z)
		### de-redshift
		rest_spectrum = ms.deredshift_spec(spectrum, z)

		if grating == 'g140m':
		### Normalize spectrum based on flux at 1500A
			spec_1500 = rest_spectrum[1450*u.AA:1550*u.AA].data
			flux_1500 = np.nanmedian(spec_1500)
			if flux_1500 < 0.:
				flux_1500 = flux_1500 + abs(np.nanstd(spec_1500))
			rest_spectrum_normed = rest_spectrum/flux_1500
			rest_spectra.append(rest_spectrum_normed)

		elif grating == 'g395m':
			### Normalize spectrum based on flux at 4800A
			spec_4800 = rest_spectrum[4780*u.AA:4820*u.AA].data
			flux_4800 = np.nanmedian(spec_4800)
			if flux_4800 < 0.:
				flux_4800 = flux_4800 + abs(np.nanstd(spec_4800))
			rest_spectrum_normed = rest_spectrum/flux_4800
			rest_spectra.append(rest_spectrum_normed)

	return(rest_spectra)

def deredshifted_stacking(wavegrid, rest_fluxes):
	# Assign the wavelength grid from the first spectrum, assuming that the input spectra are all resampled 
	stack_wavegrid = wavegrid * u.AA

	# Calculate sigma clipped statistics
	dr_mean_sc_data = np.nanmean(rest_fluxes, axis=0)
	dr_median_sc_data = np.nanmedian(rest_fluxes, axis=0)
	dr_std_sc_data = np.nanstd(rest_fluxes, axis=0)

	# Create Spectrum1D objects for all three stacks
	dr_mean_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=dr_mean_sc_data, uncertainty=StdDevUncertainty(dr_std_sc_data))
	dr_median_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=dr_median_sc_data, uncertainty=StdDevUncertainty(dr_std_sc_data))
	dr_std_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=dr_std_sc_data)

	return(dr_mean_sc_stack, dr_median_sc_stack, dr_std_sc_stack)

### Simplest stacking, given a bunch of deredshifted spectra, just take the mean along the axis
def simple_stack(resampled_spectra):
    ### Mean/median combine
    mean_stacked_spectrum = np.mean(resampled_spectra, axis=0)
    return mean_stacked_spectrum



def mean_stacking(speclist, redshifts, resolution=3.):
	### Define an empty array to append the de-redshifted and normalized spectra
	rest_spectra, weights = renormalize_spec(speclist, redshifts)

	### Create a uniform wavelength grid
	### Min and max wavelengths of NIRSpec PRISM
	wave_min = 7000.
	wave_max = 55000.

	### Median redshift will be used to pick the start and end wavelengths of the stacked spectrum
	median_redshift = np.median(redshifts)
	
	res = resolution  ### Size of the wavelength element in the stacked spectrum
	stack_wavegrid = np.arange(wave_min/(1+median_redshift), wave_max/(1+median_redshift), res)*u.AA
	
	# Resample the spectra
	resampled_spec = Parallel(n_jobs=-1)(delayed(resample_spec)(source, stack_wavegrid) for source in rest_spectra)

	### Mean/median combine
	mean_stack = np.average(resampled_spec, weights=weights)
	# err_stack = np.std(resampled_spec, axis=0)    

	return(mean_stack, resampled_spec, weights)


def sigma_clipped_stacking(speclist, redshifts, resolution=3.0, sigma=20.0):
	# Renormalize spectra
	rest_spectra = Parallel(n_jobs=-1)(delayed(renormalize_spec)(speclist[i], redshifts[i]) for i in range(len(speclist)))

	### Create a uniform wavelength grid
	### Min and max wavelengths of NIRSpec PRISM
	wave_min = 7000.
	wave_max = 55000.

	### Median redshift will be used to pick the start and end wavelengths of the stacked spectrum
	median_redshift = np.median(redshifts)
	
	res = resolution  ### Size of the wavelength element in the stacked spectrum
	stack_wavegrid = np.arange(wave_min/(1+median_redshift), wave_max/(1+median_redshift), res)*u.AA

	# Resample the spectra
	resampled_spec = Parallel(n_jobs=-1)(delayed(resample_spec)(source, stack_wavegrid) for source in rest_spectra)

	# Mask the nans and infs, then append to a new array
	masked_resampled_spec = Parallel(n_jobs=-1)(delayed(np.nan_to_num)(input_spec.data, nan=0.0, posinf=0.0, neginf=0.0) for input_spec in resampled_spec)

	# Convert masked_resampled_spec to float32 for speeding up
	masked_resampled_spec = np.array(masked_resampled_spec, dtype=np.float32)

	# Now implement sigma clipped statistics 
	mean_sc_data, median_sc_data, std_sc_data = sigma_clipped_stats(masked_resampled_spec, mask_value=0.0, sigma=sigma, axis=0)
	# Create Spectrum1D objects for all three stacks
	mean_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=mean_sc_data, uncertainty=StdDevUncertainty(std_sc_data))
	median_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=median_sc_data, uncertainty=StdDevUncertainty(std_sc_data))
	std_sc_stack = Spectrum1D(spectral_axis=stack_wavegrid, data=std_sc_data)

	return(mean_sc_stack, median_sc_stack, std_sc_stack, resampled_spec)


def stack_grating(speclist, redshifts, grating='g395m', resolution=1.0):
	### Define an empty array to append the de-redshifted spectra
	rest_spectra = renormalize_spec_gratings(speclist, redshifts, grating)

	### Create a uniform wavelength grid
	### Min and max wavelengths of NIRSpec PRISM
	if grating == 'prism':
		wave_min = 7000.
		wave_max = 55000.
	elif grating == 'g395m':
		wave_min = 28700.
		wave_max = 51000.
	elif grating == 'g235m':
		wave_min = 16600.
		wave_max = 30700.
	elif grating == 'g140m':
		wave_min = 8000.
		wave_max = 16000.
	else:
		print("No grating supplied, defaulting to G140M.")
		wave_min = 8000.
		wave_max = 16000.

	### Median redshift will be used to pick the start and end wavelengths of the stacked spectrum
	median_redshift = np.median(redshifts)
	
	res = resolution  ### Size of the wavelength element in the stacked spectrum
	stack_wavegrid = np.arange(wave_min/(1+median_redshift), wave_max/(1+median_redshift), res)*u.AA

	# Resample the spectra
	resampled_spec = Parallel(n_jobs=-1)(delayed(resample_spec)(source, stack_wavegrid) for source in rest_spectra)

	# Mask the nans and infs, then append to a new array
	masked_resampled_spec = Parallel(n_jobs=-1)(delayed(np.nan_to_num)(input_spec.data, nan=0.0, posinf=0.0, neginf=0.0) for input_spec in resampled_spec)

	# Convert masked_resampled_spec to float32 for speeding up
	masked_resampled_spec = np.array(masked_resampled_spec, dtype=np.float32)

	# Sigma clipped stats stack
	# mean_stack_data, median_stack_data, std_stack_data = sigma_clipped_stats(masked_resampled_spec, mask_value=0.0, sigma=sigma, axis=1)
	mean_stack_data = np.nanmean(masked_resampled_spec, axis=0)
	median_stack_data = np.nanmedian(masked_resampled_spec, axis=0)
	std_stack_data = np.nanstd(masked_resampled_spec, axis=0)

	# Create specutils Spectrum1D object
	mean_stack_g395m = Spectrum1D(spectral_axis=stack_wavegrid, data=mean_stack_data)
	median_stack_g395m = Spectrum1D(spectral_axis=stack_wavegrid, data=median_stack_data)
	std_stack_g395m = Spectrum1D(spectral_axis=stack_wavegrid, data=std_stack_data)
	
	return(mean_stack_g395m, median_stack_g395m, std_stack_g395m, resampled_spec)


### Function to plot the stacked spectra
def plot_stack(spectrum):
    fig = plt.figure(figsize=(8, 6))
    plt.step(spectrum.wavelength, spectrum.data, label="Stacked spectrum")
    plt.xlabel("Wavelength (microns)")
    plt.ylabel(r"F_$\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
    plt.legend()
    plt.show()