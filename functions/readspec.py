### Functions to read in JADES spectra and deredshift them

import numpy as np
import specutils as spec 
import astropy.units as u
from astropy.io import fits 
from astropy.nddata import StdDevUncertainty


def read_spectrum(specfile, z=0, micron=False):
    """
    Inputs: filename, redshift (optional)
    If no redshift is supplied, it is not added to the specutils object

    Read 1D spectrum using specutils

    """
    f = fits.open(specfile)
    
    if micron:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.micron)
    else:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.angstrom)

    ### Remove nan
    flux_val = f['DATA'].data 
    flux_val = np.nan_to_num(flux_val)
    flux_watt = flux_val * u.watt/u.m**2/u.m
    flux = flux_watt.to(u.erg/u.s/u.cm**2/u.AA)

    unc_val = f['ERR'].data
    unc_val = np.nan_to_num(unc_val)
    unc_watt = unc_val * u.watt/u.m**2/u.m
    unc = unc_watt.to(u.erg/u.s/u.cm**2/u.AA)

    ### First create a specutils 1D spectrum
    if z==0:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc))  

    else:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc), 
                                        redshift=z)  
    
    return(specutil_spec)


def deredshift_spec(spectrum, z):
    """
    Input: specutils spectrum object

    De-redshift the spectrum, with spectroscopic redshift info that we previously added to the object
    
    """

    rest_lamb = spectrum.spectral_axis/(1+z)
    rest_flux = spectrum.flux*(1+z)
    rest_unc = spectrum.uncertainty.array*(1+z) * u.erg/u.s/u.cm**2/u.AA
    
    rest_specutil_spec = spec.Spectrum1D(spectral_axis=rest_lamb, flux=rest_flux, 
                                         uncertainty=StdDevUncertainty(rest_unc))

    return rest_specutil_spec