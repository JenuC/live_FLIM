"""
Phasor computation module for FLIM data processing.

This module provides functionality to compute phasor coordinates (g, s) from
photon count data using the Fourier transform method via flimlib.
"""

import numpy as np
import flimlib
from flim_processing._dataclasses import FlimParams


class PhasorComputer:
    """Computes phasor coordinates from photon count data.
    
    The phasor approach represents fluorescence lifetime data in the frequency
    domain using phasor coordinates (g, s), which are computed via Fourier
    transform of the fluorescence decay curve.
    """
    
    @staticmethod
    def compute_phasor(photon_count: np.ndarray, params: FlimParams) -> np.ndarray:
        """
        Compute phasor coordinates (g, s) for each pixel.
        
        This method uses the Fourier transform method to compute phasor
        coordinates from photon count histograms. The fit range parameters
        are applied to use only the specified portion of the decay curve.
        
        Args:
            photon_count: 3D array with shape (height, width, time_bins)
                containing photon count histograms for each pixel
            params: FLIM parameters including period and fit range
            
        Returns:
            Array of shape (height, width, 2) containing g and s coordinates.
            The last dimension contains [g, s] for each pixel. NaN values in
            the input are propagated to the output.
            
        Notes:
            - The fit_start and fit_end parameters determine which portion of
              the decay curve is used for computation
            - If fit_end exceeds the data size, it is adjusted to the available data
            - NaN values in photon_count result in NaN phasor coordinates
        """
        period = params.period
        
        # Adjust fit range to not exceed data size
        fstart = params.fit_start if params.fit_start < photon_count.shape[-1] else photon_count.shape[-1]
        fend = params.fit_end if params.fit_end <= photon_count.shape[-1] else photon_count.shape[-1]
        
        # Compute phasor using flimlib
        # GCI_Phasor returns an object with .u and .v attributes
        # u corresponds to s (imaginary part), v corresponds to g (real part)
        phasor = flimlib.GCI_Phasor(
            period,
            photon_count,
            fit_start=fstart,
            fit_end=fend,
            compute_fitted=False,
            compute_residuals=False,
            compute_chisq=False
        )
        
        # Reshape to (height, width, 2) with [g, s] in the last dimension
        # Note: flimlib returns v (g) and u (s)
        result = np.dstack([phasor.v, phasor.u])
        
        return result
