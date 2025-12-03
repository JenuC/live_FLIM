"""
Fitting engine module for FLIM data processing.

This module provides functionality to perform curve fitting on fluorescence
decay data using RLD (Rapid Lifetime Determination) and LMA (Levenberg-Marquardt)
algorithms via flimlib.
"""

import numpy as np
import flimlib
from flim_processing._dataclasses import FlimParams, RLDResult, LMAResult


class FittingEngine:
    """Performs fluorescence decay curve fitting.
    
    This class provides static methods for fitting fluorescence decay curves
    using two complementary algorithms:
    - RLD (Rapid Lifetime Determination): Fast triple integral method
    - LMA (Levenberg-Marquardt): Iterative refinement using RLD as initialization
    """
    
    @staticmethod
    def compute_rld(photon_count: np.ndarray, params: FlimParams) -> RLDResult:
        """
        Compute Rapid Lifetime Determination using triple integral method.
        
        RLD is a fast, non-iterative method for estimating fluorescence lifetime
        based on the triple integral of the decay curve. It provides good initial
        estimates for more sophisticated fitting algorithms.
        
        Args:
            photon_count: Photon count array. Can be:
                - 1D array (time_bins,) for single decay curve
                - 3D array (height, width, time_bins) for image data
            params: FLIM parameters including period and fit range
            
        Returns:
            RLDResult containing:
                - tau: Lifetime values (same shape as input without time dimension)
                - chisq: Chi-squared goodness-of-fit values
                - Z: Background parameter
                - A: Amplitude parameter
                - fitted: Fitted decay curves (optional)
                
        Notes:
            - If fit_end exceeds data size, it is adjusted to available data
            - NaN values in input propagate to output
            - The fit range [fit_start, fit_end) determines which portion
              of the decay curve is used for fitting
        """
        period = params.period
        
        # Adjust fit range to not exceed data size
        time_bins = photon_count.shape[-1]
        fit_start = params.fit_start if params.fit_start < time_bins else time_bins
        fit_end = params.fit_end if params.fit_end <= time_bins else time_bins
        
        # Call flimlib RLD fitting
        # GCI_triple_integral_fitting_engine performs RLD fitting
        result = flimlib.GCI_triple_integral_fitting_engine(
            period,
            photon_count,
            fit_start=fit_start,
            fit_end=fit_end,
            compute_fitted=False,
            compute_residuals=False,
            compute_chisq=True
        )
        
        # Extract results from flimlib result object
        # The result object has attributes: Z, A, tau, chisq
        return RLDResult(
            tau=result.tau,
            chisq=result.chisq,
            Z=result.Z,
            A=result.A,
            fitted=None
        )
    
    @staticmethod
    def compute_lma(photon_count: np.ndarray, params: FlimParams,
                    initial_params: np.ndarray = None) -> LMAResult:
        """
        Compute Levenberg-Marquardt fitting.
        
        LMA is an iterative optimization algorithm that refines lifetime estimates.
        It typically uses RLD results as initial parameter estimates for faster
        convergence and better results.
        
        Args:
            photon_count: Photon count array. Can be:
                - 1D array (time_bins,) for single decay curve
                - 3D array (height, width, time_bins) for image data
            params: FLIM parameters including period and fit range
            initial_params: Initial parameter estimates [Z, A, tau].
                If None, RLD is computed automatically for initialization.
                Shape should match the spatial dimensions of photon_count.
                
        Returns:
            LMAResult containing:
                - param: Refined parameters array with shape (..., 3) for [Z, A, tau]
                - chisq: Chi-squared goodness-of-fit values
                - fitted: Fitted decay curves (optional)
                
        Notes:
            - If initial_params is None, RLD is computed first
            - If fit_end exceeds data size, it is adjusted to available data
            - NaN values in input propagate to output
            - The fit range [fit_start, fit_end) determines which portion
              of the decay curve is used for fitting
        """
        period = params.period
        
        # Adjust fit range to not exceed data size
        time_bins = photon_count.shape[-1]
        fit_start = params.fit_start if params.fit_start < time_bins else time_bins
        fit_end = params.fit_end if params.fit_end <= time_bins else time_bins
        
        # If no initial parameters provided, compute RLD first
        if initial_params is None:
            rld_result = FittingEngine.compute_rld(photon_count, params)
            # Stack Z, A, tau into initial parameter array
            # Shape will be (..., 3) where ... matches spatial dimensions
            initial_params = np.stack([rld_result.Z, rld_result.A, rld_result.tau], axis=-1)
        
        # Call flimlib LMA fitting
        # GCI_marquardt_fitting_engine performs LMA fitting
        result = flimlib.GCI_marquardt_fitting_engine(
            period,
            photon_count,
            fit_start=fit_start,
            fit_end=fit_end,
            compute_fitted=False,
            compute_residuals=False,
            compute_chisq=True,
            param=initial_params
        )
        
        # Extract results from flimlib result object
        # The result object has attributes: param (shape (..., 3)), chisq
        return LMAResult(
            param=result.param,
            chisq=result.chisq,
            fitted=None
        )

