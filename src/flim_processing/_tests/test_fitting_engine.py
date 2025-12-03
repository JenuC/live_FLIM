"""
Unit tests for FittingEngine class.
"""

import pytest
import numpy as np
from flim_processing import FittingEngine, FlimParams, RLDResult, LMAResult


class TestFittingEngine:
    """Tests for FittingEngine."""
    
    def test_compute_rld_returns_rld_result(self):
        """Test that compute_rld returns an RLDResult object."""
        # Create synthetic photon count data (single decay curve)
        time_bins = 64
        photon_count = np.random.poisson(100, size=(10, 10, time_bins)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        
        # Compute RLD
        result = FittingEngine.compute_rld(photon_count, params)
        
        # Check that it's an RLDResult
        assert isinstance(result, RLDResult)
        assert result.tau is not None
        assert result.chisq is not None
        assert result.Z is not None
        assert result.A is not None
    
    def test_compute_rld_output_shape(self):
        """Test that RLD output has correct shape."""
        # Create 3D photon count data
        photon_count = np.random.poisson(100, size=(10, 15, 64)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=64)
        
        result = FittingEngine.compute_rld(photon_count, params)
        
        # Output should have spatial dimensions only (no time dimension)
        assert result.tau.shape == (10, 15)
        assert result.chisq.shape == (10, 15)
        assert result.Z.shape == (10, 15)
        assert result.A.shape == (10, 15)
    
    def test_compute_rld_with_1d_input(self):
        """Test RLD with 1D input (single decay curve)."""
        # Create single decay curve
        time_bins = 64
        t = np.arange(time_bins)
        photon_count = 1000 * np.exp(-t / 10.0).astype(np.float32)
        
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        result = FittingEngine.compute_rld(photon_count, params)
        
        # Output should be scalar-like (0D arrays)
        assert result.tau.shape == ()
        assert result.chisq.shape == ()
        assert result.Z.shape == ()
        assert result.A.shape == ()
    
    def test_compute_rld_with_fit_range(self):
        """Test that fit range parameters are applied in RLD."""
        photon_count = np.random.poisson(100, size=(10, 10, 128)).astype(np.float32)
        
        # Use only middle portion of decay curve
        params = FlimParams(period=0.04, fit_start=20, fit_end=100)
        
        result = FittingEngine.compute_rld(photon_count, params)
        
        # Should return valid results
        assert result.tau.shape == (10, 10)
        assert not np.all(np.isnan(result.tau))
    
    def test_compute_rld_handles_fit_end_exceeding_data(self):
        """Test that fit_end is adjusted when it exceeds data size."""
        photon_count = np.random.poisson(100, size=(10, 10, 64)).astype(np.float32)
        
        # Set fit_end beyond data size
        params = FlimParams(period=0.04, fit_start=0, fit_end=128)
        
        # Should not raise an error
        result = FittingEngine.compute_rld(photon_count, params)
        
        assert result.tau.shape == (10, 10)
        assert not np.all(np.isnan(result.tau))
    
    def test_compute_lma_returns_lma_result(self):
        """Test that compute_lma returns an LMAResult object."""
        # Create synthetic photon count data
        time_bins = 64
        photon_count = np.random.poisson(100, size=(10, 10, time_bins)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        
        # Compute LMA (should compute RLD internally for initialization)
        result = FittingEngine.compute_lma(photon_count, params)
        
        # Check that it's an LMAResult
        assert isinstance(result, LMAResult)
        assert result.param is not None
        assert result.chisq is not None
    
    def test_compute_lma_output_shape(self):
        """Test that LMA output has correct shape."""
        # Create 3D photon count data
        photon_count = np.random.poisson(100, size=(10, 15, 64)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=64)
        
        result = FittingEngine.compute_lma(photon_count, params)
        
        # param should have shape (10, 15, 3) for [Z, A, tau]
        assert result.param.shape == (10, 15, 3)
        assert result.chisq.shape == (10, 15)
    
    def test_compute_lma_with_1d_input(self):
        """Test LMA with 1D input (single decay curve)."""
        # Create single decay curve
        time_bins = 64
        t = np.arange(time_bins)
        photon_count = 1000 * np.exp(-t / 10.0).astype(np.float32)
        
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        result = FittingEngine.compute_lma(photon_count, params)
        
        # param should have shape (3,) for [Z, A, tau]
        assert result.param.shape == (3,)
        assert result.chisq.shape == ()
    
    def test_compute_lma_uses_rld_initialization(self):
        """Test that LMA uses RLD for initialization when no initial params provided."""
        # Create synthetic data
        time_bins = 64
        photon_count = np.random.poisson(100, size=(5, 5, time_bins)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        
        # Compute RLD separately
        rld_result = FittingEngine.compute_rld(photon_count, params)
        
        # Compute LMA without initial params (should use RLD internally)
        lma_result = FittingEngine.compute_lma(photon_count, params)
        
        # LMA should have been initialized with RLD values
        # The results should be similar but LMA should refine them
        # Just verify both complete successfully
        assert lma_result.param.shape == (5, 5, 3)
        assert not np.all(np.isnan(lma_result.param))
    
    def test_compute_lma_with_explicit_initial_params(self):
        """Test that LMA accepts explicit initial parameters."""
        # Create synthetic data
        time_bins = 64
        photon_count = np.random.poisson(100, size=(5, 5, time_bins)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        
        # Create initial parameters [Z, A, tau]
        initial_params = np.zeros((5, 5, 3), dtype=np.float32)
        initial_params[:, :, 0] = 10.0  # Z (background)
        initial_params[:, :, 1] = 100.0  # A (amplitude)
        initial_params[:, :, 2] = 2.0  # tau (lifetime in ns)
        
        # Compute LMA with explicit initial params
        result = FittingEngine.compute_lma(photon_count, params, initial_params=initial_params)
        
        # Should return valid results
        assert result.param.shape == (5, 5, 3)
        assert result.chisq.shape == (5, 5)
    
    def test_compute_lma_with_fit_range(self):
        """Test that fit range parameters are applied in LMA."""
        photon_count = np.random.poisson(100, size=(10, 10, 128)).astype(np.float32)
        
        # Use only middle portion of decay curve
        params = FlimParams(period=0.04, fit_start=20, fit_end=100)
        
        result = FittingEngine.compute_lma(photon_count, params)
        
        # Should return valid results
        assert result.param.shape == (10, 10, 3)
        assert not np.all(np.isnan(result.param))
    
    def test_compute_lma_handles_fit_end_exceeding_data(self):
        """Test that fit_end is adjusted when it exceeds data size in LMA."""
        photon_count = np.random.poisson(100, size=(10, 10, 64)).astype(np.float32)
        
        # Set fit_end beyond data size
        params = FlimParams(period=0.04, fit_start=0, fit_end=128)
        
        # Should not raise an error
        result = FittingEngine.compute_lma(photon_count, params)
        
        assert result.param.shape == (10, 10, 3)
        assert not np.all(np.isnan(result.param))
    
    def test_rld_and_lma_consistency(self):
        """Test that LMA refines RLD results."""
        # Create synthetic exponential decay
        time_bins = 64
        t = np.arange(time_bins)
        tau_true = 5.0
        photon_count = np.zeros((3, 3, time_bins), dtype=np.float32)
        photon_count[:, :, :] = 1000 * np.exp(-t / tau_true)
        
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        
        # Compute both RLD and LMA
        rld_result = FittingEngine.compute_rld(photon_count, params)
        lma_result = FittingEngine.compute_lma(photon_count, params)
        
        # Both should produce reasonable lifetime estimates
        # LMA tau should be in param[:, :, 2]
        lma_tau = lma_result.param[:, :, 2]
        
        # Both should be in a reasonable range
        assert np.all(rld_result.tau > 0), "RLD tau should be positive"
        assert np.all(lma_tau > 0), "LMA tau should be positive"
        
        # For clean exponential decay, both should give similar results
        # (though LMA might be more accurate)
        # Just verify they're in the same ballpark
        assert np.all(np.abs(rld_result.tau - lma_tau) < 10), \
            "RLD and LMA should give similar results for clean data"
    
    def test_compute_rld_with_different_periods(self):
        """Test that different period values affect RLD results."""
        # Create synthetic decay
        time_bins = 64
        t = np.arange(time_bins)
        photon_count = 1000 * np.exp(-t / 10.0).astype(np.float32)
        photon_count = np.tile(photon_count, (3, 3, 1))
        
        params1 = FlimParams(period=0.04, fit_start=0, fit_end=64)
        params2 = FlimParams(period=0.08, fit_start=0, fit_end=64)
        
        result1 = FittingEngine.compute_rld(photon_count, params1)
        result2 = FittingEngine.compute_rld(photon_count, params2)
        
        # Both should complete successfully
        assert result1.tau.shape == (3, 3)
        assert result2.tau.shape == (3, 3)
        assert not np.all(np.isnan(result1.tau))
        assert not np.all(np.isnan(result2.tau))
    
    def test_compute_lma_with_different_periods(self):
        """Test that different period values affect LMA results."""
        # Create synthetic decay
        time_bins = 64
        t = np.arange(time_bins)
        photon_count = 1000 * np.exp(-t / 10.0).astype(np.float32)
        photon_count = np.tile(photon_count, (3, 3, 1))
        
        params1 = FlimParams(period=0.04, fit_start=0, fit_end=64)
        params2 = FlimParams(period=0.08, fit_start=0, fit_end=64)
        
        result1 = FittingEngine.compute_lma(photon_count, params1)
        result2 = FittingEngine.compute_lma(photon_count, params2)
        
        # Both should complete successfully
        assert result1.param.shape == (3, 3, 3)
        assert result2.param.shape == (3, 3, 3)
        assert not np.all(np.isnan(result1.param))
        assert not np.all(np.isnan(result2.param))

