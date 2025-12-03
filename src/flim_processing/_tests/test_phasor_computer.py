"""
Unit tests for PhasorComputer class.
"""

import pytest
import numpy as np
from flim_processing import PhasorComputer, FlimParams


class TestPhasorComputer:
    """Tests for PhasorComputer."""
    
    def test_compute_phasor_returns_correct_shape(self):
        """Test that phasor computation returns correct output shape."""
        # Create synthetic photon count data
        photon_count = np.random.poisson(100, size=(10, 10, 64)).astype(np.float32)
        params = FlimParams(period=0.04, fit_start=0, fit_end=64)
        
        # Compute phasor
        phasor = PhasorComputer.compute_phasor(photon_count, params)
        
        # Check shape
        assert phasor.shape == (10, 10, 2), f"Expected shape (10, 10, 2), got {phasor.shape}"
    
    def test_compute_phasor_with_different_sizes(self):
        """Test phasor computation with various input sizes."""
        sizes = [(5, 5, 32), (20, 30, 128), (1, 1, 256)]
        
        for height, width, time_bins in sizes:
            photon_count = np.random.poisson(50, size=(height, width, time_bins)).astype(np.float32)
            params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
            
            phasor = PhasorComputer.compute_phasor(photon_count, params)
            
            assert phasor.shape == (height, width, 2), \
                f"For input {(height, width, time_bins)}, expected shape {(height, width, 2)}, got {phasor.shape}"
    
    def test_compute_phasor_with_fit_range(self):
        """Test that fit range parameters are applied."""
        photon_count = np.random.poisson(100, size=(10, 10, 128)).astype(np.float32)
        
        # Use only middle portion of decay curve
        params = FlimParams(period=0.04, fit_start=20, fit_end=100)
        
        phasor = PhasorComputer.compute_phasor(photon_count, params)
        
        # Should still return correct shape
        assert phasor.shape == (10, 10, 2)
        # Should contain valid values (not all NaN)
        assert not np.all(np.isnan(phasor))
    
    def test_compute_phasor_handles_nan_input(self):
        """Test that NaN values in input are propagated to output."""
        photon_count = np.random.poisson(100, size=(10, 10, 64)).astype(np.float32)
        
        # Set some pixels to NaN
        photon_count[0, 0, :] = np.nan
        photon_count[5, 5, :] = np.nan
        
        params = FlimParams(period=0.04, fit_start=0, fit_end=64)
        phasor = PhasorComputer.compute_phasor(photon_count, params)
        
        # Check that NaN pixels have NaN phasor values
        assert np.all(np.isnan(phasor[0, 0, :])), "Expected NaN at pixel (0, 0)"
        assert np.all(np.isnan(phasor[5, 5, :])), "Expected NaN at pixel (5, 5)"
        
        # Check that other pixels have valid values
        assert not np.all(np.isnan(phasor[1, 1, :])), "Expected valid values at pixel (1, 1)"
    
    def test_compute_phasor_with_fit_end_exceeding_data(self):
        """Test that fit_end is adjusted when it exceeds data size."""
        photon_count = np.random.poisson(100, size=(10, 10, 64)).astype(np.float32)
        
        # Set fit_end beyond data size
        params = FlimParams(period=0.04, fit_start=0, fit_end=128)
        
        # Should not raise an error
        phasor = PhasorComputer.compute_phasor(photon_count, params)
        
        assert phasor.shape == (10, 10, 2)
        assert not np.all(np.isnan(phasor))
    
    def test_compute_phasor_values_in_valid_range(self):
        """Test that phasor values are in reasonable range."""
        # Create synthetic decay with known characteristics
        time_bins = 64
        photon_count = np.zeros((5, 5, time_bins), dtype=np.float32)
        
        # Create exponential decay
        t = np.arange(time_bins)
        decay = 1000 * np.exp(-t / 10.0)
        photon_count[:, :, :] = decay
        
        params = FlimParams(period=0.04, fit_start=0, fit_end=time_bins)
        phasor = PhasorComputer.compute_phasor(photon_count, params)
        
        # Phasor g values should typically be in [0, 1]
        # Phasor s values should typically be in [-0.5, 0.5] or similar range
        # Just check they're not wildly out of range
        assert np.all(np.abs(phasor[:, :, 0]) < 10), "g values seem out of range"
        assert np.all(np.abs(phasor[:, :, 1]) < 10), "s values seem out of range"
    
    def test_compute_phasor_different_periods(self):
        """Test that different period values produce different results."""
        photon_count = np.random.poisson(100, size=(5, 5, 64)).astype(np.float32)
        
        params1 = FlimParams(period=0.04, fit_start=0, fit_end=64)
        params2 = FlimParams(period=0.08, fit_start=0, fit_end=64)
        
        phasor1 = PhasorComputer.compute_phasor(photon_count, params1)
        phasor2 = PhasorComputer.compute_phasor(photon_count, params2)
        
        # Different periods should produce different results (unless data is degenerate)
        # Check that at least some values are different
        assert not np.allclose(phasor1, phasor2, equal_nan=True), \
            "Different periods should produce different phasor values"
