"""
Phasor computation module for FLIM data processing.

This module provides functionality to compute phasor coordinates (g, s) from
photon count data using the Fourier transform method via flimlib.
"""

import numpy as np
import flimlib
from scipy.spatial import KDTree
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
    
    @staticmethod
    def build_kdtree(phasor: np.ndarray, scale: float = 1000.0) -> KDTree:
        """
        Build KD-tree for efficient phasor space queries.
        
        This method constructs a KD-tree spatial index from phasor coordinates
        to enable fast nearest-neighbor queries in phasor space. NaN values
        are replaced with infinity to exclude them from queries.
        
        Args:
            phasor: Phasor array with shape (height, width, 2) containing
                g and s coordinates
            scale: Scaling factor applied to phasor coordinates before
                building the tree (default: 1000.0)
            
        Returns:
            KDTree object for spatial queries in phasor space
            
        Notes:
            - Phasor coordinates are scaled by the scale factor before building
              the tree to improve numerical precision
            - NaN values are replaced with infinity, effectively excluding them
              from query results
            - The tree uses the infinity norm (Chebyshev distance) for queries
        """
        # Get the original shape
        height, width = phasor.shape[:2]
        
        # Reshape to (height * width, 2) for KDTree construction
        phasor_flat = phasor.reshape(-1, 2)
        
        # Replace NaN values with infinity
        # This ensures NaN pixels are excluded from queries
        phasor_clean = np.where(np.isnan(phasor_flat), np.inf, phasor_flat)
        
        # Scale the coordinates
        phasor_scaled = phasor_clean * scale
        
        # Build and return the KDTree
        return KDTree(phasor_scaled)
    
    @staticmethod
    def query_kdtree(kdtree: KDTree, center: np.ndarray, radius: float, 
                     scale: float = 1000.0, shape: tuple = None) -> np.ndarray:
        """
        Query the KD-tree for pixels within a specified distance.
        
        This method finds all pixels whose phasor coordinates fall within
        a specified distance from a center point, using the infinity norm
        (Chebyshev distance).
        
        Args:
            kdtree: KDTree object built from phasor coordinates
            center: Center point for the query as [g, s]
            radius: Distance threshold for the query (in phasor space)
            scale: Scaling factor used when building the tree (default: 1000.0)
            shape: Optional tuple (height, width) to convert flat indices to 2D
            
        Returns:
            Array of pixel indices within the distance threshold.
            If shape is provided, returns array of shape (n, 2) with [y, x] coordinates.
            Otherwise, returns flat indices of shape (n,).
            Returns empty array if no points are within the threshold.
            
        Notes:
            - Uses infinity norm (p=np.inf) for distance calculations
            - The center and radius should be in the original phasor space
              (they will be scaled internally)
            - Points at infinity (NaN in original data) are never returned
        """
        # Scale the center and radius to match the tree's coordinate system
        center_scaled = np.array(center) * scale
        radius_scaled = radius * scale
        
        # Query the tree using infinity norm
        # query_ball_point returns indices of all points within the radius
        indices = kdtree.query_ball_point(center_scaled, radius_scaled, p=np.inf)
        
        # Convert to numpy array
        indices = np.array(indices, dtype=np.int64)
        
        # If no points found, return empty array
        if len(indices) == 0:
            if shape is not None:
                return np.empty((0, 2), dtype=np.int64)
            else:
                return np.empty(0, dtype=np.int64)
        
        # If shape is provided, convert flat indices to 2D coordinates
        if shape is not None:
            height, width = shape
            y_coords = indices // width
            x_coords = indices % width
            return np.column_stack([y_coords, x_coords])
        
        return indices
