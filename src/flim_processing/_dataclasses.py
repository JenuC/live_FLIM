"""
Core data classes for FLIM processing.

This module defines the fundamental data structures used throughout the library,
including parameters, settings, metadata, and results.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class FlimParams:
    """Parameters for FLIM computation.
    
    Attributes:
        period: Laser period in nanoseconds (must be positive)
        fit_start: Start index for fitting (must be less than fit_end)
        fit_end: End index for fitting (exclusive)
    """
    period: float
    fit_start: int
    fit_end: int
    
    def __post_init__(self):
        """Validate parameters.
        
        Raises:
            ValueError: If period is not positive or fit_start >= fit_end
        """
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.fit_start >= self.fit_end:
            raise ValueError(
                f"fit_start ({self.fit_start}) must be less than fit_end ({self.fit_end})"
            )


@dataclass(frozen=True)
class DisplaySettings:
    """Settings for lifetime image display.
    
    Attributes:
        max_chisq: Maximum chi-squared for filtering
        min_tau: Minimum lifetime in nanoseconds (must be less than max_tau)
        max_tau: Maximum lifetime in nanoseconds
        colormap: Colormap name for visualization
    """
    max_chisq: float
    min_tau: float
    max_tau: float
    colormap: str
    
    def __post_init__(self):
        """Validate settings.
        
        Raises:
            ValueError: If min_tau >= max_tau
        """
        if self.min_tau >= self.max_tau:
            raise ValueError(
                f"min_tau ({self.min_tau}) must be less than max_tau ({self.max_tau})"
            )


@dataclass(frozen=True)
class ProcessingSettings:
    """Combined processing settings.
    
    Attributes:
        flim_params: FLIM computation parameters
        display_settings: Display and filtering settings
        delta_snapshots: Whether to use delta snapshot mode
    """
    flim_params: FlimParams
    display_settings: DisplaySettings
    delta_snapshots: bool = False


@dataclass(frozen=True)
class SeriesMetadata:
    """Metadata for a FLIM data series.
    
    Attributes:
        series_no: Series number identifier
        port: UDP port number
        shape: Data shape as (height, width, time_bins)
        dtype: NumPy data type
    """
    series_no: int
    port: int
    shape: tuple
    dtype: np.dtype


@dataclass
class ElementData:
    """A single frame of FLIM data.
    
    Attributes:
        series_no: Series number identifier
        seqno: Sequence number for ordering
        frame: Photon count array
    """
    series_no: int
    seqno: int
    frame: np.ndarray


@dataclass(frozen=True)
class EndSeries:
    """Marker for end of a data series.
    
    Attributes:
        series_no: Series number identifier
    """
    series_no: int


@dataclass(frozen=True)
class RLDResult:
    """Result from Rapid Lifetime Determination.
    
    Attributes:
        tau: Lifetime values array
        chisq: Chi-squared values array
        Z: Background parameter array
        A: Amplitude parameter array
        fitted: Optional fitted curve array
    """
    tau: np.ndarray
    chisq: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    fitted: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LMAResult:
    """Result from Levenberg-Marquardt fitting.
    
    Attributes:
        param: Parameter array [Z, A, tau]
        chisq: Chi-squared values array
        fitted: Optional fitted curve array
    """
    param: np.ndarray
    chisq: np.ndarray
    fitted: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SelectionResult:
    """Result from analyzing a selection.
    
    Attributes:
        histogram: Averaged photon count histogram
        points: Selected pixel coordinates or phasor points
        rld: RLD fit result
        lma: LMA fit result
        pixel_count: Number of pixels in selection
    """
    histogram: np.ndarray
    points: np.ndarray
    rld: RLDResult
    lma: LMAResult
    pixel_count: int
