"""
FLIM Processing Library

A standalone Python library for Fluorescence Lifetime Imaging Microscopy (FLIM) data processing.
Provides core FLIM analysis capabilities including phasor analysis, curve fitting, and data streaming
without dependencies on napari or Qt.
"""

from flim_processing._dataclasses import (
    FlimParams,
    DisplaySettings,
    ProcessingSettings,
    SeriesMetadata,
    ElementData,
    RLDResult,
    LMAResult,
    SelectionResult,
    EndSeries,
)
from flim_processing._stream_receiver import StreamReceiver
from flim_processing._data_manager import DataManager
from flim_processing._phasor_computer import PhasorComputer

__all__ = [
    "FlimParams",
    "DisplaySettings",
    "ProcessingSettings",
    "SeriesMetadata",
    "ElementData",
    "RLDResult",
    "LMAResult",
    "SelectionResult",
    "EndSeries",
    "StreamReceiver",
    "DataManager",
    "PhasorComputer",
]

__version__ = "0.1.0"
