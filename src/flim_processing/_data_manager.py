"""
Data manager for FLIM data storage and retrieval.

This module provides the DataManager class for managing FLIM data frames,
including snapshot storage, delta mode computation, and frame retrieval.
"""

import logging
from typing import Optional
import numpy as np


class DataManager:
    """Manages FLIM data storage and retrieval.
    
    The DataManager handles storage of FLIM data frames with support for:
    - Sequential element storage with sequence number tracking
    - Snapshot creation for freezing frames
    - Delta mode for computing frame differences
    - Frame retrieval with proper ordering
    
    Attributes:
        shape: Data shape as (height, width, time_bins)
        dtype: NumPy data type for frames
        delta_mode: Whether to compute frame differences
    """
    
    def __init__(self, shape: tuple, dtype: np.dtype, delta_mode: bool = False):
        """Initialize data manager with series parameters.
        
        Args:
            shape: Data shape as (height, width, time_bins)
            dtype: NumPy data type for frames
            delta_mode: Whether to use delta snapshot mode (default: False)
        """
        self.shape = shape
        self.dtype = dtype
        self.delta_mode = delta_mode
        
        # Storage for snapshots (frozen frames)
        self._snapshots = []
        
        # Live frame storage
        self._live_frame = None
        self._live_seqno = -1
        
        # Previous frame for delta computation
        self._previous_frame = None
        
        # Track all received sequence numbers for ordering
        self._received_seqnos = []
        
        logging.info(
            f"DataManager initialized: shape={shape}, dtype={dtype}, "
            f"delta_mode={delta_mode}"
        )
    
    def add_element(self, seqno: int, frame: np.ndarray):
        """Add a new data element to the series.
        
        Elements are tracked by sequence number to ensure correct ordering.
        In delta mode, the frame is stored as-is (delta computation happens
        during retrieval).
        
        Args:
            seqno: Sequence number for ordering
            frame: Photon count array with shape matching self.shape
            
        Raises:
            ValueError: If frame shape doesn't match expected shape
        """
        if frame.shape != self.shape:
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match expected shape {self.shape}"
            )
        
        # Store the frame as the live frame
        self._live_frame = frame.copy()
        self._live_seqno = seqno
        
        # Track sequence number for ordering
        self._received_seqnos.append(seqno)
        
        logging.debug(f"Added element with seqno={seqno}")
    
    def snapshot(self):
        """Create a snapshot of the current live frame.
        
        Snapshots are frozen copies of frames that can be accessed later.
        The snapshot is stored independently, so modifications to the live
        frame don't affect it.
        
        In delta mode, the snapshot stores the actual frame data (not deltas).
        """
        if self._live_frame is None:
            logging.warning("No live frame to snapshot")
            return
        
        # Create independent copy for snapshot
        snapshot_frame = self._live_frame.copy()
        self._snapshots.append(snapshot_frame)
        
        # Update previous frame for delta computation
        if self.delta_mode:
            self._previous_frame = snapshot_frame.copy()
        
        logging.info(f"Created snapshot {len(self._snapshots) - 1}")
    
    def get_photon_count(self, index: int) -> np.ndarray:
        """
        Get photon count array for a specific frame.
        
        In snapshot mode, returns the frame as-is.
        In delta mode, returns the difference from the previous frame.
        
        Args:
            index: Frame index (0 to get_frame_count()-1)
                  Negative indices count from the end (-1 is live frame)
                  
        Returns:
            Photon count array with shape self.shape
            
        Raises:
            IndexError: If index is out of range
        """
        frame_count = self.get_frame_count()
        
        if frame_count == 0:
            raise IndexError("No frames available")
        
        # Handle negative indices
        if index < 0:
            index = frame_count + index
        
        if index < 0 or index >= frame_count:
            raise IndexError(
                f"Frame index {index} out of range [0, {frame_count})"
            )
        
        # Determine if this is a snapshot or the live frame
        if index < len(self._snapshots):
            # Accessing a snapshot
            frame = self._snapshots[index]
            
            if self.delta_mode and index > 0:
                # Compute delta from previous snapshot
                previous = self._snapshots[index - 1]
                return frame - previous
            else:
                # First frame or snapshot mode - return as-is
                return frame.copy()
        else:
            # Accessing the live frame
            if self._live_frame is None:
                raise IndexError("Live frame not available")
            
            if self.delta_mode and len(self._snapshots) > 0:
                # Compute delta from last snapshot
                previous = self._snapshots[-1]
                return self._live_frame - previous
            else:
                # First frame or snapshot mode - return as-is
                return self._live_frame.copy()
    
    def get_frame_count(self) -> int:
        """Return the number of frames (snapshots + live).
        
        Returns:
            Total number of accessible frames
        """
        count = len(self._snapshots)
        if self._live_frame is not None:
            count += 1
        return count
