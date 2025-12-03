"""
Unit tests for DataManager class.
"""

import pytest
import numpy as np
from flim_processing import DataManager


class TestDataManagerBasics:
    """Tests for basic DataManager functionality."""
    
    def test_initialization(self):
        """Test that DataManager initializes correctly."""
        shape = (256, 256, 128)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=False)
        
        assert dm.shape == shape
        assert dm.dtype == dtype
        assert dm.delta_mode is False
        assert dm.get_frame_count() == 0
    
    def test_initialization_with_delta_mode(self):
        """Test that DataManager initializes with delta mode."""
        shape = (256, 256, 128)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=True)
        
        assert dm.delta_mode is True
    
    def test_add_element(self):
        """Test adding a single element."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        frame = np.random.randint(0, 100, shape, dtype=dtype)
        dm.add_element(seqno=0, frame=frame)
        
        assert dm.get_frame_count() == 1
    
    def test_add_element_wrong_shape_raises_error(self):
        """Test that adding element with wrong shape raises ValueError."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        wrong_frame = np.random.randint(0, 100, (5, 5, 8), dtype=dtype)
        
        with pytest.raises(ValueError, match="Frame shape.*doesn't match"):
            dm.add_element(seqno=0, frame=wrong_frame)
    
    def test_add_multiple_elements(self):
        """Test adding multiple elements."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        for i in range(3):
            frame = np.random.randint(0, 100, shape, dtype=dtype)
            dm.add_element(seqno=i, frame=frame)
        
        # Only live frame exists (no snapshots yet)
        assert dm.get_frame_count() == 1


class TestDataManagerSnapshot:
    """Tests for snapshot functionality."""
    
    def test_snapshot_creates_frozen_copy(self):
        """Test that snapshot creates an independent copy."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        # Add first frame
        frame1 = np.ones(shape, dtype=dtype) * 10
        dm.add_element(seqno=0, frame=frame1)
        
        # Create snapshot
        dm.snapshot()
        assert dm.get_frame_count() == 1  # Only snapshot exists now
        
        # Add second frame
        frame2 = np.ones(shape, dtype=dtype) * 20
        dm.add_element(seqno=1, frame=frame2)
        
        assert dm.get_frame_count() == 2  # Snapshot + live frame
        
        # Verify snapshot is independent
        snapshot_data = dm.get_photon_count(0)
        live_data = dm.get_photon_count(1)
        
        assert np.allclose(snapshot_data, 10)
        assert np.allclose(live_data, 20)
    
    def test_multiple_snapshots(self):
        """Test creating multiple snapshots."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        # Add frames and create snapshots
        for i in range(3):
            frame = np.ones(shape, dtype=dtype) * (i + 1) * 10
            dm.add_element(seqno=i, frame=frame)
            dm.snapshot()
        
        assert dm.get_frame_count() == 3
        
        # Verify each snapshot
        for i in range(3):
            data = dm.get_photon_count(i)
            expected = (i + 1) * 10
            assert np.allclose(data, expected)
    
    def test_snapshot_without_live_frame(self):
        """Test that snapshot without live frame logs warning."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        # Try to snapshot without adding any frame
        dm.snapshot()  # Should not crash
        assert dm.get_frame_count() == 0


class TestDataManagerRetrieval:
    """Tests for frame retrieval."""
    
    def test_get_photon_count_live_frame(self):
        """Test retrieving the live frame."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        frame = np.random.randint(0, 100, shape, dtype=dtype)
        dm.add_element(seqno=0, frame=frame)
        
        retrieved = dm.get_photon_count(0)
        assert np.array_equal(retrieved, frame)
    
    def test_get_photon_count_negative_index(self):
        """Test retrieving frame with negative index."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        frame = np.random.randint(0, 100, shape, dtype=dtype)
        dm.add_element(seqno=0, frame=frame)
        
        # -1 should get the last frame (live frame)
        retrieved = dm.get_photon_count(-1)
        assert np.array_equal(retrieved, frame)
    
    def test_get_photon_count_out_of_range(self):
        """Test that out of range index raises IndexError."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        frame = np.random.randint(0, 100, shape, dtype=dtype)
        dm.add_element(seqno=0, frame=frame)
        
        with pytest.raises(IndexError, match="out of range"):
            dm.get_photon_count(5)
    
    def test_get_photon_count_no_frames(self):
        """Test that accessing frames when none exist raises IndexError."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        with pytest.raises(IndexError, match="No frames available"):
            dm.get_photon_count(0)
    
    def test_get_frame_count(self):
        """Test frame count with snapshots and live frame."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype)
        
        assert dm.get_frame_count() == 0
        
        # Add frame
        frame = np.random.randint(0, 100, shape, dtype=dtype)
        dm.add_element(seqno=0, frame=frame)
        assert dm.get_frame_count() == 1
        
        # Create snapshot
        dm.snapshot()
        assert dm.get_frame_count() == 1
        
        # Add another frame
        dm.add_element(seqno=1, frame=frame)
        assert dm.get_frame_count() == 2


class TestDataManagerDeltaMode:
    """Tests for delta mode functionality."""
    
    def test_delta_mode_first_frame(self):
        """Test that first frame in delta mode returns frame without subtraction."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=True)
        
        frame1 = np.ones(shape, dtype=dtype) * 100
        dm.add_element(seqno=0, frame=frame1)
        dm.snapshot()
        
        # First frame should be returned as-is
        retrieved = dm.get_photon_count(0)
        assert np.array_equal(retrieved, frame1)
    
    def test_delta_mode_second_frame(self):
        """Test that second frame in delta mode returns difference."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=True)
        
        # First frame
        frame1 = np.ones(shape, dtype=dtype) * 100
        dm.add_element(seqno=0, frame=frame1)
        dm.snapshot()
        
        # Second frame
        frame2 = np.ones(shape, dtype=dtype) * 150
        dm.add_element(seqno=1, frame=frame2)
        dm.snapshot()
        
        # Second frame should return delta
        retrieved = dm.get_photon_count(1)
        expected_delta = frame2 - frame1
        assert np.array_equal(retrieved, expected_delta)
    
    def test_delta_mode_live_frame(self):
        """Test that live frame in delta mode returns difference from last snapshot."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=True)
        
        # First frame and snapshot
        frame1 = np.ones(shape, dtype=dtype) * 100
        dm.add_element(seqno=0, frame=frame1)
        dm.snapshot()
        
        # Live frame (not snapshotted)
        frame2 = np.ones(shape, dtype=dtype) * 150
        dm.add_element(seqno=1, frame=frame2)
        
        # Live frame should return delta from last snapshot
        retrieved = dm.get_photon_count(1)
        expected_delta = frame2 - frame1
        assert np.array_equal(retrieved, expected_delta)
    
    def test_delta_mode_multiple_frames(self):
        """Test delta mode with multiple frames."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=True)
        
        frames = []
        for i in range(4):
            frame = np.ones(shape, dtype=dtype) * (i + 1) * 50
            frames.append(frame)
            dm.add_element(seqno=i, frame=frame)
            dm.snapshot()
        
        # First frame should be as-is
        retrieved0 = dm.get_photon_count(0)
        assert np.array_equal(retrieved0, frames[0])
        
        # Subsequent frames should be deltas
        for i in range(1, 4):
            retrieved = dm.get_photon_count(i)
            expected_delta = frames[i] - frames[i-1]
            assert np.array_equal(retrieved, expected_delta)


class TestDataManagerSnapshotModeIndependence:
    """Tests for snapshot mode independence (Property 30)."""
    
    def test_snapshot_independence(self):
        """Test that snapshots are independent in snapshot mode."""
        shape = (10, 10, 8)
        dtype = np.uint16
        dm = DataManager(shape, dtype, delta_mode=False)
        
        # Create multiple snapshots
        frames = []
        for i in range(3):
            frame = np.ones(shape, dtype=dtype) * (i + 1) * 10
            frames.append(frame.copy())
            dm.add_element(seqno=i, frame=frame)
            dm.snapshot()
        
        # Retrieve all snapshots
        retrieved_frames = [dm.get_photon_count(i) for i in range(3)]
        
        # Verify each snapshot is independent and matches original
        for i, (retrieved, original) in enumerate(zip(retrieved_frames, frames)):
            assert np.array_equal(retrieved, original)
            
            # Modify retrieved frame and verify it doesn't affect storage
            retrieved[:] = 999
            retrieved_again = dm.get_photon_count(i)
            assert np.array_equal(retrieved_again, original)

