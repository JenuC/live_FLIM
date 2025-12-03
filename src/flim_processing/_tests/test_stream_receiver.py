"""
Unit tests for StreamReceiver.
"""

import pytest
import numpy as np
import tempfile
import threading
import time
from flim_processing import StreamReceiver, SeriesMetadata, ElementData, EndSeries


class TestStreamReceiverMessageParsing:
    """Tests for message parsing methods."""
    
    def test_parse_dtype_u16(self):
        """Test parsing u16 dtype."""
        receiver = StreamReceiver(port=5555)
        dtype = receiver._parse_dtype("u16")
        assert dtype == np.dtype(np.uint16)
        receiver.stop_receiving()
    
    def test_parse_dtype_u8(self):
        """Test parsing u8 dtype."""
        receiver = StreamReceiver(port=5556)
        dtype = receiver._parse_dtype("u8")
        assert dtype == np.dtype(np.uint8)
        receiver.stop_receiving()
    
    def test_parse_dtype_invalid(self):
        """Test parsing invalid dtype raises IOError."""
        receiver = StreamReceiver(port=5557)
        with pytest.raises(IOError, match="Unknown dtype"):
            receiver._parse_dtype("invalid")
        receiver.stop_receiving()
    
    def test_parse_message_new_series(self):
        """Test parsing new_series message."""
        receiver = StreamReceiver(port=5558)
        msg = b"new_series\tu16\t3\t256\t256\t128\t/tmp/test"
        parsed = receiver._parse_message(msg)
        assert parsed.dtype == np.dtype(np.uint16)
        assert parsed.shape == (256, 256, 128)
        assert parsed.dirpath == "/tmp/test"
        receiver.stop_receiving()
    
    def test_parse_message_element(self):
        """Test parsing element message."""
        receiver = StreamReceiver(port=5559)
        msg = b"element\t42"
        parsed = receiver._parse_message(msg)
        assert parsed.seqno == 42
        receiver.stop_receiving()
    
    def test_parse_message_end_series(self):
        """Test parsing end_series message."""
        receiver = StreamReceiver(port=5560)
        msg = b"end_series"
        parsed = receiver._parse_message(msg)
        assert parsed is not None
        receiver.stop_receiving()
    
    def test_parse_message_quit(self):
        """Test parsing quit message."""
        receiver = StreamReceiver(port=5561)
        msg = b"quit"
        parsed = receiver._parse_message(msg)
        assert parsed is not None
        receiver.stop_receiving()
    
    def test_parse_message_invalid_command(self):
        """Test parsing invalid command raises IOError."""
        receiver = StreamReceiver(port=5562)
        msg = b"invalid_command\tdata"
        with pytest.raises(IOError, match="Unknown message command"):
            receiver._parse_message(msg)
        receiver.stop_receiving()


class TestStreamReceiverInitialization:
    """Tests for StreamReceiver initialization."""
    
    def test_initialization_success(self):
        """Test successful initialization."""
        receiver = StreamReceiver(port=5563)
        assert receiver.port == 5563
        assert receiver.addr == "127.0.0.1"
        receiver.stop_receiving()
    
    def test_initialization_custom_addr(self):
        """Test initialization with custom address."""
        receiver = StreamReceiver(port=5564, addr="0.0.0.0")
        assert receiver.addr == "0.0.0.0"
        receiver.stop_receiving()
    
    def test_initialization_port_in_use(self):
        """Test that binding to a port already in use raises IOError."""
        receiver1 = StreamReceiver(port=5565)
        with pytest.raises(IOError, match="Failed to bind to port"):
            receiver2 = StreamReceiver(port=5565)
        receiver1.stop_receiving()


class TestStreamReceiverStopReceiving:
    """Tests for stop_receiving method."""
    
    def test_stop_receiving_closes_socket(self):
        """Test that stop_receiving closes the socket."""
        receiver = StreamReceiver(port=5566)
        receiver.stop_receiving()
        # Socket should be closed, so binding to same port should work
        receiver2 = StreamReceiver(port=5566)
        receiver2.stop_receiving()



class TestStreamReceiverIntegration:
    """Integration tests for StreamReceiver with actual UDP communication."""
    
    def test_receive_single_series_single_element(self):
        """Test receiving a single series with one element."""
        from flim_processing._tests._test_utils import SeriesSender
        
        port = 5570
        shape = (64, 64, 32)
        dtype = np.uint16
        
        # Create test data
        test_data = np.random.randint(0, 1000, size=shape, dtype=dtype)
        
        # Start receiver in a separate thread
        receiver = StreamReceiver(port=port)
        events = []
        
        def receive_data():
            for event in receiver.start_receiving():
                events.append(event)
                if isinstance(event, EndSeries):
                    break
        
        receiver_thread = threading.Thread(target=receive_data)
        receiver_thread.start()
        
        # Give receiver time to start
        time.sleep(0.1)
        
        # Send data
        sender = SeriesSender(dtype, shape, port)
        sender.start()
        sender.send_element(0, test_data)
        sender.end()
        
        # Wait for receiver to finish
        receiver_thread.join(timeout=2.0)
        receiver.stop_receiving()
        
        # Verify events
        assert len(events) == 3  # SeriesMetadata, ElementData, EndSeries
        assert isinstance(events[0], SeriesMetadata)
        assert events[0].shape == shape
        assert events[0].dtype == dtype
        assert events[0].port == port
        assert events[0].series_no == 0
        
        assert isinstance(events[1], ElementData)
        assert events[1].series_no == 0
        assert events[1].seqno == 0
        assert events[1].frame.shape == shape
        assert np.array_equal(events[1].frame, test_data)
        
        assert isinstance(events[2], EndSeries)
        assert events[2].series_no == 0
    
    def test_receive_single_series_multiple_elements(self):
        """Test receiving a single series with multiple elements."""
        from flim_processing._tests._test_utils import SeriesSender
        
        port = 5571
        shape = (32, 32, 16)
        dtype = np.uint16
        num_frames = 3
        
        # Create test data
        test_frames = [
            np.random.randint(0, 1000, size=shape, dtype=dtype)
            for _ in range(num_frames)
        ]
        
        # Start receiver
        receiver = StreamReceiver(port=port)
        events = []
        
        def receive_data():
            for event in receiver.start_receiving():
                events.append(event)
                if isinstance(event, EndSeries):
                    break
        
        receiver_thread = threading.Thread(target=receive_data)
        receiver_thread.start()
        time.sleep(0.1)
        
        # Send data
        sender = SeriesSender(dtype, shape, port)
        sender.start()
        for i, frame in enumerate(test_frames):
            sender.send_element(i, frame)
        sender.end()
        
        # Wait for receiver
        receiver_thread.join(timeout=2.0)
        receiver.stop_receiving()
        
        # Verify events
        assert len(events) == 1 + num_frames + 1  # Metadata + frames + EndSeries
        assert isinstance(events[0], SeriesMetadata)
        
        for i in range(num_frames):
            assert isinstance(events[i + 1], ElementData)
            assert events[i + 1].seqno == i
            assert np.array_equal(events[i + 1].frame, test_frames[i])
        
        assert isinstance(events[-1], EndSeries)
    
    def test_receive_multiple_series(self):
        """Test receiving multiple series."""
        from flim_processing._tests._test_utils import SeriesSender
        
        port = 5572
        shape = (16, 16, 8)
        dtype = np.uint8
        
        # Start receiver
        receiver = StreamReceiver(port=port)
        events = []
        
        def receive_data():
            series_count = 0
            for event in receiver.start_receiving():
                events.append(event)
                if isinstance(event, EndSeries):
                    series_count += 1
                    if series_count >= 2:
                        break
        
        receiver_thread = threading.Thread(target=receive_data)
        receiver_thread.start()
        time.sleep(0.1)
        
        # Send two series
        for series_idx in range(2):
            sender = SeriesSender(dtype, shape, port)
            sender.start()
            test_data = np.random.randint(0, 255, size=shape, dtype=dtype)
            sender.send_element(0, test_data)
            sender.end()
            time.sleep(0.05)
        
        # Wait for receiver
        receiver_thread.join(timeout=2.0)
        receiver.stop_receiving()
        
        # Verify we got two series
        metadata_events = [e for e in events if isinstance(e, SeriesMetadata)]
        assert len(metadata_events) == 2
        assert metadata_events[0].series_no == 0
        assert metadata_events[1].series_no == 1
        
        end_events = [e for e in events if isinstance(e, EndSeries)]
        assert len(end_events) == 2
