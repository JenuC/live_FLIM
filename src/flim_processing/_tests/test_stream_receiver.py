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
