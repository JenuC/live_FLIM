"""
UDP stream receiver for FLIM data.

This module provides the StreamReceiver class for receiving FLIM data via UDP
from TCSPC hardware, parsing messages, and reading memory-mapped files.
"""

import logging
import socket
import os.path
from typing import Iterator, Union
import numpy as np

from flim_processing._dataclasses import SeriesMetadata, ElementData, EndSeries


class StreamReceiver:
    """Receives FLIM data streams via UDP.
    
    The receiver listens on a UDP port for messages from TCSPC hardware.
    Messages follow a tab-separated protocol:
    - new_series: Starts a new data series with metadata
    - element: Indicates a new data frame is available
    - end_series: Marks the end of a series
    - quit: Stops the receiver
    
    Data frames are stored in memory-mapped files and referenced by sequence number.
    
    Attributes:
        port: UDP port number to listen on
        addr: IP address to bind to (default: "127.0.0.1")
        socket: UDP socket for receiving messages
    """
    
    def __init__(self, port: int, addr: str = "127.0.0.1"):
        """Initialize receiver on specified port.
        
        Args:
            port: UDP port number to listen on
            addr: IP address to bind to (default: "127.0.0.1")
            
        Raises:
            IOError: If socket binding fails
        """
        self.port = port
        self.addr = addr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind((self.addr, self.port))
            logging.info(f"StreamReceiver bound to {self.addr}:{self.port}")
        except OSError as e:
            self.socket.close()
            raise IOError(f"Failed to bind to port {self.port}: {e}")
        
        self._running = False
        self._current_series = None
    
    def start_receiving(self) -> Iterator[Union[SeriesMetadata, ElementData, EndSeries]]:
        """
        Generator that yields data events as they arrive.
        
        This method blocks waiting for UDP messages and yields events:
        - SeriesMetadata: When a new series starts
        - ElementData: When a data frame arrives
        - EndSeries: When a series ends
        
        The generator continues until stop_receiving() is called or a quit
        message is received.
        
        Yields:
            SeriesMetadata: Metadata for a new series
            ElementData: A single frame of FLIM data
            EndSeries: Marker indicating series end
            
        Example:
            >>> receiver = StreamReceiver(port=4444)
            >>> for event in receiver.start_receiving():
            ...     if isinstance(event, SeriesMetadata):
            ...         print(f"New series: {event.shape}")
            ...     elif isinstance(event, ElementData):
            ...         print(f"Frame {event.seqno}")
            ...     elif isinstance(event, EndSeries):
            ...         print("Series ended")
        """
        self._running = True
        series_no = -1
        
        while self._running:
            # Wait for new series
            msg = self._parse_message(self._recvmsg())
            
            if isinstance(msg, _QuitMessage):
                logging.info("Received quit message")
                break
            
            if not isinstance(msg, _NewSeriesMessage):
                # Ignore messages until start of new series
                continue
            
            # New series started
            series_no += 1
            self._current_series = _SeriesContext(
                series_no=series_no,
                dtype=msg.dtype,
                shape=msg.shape,
                dirpath=msg.dirpath
            )
            
            yield SeriesMetadata(
                series_no=series_no,
                port=self.port,
                shape=msg.shape,
                dtype=msg.dtype
            )
            
            # Process elements in this series
            while self._running:
                msg = self._parse_message(self._recvmsg())
                
                if isinstance(msg, _QuitMessage):
                    logging.info("Received quit message")
                    self._running = False
                    break
                
                if isinstance(msg, _EndSeriesMessage):
                    yield EndSeries(series_no=series_no)
                    break
                
                if isinstance(msg, _SeriesElementMessage):
                    # Read array from memory-mapped file
                    array = self._map_array(
                        self._current_series.dtype,
                        self._current_series.shape,
                        self._current_series.dirpath,
                        msg.seqno
                    )
                    
                    # Handle multi-channel data - take only first channel
                    if array.ndim > 3:
                        array = array[tuple([0] * (array.ndim - 3))]
                    
                    yield ElementData(
                        series_no=series_no,
                        seqno=msg.seqno,
                        frame=array
                    )
    
    def stop_receiving(self):
        """Stop the receiver and close socket.
        
        This method sends a quit message to unblock any waiting receive
        operations and closes the socket.
        """
        self._running = False
        
        # Send quit message to unblock socket
        try:
            quit_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            quit_socket.sendto("quit".encode(), (self.addr, self.port))
            quit_socket.close()
            logging.info("Sent quit message to self")
        except Exception as e:
            logging.warning(f"Failed to send quit message: {e}")
        
        # Close the socket
        try:
            self.socket.close()
            logging.info("StreamReceiver socket closed")
        except Exception as e:
            logging.warning(f"Error closing socket: {e}")
    
    def _recvmsg(self) -> bytes:
        """Receive a message from the UDP socket.
        
        Returns:
            Raw message bytes
        """
        msg, addr = self.socket.recvfrom(512)
        msgstr = msg.decode()
        logging.debug(f"Received message: {msgstr}")
        return msg
    
    def _parse_message(self, msg: bytes):
        """Parse a UDP message into a message object.
        
        Args:
            msg: Raw message bytes
            
        Returns:
            Message object (_NewSeriesMessage, _SeriesElementMessage, etc.)
            
        Raises:
            IOError: If message format is invalid
        """
        fields = msg.decode().split('\t')
        cmd = fields.pop(0)
        
        if cmd == "new_series":
            return self._parse_message_new_series(fields)
        elif cmd == "element":
            return self._parse_message_element(fields)
        elif cmd == "end_series":
            return self._parse_message_end_series(fields)
        elif cmd == "quit":
            return self._parse_message_quit(fields)
        else:
            logging.error(f"Unknown message command: {cmd}")
            raise IOError(f"Unknown message command: {cmd}")
    
    def _parse_message_new_series(self, fields: list):
        """Parse a new_series message.
        
        Format: new_series\t<dtype>\t<ndim>\t<dim1>\t<dim2>\t...\t<dirpath>
        
        Args:
            fields: Message fields (without command)
            
        Returns:
            _NewSeriesMessage object
            
        Raises:
            IOError: If message format is invalid
        """
        if len(fields) < 3:
            raise IOError("Invalid new_series message: insufficient fields")
        
        dtype = self._parse_dtype(fields.pop(0))
        ndim = int(fields.pop(0))
        
        if len(fields) < ndim + 1:
            raise IOError("Invalid new_series message: shape dimensions missing")
        
        shape = tuple(int(fields.pop(0)) for _ in range(ndim))
        dirpath = fields.pop(0)
        
        if len(fields) > 0:
            raise IOError("Invalid new_series message: extra fields")
        
        return _NewSeriesMessage(dtype, shape, dirpath)
    
    def _parse_message_element(self, fields: list):
        """Parse an element message.
        
        Format: element\t<seqno>
        
        Args:
            fields: Message fields (without command)
            
        Returns:
            _SeriesElementMessage object
            
        Raises:
            IOError: If message format is invalid
        """
        if len(fields) < 1:
            raise IOError("Invalid element message: missing seqno")
        
        return _SeriesElementMessage(int(fields[0]))
    
    def _parse_message_end_series(self, fields: list):
        """Parse an end_series message.
        
        Format: end_series
        
        Args:
            fields: Message fields (without command)
            
        Returns:
            _EndSeriesMessage object
        """
        return _EndSeriesMessage()
    
    def _parse_message_quit(self, fields: list):
        """Parse a quit message.
        
        Format: quit
        
        Args:
            fields: Message fields (without command)
            
        Returns:
            _QuitMessage object
        """
        return _QuitMessage()
    
    def _parse_dtype(self, dtype_str: str) -> np.dtype:
        """Parse a dtype string into a NumPy dtype.
        
        Args:
            dtype_str: Data type string ("u16" or "u8")
            
        Returns:
            NumPy dtype object
            
        Raises:
            IOError: If dtype string is not recognized
        """
        if dtype_str == "u16":
            return np.dtype(np.uint16)
        elif dtype_str == "u8":
            return np.dtype(np.uint8)
        else:
            raise IOError(f"Unknown dtype: {dtype_str}")
    
    def _map_array(self, dtype: np.dtype, shape: tuple, dirpath: str, index: int) -> np.ndarray:
        """Read array from memory-mapped file.
        
        Args:
            dtype: NumPy data type
            shape: Array shape
            dirpath: Directory containing memory-mapped files
            index: Sequence number (file index)
            
        Returns:
            Memory-mapped NumPy array
            
        Raises:
            IOError: If file cannot be opened
        """
        path = os.path.join(dirpath, str(index))
        try:
            arr = np.memmap(path, dtype=dtype, shape=shape, mode="c")
            logging.debug(f"Mapped {dtype} array {shape} at {path}")
            return arr
        except Exception as e:
            raise IOError(f"Failed to map array at {path}: {e}")


# Internal message classes

class _NewSeriesMessage:
    """Internal message class for new series."""
    def __init__(self, dtype, shape, dirpath):
        self.dtype = dtype
        self.shape = shape
        self.dirpath = dirpath


class _SeriesElementMessage:
    """Internal message class for series element."""
    def __init__(self, seqno):
        self.seqno = seqno


class _EndSeriesMessage:
    """Internal message class for end of series."""
    pass


class _QuitMessage:
    """Internal message class for quit command."""
    pass


class _SeriesContext:
    """Internal class to track current series state."""
    def __init__(self, series_no, dtype, shape, dirpath):
        self.series_no = series_no
        self.dtype = dtype
        self.shape = shape
        self.dirpath = dirpath
