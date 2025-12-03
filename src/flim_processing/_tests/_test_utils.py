"""
Test utilities for FLIM processing library.

This module provides helper classes for testing, including SeriesSender
for simulating UDP data streams.
"""

import logging
import socket
import tempfile
import os.path
import mmap
import numpy as np


class SeriesSender:
    """
    A class to send a series via UDP and temporary memory mapped files.
    Intended to be used for testing the StreamReceiver.
    
    This is a simplified version extracted from napari-live-flim for testing purposes.
    """
    
    def __init__(self, dtype, element_shape, port, addr=None, dirpath=None):
        """Initialize SeriesSender.
        
        Args:
            dtype: NumPy data type (np.uint16 or np.uint8)
            element_shape: Shape of each data element
            port: UDP port to send to
            addr: IP address to send to (default: "127.0.0.1")
            dirpath: Directory for temporary files (default: system temp)
        """
        self.dtype = dtype
        self.element_shape = element_shape
        self.port = port
        self.addr = addr if addr else "127.0.0.1"
        
        self.tempdir = tempfile.TemporaryDirectory(dir=dirpath)
        self.dirpath = self.tempdir.name
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def _send_message(self, msgbytes):
        """Send a UDP message."""
        self.socket.sendto(msgbytes, (self.addr, self.port))
        msgstr = msgbytes.decode()
        logging.debug(f"Sent message: {msgstr}")
    
    def _write_array(self, dirpath, seqno, dtype, arr):
        """Write array to memory-mapped file."""
        path = os.path.join(dirpath, str(seqno))
        size = arr.size * dtype.itemsize
        with open(path, "wb+") as f:
            with mmap.mmap(f.fileno(), size) as m:
                b = np.frombuffer(m, dtype=dtype).reshape(arr.shape)
                b[:] = arr
                del b
        logging.debug(f"Wrote {dtype} array {arr.shape} to {path}")
    
    def start(self):
        """Send new_series message to start a series."""
        dt = None
        if self.dtype == np.uint16:
            dt = "u16"
        elif self.dtype == np.uint8:
            dt = "u8"
        assert dt, f"Unsupported dtype: {self.dtype}"
        
        ndim = len(self.element_shape)
        shape = "\t".join(str(d) for d in self.element_shape)
        dirpath = self.dirpath
        
        self._send_message(f"new_series\t{dt}\t{ndim}\t{shape}\t{dirpath}".encode())
    
    def send_element(self, seqno, element):
        """Send a data element.
        
        Args:
            seqno: Sequence number
            element: NumPy array with data
        """
        assert element.shape == self.element_shape, \
            f"Element shape {element.shape} doesn't match expected {self.element_shape}"
        self._write_array(self.dirpath, seqno, self.dtype, element)
        self._send_message(f"element\t{seqno}".encode())
    
    def end(self):
        """Send end_series message to end the series."""
        self._send_message("end_series".encode())
    
    def __del__(self):
        """Cleanup temporary directory."""
        try:
            self.tempdir.cleanup()
        except Exception:
            pass
