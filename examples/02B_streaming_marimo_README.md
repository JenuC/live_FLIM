This folder contains a minimal Flask-based wrapper (examples/02B_streaming_marimo.py) that provides simple HTTP endpoints to start the StreamReceiver and fetch frame snapshots as PNGs.

Usage:
- Ensure dependencies are installed (flask, matplotlib, numpy and the project's flim_processing package). Example: pip install flask matplotlib numpy
- Run: python examples\02B_streaming_marimo.py
- Open http://localhost:8765/start to start receiving data. Check http://localhost:8765/status and http://localhost:8765/frame/0

This is not a full marimo app but provides a simple web UI to the streaming example for local testing and demos.
