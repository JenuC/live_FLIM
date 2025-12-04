"""
Minimal Marimo-like app wrapping the streaming example into a lightweight Flask UI.
Run: python examples\02B_streaming_marimo.py

This creates endpoints:
- /start : start the StreamReceiver
- /status: show receiver status and frame count
- /frame/<int:i>: return PNG of snapshot frame intensity

Notes: This is a small conversion for local use; a full Marimo app would use the marimo framework.
"""
from flask import Flask, send_file, jsonify
import io
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
import time

from flim_processing import StreamReceiver, DataManager

app = Flask(__name__)
receiver = None
manager = None
thread = None
running = False

def receiver_thread(port, addr):
    global receiver, manager, running
    receiver = StreamReceiver(port=port, addr=addr)
    manager = DataManager()
    running = True
    # Very small loop: receive up to 10 messages then stop
    try:
        count = 0
        while running and count < 10:
            pkt = receiver.receive(timeout=1.0)
            if pkt is None:
                time.sleep(0.1)
                continue
            manager.add_packet(pkt)
            count += 1
    finally:
        running = False

@app.route('/start')
def start():
    global thread, running
    if running:
        return jsonify({'status': 'already running'})
    thread = Thread(target=receiver_thread, args=(4444, '127.0.0.1'), daemon=True)
    thread.start()
    return jsonify({'status': 'started'})

@app.route('/status')
def status():
    if manager is None:
        return jsonify({'running': running, 'frames': 0})
    return jsonify({'running': running, 'frames': manager.get_frame_count()})

@app.route('/frame/<int:i>')
def frame(i):
    if manager is None or manager.get_frame_count() == 0:
        return jsonify({'error': 'no frames'}), 404
    snapshot = manager.get_photon_count(i)
    intensity = np.sum(snapshot, axis=-1)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(intensity, cmap='gray')
    ax.set_title(f'Frame {i} intensity')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=8765)

