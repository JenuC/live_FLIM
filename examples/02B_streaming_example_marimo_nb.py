import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():

    from flask import Flask, send_file, jsonify
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    from threading import Thread
    import time
    return Flask, Thread, jsonify, time


@app.cell
def _():
    from flim_processing import StreamReceiver, DataManager
    return DataManager, StreamReceiver


@app.cell
def _(DataManager, Flask, StreamReceiver, time):
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
    return receiver_thread, running


@app.cell
def _(Thread, jsonify, receiver_thread, running):
    def start():
        global thread, running
        if running:
            return jsonify({'status': 'already running'})
        thread = Thread(target=receiver_thread, args=(4444, '127.0.0.1'), daemon=True)
        thread.start()
        return jsonify({'status': 'started'})
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
