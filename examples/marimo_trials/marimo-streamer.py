import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import marimo
    return (marimo,)


@app.cell
def _():
    import threading, time
    import numpy as np
    import matplotlib.pyplot as plt
    from flim_processing import StreamReceiver, DataManager

    receiver = None
    manager = None
    running = False
    return manager, running


@app.cell
def _(marimo):
    start_button = marimo.ui.button("Start Stream Receiver")
    start_button
    return


app._unparsable_cell(
    r"""
    import threading, time
    from flim_processing import StreamReceiver, DataManager

    def start_receiver():
        nonlocal receiver, manager, running

        if running:
            return \"Already running.\"

        receiver = StreamReceiver(port=4444, addr=\"127.0.0.1\")
        manager = DataManager()
        running = True

        def loop():
            nonlocal running, receiver, manager
            count = 0
            while running and count < 10:
                pkt = receiver.receive(timeout=1.0)
                if pkt:
                    manager.add_packet(pkt)
                    count += 1
                else:
                    time.sleep(0.1)
            running = False

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        return \"Started.\"

    status_message = None
    if start_button.value:
        status_message = start_receiver()

    status_message
    """,
    name="_"
)


@app.cell
def _(manager, running):
    if manager is None:
        result = {"running": running, "frames": 0}
    else:
        try:
            result = {"running": running, "frames": manager.get_frame_count()}
        except Exception:
            result = {"running": running, "frames": 0}

    result
    return


app._unparsable_cell(
    r"""
    import matplotlib.pyplot as plt
    import numpy as np

    if manager is None or manager.get_frame_count() == 0:
        result = \"No frames yet.\"
        result
        return result

    idx = manager.get_frame_count() - 1
    snapshot = manager.get_photon_count(idx)
    intensity = np.sum(snapshot, axis=-1)

    fig, ax = plt.subplots()
    ax.imshow(intensity, cmap=\"gray\")
    ax.set_title(f\"Frame {idx}\")

    fig
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
