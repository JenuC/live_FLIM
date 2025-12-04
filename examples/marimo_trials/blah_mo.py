import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import threading, time
    import numpy as np
    import matplotlib.pyplot as plt
    from flim_processing import StreamReceiver, DataManager
    import marimo

    # single mutable state dictionary
    state = {
        "receiver": None,
        "manager": None,
        "running": False,
        "thread": None
    }
    return DataManager, StreamReceiver, marimo, state


@app.cell
def _(marimo):
    start_button = marimo.ui.button("Start Stream Receiver")
    start_button
    return (start_button,)


@app.cell
def _(DataManager, StreamReceiver, start_button, state):
    import threading, time
    from flim_processing import StreamReceive

    def _receiver_loop(port=4444, addr="127.0.0.1"):
        state["receiver"] = StreamReceiver(port=port, addr=addr)
        state["manager"] = DataManager()
        state["running"] = True

        try:
            count = 0
            while state["running"] and count < 10:
                pkt = state["receiver"].receive(timeout=1.0)
                if pkt is None:
                    time.sleep(0.1)
                    continue
                state["manager"].add_packet(pkt)
                count += 1
        finally:
            state["running"] = False

    status_message = None
    if start_button.value and (state.get("thread") is None or not state["thread"].is_alive()):
        t = threading.Thread(target=_receiver_loop, args=(4444, "127.0.0.1"), daemon=True)
        t.start()
        state["thread"] = t
        status_message = "Started receiver thread."
    elif start_button.value and state.get("thread") and state["thread"].is_alive():
        status_message = "Already running."
    else:
        status_message = "Waiting to start."

    status_message
    return


@app.cell
def _(state):
    try:
        manager = state["manager"]
        frames = 0 if manager is None else manager.get_frame_count()
    except Exception:
        frames = 0

    status = {
        "running": bool(state.get("running")),
        "frames": int(frames)
    }
    status
    return


app._unparsable_cell(
    r"""
    import matplotlib.pyplot as plt
    import numpy as np

    mgr = state.get(\"manager\")
    if mgr is None:
        return \"No frames yet.\"

    try:
        count = mgr.get_frame_count()
    except Exception:
        return \"No frames yet.\"

    if count == 0:
        return \"No frames yet.\"

    idx = count - 1
    snapshot = mgr.get_photon_count(idx)
    intensity = np.sum(snapshot, axis=-1)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(intensity, cmap=\"gray\")
    ax.set_title(f\"Frame {idx}\")
    fig
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
