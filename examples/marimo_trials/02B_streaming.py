import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    from flim_processing import (
        FlimParams,
        StreamReceiver,
        DataManager,
        PhasorComputer,
        FittingEngine,
        SeriesMetadata,
        ElementData,
        EndSeries
    )
    return (
        DataManager,
        ElementData,
        EndSeries,
        FittingEngine,
        FlimParams,
        PhasorComputer,
        SeriesMetadata,
        StreamReceiver,
    )


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    # '%matplotlib inline' command supported automatically in marimo
    plt.rcParams['figure.figsize'] = (14, 5)
    return np, plt


@app.cell
def _():
    UDP_PORT = 4444
    UDP_ADDR = "127.0.0.1"
    return UDP_ADDR, UDP_PORT


@app.cell
def _(StreamReceiver, UDP_ADDR, UDP_PORT):
    # Create a stream receiver
    receiver = StreamReceiver(port=UDP_PORT, addr=UDP_ADDR)
    print(f"StreamReceiver created and ready to receive data")
    print(f"Listening on port {UDP_PORT}")
    return (receiver,)


@app.cell
def _(FlimParams):
    period = 12.5  # nanoseconds (80 MHz laser)
    time_bins = 256

    flim_params = FlimParams(
        period=period,
        fit_start=10,
        fit_end=200
    )
    return (flim_params,)


@app.cell
def _(
    DataManager,
    ElementData,
    EndSeries,
    FittingEngine,
    PhasorComputer,
    SeriesMetadata,
    flim_params,
    np,
    receiver,
):
    # Initialize variables for data processinga
    data_manager = None
    frame_count = 0
    max_frames = 10  # Process up to 10 frames for this example

    # Storage for results
    phasor_results = []
    lifetime_results = []

    print("Starting to receive data...")
    print("(Press Ctrl+C to stop)\n")

    try:
        for event in receiver.start_receiving():
            if isinstance(event, SeriesMetadata):
                # New series started
                print(f"\n=== New Series {event.series_no} ===")
                print(f"Shape: {event.shape}")
                print(f"Data type: {event.dtype}")
            
                # Initialize data manager for this series
                data_manager = DataManager(
                    shape=event.shape,
                    dtype=event.dtype,
                    delta_mode=False
                )
                frame_count = 0
            
            elif isinstance(event, ElementData):
                # New frame received
                frame_count += 1
                print(f"Frame {frame_count}: seqno={event.seqno}, shape={event.frame.shape}")
            
                # Add to data manager
                data_manager.add_element(event.seqno, event.frame)
            
                # Process the frame
                photon_count = event.frame
            
                # Compute phasor
                phasor = PhasorComputer.compute_phasor(photon_count, flim_params)
                phasor_results.append(phasor)
            
                # Compute lifetime (RLD)
                rld_result = FittingEngine.compute_rld(photon_count, flim_params)
                lifetime_results.append(rld_result.tau)
            
                print(f"  Phasor: g=[{np.nanmin(phasor[..., 0]):.3f}, {np.nanmax(phasor[..., 0]):.3f}], "
                      f"s=[{np.nanmin(phasor[..., 1]):.3f}, {np.nanmax(phasor[..., 1]):.3f}]")
                print(f"  Lifetime: [{np.nanmin(rld_result.tau):.3f}, {np.nanmax(rld_result.tau):.3f}] ns")
            
                # Create snapshot every few frames
                if frame_count % 3 == 0:
                    data_manager.snapshot()
                    print(f"  Created snapshot (total snapshots: {data_manager.get_frame_count() - 1})")
            
                # Stop after max_frames
                if frame_count >= max_frames:
                    print(f"\nReached {max_frames} frames, stopping...")
                    break
                
            elif isinstance(event, EndSeries):
                # Series ended
                print(f"\n=== Series {event.series_no} Ended ===")
                break
            
    except KeyboardInterrupt:
        print("\nReceiving interrupted by user")
    finally:
        receiver.stop_receiving()
        print("Receiver stopped")

    print(f"\nProcessed {frame_count} frames")
    return data_manager, lifetime_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## lifetime results
    """)
    return


@app.cell
def _(lifetime_results, plt):
    if len(lifetime_results) > 0:
        n_frames = min(6, len(lifetime_results))  # Show lifetime evolution over frames
        _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))
        _axes = _axes.flatten()
        for _i in range(n_frames):
            _im = _axes[_i].imshow(lifetime_results[_i], cmap='viridis', vmin=2.0, vmax=3.5)
            _axes[_i].set_title(f'Frame {_i + 1} Lifetime', fontsize=12)
            _axes[_i].set_xlabel('X (pixels)')
            _axes[_i].set_ylabel('Y (pixels)')
            plt.colorbar(_im, ax=_axes[_i], label='Lifetime (ns)')
        for _i in range(n_frames, 6):
            _axes[_i].axis('off')
        plt.tight_layout()
        plt.show()  # Hide unused subplots
    else:
        print('No frames were processed. Make sure the data sender is running.')
    return


@app.cell
def _(data_manager):
    data_manager.get_frame_count()
    return


@app.cell
def _(data_manager, plt):
    x = data_manager.get_photon_count(0)
    plt.plot(x.sum((0,1)))
    return


@app.cell
def _(data_manager, np, plt):
    if data_manager is not None and data_manager.get_frame_count() > 1:
        print(f'Total frames in DataManager: {data_manager.get_frame_count()}')
        n_snapshots = min(3, data_manager.get_frame_count())
        _fig, _axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5))  # Retrieve and compare snapshots
        if n_snapshots == 1:
            _axes = [_axes]
        for _i in range(n_snapshots):
            snapshot = data_manager.get_photon_count(_i)
            intensity = np.sum(snapshot, axis=-1)
            _im = _axes[_i].imshow(intensity, cmap='gray')
            _axes[_i].set_title(f'Snapshot {_i} Intensity', fontsize=13)
            _axes[_i].set_xlabel('X (pixels)')  # Get snapshot
            _axes[_i].set_ylabel('Y (pixels)')
            plt.colorbar(_im, ax=_axes[_i], label='Total Counts')
        plt.tight_layout()  # Compute intensity (sum over time)
        plt.show()
    else:
        print('No snapshots available.')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
