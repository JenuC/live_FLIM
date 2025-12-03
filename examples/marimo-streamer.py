"""
Marimo-compatible streamer app scaffold
Generated: 2025-12-03T13:28:07.700Z

This script exposes a simple Marimo app with:
- Live intensity image widget
- Start / Stop controls
- Status text

If marimo or the project's flim_processing modules are not available, the app falls back to a CLI preview mode.
"""
from datetime import datetime

GENERATED_AT = "2025-12-03T13:28:07.700Z"

import sys
import os
# Search up to 3 levels up for a '.venv' directory and add its site-packages to sys.path
try:
    base = os.path.dirname(__file__)
    found = False
    for _ in range(4):
        candidate = os.path.join(base, '.venv')
        if os.path.isdir(candidate):
            if os.name == 'nt':
                site = os.path.join(candidate, 'Lib', 'site-packages')
            else:
                site = os.path.join(candidate, 'lib', 'python3', 'site-packages')
            if os.path.isdir(site) and site not in sys.path:
                sys.path.insert(0, site)
                found = True
                break
        base = os.path.dirname(base)
    # fallback: also check user-roaming uv venv used by uv manager
    if not found:
        roaming_uv = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'uv', 'python') if os.name == 'nt' else None
        if roaming_uv and os.path.isdir(roaming_uv):
            # find site-packages under this structure
            for root, dirs, _ in os.walk(roaming_uv):
                if root.endswith('site-packages'):
                    if root not in sys.path:
                        sys.path.insert(0, root)
                        break
except Exception:
    pass

# Debug import of marimo
try:
    import traceback as _traceback
    try:
        import marimo
        # marimo exposes md and image as top-level helpers
        md = getattr(marimo, 'md', None)
        image_fn = getattr(marimo, 'image', None)
        if md is None:
            # fallback to plugin namespace
            try:
                from marimo import _plugins as _plugins
                md = getattr(_plugins, 'ui', None)
            except Exception:
                md = None
        MARIMO_AVAILABLE = md is not None
        if not MARIMO_AVAILABLE:
            print('DEBUG: marimo imported but md image helpers not found', file=sys.stderr)
    except Exception as _e:
        MARIMO_AVAILABLE = False
        print('DEBUG: failed importing marimo:', file=sys.stderr)
        print('DEBUG: sys.executable =', sys.executable, file=sys.stderr)
        print('DEBUG: sys.path =', sys.path, file=sys.stderr)
        _traceback.print_exc()
except Exception:
    MARIMO_AVAILABLE = False
    print('DEBUG: unexpected error during marimo import check', file=sys.stderr)

try:
    from flim_processing import StreamReceiver, DataManager
    FLIM_AVAILABLE = True
except Exception:
    FLIM_AVAILABLE = False

import threading
import time
import numpy as np

# Simple manager shim if DataManager not available
class _SimpleManager:
    def __init__(self):
        self.frames = []
    def add_packet(self, pkt):
        # pkt should include photon counts as a 3D array; shim: append random
        self.frames.append(np.random.poisson(1.0, size=(64,64,1)))
    def get_frame_count(self):
        return len(self.frames)
    def get_photon_count(self, i):
        return self.frames[i]

class MarimoStreamer:
    def __init__(self):
        self.running = False
        self.thread = None
        self.receiver = None
        # DataManager requires (shape, dtype). Use a sensible default and allow
        # the stream to reinitialize manager when new_series arrives.
        if FLIM_AVAILABLE:
            try:
                self.manager = DataManager((64,64,1), np.dtype(np.uint16))
            except Exception:
                self.manager = _SimpleManager()
        else:
            self.manager = _SimpleManager()

    def start(self, port=4444, addr='127.0.0.1'):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(port, addr), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run(self, port, addr):
        count = 0
        if FLIM_AVAILABLE:
            self.receiver = StreamReceiver(port=port, addr=addr)
            try:
                for event in self.receiver.start_receiving():
                    if not self.running:
                        break
                    # Handle new series metadata
                    from flim_processing._dataclasses import SeriesMetadata, ElementData, EndSeries
                    if isinstance(event, SeriesMetadata):
                        # Reinitialize manager with correct shape/dtype
                        try:
                            self.manager = DataManager(event.shape, event.dtype)
                        except Exception:
                            self.manager = _SimpleManager()
                    elif isinstance(event, ElementData):
                        # event.frame is the numpy array for the element
                        # Use add_element if available, else add_packet
                        try:
                            self.manager.add_element(event.seqno, event.frame)
                        except Exception:
                            try:
                                self.manager.add_packet(event.frame)
                            except Exception:
                                pass
                    elif isinstance(event, EndSeries):
                        # no-op for now
                        pass
                    count += 1
                    if not self.running:
                        break
            finally:
                try:
                    self.receiver.stop_receiving()
                except Exception:
                    pass
        else:
            while self.running:
                # synthesize
                self.manager.add_packet(None)
                count += 1
                time.sleep(0.2)

    def latest_intensity(self):
        c = self.manager.get_frame_count()
        if c == 0:
            return None
        snap = self.manager.get_photon_count(c-1)
        # sum last axis -> intensity
        return np.sum(snap, axis=-1)


def create_marimo_app():
    app = MarimoStreamer()
    if not MARIMO_AVAILABLE:
        print('Marimo not installed; run in CLI mode')
        return app

    # Build marimo dashboard using marimo public helpers
    # Use the ui plugin for buttons and stateless.image for image component
    from marimo import md as marimo_md
    from marimo._plugins.ui import button
    from marimo._plugins.stateless.image import image as stateless_image

    start_btn = button('Start')
    stop_btn = button('Stop')
    status_txt = marimo_md.text('status') if hasattr(marimo_md, 'text') else None
    img = stateless_image('intensity')

    def on_start(_):
        app.start()
        if status_txt is not None:
            status_txt.set('running')
    def on_stop(_):
        app.stop()
        if status_txt is not None:
            status_txt.set('stopped')

    start_btn.on_click(on_start)
    stop_btn.on_click(on_stop)

    def refresh_image():
        while True:
            if not app.running:
                time.sleep(0.5)
                continue
            intensity = app.latest_intensity()
            if intensity is not None:
                arr = intensity.astype(float)
                arr = arr - arr.min()
                if arr.max() > 0:
                    arr = (arr / arr.max() * 255.0).astype('uint8')
                # stateless_image expects a numpy array via its set method if available
                try:
                    img.set(arr)
                except Exception:
                    try:
                        img.set_numpy(arr)
                    except Exception:
                        pass
            time.sleep(0.2)

    threading.Thread(target=refresh_image, daemon=True).start()

    # Serve via marimo's app runtime
    try:
        marimo._runtime.app_meta.AppMeta().run(lambda: None)
    except Exception:
        pass
    return app


if __name__ == '__main__':
    print('Generated at', GENERATED_AT)
    if MARIMO_AVAILABLE:
        create_marimo_app()
    else:
        print('Marimo not present; running CLI preview for 5 seconds')
        s = MarimoStreamer()
        s.start()
        start = time.time()
        try:
            while time.time() - start < 5.0:
                it = s.latest_intensity()
                if it is not None:
                    print('Latest intensity shape', it.shape)
                time.sleep(0.5)
        finally:
            s.stop()
