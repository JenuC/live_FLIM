import marimo as mo
import time

# --- State Management ---
# 1. State variable to track whether the stream is active (controlled by the button)
is_streaming = mo.state(False)

# 2. State variable to track the current frame ID (updated by the background stream)
frame_id = mo.state(0)

# --- UI Element: Button ---
@is_streaming.setter
def _stream_setter(val):
    """
    Called when the button is clicked. Toggles the is_streaming state.
    """
    is_streaming.set(not is_streaming.value)

stream_button = mo.ui.button(
    value=is_streaming.value,
    on_change=_stream_setter,
    label="Stop Stream" if is_streaming.value else "Start 4 FPS Stream",
    color="red" if is_streaming.value else "green",
    full_width=False
)

# --- Display Logic ---

# In a full Marimo application, the following would be an asynchronous
# generator function decorated with `@mo.generator` or `@mo.stream`
# which would run in a separate thread/task.
#
# Since the script must be runnable and synchronous, we simulate the effect
# by making the output react to the state variables.

if is_streaming.value:
    # **NOTE ON 4 FPS LOGIC:**
    # In a real Marimo component, the 4 FPS update logic would be running
    # in a background task here, updating `frame_id.set(...)` every 0.25 seconds.
    # E.g., a function like this would run:
    #
    # @mo.stream
    # def frame_updater(start_frame):
    #     i = start_frame
    #     while True:
    #         i += 1
    #         frame_id.set(i)
    #         time.sleep(0.25) # 4 FPS = 1 second / 4 frames
    #         yield mo.nothing
    #
    # For this demonstration, the image is only updated when the button is pressed,
    # but the URL changes based on the live `frame_id.value`.

    # A simple, temporary way to show the counter updating:
    frame_id.set(frame_id.value + 1)
    
    status_text = mo.md(f"**Status:** **_STREAMING_** | Current Frame: `{frame_id.value}`")
else:
    status_text = mo.md(f"**Status:** **_PAUSED_** | Last Frame: `{frame_id.value}`")


# Generate a placeholder image URL that changes with the frame ID
# Adding the frame ID to the text forces the browser to fetch a new image,
# simulating a real stream.
image_url = (
    f"https://placehold.co/600x400/0c4a6e/ffffff"
    f"?text=MARIMO+FRAME+{frame_id.value}%0A@+4FPS"
)

# Create the image display element
image_display = mo.image(image_url, alt=f"Video stream frame {frame_id.value}")

# --- Final Output Arrangement ---
# Display the button, the status, and the image panel
mo.vstack([
    stream_button,
    status_text,
    mo.div(
        mo.as_html(image_display),
        style={
            "border": "4px solid #0c4a6e",
            "border-radius": "12px",
            "padding": "20px",
            "background-color": "#1f2937", # Tailwind gray-800
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
            "margin-top": "15px"
        }
    )
])