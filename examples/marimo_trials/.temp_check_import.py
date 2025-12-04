import sys, traceback
print('PYTHON', sys.executable)
print('PATHS:', sys.path[:8])
try:
    import marimo
    print('IMPORT OK', getattr(marimo, '__file__', repr(marimo)))
except Exception:
    traceback.print_exc()
