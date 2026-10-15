"""Implementation package of the image processing pipeline."""
import sys

# The Windows console is often cp1252 and breaks on non-ASCII symbols (σ, ∇, —).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
