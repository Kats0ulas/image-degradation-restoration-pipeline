"""Πακέτο υλοποίησης της αλυσίδας επεξεργασίας εικόνας (ΕΥ631)."""
import sys

# Η κονσόλα των Windows είναι συχνά cp1252 και «σπάει» στα ελληνικά.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
