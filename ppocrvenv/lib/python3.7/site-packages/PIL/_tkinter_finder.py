""" Find compiled module linking to Tcl / Tk libraries
"""
import sys
import tkinter
import warnings
from tkinter import _tkinter as tk

if hasattr(sys, "pypy_find_executable"):
    TKINTER_LIB = tk.tklib_cffi.__file__
else:
    TKINTER_LIB = tk.__file__

tk_version = str(tkinter.TkVersion)
if tk_version == "8.4":
    warnings.warn(
        "Support for Tk/Tcl 8.4 is deprecated and will be removed"
        " in Pillow 10 (2023-07-01). Please upgrade to Tk/Tcl 8.5 "
        "or newer.",
        DeprecationWarning,
    )
