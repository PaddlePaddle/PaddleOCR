from ._pyclipper import *

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
