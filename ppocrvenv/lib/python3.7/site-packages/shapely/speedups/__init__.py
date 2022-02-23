import warnings
from functools import wraps

from shapely.geometry import linestring, polygon
from shapely import coords
import shapely.affinity


try:
    from shapely.speedups import _speedups
    available = True
    import_error_msg = None
except ImportError:
    import sys
    available = False
    # TODO: This does not appear to do anything useful
    import_error_msg = sys.exc_info()[1]


def method_wrapper(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wraps(f)(wrapper)

__all__ = ['available', 'enable', 'disable', 'enabled']
_orig = {}

# keep track of whether speedups are enabled
enabled = False


def enable():
    """Enable Cython speedups

    The shapely.speedups module contains performance enhancements written in C.
    They are automatically installed when Python has access to a compiler and
    GEOS development headers during installation, and are enabled by default.

    You can check if speedups are installed with the `available` attribute, and
    check if they have been enabled with the `enabled` attribute.

    >>> from shapely import speedups
    >>> speedups.available
    True
    >>> speedups.enable()
    >>> speedups.enabled
    True
    """
    if not available:
        warnings.warn("shapely.speedups not available", RuntimeWarning)
        return

    if _orig:
        return

    _orig['CoordinateSequence._ctypes'] = coords.CoordinateSequence._ctypes
    coords.CoordinateSequence._ctypes = property(_speedups.coordseq_ctypes)

    _orig['CoordinateSequence.__iter__'] = coords.CoordinateSequence.__iter__
    coords.CoordinateSequence.__iter__ = method_wrapper(
        _speedups.coordseq_iter)

    _orig['geos_linestring_from_py'] = linestring.geos_linestring_from_py
    linestring.geos_linestring_from_py = _speedups.geos_linestring_from_py

    _orig['geos_linearring_from_py'] = polygon.geos_linearring_from_py
    polygon.geos_linearring_from_py = _speedups.geos_linearring_from_py

    _orig['affine_transform'] = shapely.affinity.affine_transform

    # copy docstring from original function
    def affine_transform(geom, matrix):
        return _speedups.affine_transform(geom, matrix)
    affine_transform.__doc__ = shapely.affinity.affine_transform.__doc__
    shapely.affinity.affine_transform = affine_transform

    global enabled
    enabled = True


def disable():
    """Disable Cython speedups
    """
    if not _orig:
        return

    coords.CoordinateSequence._ctypes = _orig['CoordinateSequence._ctypes']
    coords.CoordinateSequence.__iter__ = _orig['CoordinateSequence.__iter__']
    linestring.geos_linestring_from_py = _orig['geos_linestring_from_py']
    polygon.geos_linearring_from_py = _orig['geos_linearring_from_py']
    shapely.affinity.affine_transform = _orig['affine_transform']
    _orig.clear()

    global enabled
    enabled = False

# if cython speedups are available, use them by default
if available:
    enable()
