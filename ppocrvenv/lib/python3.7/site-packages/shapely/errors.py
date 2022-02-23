"""Shapely errors."""
import warnings


class ShapelyError(Exception):
    """Base error class."""


class UnsupportedGEOSVersionError(ShapelyError):
    """Raised when the system's GEOS library version is unsupported."""


class ReadingError(ShapelyError):
    """A WKT or WKB reading error."""


class WKBReadingError(ReadingError):
    """A WKB reading error."""


class WKTReadingError(ReadingError):
    """A WKT reading error."""


class DimensionError(ShapelyError):
    """An error in the number of coordinate dimensions."""


class TopologicalError(ShapelyError):
    """A geometry is invalid or topologically incorrect."""


class PredicateError(ShapelyError):
    """A geometric predicate has failed to return True/False."""


class ShapelyDeprecationWarning(FutureWarning):
    """
    Warning for features that will be removed or behaviour that will be
    changed in a future release.
    """


class EmptyPartError(ShapelyError):
    """An error signifying an empty part was encountered when creating a multi-part."""


class GeometryTypeError(ShapelyError, TypeError, ValueError):
    """
    An error raised when the type of the geometry in question is
    unrecognized or inappropriate.
    """
    def __init__(self, msg):
        warnings.warn("GeometryTypeError will derive from ShapelyError and not TypeError or ValueError in Shapely 2.0.", ShapelyDeprecationWarning, stacklevel=2)
        super().__init__(msg)


class InvalidGeometryError(ShapelyError, TypeError, ValueError):
    """
    An error raised when an operation is attempted on a null geometry
    """
    def __init__(self, msg):
        warnings.warn("InvalidGeometryError will derive from ShapelyError and not TypeError or ValueError in Shapely 2.0.", ShapelyDeprecationWarning, stacklevel=2)
        super().__init__(msg)
