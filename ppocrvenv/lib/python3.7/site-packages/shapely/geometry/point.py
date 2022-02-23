"""Points and related utilities
"""

from ctypes import c_double
import warnings

from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError, ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, geos_geom_from_py
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['Point', 'asPoint']


class Point(BaseGeometry):
    """
    A zero dimensional feature

    A point has zero length and zero area.

    Attributes
    ----------
    x, y, z : float
        Coordinate values

    Example
    -------
      >>> p = Point(1.0, -1.0)
      >>> print(p)
      POINT (1 -1)
      >>> p.y
      -1.0
      >>> p.x
      1.0
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        There are 2 cases:

        1) 1 parameter: this must satisfy the numpy array protocol.
        2) 2 or more parameters: x, y, z : float
            Easting, northing, and elevation.
        """
        BaseGeometry.__init__(self)
        if len(args) > 0:
            if len(args) == 1:
                geom, n = geos_point_from_py(args[0])
            elif len(args) > 3:
                raise TypeError(
                    "Point() takes at most 3 arguments ({} given)".format(len(args))
                )
            else:
                geom, n = geos_point_from_py(tuple(args))
            self._set_geom(geom)
            self._ndim = n

    # Coordinate getters and setters

    @property
    def x(self):
        """Return x coordinate."""
        return self.coords[0][0]

    @property
    def y(self):
        """Return y coordinate."""
        return self.coords[0][1]

    @property
    def z(self):
        """Return z coordinate."""
        if self._ndim != 3:
            raise DimensionError("This point has no z coordinate.")
        return self.coords[0][2]

    @property
    def __geo_interface__(self):
        return {
            'type': 'Point',
            'coordinates': self.coords[0]
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns SVG circle element for the Point geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameter.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is to use "#66cc99" if
            geometry is valid, and "#ff3333" if invalid.
        opacity : float
            Float number between 0 and 1 for color opacity. Defaul value is 0.6
        """
        if self.is_empty:
            return '<g />'
        if fill_color is None:
            fill_color = "#66cc99" if self.is_valid else "#ff3333"
        if opacity is None:
            opacity = 0.6 
        return (
            '<circle cx="{0.x}" cy="{0.y}" r="{1}" '
            'stroke="#555555" stroke-width="{2}" fill="{3}" opacity="{4}" />'
            ).format(self, 3. * scale_factor, 1. * scale_factor, fill_color, opacity)

    @property
    def _ctypes(self):
        if not self._ctypes_data:
            array_type = c_double * self._ndim
            array = array_type()
            xy = self.coords[0]
            array[0] = xy[0]
            array[1] = xy[1]
            if self._ndim == 3:
                array[2] = xy[2]
            self._ctypes_data = array
        return self._ctypes_data

    def _array_interface(self):
        """Provide the Numpy array protocol."""
        if self.is_empty:
            ai = {'version': 3, 'typestr': '<f8', 'shape': (0,), 'data': (c_double * 0)()}
        else:
            ai = self._array_interface_base
            ai.update({'shape': (self._ndim,)})
        return ai

    def array_interface(self):
        """Provide the Numpy array protocol."""
        warnings.warn(
            "The 'array_interface' method is deprecated and will be removed "
            "in Shapely 2.0.",
            ShapelyDeprecationWarning, stacklevel=2)
        return self._array_interface()

    @property
    def __array_interface__(self):
        warnings.warn(
            "The array interface is deprecated and will no longer work in "
            "Shapely 2.0. Convert the '.coords' to a numpy array instead.",
            ShapelyDeprecationWarning, stacklevel=3)
        return self._array_interface()

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        try:
            xy = self.coords[0]
        except IndexError:
            return ()
        return (xy[0], xy[1], xy[0], xy[1])

    # Coordinate access

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        return CoordinateSequence(self)

    def _set_coords(self, *args):
        warnings.warn(
            "Setting the 'coords' to mutate a Geometry in place is deprecated,"
            " and will not be possible any more in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        self._empty()
        if len(args) == 1:
            geom, n = geos_point_from_py(args[0])
        elif len(args) > 3:
            raise TypeError("Point() takes at most 3 arguments ({} given)".format(len(args)))
        else:
            geom, n = geos_point_from_py(tuple(args))
        self._set_geom(geom)
        self._ndim = n

    coords = property(_get_coords, _set_coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values

        Example:
          >>> x, y = Point(0, 0).xy
          >>> list(x)
          [0.0]
          >>> list(y)
          [0.0]
        """
        return self.coords.xy


class PointAdapter(CachingGeometryProxy, Point):

    _other_owned = False

    def __init__(self, context):
        warnings.warn(
            "The proxy geometries (through the 'asShape()', 'asPoint()' or "
            "'PointAdapter()' constructors) are deprecated and will be "
            "removed in Shapely 2.0. Use the 'shape()' function or the "
            "standard 'Point()' constructor instead.",
            ShapelyDeprecationWarning, stacklevel=4)
        self.context = context
        self.factory = geos_point_from_py

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context.__array_interface__
            n = array['shape'][0]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context)

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        try:
            return self.context.__array_interface__
        except AttributeError:
            return self.array_interface()

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        return CoordinateSequence(self)

    def _set_coords(self, ob):
        raise NotImplementedError("Adapters can not modify their sources")

    coords = property(_get_coords)


def asPoint(context):
    """Adapt an object to the Point interface"""
    return PointAdapter(context)


def geos_point_from_py(ob, update_geom=None, update_ndim=0):
    """Create a GEOS geom from an object that is a Point, a coordinate sequence
    or that provides the array interface.

    Returns the GEOS geometry and the number of its dimensions.
    """
    if isinstance(ob, Point):
        return geos_geom_from_py(ob)

    # Accept either (x, y) or [(x, y)]
    if not hasattr(ob, '__getitem__'):  # generators
        ob = list(ob)

    if isinstance(ob[0], tuple):
        coords = ob[0]
    else:
        coords = ob
    n = len(coords)
    dx = c_double(coords[0])
    dy = c_double(coords[1])
    dz = None
    if n == 3:
        dz = c_double(coords[2])

    if update_geom:
        cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
        if n != update_ndim:
            raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: "
                "%d" % update_ndim)
    else:
        cs = lgeos.GEOSCoordSeq_create(1, n)

    # Because of a bug in the GEOS C API, always set X before Y
    lgeos.GEOSCoordSeq_setX(cs, 0, dx)
    lgeos.GEOSCoordSeq_setY(cs, 0, dy)
    if n == 3:
        lgeos.GEOSCoordSeq_setZ(cs, 0, dz)

    if update_geom:
        return None
    else:
        return lgeos.GEOSGeom_createPoint(cs), n


def update_point_from_py(geom, ob):
    geos_point_from_py(ob, geom._geom, geom._ndim)
