"""Collections of points and related utilities
"""

from ctypes import byref, c_double, c_void_p, cast
import warnings

from shapely.errors import EmptyPartError, ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry.base import (
    BaseMultipartGeometry, exceptNull, geos_geom_from_py)
from shapely.geometry import point
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiPoint', 'asMultiPoint']


class MultiPoint(BaseMultipartGeometry):

    """A collection of one or more points

    A MultiPoint has zero area and zero length.

    Attributes
    ----------
    geoms : sequence
        A sequence of Points
    """

    def __init__(self, points=None):
        """
        Parameters
        ----------
        points : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples or a
            sequence of objects that implement the numpy array interface,
            including instances of Point.

        Example
        -------
        Construct a 2 point collection

          >>> from shapely.geometry import Point
          >>> ob = MultiPoint([[0.0, 0.0], [1.0, 2.0]])
          >>> len(ob.geoms)
          2
          >>> type(ob.geoms[0]) == Point
          True
        """
        super().__init__()

        if points is None or (not isinstance(points, MultiPoint) and len(points) == 0):
            # allow creation of empty multipoints, to support unpickling
            pass
        else:
            geom, n = geos_multipoint_from_py(points)
            self._set_geom(geom)
            self._ndim = n

    def shape_factory(self, *args):
        return point.Point(*args)

    @property
    def __geo_interface__(self):
        return {
            'type': 'MultiPoint',
            'coordinates': tuple([g.coords[0] for g in self.geoms])
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns a group of SVG circle elements for the MultiPoint geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG circle diameters.  Default is 1.
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
        return '<g>' + \
            ''.join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms) + \
            '</g>'

    @property
    @exceptNull
    def _ctypes(self):
        if not self._ctypes_data:
            temp = c_double()
            n = self._ndim
            m = len(self.geoms)
            array_type = c_double * (m * n)
            data = array_type()
            for i in range(m):
                g = self.geoms[i]._geom
                cs = lgeos.GEOSGeom_getCoordSeq(g)
                lgeos.GEOSCoordSeq_getX(cs, 0, byref(temp))
                data[n*i] = temp.value
                lgeos.GEOSCoordSeq_getY(cs, 0, byref(temp))
                data[n*i+1] = temp.value
                if n == 3: # TODO: use hasz
                    lgeos.GEOSCoordSeq_getZ(cs, 0, byref(temp))
                    data[n*i+2] = temp.value
            self._ctypes_data = data
        return self._ctypes_data

    @property
    def ctypes(self):
        warnings.warn(
            "Accessing the 'ctypes' attribute is deprecated,"
            " and will not be possible any more in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        return self._ctypes

    @exceptNull
    def _array_interface(self):
        """Provide the Numpy array protocol."""
        ai = self._array_interface_base
        ai.update({'shape': (len(self.geoms), self._ndim)})
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


class MultiPointAdapter(CachingGeometryProxy, MultiPoint):

    context = None
    _other_owned = False

    def __init__(self, context):
        warnings.warn(
            "The proxy geometries (through the 'asShape()', 'asMultiPoint()' "
            "or 'MultiPointAdapter()' constructors) are deprecated and will be "
            "removed in Shapely 2.0. Use the 'shape()' function or the "
            "standard 'MultiPoint()' constructor instead.",
            ShapelyDeprecationWarning, stacklevel=4)
        self.context = context
        self.factory = geos_multipoint_from_py

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context.__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context[0])

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        try:
            return self.context.__array_interface__
        except AttributeError:
            return self.array_interface()


def asMultiPoint(context):
    """Adapt a sequence of objects to the MultiPoint interface"""
    return MultiPointAdapter(context)


def geos_multipoint_from_py(ob):
    if isinstance(ob, MultiPoint):
        return geos_geom_from_py(ob)

    m = len(ob)
    try:
        n = len(ob[0])
    except TypeError:
        n = ob[0]._ndim
    assert n == 2 or n == 3

    # Array of pointers to point geometries
    subs = (c_void_p * m)()

    # add to coordinate sequence
    for i in range(m):
        coords = ob[i]
        geom, ndims = point.geos_point_from_py(coords)

        if lgeos.GEOSisEmpty(geom):
            raise EmptyPartError("Can't create MultiPoint with empty component")

        subs[i] = cast(geom, c_void_p)

    return lgeos.GEOSGeom_createCollection(4, subs, m), n
