"""Polygons and their linear ring components
"""

import sys
import warnings

from ctypes import c_void_p, cast, POINTER
import weakref

from shapely.algorithms.cga import signed_area
from shapely.coords import CoordinateSequence
from shapely.geos import lgeos
from shapely.geometry.base import BaseGeometry, geos_geom_from_py
from shapely.geometry.linestring import LineString, LineStringAdapter
from shapely.geometry.point import Point
from shapely.geometry.proxy import PolygonProxy
from shapely.errors import TopologicalError, ShapelyDeprecationWarning


__all__ = ['Polygon', 'asPolygon', 'LinearRing', 'asLinearRing']


class LinearRing(LineString):
    """
    A closed one-dimensional feature comprising one or more line segments

    A LinearRing that crosses itself or touches itself at a single point is
    invalid and operations on it may fail.
    """

    def __init__(self, coordinates=None):
        """
        Parameters
        ----------
        coordinates : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples.
            Also can be a sequence of Point objects.

        Rings are implicitly closed. There is no need to specific a final
        coordinate pair identical to the first.

        Example
        -------
        Construct a square ring.

          >>> ring = LinearRing( ((0, 0), (0, 1), (1 ,1 ), (1 , 0)) )
          >>> ring.is_closed
          True
          >>> ring.length
          4.0
        """
        BaseGeometry.__init__(self)
        if coordinates is not None:
            ret = geos_linearring_from_py(coordinates)
            if ret is not None:
                geom, n = ret
                self._set_geom(geom)
                self._ndim = n

    @property
    def __geo_interface__(self):
        return {
            'type': 'LinearRing',
            'coordinates': tuple(self.coords)
            }

    # Coordinate access

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        return CoordinateSequence(self)

    def _set_coords(self, coordinates):
        warnings.warn(
            "Setting the 'coords' to mutate a Geometry in place is deprecated,"
            " and will not be possible any more in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        self._empty()
        ret = geos_linearring_from_py(coordinates)
        if ret is not None:
            geom, n = ret
            self._set_geom(geom)
            self._ndim = n

    coords = property(_get_coords, _set_coords)

    def __setstate__(self, state):
        """WKB doesn't differentiate between LineString and LinearRing so we
        need to move the coordinate sequence into the correct geometry type"""
        super().__setstate__(state)
        cs = lgeos.GEOSGeom_getCoordSeq(self.__geom__)
        cs_clone = lgeos.GEOSCoordSeq_clone(cs)
        lgeos.GEOSGeom_destroy(self.__geom__)
        self.__geom__ = lgeos.GEOSGeom_createLinearRing(cs_clone)

    @property
    def is_ccw(self):
        """True is the ring is oriented counter clock-wise"""
        return bool(self.impl['is_ccw'](self))

    @property
    def is_simple(self):
        """True if the geometry is simple, meaning that any self-intersections
        are only at boundary points, else False"""
        return LineString(self).is_simple


class LinearRingAdapter(LineStringAdapter):

    __p__ = None

    def __init__(self, context):
        warnings.warn(
            "The proxy geometries (through the 'asShape()', 'asLinearRing()' or "
            "'LinearRingAdapter()' constructors) are deprecated and will be "
            "removed in Shapely 2.0. Use the 'shape()' function or the "
            "standard 'LinearRing()' constructor instead.",
            ShapelyDeprecationWarning, stacklevel=3)
        self.context = context
        self.factory = geos_linearring_from_py

    @property
    def __geo_interface__(self):
        return {
            'type': 'LinearRing',
            'coordinates': tuple(self.coords)
            }

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        return CoordinateSequence(self)

    coords = property(_get_coords)


def asLinearRing(context):
    """Adapt an object to the LinearRing interface"""
    return LinearRingAdapter(context)


class InteriorRingSequence:

    _factory = None
    _geom = None
    __p__ = None
    _ndim = None
    _index = 0
    _length = 0
    __rings__ = None
    _gtag = None

    def __init__(self, parent):
        self.__p__ = parent
        self._geom = parent._geom
        self._ndim = parent._ndim

    def __iter__(self):
        self._index = 0
        self._length = self.__len__()
        return self

    def __next__(self):
        if self._index < self._length:
            ring = self._get_ring(self._index)
            self._index += 1
            return ring
        else:
            raise StopIteration

    def __len__(self):
        return lgeos.GEOSGetNumInteriorRings(self._geom)

    def __getitem__(self, key):
        m = self.__len__()
        if isinstance(key, int):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_ring(i)
        elif isinstance(key, slice):
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_ring(i))
            return res
        else:
            raise TypeError("key must be an index or slice")

    @property
    def _longest(self):
        max = 0
        for g in iter(self):
            l = len(g.coords)
            if l > max:
                max = l

    def gtag(self):
        return hash(repr(self.__p__))

    def _get_ring(self, i):
        gtag = self.gtag()
        if gtag != self._gtag:
            self.__rings__ = {}
        if i not in self.__rings__:
            g = lgeos.GEOSGetInteriorRingN(self._geom, i)
            ring = LinearRing()
            ring._set_geom(g)
            ring.__p__ = self
            ring._other_owned = True
            ring._ndim = self._ndim
            self.__rings__[i] = weakref.ref(ring)
        return self.__rings__[i]()


class Polygon(BaseGeometry):
    """
    A two-dimensional figure bounded by a linear ring

    A polygon has a non-zero area. It may have one or more negative-space
    "holes" which are also bounded by linear rings. If any rings cross each
    other, the feature is invalid and operations on it may fail.

    Attributes
    ----------
    exterior : LinearRing
        The ring which bounds the positive space of the polygon.
    interiors : sequence
        A sequence of rings which bound all existing holes.
    """

    _exterior = None
    _interiors = []
    _ndim = 2

    def __init__(self, shell=None, holes=None):
        """
        Parameters
        ----------
        shell : sequence
            A sequence of (x, y [,z]) numeric coordinate pairs or triples.
            Also can be a sequence of Point objects.
        holes : sequence
            A sequence of objects which satisfy the same requirements as the
            shell parameters above

        Example
        -------
        Create a square polygon with no holes

          >>> coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
          >>> polygon = Polygon(coords)
          >>> polygon.area
          1.0
        """
        BaseGeometry.__init__(self)

        if shell is not None:
            ret = geos_polygon_from_py(shell, holes)
            if ret is not None:
                geom, n = ret
                self._set_geom(geom)
                self._ndim = n
            else:
                self._empty()

    @property
    def exterior(self):
        if self.is_empty:
            return LinearRing()
        elif self._exterior is None or self._exterior() is None:
            g = lgeos.GEOSGetExteriorRing(self._geom)
            ring = LinearRing()
            ring._set_geom(g)
            ring.__p__ = self
            ring._other_owned = True
            ring._ndim = self._ndim
            self._exterior = weakref.ref(ring)
        return self._exterior()

    @property
    def interiors(self):
        if self.is_empty:
            return []
        return InteriorRingSequence(self)

    def __eq__(self, other):
        if not isinstance(other, Polygon):
            return False
        check_empty = (self.is_empty, other.is_empty)
        if all(check_empty):
            return True
        elif any(check_empty):
            return False
        my_coords = [
            tuple(self.exterior.coords),
            [tuple(interior.coords) for interior in self.interiors]
        ]
        other_coords = [
            tuple(other.exterior.coords),
            [tuple(interior.coords) for interior in other.interiors]
        ]
        return my_coords == other_coords

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    @property
    def _ctypes(self):
        if not self._ctypes_data:
            self._ctypes_data = self.exterior._ctypes
        return self._ctypes_data

    @property
    def __array_interface__(self):
        raise AttributeError(
        "A polygon does not itself provide the array interface. Its rings do.")

    def _get_coords(self):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    def _set_coords(self, ob):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    @property
    def coords(self):
        raise NotImplementedError(
        "Component rings have coordinate sequences, but the polygon does not")

    @property
    def __geo_interface__(self):
        if self.exterior == LinearRing():
            coords = []
        else:
            coords = [tuple(self.exterior.coords)]
            for hole in self.interiors:
                coords.append(tuple(hole.coords))
        return {
            'type': 'Polygon',
            'coordinates': tuple(coords)}

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns SVG path element for the Polygon geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
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
        exterior_coords = [
            ["{},{}".format(*c) for c in self.exterior.coords]]
        interior_coords = [
            ["{},{}".format(*c) for c in interior.coords]
            for interior in self.interiors]
        path = " ".join([
            "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
            for coords in exterior_coords + interior_coords])
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="{3}" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color, opacity)

    @classmethod
    def from_bounds(cls, xmin, ymin, xmax, ymax):
        """Construct a `Polygon()` from spatial bounds."""
        return cls([
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin)])


class PolygonAdapter(PolygonProxy, Polygon):

    def __init__(self, shell, holes=None):
        warnings.warn(
            "The proxy geometries (through the 'asShape()', 'asPolygon()' or "
            "'PolygonAdapter()' constructors) are deprecated and will be "
            "removed in Shapely 2.0. Use the 'shape()' function or the "
            "standard 'Polygon()' constructor instead.",
            ShapelyDeprecationWarning, stacklevel=4)
        self.shell = shell
        self.holes = holes
        self.context = (shell, holes)
        self.factory = geos_polygon_from_py

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.shell.__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.shell[0])


def asPolygon(shell, holes=None):
    """Adapt objects to the Polygon interface"""
    return PolygonAdapter(shell, holes)


def orient(polygon, sign=1.0):
    s = float(sign)
    rings = []
    ring = polygon.exterior
    if signed_area(ring)/s >= 0.0:
        rings.append(ring)
    else:
        rings.append(list(ring.coords)[::-1])
    for ring in polygon.interiors:
        if signed_area(ring)/s <= 0.0:
            rings.append(ring)
        else:
            rings.append(list(ring.coords)[::-1])
    return Polygon(rings[0], rings[1:])


def geos_linearring_from_py(ob, update_geom=None, update_ndim=0):
    # If a LinearRing is passed in, clone it and return
    # If a valid LineString is passed in, clone the coord seq and return a
    # LinearRing.
    #
    # NB: access to coordinates using the array protocol has been moved
    # entirely to the speedups module.

    if isinstance(ob, LineString):
        if type(ob) == LinearRing:
            return geos_geom_from_py(ob)
        elif not ob.is_valid:
            raise TopologicalError("An input LineString must be valid.")
        elif ob.is_closed and len(ob.coords) >= 4:
            return geos_geom_from_py(ob, lgeos.GEOSGeom_createLinearRing)
        else:
            ob = list(ob.coords)

    try:
        m = len(ob)
    except TypeError:  # generators
        ob = list(ob)
        m = len(ob)

    if m == 0:
        return None

    def _coords(o):
        if isinstance(o, Point):
            return o.coords[0]
        else:
            return o

    n = len(_coords(ob[0]))
    if m < 3:
        raise ValueError(
            "A LinearRing must have at least 3 coordinate tuples")
    assert (n == 2 or n == 3)

    # Add closing coordinates if not provided
    if (
        m == 3
        or _coords(ob[0])[0] != _coords(ob[-1])[0]
        or _coords(ob[0])[1] != _coords(ob[-1])[1]
    ):
        M = m + 1
    else:
        M = m

    # Create a coordinate sequence
    if update_geom is not None:
        if n != update_ndim:
            raise ValueError(
                "Coordinate dimensions mismatch: target geom has {} dims, "
                "update geom has {} dims".format(n, update_ndim))
        cs = lgeos.GEOSGeom_getCoordSeq(update_geom)
    else:
        cs = lgeos.GEOSCoordSeq_create(M, n)

    # add to coordinate sequence
    for i in range(m):
        coords = _coords(ob[i])
        # Because of a bug in the GEOS C API,
        # always set X before Y
        lgeos.GEOSCoordSeq_setX(cs, i, coords[0])
        lgeos.GEOSCoordSeq_setY(cs, i, coords[1])
        if n == 3:
            try:
                lgeos.GEOSCoordSeq_setZ(cs, i, coords[2])
            except IndexError:
                raise ValueError("Inconsistent coordinate dimensionality")

    # Add closing coordinates to sequence?
    if M > m:
        coords = _coords(ob[0])
        # Because of a bug in the GEOS C API,
        # always set X before Y
        lgeos.GEOSCoordSeq_setX(cs, M-1, coords[0])
        lgeos.GEOSCoordSeq_setY(cs, M-1, coords[1])
        if n == 3:
            lgeos.GEOSCoordSeq_setZ(cs, M-1, coords[2])

    if update_geom is not None:
        return None
    else:
        return lgeos.GEOSGeom_createLinearRing(cs), n


def update_linearring_from_py(geom, ob):
    geos_linearring_from_py(ob, geom._geom, geom._ndim)


def geos_polygon_from_py(shell, holes=None):

    if shell is None:
        return None

    if isinstance(shell, Polygon):
        return geos_geom_from_py(shell)

    if shell is not None:
        ret = geos_linearring_from_py(shell)
        if ret is None:
            return None

        geos_shell, ndim = ret
        if holes is not None and len(holes) > 0:
            ob = holes
            L = len(ob)
            exemplar = ob[0]
            try:
                N = len(exemplar[0])
            except TypeError:
                N = exemplar._ndim
            if not L >= 1:
                raise ValueError("number of holes must be non zero")
            if N not in (2, 3):
                raise ValueError("insufficiant coordinate dimension")

            # Array of pointers to ring geometries
            geos_holes = (c_void_p * L)()

            # add to coordinate sequence
            for l in range(L):
                geom, ndim = geos_linearring_from_py(ob[l])
                geos_holes[l] = cast(geom, c_void_p)
        else:
            geos_holes = POINTER(c_void_p)()
            L = 0

        return (
            lgeos.GEOSGeom_createPolygon(
                c_void_p(geos_shell), geos_holes, L), ndim)
