"""Base geometry class and utilities

Note: a third, z, coordinate value may be used when constructing
geometry objects, but has no effect on geometric analysis. All
operations are performed in the x-y plane. Thus, geometries with
different z values may intersect or be equal.
"""

from binascii import a2b_hex
from ctypes import pointer, c_size_t, c_char_p, c_void_p
from itertools import islice
import logging
import math
import sys
from warnings import warn
from functools import wraps
import warnings

from shapely.affinity import affine_transform
from shapely.coords import CoordinateSequence
from shapely.errors import GeometryTypeError, WKBReadingError, WKTReadingError
from shapely.errors import ShapelyDeprecationWarning
from shapely.geos import WKBWriter, WKTWriter
from shapely.geos import lgeos
from shapely.impl import DefaultImplementation, delegated

log = logging.getLogger(__name__)

try:
    import numpy as np
    integer_types = (int, np.integer)
except ImportError:
    integer_types = (int,)


GEOMETRY_TYPES = [
    'Point',
    'LineString',
    'LinearRing',
    'Polygon',
    'MultiPoint',
    'MultiLineString',
    'MultiPolygon',
    'GeometryCollection',
]


def dump_coords(geom):
    """Dump coordinates of a geometry in the same order as data packing"""
    if not isinstance(geom, BaseGeometry):
        raise ValueError('Must be instance of a geometry class; found ' +
                         geom.__class__.__name__)
    elif geom.type in ('Point', 'LineString', 'LinearRing'):
        return geom.coords[:]
    elif geom.type == 'Polygon':
        return geom.exterior.coords[:] + [i.coords[:] for i in geom.interiors]
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        # Recursive call
        return [dump_coords(part) for part in geom.geoms]
    else:
        raise GeometryTypeError('Unhandled geometry type: ' + repr(geom.type))


def geometry_type_name(g):
    if g is None:
        raise ValueError("Null geometry has no type")
    return GEOMETRY_TYPES[lgeos.GEOSGeomTypeId(g)]


def geom_factory(g, parent=None):
    # Abstract geometry factory for use with topological methods below
    if not g:
        raise ValueError("No Shapely geometry can be created from null value")
    ob = BaseGeometry()
    geom_type = geometry_type_name(g)
    # TODO: check cost of dynamic import by profiling
    mod = __import__(
        'shapely.geometry',
        globals(),
        locals(),
        [geom_type],
        )
    ob.__class__ = getattr(mod, geom_type)
    ob._set_geom(g)
    ob.__p__ = parent
    if lgeos.methods['has_z'](g):
        ob._ndim = 3
    else:
        ob._ndim = 2
    ob._is_empty = False
    return ob


def deserialize_wkb(data):
    geom = lgeos.GEOSGeomFromWKB_buf(c_char_p(data), c_size_t(len(data)))
    if not geom:
        raise WKBReadingError(
            "Could not create geometry because of errors while reading input.")
    return geom


def geos_geom_from_py(ob, create_func=None):
    """Helper function for geos_*_from_py functions in each geom type.

    If a create_func is specified the coordinate sequence is cloned and a new
    geometry is created with it, otherwise the geometry is cloned directly.
    This behaviour is useful for converting between LineString and LinearRing
    objects.
    """
    if create_func is None:
        geom = lgeos.GEOSGeom_clone(ob._geom)
    else:
        cs = lgeos.GEOSGeom_getCoordSeq(ob._geom)
        cs = lgeos.GEOSCoordSeq_clone(cs)
        geom = create_func(cs)

    N = ob._ndim

    return geom, N


def exceptNull(func):
    """Decorator which helps avoid GEOS operations on null pointers."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0]._geom or args[0].is_empty:
            raise ValueError("Null/empty geometry supports no operations")
        return func(*args, **kwargs)
    return wrapper


class CAP_STYLE:
    round = 1
    flat = 2
    square = 3


class JOIN_STYLE:
    round = 1
    mitre = 2
    bevel = 3

EMPTY = deserialize_wkb(a2b_hex(b'010700000000000000'))


class BaseGeometry:
    """
    Provides GEOS spatial predicates and topological operations.

    """

    # Attributes
    # ----------
    # __geom__ : c_void_p
    #     Cached ctypes pointer to GEOS geometry. Not to be accessed.
    # _geom : c_void_p
    #     Property by which the GEOS geometry is accessed.
    # __p__ : object
    #     Parent (Shapely) geometry
    # _ctypes_data : object
    #     Cached ctypes data buffer
    # _ndim : int
    #     Number of dimensions (2 or 3, generally)
    # _crs : object
    #     Coordinate reference system. Available for Shapely extensions, but
    #     not implemented here.
    # _other_owned : bool
    #     True if this object's GEOS geometry is owned by another as in the
    #     case of a multipart geometry member.
    __geom__ = EMPTY
    __p__ = None
    _ctypes_data = None
    _ndim = None
    _crs = None
    _other_owned = False
    _is_empty = True

    # Backend config
    impl = DefaultImplementation

    # a reference to the so/dll proxy to preserve access during clean up
    _lgeos = lgeos

    def empty(self, val=EMPTY):
        warn(
            "The 'empty()' method is deprecated and will be removed in "
            "Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        self._empty(val=val)

    def _empty(self, val=EMPTY):
        if not self._other_owned and self.__geom__ and self.__geom__ != EMPTY:
            try:
                self._lgeos.GEOSGeom_destroy(self.__geom__)
            except (AttributeError, TypeError):
                # _lgeos might be empty on shutdown
                log.exception("Failed to delete GEOS geom")

        self._is_empty = True
        self.__geom__ = val

    def __bool__(self):
        return self.is_empty is False

    def __nonzero__(self):
        return self.__bool__()

    def __del__(self):
        self._empty(val=None)
        self.__p__ = None

    def __str__(self):
        return self.wkt

    # To support pickling
    def __reduce__(self):
        return (self.__class__, (), self.wkb)

    def __setstate__(self, state):
        self._empty()
        self.__geom__ = deserialize_wkb(state)
        self._is_empty = False
        if lgeos.methods['has_z'](self.__geom__):
            self._ndim = 3
        else:
            self._ndim = 2

    @property
    def _geom(self):
        return self.__geom__

    @_geom.setter
    def _geom(self, val):
        warn(
            "Setting the '_geom' attribute directly is deprecated, and will "
            "no longer be possible in Shapely 2.0.",
            ShapelyDeprecationWarning, stacklevel=2)
        self._set_geom(val)

    def _set_geom(self, val):
        self._empty()
        self._is_empty = val in [EMPTY, None]
        self.__geom__ = val

    def __setattr__(self, name, value):
        # first try regular attribute access via __getattribute__, so that
        # our own (existing) attributes don't raise a warning
        try:
            object.__getattribute__(self, name)
            super().__setattr__(name, value)
            return
        except AttributeError:
            pass

        # if custom atribute, raise the warning
        warn(
            "Setting custom attributes on geometry objects is deprecated, "
            "and will raise an AttributeError in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2
        )
        super().__setattr__(name, value)

    # Operators
    # ---------

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __eq__(self, other):
        return (
            type(other) == type(self) and
            tuple(self.coords) == tuple(other.coords)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    # Array and ctypes interfaces
    # ---------------------------

    @property
    def _ctypes(self):
        raise NotImplementedError

    @property
    def ctypes(self):
        """Return ctypes buffer"""
        warn(
            "Accessing the 'ctypes' attribute is deprecated,"
            " and will not be possible any more in Shapely 2.0",
            ShapelyDeprecationWarning, stacklevel=2)
        return self._ctypes

    @property
    def _array_interface_base(self):
        if sys.byteorder == 'little':
            typestr = '<f8'
        elif sys.byteorder == 'big':
            typestr = '>f8'
        else:
            raise ValueError(
                "Unsupported byteorder: neither little nor big-endian")
        return {
            'version': 3,
            'typestr': typestr,
            'data': self._ctypes,
            }

    @property
    def array_interface_base(self):
        warn(
            "The 'array_interface_base' property is deprecated and will be "
            "removed in Shapely 2.0.",
            ShapelyDeprecationWarning, stacklevel=2)
        return self._array_interface_base()

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise AttributeError("Not implemented")

    # Coordinate access
    # -----------------

    def _get_coords(self):
        """Access to geometry's coordinates (CoordinateSequence)"""
        raise NotImplementedError("set_coords must be provided by derived classes")

    def _set_coords(self, ob):
        raise NotImplementedError(
            "set_coords must be provided by derived classes")

    coords = property(_get_coords, _set_coords)

    @property
    def xy(self):
        """Separate arrays of X and Y coordinate values"""
        raise NotImplementedError

    # Python feature protocol

    @property
    def __geo_interface__(self):
        """Dictionary representation of the geometry"""
        raise NotImplementedError

    # Type of geometry and its representations
    # ----------------------------------------

    def geometryType(self):
        return geometry_type_name(self._geom)

    @property
    def type(self):
        return self.geometryType()

    @property
    def wkt(self):
        """WKT representation of the geometry"""
        return WKTWriter(lgeos).write(self)

    @property
    def wkb(self):
        """WKB representation of the geometry"""
        return WKBWriter(lgeos).write(self)

    @property
    def wkb_hex(self):
        """WKB hex representation of the geometry"""
        return WKBWriter(lgeos).write_hex(self)

    def svg(self, scale_factor=1., **kwargs):
        """Raises NotImplementedError"""
        raise NotImplementedError

    def _repr_svg_(self):
        """SVG representation for iPython notebook"""
        svg_top = '<svg xmlns="http://www.w3.org/2000/svg" ' \
            'xmlns:xlink="http://www.w3.org/1999/xlink" '
        if self.is_empty:
            return svg_top + '/>'
        else:
            # Establish SVG canvas that will fit all the data + small space
            xmin, ymin, xmax, ymax = self.bounds
            if xmin == xmax and ymin == ymax:
                # This is a point; buffer using an arbitrary size
                xmin, ymin, xmax, ymax = self.buffer(1).bounds
            else:
                # Expand bounds by a fraction of the data ranges
                expand = 0.04  # or 4%, same as R plots
                widest_part = max([xmax - xmin, ymax - ymin])
                expand_amount = widest_part * expand
                xmin -= expand_amount
                ymin -= expand_amount
                xmax += expand_amount
                ymax += expand_amount
            dx = xmax - xmin
            dy = ymax - ymin
            width = min([max([100., dx]), 300])
            height = min([max([100., dy]), 300])
            try:
                scale_factor = max([dx, dy]) / max([width, height])
            except ZeroDivisionError:
                scale_factor = 1.
            view_box = "{} {} {} {}".format(xmin, ymin, dx, dy)
            transform = "matrix(1,0,0,-1,0,{})".format(ymax + ymin)
            return svg_top + (
                'width="{1}" height="{2}" viewBox="{0}" '
                'preserveAspectRatio="xMinYMin meet">'
                '<g transform="{3}">{4}</g></svg>'
                ).format(view_box, width, height, transform,
                         self.svg(scale_factor))

    @property
    def geom_type(self):
        """Name of the geometry's type, such as 'Point'"""
        return self.geometryType()

    # Real-valued properties and methods
    # ----------------------------------

    @property
    def area(self):
        """Unitless area of the geometry (float)"""
        return self.impl['area'](self)

    def distance(self, other):
        """Unitless distance to other geometry (float)"""
        return self.impl['distance'](self, other)

    def hausdorff_distance(self, other):
        """Unitless hausdorff distance to other geometry (float)"""
        return self.impl['hausdorff_distance'](self, other)

    @property
    def length(self):
        """Unitless length of the geometry (float)"""
        return self.impl['length'](self)

    @property
    def minimum_clearance(self):
        """Unitless distance by which a node could be moved to produce an invalid geometry (float)"""
        return self.impl['minimum_clearance'](self)

    # Topological properties
    # ----------------------

    @property
    def boundary(self):
        """Returns a lower dimension geometry that bounds the object

        The boundary of a polygon is a line, the boundary of a line is a
        collection of points. The boundary of a point is an empty (null)
        collection.
        """
        return geom_factory(self.impl['boundary'](self))

    @property
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)"""
        if self.is_empty:
            return ()
        else:
            return self.impl['bounds'](self)

    @property
    def centroid(self):
        """Returns the geometric center of the object"""
        return geom_factory(self.impl['centroid'](self))

    @delegated
    def representative_point(self):
        """Returns a point guaranteed to be within the object, cheaply."""
        return geom_factory(self.impl['representative_point'](self))

    @property
    def convex_hull(self):
        """Imagine an elastic band stretched around the geometry: that's a
        convex hull, more or less

        The convex hull of a three member multipoint, for example, is a
        triangular polygon.
        """
        return geom_factory(self.impl['convex_hull'](self))

    @property
    def envelope(self):
        """A figure that envelopes the geometry"""
        return geom_factory(self.impl['envelope'](self))

    @property
    def minimum_rotated_rectangle(self):
        """Returns the general minimum bounding rectangle of
        the geometry. Can possibly be rotated. If the convex hull
        of the object is a degenerate (line or point) this same degenerate
        is returned.
        """
        # first compute the convex hull
        hull = self.convex_hull
        try:
            coords = hull.exterior.coords
        except AttributeError:  # may be a Point or a LineString
            return hull
        # generate the edge vectors between the convex hull's coords
        edges = ((pt2[0] - pt1[0], pt2[1] - pt1[1]) for pt1, pt2 in zip(
            coords, islice(coords, 1, None)))

        def _transformed_rects():
            for dx, dy in edges:
                # compute the normalized direction vector of the edge
                # vector.
                length = math.sqrt(dx ** 2 + dy ** 2)
                ux, uy = dx / length, dy / length
                # compute the normalized perpendicular vector
                vx, vy = -uy, ux
                # transform hull from the original coordinate system to
                # the coordinate system defined by the edge and compute
                # the axes-parallel bounding rectangle.
                transf_rect = affine_transform(
                    hull, (ux, uy, vx, vy, 0, 0)).envelope
                # yield the transformed rectangle and a matrix to
                # transform it back to the original coordinate system.
                yield (transf_rect, (ux, vx, uy, vy, 0, 0))

        # check for the minimum area rectangle and return it
        transf_rect, inv_matrix = min(
            _transformed_rects(), key=lambda r: r[0].area)
        return affine_transform(transf_rect, inv_matrix)

    def buffer(self, distance, resolution=16, quadsegs=None,
               cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round,
               mitre_limit=5.0, single_sided=False):
        """Get a geometry that represents all points within a distance
        of this geometry.

        A positive distance produces a dilation, a negative distance an
        erosion. A very small or zero distance may sometimes be used to
        "tidy" a polygon.

        Parameters
        ----------
        distance : float
            The distance to buffer around the object.
        resolution : int, optional
            The resolution of the buffer around each vertex of the
            object.
        quadsegs : int, optional
            Sets the number of line segments used to approximate an
            angle fillet.  Note: the use of a `quadsegs` parameter is
            deprecated and will be gone from the next major release.
        cap_style : int, optional
            The styles of caps are: CAP_STYLE.round (1), CAP_STYLE.flat
            (2), and CAP_STYLE.square (3).
        join_style : int, optional
            The styles of joins between offset segments are:
            JOIN_STYLE.round (1), JOIN_STYLE.mitre (2), and
            JOIN_STYLE.bevel (3).
        mitre_limit : float, optional
            The mitre limit ratio is used for very sharp corners. The
            mitre ratio is the ratio of the distance from the corner to
            the end of the mitred offset corner. When two line segments
            meet at a sharp angle, a miter join will extend the original
            geometry. To prevent unreasonable geometry, the mitre limit
            allows controlling the maximum length of the join corner.
            Corners with a ratio which exceed the limit will be beveled.
        single_side : bool, optional
            The side used is determined by the sign of the buffer
            distance:

                a positive distance indicates the left-hand side
                a negative distance indicates the right-hand side

            The single-sided buffer of point geometries is the same as
            the regular buffer.  The End Cap Style for single-sided
            buffers is always ignored, and forced to the equivalent of
            CAP_FLAT.

        Returns
        -------
        Geometry

        Notes
        -----
        The return value is a strictly two-dimensional geometry. All
        Z coordinates of the original geometry will be ignored.

        Examples
        --------
        >>> from shapely.wkt import loads
        >>> g = loads('POINT (0.0 0.0)')
        >>> g.buffer(1.0).area        # 16-gon approx of a unit radius circle
        3.1365484905459384
        >>> g.buffer(1.0, 128).area   # 128-gon approximation
        3.141513801144299
        >>> g.buffer(1.0, 3).area     # triangle approximation
        3.0
        >>> list(g.buffer(1.0, cap_style=CAP_STYLE.square).exterior.coords)
        [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
        >>> g.buffer(1.0, cap_style=CAP_STYLE.square).area
        4.0

        """
        if quadsegs is not None:
            warn(
                "The `quadsegs` argument is deprecated. Use `resolution`.",
                DeprecationWarning)
            res = quadsegs
        else:
            res = resolution

        if mitre_limit == 0.0:
            raise ValueError(
                'Cannot compute offset from zero-length line segment')

        if 'buffer_with_params' in self.impl:
            params = self._lgeos.GEOSBufferParams_create()
            self._lgeos.GEOSBufferParams_setEndCapStyle(params, cap_style)
            self._lgeos.GEOSBufferParams_setJoinStyle(params, join_style)
            self._lgeos.GEOSBufferParams_setMitreLimit(params, mitre_limit)
            self._lgeos.GEOSBufferParams_setQuadrantSegments(params, res)
            self._lgeos.GEOSBufferParams_setSingleSided(params, single_sided)
            return geom_factory(self.impl['buffer_with_params'](self, params, distance))

        if cap_style == CAP_STYLE.round and join_style == JOIN_STYLE.round:
            return geom_factory(self.impl['buffer'](self, distance, res))

        return geom_factory(self.impl['buffer_with_style'](self, distance, res,
                                                           cap_style,
                                                           join_style,
                                                           mitre_limit))

    @delegated
    def simplify(self, tolerance, preserve_topology=True):
        """Returns a simplified geometry produced by the Douglas-Peucker
        algorithm

        Coordinates of the simplified geometry will be no more than the
        tolerance distance from the original. Unless the topology preserving
        option is used, the algorithm may produce self-intersecting or
        otherwise invalid geometries.
        """
        if preserve_topology:
            op = self.impl['topology_preserve_simplify']
        else:
            op = self.impl['simplify']
        return geom_factory(op(self, tolerance))

    def normalize(self):
        """Converts geometry to normal form (or canonical form).

        This method orders the coordinates, rings of a polygon and parts of
        multi geometries consistently. Typically useful for testing purposes
        (for example in combination with `equals_exact`).

        Examples
        --------
        >>> from shapely.wkt import loads
        >>> p = loads("MULTILINESTRING((0 0, 1 1), (3 3, 2 2))")
        >>> p.normalize().wkt
        'MULTILINESTRING ((2 2, 3 3), (0 0, 1 1))'
        """
        # self.impl['normalize'](self)
        if self._geom is None:
            raise InvalidGeometryError("Null geometry supports no operations")
        geom_cloned = lgeos.GEOSGeom_clone(self._geom)
        lgeos.GEOSNormalize(geom_cloned)
        return geom_factory(geom_cloned)

    # Binary operations
    # -----------------

    def difference(self, other):
        """Returns the difference of the geometries"""
        return geom_factory(self.impl['difference'](self, other))

    def intersection(self, other):
        """Returns the intersection of the geometries"""
        return geom_factory(self.impl['intersection'](self, other))

    def symmetric_difference(self, other):
        """Returns the symmetric difference of the geometries
        (Shapely geometry)"""
        return geom_factory(self.impl['symmetric_difference'](self, other))

    def union(self, other):
        """Returns the union of the geometries (Shapely geometry)"""
        return geom_factory(self.impl['union'](self, other))

    # Unary predicates
    # ----------------

    @property
    def has_z(self):
        """True if the geometry's coordinate sequence(s) have z values (are
        3-dimensional)"""
        return bool(self.impl['has_z'](self))

    @property
    def is_empty(self):
        """True if the set of points in this geometry is empty, else False"""
        return (self._geom is None) or bool(self.impl['is_empty'](self))

    @property
    def is_ring(self):
        """True if the geometry is a closed ring, else False"""
        return bool(self.impl['is_ring'](self))

    @property
    def is_closed(self):
        """True if the geometry is closed, else False

        Applicable only to 1-D geometries."""
        if self.geom_type == 'LinearRing':
            return True
        elif self.geom_type == 'LineString':
            if 'is_closed' in self.impl:
                return bool(self.impl['is_closed'](self))
            else:
                return self.coords[0] == self.coords[-1]
        else:
            return False

    @property
    def is_simple(self):
        """True if the geometry is simple, meaning that any self-intersections
        are only at boundary points, else False"""
        return bool(self.impl['is_simple'](self))

    @property
    def is_valid(self):
        """True if the geometry is valid (definition depends on sub-class),
        else False"""
        return bool(self.impl['is_valid'](self))

    # Binary predicates
    # -----------------

    def relate(self, other):
        """Returns the DE-9IM intersection matrix for the two geometries
        (string)"""
        return self.impl['relate'](self, other)

    def covers(self, other):
        """Returns True if the geometry covers the other, else False"""
        return bool(self.impl['covers'](self, other))

    def covered_by(self, other):
        """Returns True if the geometry is covered by the other, else False"""
        return bool(self.impl['covered_by'](self, other))

    def contains(self, other):
        """Returns True if the geometry contains the other, else False"""
        return bool(self.impl['contains'](self, other))

    def crosses(self, other):
        """Returns True if the geometries cross, else False"""
        return bool(self.impl['crosses'](self, other))

    def disjoint(self, other):
        """Returns True if geometries are disjoint, else False"""
        return bool(self.impl['disjoint'](self, other))

    def equals(self, other):
        """Returns True if geometries are equal, else False.

        This method considers point-set equality (or topological
        equality), and is equivalent to (self.within(other) &
        self.contains(other)).

        Examples
        --------
        >>> LineString(
        ...     [(0, 0), (2, 2)]
        ... ).equals(
        ...     LineString([(0, 0), (1, 1), (2, 2)])
        ... )
        True

        Returns
        -------
        bool

        """
        return bool(self.impl['equals'](self, other))

    def intersects(self, other):
        """Returns True if geometries intersect, else False"""
        return bool(self.impl['intersects'](self, other))

    def overlaps(self, other):
        """Returns True if geometries overlap, else False"""
        return bool(self.impl['overlaps'](self, other))

    def touches(self, other):
        """Returns True if geometries touch, else False"""
        return bool(self.impl['touches'](self, other))

    def within(self, other):
        """Returns True if geometry is within the other, else False"""
        return bool(self.impl['within'](self, other))

    def equals_exact(self, other, tolerance):
        """True if geometries are equal to within a specified
        tolerance.

        Parameters
        ----------
        other : BaseGeometry
            The other geometry object in this comparison.
        tolerance : float
            Absolute tolerance in the same units as coordinates.

        This method considers coordinate equality, which requires
        coordinates to be equal and in the same order for all components
        of a geometry.

        Because of this it is possible for "equals()" to be True for two
        geometries and "equals_exact()" to be False.

        Examples
        --------
        >>> LineString(
        ...     [(0, 0), (2, 2)]
        ... ).equals_exact(
        ...     LineString([(0, 0), (1, 1), (2, 2)]),
        ...     1e-6
        ... )
        False

        Returns
        -------
        bool

        """
        return bool(self.impl['equals_exact'](self, other, tolerance))

    def almost_equals(self, other, decimal=6):
        """True if geometries are equal at all coordinates to a
        specified decimal place.

        .. deprecated:: 1.8.0 The 'almost_equals()' method is deprecated
            and will be removed in Shapely 2.0 because the name is
            confusing. The 'equals_exact()' method should be used
            instead.

        Refers to approximate coordinate equality, which requires
        coordinates to be approximately equal and in the same order for
        all components of a geometry.

        Because of this it is possible for "equals()" to be True for two
        geometries and "almost_equals()" to be False.

        Examples
        --------
        >>> LineString(
        ...     [(0, 0), (2, 2)]
        ... ).equals_exact(
        ...     LineString([(0, 0), (1, 1), (2, 2)]),
        ...     1e-6
        ... )
        False

        Returns
        -------
        bool

        """
        warn(
            "The 'almost_equals()' method is deprecated and will be removed in Shapely 2.0",
            ShapelyDeprecationWarning,
            stacklevel=2,
        )
        return self.equals_exact(other, 0.5 * 10 ** (-decimal))

    def relate_pattern(self, other, pattern):
        """Returns True if the DE-9IM string code for the relationship between
        the geometries satisfies the pattern, else False"""
        pattern = c_char_p(pattern.encode('ascii'))
        return bool(self.impl['relate_pattern'](self, other, pattern))

    # Linear referencing
    # ------------------

    @delegated
    def project(self, other, normalized=False):
        """Returns the distance along this geometry to a point nearest the
        specified point

        If the normalized arg is True, return the distance normalized to the
        length of the linear geometry.
        """
        if normalized:
            op = self.impl['project_normalized']
        else:
            op = self.impl['project']
        return op(self, other)

    @delegated
    @exceptNull
    def interpolate(self, distance, normalized=False):
        """Return a point at the specified distance along a linear geometry

        Negative length values are taken as measured in the reverse
        direction from the end of the geometry. Out-of-range index
        values are handled by clamping them to the valid range of values.
        If the normalized arg is True, the distance will be interpreted as a
        fraction of the geometry's length.
        """
        if normalized:
            op = self.impl['interpolate_normalized']
        else:
            op = self.impl['interpolate']
        return geom_factory(op(self, distance))


class BaseMultipartGeometry(BaseGeometry):

    def shape_factory(self, *args):
        # Factory for part instances, usually a geometry class
        raise NotImplementedError("To be implemented by derived classes")

    @property
    def ctypes(self):
        raise NotImplementedError(
            "Multi-part geometries have no ctypes representations")

    @property
    def __array_interface__(self):
        """Provide the Numpy array protocol."""
        raise AttributeError("Multi-part geometries do not themselves "
                             "provide the array interface")

    def _get_coords(self):
        raise NotImplementedError("Sub-geometries may have coordinate "
                                  "sequences, but collections do not")

    def _set_coords(self, ob):
        raise NotImplementedError("Sub-geometries may have coordinate "
                                  "sequences, but collections do not")

    @property
    def coords(self):
        raise NotImplementedError(
            "Multi-part geometries do not provide a coordinate sequence")

    @property
    def geoms(self):
        if self.is_empty:
            return []
        return GeometrySequence(self, self.shape_factory)

    def __bool__(self):
        return self.is_empty is False

    def __iter__(self):
        """
        .. deprecated:: 1.8
           Iteration over multi-part geometries is deprecated and will be removed in
           Shapely 2.0. Use the `geoms` property to access the constituent parts of
           a multi-part geometry.
        """
        warn(
            "Iteration over multi-part geometries is deprecated and will be removed in "
            "Shapely 2.0. Use the `geoms` property to access the constituent parts of "
            "a multi-part geometry.", ShapelyDeprecationWarning, stacklevel=2)
        if not self.is_empty:
            return iter(self.geoms)
        else:
            return iter([])

    def __len__(self):
        warn(
            "__len__ for multi-part geometries is deprecated and will be removed in "
            "Shapely 2.0. Check the length of the `geoms` property instead to get the "
            " number of parts of a multi-part geometry.",
            ShapelyDeprecationWarning, stacklevel=2)
        if not self.is_empty:
            return len(self.geoms)
        else:
            return 0

    def __getitem__(self, index):
        """
        .. deprecated:: 1.8
           __getitem__ for multi-part geometries is deprecated and will be removed in
           Shapely 2.0. Use the `geoms` property to access the constituent parts of
           a multi-part geometry.
        """
        warn(
            "__getitem__ for multi-part geometries is deprecated and will be removed in "
            "Shapely 2.0. Use the `geoms` property to access the constituent parts of "
            "a multi-part geometry.", ShapelyDeprecationWarning, stacklevel=2)
        if not self.is_empty:
            return self.geoms[index]
        else:
            return ()[index]

    def __eq__(self, other):
        return (
            type(other) == type(self) and
            len(self.geoms) == len(other.geoms) and
            all(x == y for x, y in zip(self.geoms, other.geoms))
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def svg(self, scale_factor=1., color=None):
        """Returns a group of SVG elements for the multipart geometry.

        Parameters
        ==========
        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        color : str, optional
            Hex string for stroke or fill color. Default is to use "#66cc99"
            if geometry is valid, and "#ff3333" if invalid.
        """
        if self.is_empty:
            return '<g />'
        if color is None:
            color = "#66cc99" if self.is_valid else "#ff3333"
        return '<g>' + \
            ''.join(p.svg(scale_factor, color) for p in self.geoms) + \
            '</g>'


class GeometrySequence:
    """
    Iterative access to members of a homogeneous multipart geometry.
    """

    # Attributes
    # ----------
    # _factory : callable
    #     Returns instances of Shapely geometries
    # _geom : c_void_p
    #     Ctypes pointer to the parent's GEOS geometry
    # _ndim : int
    #     Number of dimensions (2 or 3, generally)
    # __p__ : object
    #     Parent (Shapely) geometry
    shape_factory = None
    _geom = None
    __p__ = None
    _ndim = None

    def __init__(self, parent, type):
        self.shape_factory = type
        self.__p__ = parent

    def _update(self):
        self._geom = self.__p__._geom
        self._ndim = self.__p__._ndim

    def _get_geom_item(self, i):
        g = self.shape_factory()
        g._other_owned = True
        g._set_geom(lgeos.GEOSGetGeometryN(self._geom, i))
        g._ndim = self._ndim
        g.__p__ = self
        return g

    def __iter__(self):
        self._update()
        for i in range(self.__len__()):
            yield self._get_geom_item(i)

    def __len__(self):
        self._update()
        return lgeos.GEOSGetNumGeometries(self._geom)

    def __getitem__(self, key):
        self._update()
        m = self.__len__()
        if isinstance(key, integer_types):
            if key + m < 0 or key >= m:
                raise IndexError("index out of range")
            if key < 0:
                i = m + key
            else:
                i = key
            return self._get_geom_item(i)
        elif isinstance(key, slice):
            if type(self) == HeterogeneousGeometrySequence:
                raise GeometryTypeError(
                    "Heterogenous geometry collections are not sliceable")
            res = []
            start, stop, stride = key.indices(m)
            for i in range(start, stop, stride):
                res.append(self._get_geom_item(i))
            return type(self.__p__)(res or None)
        else:
            raise TypeError("key must be an index or slice")

    @property
    def _longest(self):
        max = 0
        for g in iter(self):
            l = len(g.coords)
            if l > max:
                max = l


class HeterogeneousGeometrySequence(GeometrySequence):
    """
    Iterative access to a heterogeneous sequence of geometries.
    """

    def __init__(self, parent):
        super().__init__(parent, None)

    def _get_geom_item(self, i):
        sub = lgeos.GEOSGetGeometryN(self._geom, i)
        g = geom_factory(sub, parent=self)
        g._other_owned = True
        return g


class EmptyGeometry(BaseGeometry):
    def __init__(self):
        """Create an empty geometry."""
        BaseGeometry.__init__(self)
