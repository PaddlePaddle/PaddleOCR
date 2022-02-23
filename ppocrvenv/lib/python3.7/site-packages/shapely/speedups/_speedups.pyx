# cython: language_level=3

# geos_linestring_from_py was transcribed from shapely.geometry.linestring
# geos_linearring_from_py was transcribed from shapely.geometry.polygon
# coordseq_ctypes was transcribed from shapely.coords.CoordinateSequence.ctypes
#
# Copyright (c) 2007, Sean C. Gillies
# Transcription to cython: Copyright (c) 2011, Oliver Tonnhofer

import ctypes
import logging

from shapely.geos import lgeos
from shapely.geometry import Point, LineString, LinearRing
from shapely.geometry.base import geom_factory
from shapely.errors import GeometryTypeError, TopologicalError


include "../_geos.pxi"

from libc.stdint cimport uintptr_t


log = logging.getLogger(__name__)

try:
    import numpy
    has_numpy = True
except ImportError:
    log.info("Numpy was not imported, continuing without requires()")
    has_numpy = False
    numpy = None


cdef inline GEOSGeometry *cast_geom(uintptr_t geom_addr):
    return <GEOSGeometry *>geom_addr


cdef inline GEOSContextHandle_t cast_handle(uintptr_t handle_addr):
    return <GEOSContextHandle_t>handle_addr


cdef inline GEOSCoordSequence *cast_seq(uintptr_t handle_addr):
    return <GEOSCoordSequence *>handle_addr


def destroy(geom):
    GEOSGeom_destroy_r(cast_handle(lgeos.geos_handle), cast_geom(geom))


def required(ob):
    """Return an object that meets Shapely requirements for self-owned
    C-continguous data, copying if necessary, or just return the original
    object."""
    if has_numpy and hasattr(ob, '__array_interface__'):
        return numpy.require(ob, numpy.float64, ["C", "OWNDATA"])
    else:
        return ob


def geos_linestring_from_py(ob, update_geom=None, update_ndim=0):
    cdef double *cp
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef GEOSGeometry *g
    cdef double dx, dy, dz
    cdef int i, n, m, sm, sn

    # If a LineString is passed in, just clone it and return
    # If a LinearRing is passed in, clone the coord seq and return a LineString
    if isinstance(ob, LineString):
        g = cast_geom(ob._geom)
        if GEOSHasZ_r(handle, g):
            n = 3
        else:
            n = 2

        if type(ob) == LineString:
            return <uintptr_t>GEOSGeom_clone_r(handle, g), n
        else:
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, g)
            cs = GEOSCoordSeq_clone_r(handle, cs)
            return <uintptr_t>GEOSGeom_createLineString_r(handle, cs), n

    # If numpy is present, we use numpy.require to ensure that we have a
    # C-continguous array that owns its data. View data will be copied.
    ob = required(ob)
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")
        try:
            n = array['shape'][1]
        except IndexError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], ctypes.Array):
            cp = <double *><uintptr_t>ctypes.addressof(array['data'])
        else:
            cp = <double *><uintptr_t>array['data'][0]

        # Use strides to properly index into cp
        # ob[i, j] == cp[sm*i + sn*j]
        # Just to avoid a referenced before assignment warning.
        dx = 0
        if array.get('strides', None):
            sm = array['strides'][0]/sizeof(dx)
            sn = array['strides'][1]/sizeof(dx)
        else:
            sm = n
            sn = 1

        # Create a coordinate sequence
        if update_geom is not None:
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, <int>m, <int>n)

        # add to coordinate sequence
        for i in range(m):
            dx = cp[sm*i]
            dy = cp[sm*i+sn]
            dz = 0
            if n == 3:
                dz = cp[sm*i+2*sn]

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

    except AttributeError:
        # Fall back on list
        try:
            m = len(ob)
        except TypeError:  # generators
            ob = list(ob)
            m = len(ob)

        if m == 0:
            return None
        elif m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")

        if m < 2:
            raise ValueError(
                "LineStrings must have at least 2 coordinate tuples")

        def _coords(o):
            if isinstance(o, Point):
                return o.coords[0]
            else:
                return o

        try:
            n = len(_coords(ob[0]))
        except TypeError:
            raise ValueError(
                "Input %s is the wrong shape for a LineString" % str(ob))
        assert n == 2 or n == 3

        # Create a coordinate sequence
        if update_geom is not None:
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, <int>m, <int>n)

        # add to coordinate sequence
        for i in range(m):
            coords = _coords(ob[i])
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                try:
                    dz = coords[2]
                except IndexError:
                    raise ValueError("Inconsistent coordinate dimensionality")

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

    if update_geom is not None:
        return None
    else:
        g = GEOSGeom_createLineString_r(handle, cs);
        if not g:
            raise ValueError("GEOSGeom_createLineString_r returned a NULL pointer")
        return <uintptr_t>g, n


def geos_linearring_from_py(ob, update_geom=None, update_ndim=0):
    cdef double *cp
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSGeometry *g
    cdef GEOSCoordSequence *cs
    cdef double dx, dy, dz
    cdef unsigned int m
    cdef int i, n, M, sm, sn

    # If a LinearRing is passed in, just clone it and return
    # If a valid LineString is passed in, clone the coord seq and return a LinearRing
    if isinstance(ob, LineString):
        g = cast_geom(ob._geom)
        if GEOSHasZ_r(handle, g):
            n = 3
        else:
            n = 2

        if type(ob) == LinearRing:
            return <uintptr_t>GEOSGeom_clone_r(handle, g), n
        else:
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, g)
            GEOSCoordSeq_getSize_r(handle, cs, &m)
            if not GEOSisValid_r(handle, g):
                raise TopologicalError("A LineString must be valid.")
            elif GEOSisClosed_r(handle, g) and m >= 4:
                cs = GEOSCoordSeq_clone_r(handle, cs)
                return <uintptr_t>GEOSGeom_createLinearRing_r(handle, cs), n
            else:
                # else continue below.
                # (and extract coords to avoid array interface of LineString)
                ob = ob.coords

    # If numpy is present, we use numpy.require to ensure that we have a
    # C-continguous array that owns its data. View data will be copied.
    ob = required(ob)
    try:
        # From array protocol
        array = ob.__array_interface__
        assert len(array['shape']) == 2
        m = array['shape'][0]
        n = array['shape'][1]
        if m < 3:
            raise ValueError(
                "A LinearRing must have at least 3 coordinate tuples")
        assert n == 2 or n == 3

        # Make pointer to the coordinate array
        if isinstance(array['data'], ctypes.Array):
            cp = <double *><uintptr_t>ctypes.addressof(array['data'])
        else:
            cp = <double *><uintptr_t>array['data'][0]

        # Use strides to properly index into cp
        # ob[i, j] == cp[sm*i + sn*j]
        dx = 0  # Just to avoid a referenced before assignment warning.
        if array.get('strides', None):
            sm = array['strides'][0]/sizeof(dx)
            sn = array['strides'][1]/sizeof(dx)
        else:
            sm = n
            sn = 1

        # Add closing coordinates to sequence?
        # Check whether the first set of coordinates matches the last.
        # If not, we'll have to close the ring later
        if (cp[0] != cp[sm*(m-1)] or cp[sn] != cp[sm*(m-1)+sn] or
            (n == 3 and cp[2*sn] != cp[sm*(m-1)+2*sn]) or
            m == 3):
            M = m + 1
        else:
            M = m

        # Create a coordinate sequence
        if update_geom is not None:
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, M, n)

        # add to coordinate sequence
        for i in range(m):
            dx = cp[sm*i]
            dy = cp[sm*i+sn]
            dz = 0
            if n == 3:
                dz = cp[sm*i+2*sn]

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

        # Add closing coordinates to sequence?
        if M > m:
            dx = cp[0]
            dy = cp[sn]
            dz = 0
            if n == 3:
                dz = cp[2*sn]

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, M-1, dx)
            GEOSCoordSeq_setY_r(handle, cs, M-1, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, M-1, dz)

    except AttributeError:
        # Fall back on list
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
            cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, cast_geom(update_geom))
            if n != update_ndim:
                raise ValueError(
                "Wrong coordinate dimensions; this geometry has dimensions: %d" \
                % update_ndim)
        else:
            cs = GEOSCoordSeq_create_r(handle, M, n)

        # add to coordinate sequence
        for i in range(m):
            coords = _coords(ob[i])
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                try:
                    dz = coords[2]
                except IndexError:
                    raise ValueError("Inconsistent coordinate dimensionality")

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, i, dx)
            GEOSCoordSeq_setY_r(handle, cs, i, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, i, dz)

        # Add closing coordinates to sequence?
        if M > m:
            coords = _coords(ob[0])
            dx = coords[0]
            dy = coords[1]
            dz = 0
            if n == 3:
                try:
                    dz = coords[2]
                except IndexError:
                    raise ValueError("Inconsistent coordinate dimensionality")

            # Because of a bug in the GEOS C API,
            # always set X before Y
            GEOSCoordSeq_setX_r(handle, cs, M-1, dx)
            GEOSCoordSeq_setY_r(handle, cs, M-1, dy)
            if n == 3:
                GEOSCoordSeq_setZ_r(handle, cs, M-1, dz)

    if update_geom is not None:
        return None
    else:
        g = GEOSGeom_createLinearRing_r(handle, cs)
        if not g:
            raise ValueError("GEOSGeom_createLinearRing_r returned a NULL pointer")
        return <uintptr_t>g, n


def coordseq_ctypes(self):
    cdef int i, n, m
    cdef double temp = 0
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef double *data_p
    self._update()
    n = self._ndim or 0
    if n == 0:
        # ignore with NumPy 1.21 __array_interface__
        raise AttributeError("empty geometry sequence")
    m = self.__len__()
    array_type = ctypes.c_double * (m * n)
    data = array_type()

    if self._cseq:
        cs = cast_seq(self._cseq)
    data_p = <double *><uintptr_t>ctypes.addressof(data)

    for i in range(m):
        GEOSCoordSeq_getX_r(handle, cs, i, &temp)
        data_p[n*i] = temp
        GEOSCoordSeq_getY_r(handle, cs, i, &temp)
        data_p[n*i+1] = temp
        if n == 3: # TODO: use hasz
            GEOSCoordSeq_getZ_r(handle, cs, i, &temp)
            data_p[n*i+2] = temp
    return data

def coordseq_iter(self):
    cdef int i
    cdef double dx
    cdef double dy
    cdef double dz
    cdef int has_z

    self._update()

    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    if self._cseq:
        cs = cast_seq(self._cseq)

    has_z = self._ndim == 3
    for i in range(self.__len__()):
        GEOSCoordSeq_getX_r(handle, cs, i, &dx)
        GEOSCoordSeq_getY_r(handle, cs, i, &dy)
        if has_z == 1:
            GEOSCoordSeq_getZ_r(handle, cs, i, &dz)
            yield (dx, dy, dz)
        else:
            yield (dx, dy)

cdef GEOSCoordSequence* transform(GEOSCoordSequence* cs,
                                  int ndim,
                                  double a,
                                  double b,
                                  double c,
                                  double d,
                                  double e,
                                  double f,
                                  double g,
                                  double h,
                                  double i,
                                  double xoff,
                                  double yoff,
                                  double zoff):
    """Performs an affine transformation on a GEOSCoordSequence

    Returns the transformed coordinate sequence
    """
    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef unsigned int m
    cdef GEOSCoordSequence *cs_t
    cdef double x, y, z
    cdef double x_t, y_t, z_t

    # create a new coordinate sequence with the same size and dimensions
    GEOSCoordSeq_getSize_r(handle, cs, &m)
    cs_t = GEOSCoordSeq_create_r(handle, m, ndim)

    # perform the transform
    if ndim == 2:
        for n in range(0, m):
            GEOSCoordSeq_getX_r(handle, cs, n, &x)
            GEOSCoordSeq_getY_r(handle, cs, n, &y)
            x_t = a * x + b * y + xoff
            y_t = d * x + e * y + yoff
            GEOSCoordSeq_setX_r(handle, cs_t, n, x_t)
            GEOSCoordSeq_setY_r(handle, cs_t, n, y_t)
    if ndim == 3:
        for n in range(0, m):
            GEOSCoordSeq_getX_r(handle, cs, n, &x)
            GEOSCoordSeq_getY_r(handle, cs, n, &y)
            GEOSCoordSeq_getZ_r(handle, cs, n, &z)
            x_t = a * x + b * y + c * z + xoff
            y_t = d * x + e * y + f * z + yoff
            z_t = g * x + h * y + i * z + zoff
            GEOSCoordSeq_setX_r(handle, cs_t, n, x_t)
            GEOSCoordSeq_setY_r(handle, cs_t, n, y_t)
            GEOSCoordSeq_setZ_r(handle, cs_t, n, z_t)

    return cs_t

cpdef affine_transform(geom, matrix):
    cdef double a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
    if geom.is_empty:
        return geom
    if len(matrix) == 6:
        ndim = 2
        a, b, d, e, xoff, yoff = matrix
        if geom.has_z:
            ndim = 3
            i = 1.0
            c = f = g = h = zoff = 0.0
            matrix = a, b, c, d, e, f, g, h, i, xoff, yoff, zoff
    elif len(matrix) == 12:
        ndim = 3
        a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = matrix
        if not geom.has_z:
            ndim = 2
            matrix = a, b, d, e, xoff, yoff
    else:
        raise ValueError("'matrix' expects either 6 or 12 coefficients")

    cdef GEOSContextHandle_t handle = cast_handle(lgeos.geos_handle)
    cdef GEOSCoordSequence *cs
    cdef GEOSCoordSequence *cs_t
    cdef GEOSGeometry *the_geom
    cdef GEOSGeometry *the_geom_t
    cdef int m, n
    cdef double x, y, z
    cdef double x_t, y_t, z_t

    # Process coordinates from each supported geometry type
    if geom.type in ('Point', 'LineString', 'LinearRing'):
        the_geom = cast_geom(geom._geom)
        cs = <GEOSCoordSequence*>GEOSGeom_getCoordSeq_r(handle, the_geom)

        # perform the transformation
        cs_t = transform(cs, ndim, a, b, c, d, e, f, g, h, i, xoff, yoff, zoff)

        # create a new geometry from the coordinate sequence
        if geom.type == 'Point':
            the_geom_t = GEOSGeom_createPoint_r(handle, cs_t)
        elif geom.type == 'LineString':
            the_geom_t = GEOSGeom_createLineString_r(handle, cs_t)
        elif geom.type == 'LinearRing':
            the_geom_t = GEOSGeom_createLinearRing_r(handle, cs_t)

        # return the geometry as a Python object
        return geom_factory(<uintptr_t>the_geom_t)
    elif geom.type == 'Polygon':
        ring = geom.exterior
        shell = affine_transform(ring, matrix)
        holes = list(geom.interiors)
        for pos, ring in enumerate(holes):
            holes[pos] = affine_transform(ring, matrix)
        return type(geom)(shell, holes)
    elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
        return type(geom)([affine_transform(part, matrix) for part in geom.geoms])
    else:
        raise GeometryTypeError('Type %r not recognized' % geom.type)
