# The beginnings of a Cython definition of GEOS. In the future much of this
# could be auto-generated.

from libc.stdint cimport uintptr_t


cdef extern from "geos_c.h":
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSCoordSequence
    ctypedef struct GEOSPreparedGeometry

    GEOSCoordSequence *GEOSCoordSeq_create_r(GEOSContextHandle_t, unsigned int, unsigned int) nogil
    GEOSCoordSequence *GEOSGeom_getCoordSeq_r(GEOSContextHandle_t, GEOSGeometry *) nogil

    int GEOSCoordSeq_getSize_r(GEOSContextHandle_t, GEOSCoordSequence *, unsigned int *) nogil
    int GEOSCoordSeq_setX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_setY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_setZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double) nogil
    int GEOSCoordSeq_getX_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil
    int GEOSCoordSeq_getY_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil
    int GEOSCoordSeq_getZ_r(GEOSContextHandle_t, GEOSCoordSequence *, int, double *) nogil

    GEOSGeometry *GEOSGeom_createPoint_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_createLineString_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_createLinearRing_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil
    GEOSGeometry *GEOSGeom_clone_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    GEOSCoordSequence *GEOSCoordSeq_clone_r(GEOSContextHandle_t, GEOSCoordSequence *) nogil

    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry *) nogil

    char GEOSPreparedContains_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedContainsProperly_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCoveredBy_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCovers_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedCrosses_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedDisjoint_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedIntersects_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedOverlaps_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedTouches_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil
    char GEOSPreparedWithin_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry*) nogil

    char GEOSHasZ_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisRing_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisClosed_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisValid_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    char GEOSisSimple_r(GEOSContextHandle_t, GEOSGeometry *) nogil


cdef GEOSContextHandle_t get_geos_context_handle():
    # Note: This requires that lgeos is defined, so needs to be imported as:
    from shapely.geos import lgeos
    cdef uintptr_t handle = lgeos.geos_handle
    return <GEOSContextHandle_t>handle


cdef GEOSPreparedGeometry *geos_from_prepared(shapely_geom) except *:
    """Get the Prepared GEOS geometry pointer from the given shapely geometry."""
    cdef uintptr_t geos_geom = shapely_geom._geom
    return <GEOSPreparedGeometry *>geos_geom
