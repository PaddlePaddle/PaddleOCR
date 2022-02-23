'''Prototyping of the GEOS C API

See header file: geos-x.y.z/capi/geos_c.h
'''

from ctypes import CFUNCTYPE, POINTER, c_void_p, c_char_p, \
    c_size_t, c_byte, c_uint, c_int, c_double, py_object

from .errors import UnsupportedGEOSVersionError


EXCEPTION_HANDLER_FUNCTYPE = CFUNCTYPE(None, c_char_p, c_void_p)

# Derived pointer types
c_size_t_p = POINTER(c_size_t)


class allocated_c_char_p(c_char_p):
    '''char pointer return type'''
    pass


def prototype(lgeos, geos_version):
    """Protype functions in geos_c.h for different version of GEOS

    Use the GEOS version, not the C API version.
    """

    if not geos_version >= (3, 3, 0):
        raise UnsupportedGEOSVersionError(
            "Shapely requires GEOS version 3.3.0 or newer.")

    # Initialization, cleanup, version.

    lgeos.initGEOS.restype = None
    lgeos.initGEOS.argtypes = [
        EXCEPTION_HANDLER_FUNCTYPE, EXCEPTION_HANDLER_FUNCTYPE]

    lgeos.finishGEOS.restype = None
    lgeos.finishGEOS.argtypes = []

    lgeos.GEOSversion.restype = c_char_p
    lgeos.GEOSversion.argtypes = []

    # These functions are DEPRECATED.  Please use the new Reader and
    # writer APIS!

    lgeos.GEOSGeomFromWKT.restype = c_void_p
    lgeos.GEOSGeomFromWKT.argtypes = [c_char_p]

    lgeos.GEOSGeomToWKT.restype = allocated_c_char_p
    lgeos.GEOSGeomToWKT.argtypes = [c_void_p]

    lgeos.GEOS_setWKBOutputDims.restype = c_int
    lgeos.GEOS_setWKBOutputDims.argtypes = [c_int]

    lgeos.GEOSGeomFromWKB_buf.restype = c_void_p
    lgeos.GEOSGeomFromWKB_buf.argtypes = [c_void_p, c_size_t]

    lgeos.GEOSGeomToWKB_buf.restype = allocated_c_char_p
    lgeos.GEOSGeomToWKB_buf.argtypes = [c_void_p, c_size_t_p]

    # Coordinate sequence

    lgeos.GEOSCoordSeq_create.restype = c_void_p
    lgeos.GEOSCoordSeq_create.argtypes = [c_uint, c_uint]

    lgeos.GEOSCoordSeq_clone.restype = c_void_p
    lgeos.GEOSCoordSeq_clone.argtypes = [c_void_p]

    lgeos.GEOSGeom_clone.restype = c_void_p
    lgeos.GEOSGeom_clone.argtypes = [c_void_p]

    lgeos.GEOSCoordSeq_destroy.restype = None
    lgeos.GEOSCoordSeq_destroy.argtypes = [c_void_p]

    lgeos.GEOSCoordSeq_setX.restype = c_int
    lgeos.GEOSCoordSeq_setX.argtypes = [c_void_p, c_uint, c_double]

    lgeos.GEOSCoordSeq_setY.restype = c_int
    lgeos.GEOSCoordSeq_setY.argtypes = [c_void_p, c_uint, c_double]

    lgeos.GEOSCoordSeq_setZ.restype = c_int
    lgeos.GEOSCoordSeq_setZ.argtypes = [c_void_p, c_uint, c_double]

    lgeos.GEOSCoordSeq_setOrdinate.restype = c_int
    lgeos.GEOSCoordSeq_setOrdinate.argtypes = [
        c_void_p, c_uint, c_uint, c_double]

    lgeos.GEOSCoordSeq_getX.restype = c_int
    lgeos.GEOSCoordSeq_getX.argtypes = [c_void_p, c_uint, c_void_p]

    lgeos.GEOSCoordSeq_getY.restype = c_int
    lgeos.GEOSCoordSeq_getY.argtypes = [c_void_p, c_uint, c_void_p]

    lgeos.GEOSCoordSeq_getZ.restype = c_int
    lgeos.GEOSCoordSeq_getZ.argtypes = [c_void_p, c_uint, c_void_p]

    lgeos.GEOSCoordSeq_getSize.restype = c_int
    lgeos.GEOSCoordSeq_getSize.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSCoordSeq_getDimensions.restype = c_int
    lgeos.GEOSCoordSeq_getDimensions.argtypes = [c_void_p, c_void_p]

    # Linear referencing

    lgeos.GEOSProject.restype = c_double
    lgeos.GEOSProject.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSInterpolate.restype = c_void_p
    lgeos.GEOSInterpolate.argtypes = [c_void_p, c_double]

    lgeos.GEOSProjectNormalized.restype = c_double
    lgeos.GEOSProjectNormalized.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSInterpolateNormalized.restype = c_void_p
    lgeos.GEOSInterpolateNormalized.argtypes = [c_void_p, c_double]

    # Buffer related

    lgeos.GEOSBuffer.restype = c_void_p
    lgeos.GEOSBuffer.argtypes = [c_void_p, c_double, c_int]

    lgeos.GEOSBufferWithStyle.restype = c_void_p
    lgeos.GEOSBufferWithStyle.argtypes = [
        c_void_p, c_double, c_int, c_int, c_int, c_double]

    lgeos.GEOSOffsetCurve.restype = c_void_p
    lgeos.GEOSOffsetCurve.argtypes = [
        c_void_p, c_double, c_int, c_int, c_double]

    lgeos.GEOSBufferParams_create.restype = c_void_p
    lgeos.GEOSBufferParams_create.argtypes = None

    lgeos.GEOSBufferParams_setEndCapStyle.restype = c_int
    lgeos.GEOSBufferParams_setEndCapStyle.argtypes = [c_void_p, c_int]

    lgeos.GEOSBufferParams_setJoinStyle.restype = c_int
    lgeos.GEOSBufferParams_setJoinStyle.argtypes = [c_void_p, c_int]

    lgeos.GEOSBufferParams_setMitreLimit.restype = c_int
    lgeos.GEOSBufferParams_setMitreLimit.argtypes = [c_void_p, c_double]

    lgeos.GEOSBufferParams_setQuadrantSegments.restype = c_int
    lgeos.GEOSBufferParams_setQuadrantSegments.argtypes = [c_void_p, c_int]

    lgeos.GEOSBufferParams_setSingleSided.restype = c_int
    lgeos.GEOSBufferParams_setSingleSided.argtypes = [c_void_p, c_int]

    lgeos.GEOSBufferWithParams.restype = c_void_p
    lgeos.GEOSBufferWithParams.argtypes = [c_void_p, c_void_p, c_double]

    # Geometry constructors

    lgeos.GEOSGeom_createPoint.restype = c_void_p
    lgeos.GEOSGeom_createPoint.argtypes = [c_void_p]

    lgeos.GEOSGeom_createLinearRing.restype = c_void_p
    lgeos.GEOSGeom_createLinearRing.argtypes = [c_void_p]

    lgeos.GEOSGeom_createLineString.restype = c_void_p
    lgeos.GEOSGeom_createLineString.argtypes = [c_void_p]

    lgeos.GEOSGeom_createPolygon.restype = c_void_p
    lgeos.GEOSGeom_createPolygon.argtypes = [c_void_p, c_void_p, c_uint]

    lgeos.GEOSGeom_createCollection.restype = c_void_p
    lgeos.GEOSGeom_createCollection.argtypes = [c_int, c_void_p, c_uint]

    lgeos.GEOSGeom_createEmptyCollection.restype = c_void_p
    lgeos.GEOSGeom_createEmptyCollection.argtypes = [c_int]

    lgeos.GEOSGeom_clone.restype = c_void_p
    lgeos.GEOSGeom_clone.argtypes = [c_void_p]

    # Memory management

    lgeos.GEOSGeom_destroy.restype = None
    lgeos.GEOSGeom_destroy.argtypes = [c_void_p]

    # Topology operations
    # Return NULL on exception

    lgeos.GEOSEnvelope.restype = c_void_p
    lgeos.GEOSEnvelope.argtypes = [c_void_p]

    lgeos.GEOSIntersection.restype = c_void_p
    lgeos.GEOSIntersection.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSConvexHull.restype = c_void_p
    lgeos.GEOSConvexHull.argtypes = [c_void_p]

    lgeos.GEOSDifference.restype = c_void_p
    lgeos.GEOSDifference.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSSymDifference.restype = c_void_p
    lgeos.GEOSSymDifference.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSBoundary.restype = c_void_p
    lgeos.GEOSBoundary.argtypes = [c_void_p]

    lgeos.GEOSUnion.restype = c_void_p
    lgeos.GEOSUnion.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSUnaryUnion.restype = c_void_p
    lgeos.GEOSUnaryUnion.argtypes = [c_void_p]

    # deprecated in 3.3.0: use GEOSUnaryUnion instead
    lgeos.GEOSUnionCascaded.restype = c_void_p
    lgeos.GEOSUnionCascaded.argtypes = [c_void_p]

    lgeos.GEOSPointOnSurface.restype = c_void_p
    lgeos.GEOSPointOnSurface.argtypes = [c_void_p]

    lgeos.GEOSGetCentroid.restype = c_void_p
    lgeos.GEOSGetCentroid.argtypes = [c_void_p]

    if geos_version >= (3, 5, 0):
        lgeos.GEOSClipByRect.restype = c_void_p
        lgeos.GEOSClipByRect.argtypes = [c_void_p, c_double, c_double, c_double, c_double]

    lgeos.GEOSPolygonize.restype = c_void_p
    lgeos.GEOSPolygonize.argtypes = [c_void_p, c_uint]

    lgeos.GEOSPolygonize_full.restype = c_void_p
    lgeos.GEOSPolygonize_full.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p]

    if geos_version >= (3, 4, 0):
        lgeos.GEOSDelaunayTriangulation.restype = c_void_p
        lgeos.GEOSDelaunayTriangulation.argtypes = [c_void_p, c_double, c_int]

    if geos_version >= (3, 5, 0):
        lgeos.GEOSVoronoiDiagram.restype = c_void_p
        lgeos.GEOSVoronoiDiagram.argtypes = [c_void_p, c_void_p, c_double, c_int]

    lgeos.GEOSLineMerge.restype = c_void_p
    lgeos.GEOSLineMerge.argtypes = [c_void_p]

    lgeos.GEOSSimplify.restype = c_void_p
    lgeos.GEOSSimplify.argtypes = [c_void_p, c_double]

    lgeos.GEOSTopologyPreserveSimplify.restype = c_void_p
    lgeos.GEOSTopologyPreserveSimplify.argtypes = [c_void_p, c_double]

    # Binary predicates
    # Return 2 on exception, 1 on true, 0 on false

    lgeos.GEOSDisjoint.restype = c_byte
    lgeos.GEOSDisjoint.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSTouches.restype = c_byte
    lgeos.GEOSTouches.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSIntersects.restype = c_byte
    lgeos.GEOSIntersects.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSCrosses.restype = c_byte
    lgeos.GEOSCrosses.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSWithin.restype = c_byte
    lgeos.GEOSWithin.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSContains.restype = c_byte
    lgeos.GEOSContains.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSOverlaps.restype = c_byte
    lgeos.GEOSOverlaps.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSCovers.restype = c_byte
    lgeos.GEOSCovers.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSCoveredBy.restype = c_byte
    lgeos.GEOSCoveredBy.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSEquals.restype = c_byte
    lgeos.GEOSEquals.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSEqualsExact.restype = c_byte
    lgeos.GEOSEqualsExact.argtypes = [c_void_p, c_void_p, c_double]

    # Unary predicate
    # Return 2 on exception, 1 on true, 0 on false

    lgeos.GEOSisEmpty.restype = c_byte
    lgeos.GEOSisEmpty.argtypes = [c_void_p]

    lgeos.GEOSisValid.restype = c_byte
    lgeos.GEOSisValid.argtypes = [c_void_p]

    lgeos.GEOSisValidReason.restype = allocated_c_char_p
    lgeos.GEOSisValidReason.argtypes = [c_void_p]

    lgeos.GEOSisSimple.restype = c_byte
    lgeos.GEOSisSimple.argtypes = [c_void_p]

    lgeos.GEOSisRing.restype = c_byte
    lgeos.GEOSisRing.argtypes = [c_void_p]

    lgeos.GEOSisClosed.restype = c_byte
    lgeos.GEOSisClosed.argtypes = [c_void_p]

    lgeos.GEOSHasZ.restype = c_byte
    lgeos.GEOSHasZ.argtypes = [c_void_p]

    # Dimensionally Extended 9 Intersection Model related

    lgeos.GEOSRelate.restype = allocated_c_char_p
    lgeos.GEOSRelate.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSRelatePattern.restype = c_byte
    lgeos.GEOSRelatePattern.argtypes = [c_void_p, c_void_p, c_char_p]

    lgeos.GEOSRelatePatternMatch.restype = c_byte
    lgeos.GEOSRelatePatternMatch.argtypes = [c_char_p, c_char_p]

    # Prepared Geometry Binary predicates
    # Return 2 on exception, 1 on true, 0 on false

    lgeos.GEOSPrepare.restype = c_void_p
    lgeos.GEOSPrepare.argtypes = [c_void_p]

    lgeos.GEOSPreparedGeom_destroy.restype = None
    lgeos.GEOSPreparedGeom_destroy.argtypes = [c_void_p]

    lgeos.GEOSPreparedDisjoint.restype = c_byte
    lgeos.GEOSPreparedDisjoint.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedTouches.restype = c_byte
    lgeos.GEOSPreparedTouches.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedIntersects.restype = c_byte
    lgeos.GEOSPreparedIntersects.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedCrosses.restype = c_byte
    lgeos.GEOSPreparedCrosses.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedWithin.restype = c_byte
    lgeos.GEOSPreparedWithin.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedContains.restype = c_byte
    lgeos.GEOSPreparedContains.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedContainsProperly.restype = c_byte
    lgeos.GEOSPreparedContainsProperly.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedOverlaps.restype = c_byte
    lgeos.GEOSPreparedOverlaps.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSPreparedCovers.restype = c_byte
    lgeos.GEOSPreparedCovers.argtypes = [c_void_p, c_void_p]

    # Geometry info

    lgeos.GEOSGeomType.restype = c_char_p
    lgeos.GEOSGeomType.argtypes = [c_void_p]

    lgeos.GEOSGeomTypeId.restype = c_int
    lgeos.GEOSGeomTypeId.argtypes = [c_void_p]

    lgeos.GEOSGetSRID.restype = c_int
    lgeos.GEOSGetSRID.argtypes = [c_void_p]

    lgeos.GEOSSetSRID.restype = None
    lgeos.GEOSSetSRID.argtypes = [c_void_p, c_int]

    lgeos.GEOSGetNumGeometries.restype = c_int
    lgeos.GEOSGetNumGeometries.argtypes = [c_void_p]

    lgeos.GEOSGetGeometryN.restype = c_void_p
    lgeos.GEOSGetGeometryN.argtypes = [c_void_p, c_int]

    lgeos.GEOSGetNumInteriorRings.restype = c_int
    lgeos.GEOSGetNumInteriorRings.argtypes = [c_void_p]

    lgeos.GEOSGetInteriorRingN.restype = c_void_p
    lgeos.GEOSGetInteriorRingN.argtypes = [c_void_p, c_int]

    lgeos.GEOSGetExteriorRing.restype = c_void_p
    lgeos.GEOSGetExteriorRing.argtypes = [c_void_p]

    lgeos.GEOSGetNumCoordinates.restype = c_int
    lgeos.GEOSGetNumCoordinates.argtypes = [c_void_p]

    lgeos.GEOSGeom_getCoordSeq.restype = c_void_p
    lgeos.GEOSGeom_getCoordSeq.argtypes = [c_void_p]

    lgeos.GEOSGeom_getDimensions.restype = c_int
    lgeos.GEOSGeom_getDimensions.argtypes = [c_void_p]

    lgeos.GEOSNormalize.restype = c_int
    lgeos.GEOSNormalize.argtypes = [c_void_p]

    # Misc functions

    lgeos.GEOSArea.restype = c_double
    lgeos.GEOSArea.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSLength.restype = c_int
    lgeos.GEOSLength.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSDistance.restype = c_int
    lgeos.GEOSDistance.argtypes = [c_void_p, c_void_p, c_void_p]

    lgeos.GEOSHausdorffDistance.restype = c_int
    lgeos.GEOSHausdorffDistance.argtypes = [c_void_p, c_void_p, c_void_p]

    # Reader and Writer APIs

    lgeos.GEOSWKTReader_create.restype = c_void_p
    lgeos.GEOSWKTReader_create.argtypes = []

    lgeos.GEOSWKTReader_destroy.restype = None
    lgeos.GEOSWKTReader_destroy.argtypes = [c_void_p]

    lgeos.GEOSWKTReader_read.restype = c_void_p
    lgeos.GEOSWKTReader_read.argtypes = [c_void_p, c_char_p]

    lgeos.GEOSWKTWriter_create.restype = c_void_p
    lgeos.GEOSWKTWriter_create.argtypes = []

    lgeos.GEOSWKTWriter_destroy.restype = None
    lgeos.GEOSWKTWriter_destroy.argtypes = [c_void_p]

    lgeos.GEOSWKTWriter_write.restype = allocated_c_char_p
    lgeos.GEOSWKTWriter_write.argtypes = [c_void_p, c_void_p]

    lgeos.GEOSWKTWriter_setTrim.restype = None
    lgeos.GEOSWKTWriter_setTrim.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKTWriter_setRoundingPrecision.restype = None
    lgeos.GEOSWKTWriter_setRoundingPrecision.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKTWriter_setOutputDimension.restype = None
    lgeos.GEOSWKTWriter_setOutputDimension.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKTWriter_getOutputDimension.restype = c_int
    lgeos.GEOSWKTWriter_getOutputDimension.argtypes = [c_void_p]

    lgeos.GEOSWKTWriter_setOld3D.restype = None
    lgeos.GEOSWKTWriter_setOld3D.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKBReader_create.restype = c_void_p
    lgeos.GEOSWKBReader_create.argtypes = []

    lgeos.GEOSWKBReader_destroy.restype = None
    lgeos.GEOSWKBReader_destroy.argtypes = [c_void_p]

    lgeos.GEOSWKBReader_read.restype = c_void_p
    lgeos.GEOSWKBReader_read.argtypes = [c_void_p, c_char_p, c_size_t]

    lgeos.GEOSWKBReader_readHEX.restype = c_void_p
    lgeos.GEOSWKBReader_readHEX.argtypes = [c_void_p, c_char_p, c_size_t]

    lgeos.GEOSWKBWriter_create.restype = c_void_p
    lgeos.GEOSWKBWriter_create.argtypes = []

    lgeos.GEOSWKBWriter_destroy.restype = None
    lgeos.GEOSWKBWriter_destroy.argtypes = [c_void_p]

    lgeos.GEOSWKBWriter_write.restype = allocated_c_char_p
    lgeos.GEOSWKBWriter_write.argtypes = [c_void_p, c_void_p, c_size_t_p]

    lgeos.GEOSWKBWriter_writeHEX.restype = allocated_c_char_p
    lgeos.GEOSWKBWriter_writeHEX.argtypes = [c_void_p, c_void_p, c_size_t_p]

    lgeos.GEOSWKBWriter_getOutputDimension.restype = c_int
    lgeos.GEOSWKBWriter_getOutputDimension.argtypes = [c_void_p]

    lgeos.GEOSWKBWriter_setOutputDimension.restype = None
    lgeos.GEOSWKBWriter_setOutputDimension.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKBWriter_getByteOrder.restype = c_int
    lgeos.GEOSWKBWriter_getByteOrder.argtypes = [c_void_p]

    lgeos.GEOSWKBWriter_setByteOrder.restype = None
    lgeos.GEOSWKBWriter_setByteOrder.argtypes = [c_void_p, c_int]

    lgeos.GEOSWKBWriter_getIncludeSRID.restype = c_int
    lgeos.GEOSWKBWriter_getIncludeSRID.argtypes = [c_void_p]

    lgeos.GEOSWKBWriter_setIncludeSRID.restype = None
    lgeos.GEOSWKBWriter_setIncludeSRID.argtypes = [c_void_p, c_int]

    lgeos.GEOSFree.restype = None
    lgeos.GEOSFree.argtypes = [c_void_p]

    lgeos.GEOSSnap.restype = c_void_p
    lgeos.GEOSSnap.argtypes = [c_void_p, c_void_p, c_double]

    lgeos.GEOSSharedPaths.restype = c_void_p
    lgeos.GEOSSharedPaths.argtypes = [c_void_p, c_void_p]

    if geos_version >= (3, 4, 0):
        lgeos.GEOSNearestPoints.restype = c_void_p
        lgeos.GEOSNearestPoints.argtypes = [c_void_p, c_void_p]

    if geos_version >= (3, 4, 2):
        lgeos.GEOSQueryCallback = CFUNCTYPE(None, c_void_p, c_void_p)

        lgeos.GEOSSTRtree_query.argtypes = [
            c_void_p, c_void_p, lgeos.GEOSQueryCallback, py_object]
        lgeos.GEOSSTRtree_query.restype = None

        lgeos.GEOSSTRtree_create.argtypes = [c_int]
        lgeos.GEOSSTRtree_create.restype = c_void_p

        lgeos.GEOSSTRtree_insert.argtypes = [c_void_p, c_void_p, py_object]
        lgeos.GEOSSTRtree_insert.restype = None

        lgeos.GEOSSTRtree_remove.argtypes = [c_void_p, c_void_p, py_object]
        lgeos.GEOSSTRtree_remove.restype = None

        lgeos.GEOSSTRtree_destroy.argtypes = [c_void_p]
        lgeos.GEOSSTRtree_destroy.restype = None

    if geos_version >= (3, 6, 0):
        lgeos.GEOSDistanceCallback = CFUNCTYPE(c_int, c_void_p, c_void_p, c_void_p, c_void_p)

        lgeos.GEOSSTRtree_nearest_generic.argtypes = [
            c_void_p, py_object, c_void_p, lgeos.GEOSDistanceCallback, py_object]
        lgeos.GEOSSTRtree_nearest_generic.restype = c_void_p

        lgeos.GEOSMinimumClearance.argtypes = [c_void_p]
        lgeos.GEOSMinimumClearance.restype = c_double

    if geos_version >= (3, 8, 0):
        lgeos.GEOSMakeValid.restype = c_void_p
        lgeos.GEOSMakeValid.argtypes = [c_void_p]

