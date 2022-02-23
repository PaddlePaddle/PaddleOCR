"""
Geometry factories based on the geo interface
"""
import warnings

from shapely.errors import GeometryTypeError
from shapely.errors import ShapelyDeprecationWarning

from .point import Point, asPoint
from .linestring import LineString, asLineString
from .polygon import LinearRing, Polygon, asPolygon
from .multipoint import MultiPoint, asMultiPoint
from .multilinestring import MultiLineString, asMultiLineString
from .multipolygon import MultiPolygon, MultiPolygonAdapter
from .collection import GeometryCollection

# numpy is an optional dependency
try:
    import numpy as np
except ImportError:
    _has_numpy = False
else:
    _has_numpy = True


def _is_coordinates_empty(coordinates):
    """Helper to identify if coordinates or subset of coordinates are empty"""

    if coordinates is None:
        return True

    is_numpy_array = _has_numpy and isinstance(coordinates, np.ndarray)
    if isinstance(coordinates, (list, tuple)) or is_numpy_array:
        if len(coordinates) == 0:
            return True
        return all(map(_is_coordinates_empty, coordinates))
    else:
        return False


def _empty_shape_for_no_coordinates(geom_type):
    """Return empty counterpart for geom_type"""
    if geom_type == 'point':
        return Point()
    elif geom_type == 'multipoint':
        return MultiPoint()
    elif geom_type == 'linestring':
        return LineString()
    elif geom_type == 'multilinestring':
        return MultiLineString()
    elif geom_type == 'polygon':
        return Polygon()
    elif geom_type == 'multipolygon':
        return MultiPolygon()
    else:
        raise GeometryTypeError("Unknown geometry type: %s" % geom_type)


def box(minx, miny, maxx, maxy, ccw=True):
    """Returns a rectangular polygon with configurable normal vector"""
    coords = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    if not ccw:
        coords = coords[::-1]
    return Polygon(coords)


def shape(context):
    """
    Returns a new, independent geometry with coordinates *copied* from the
    context. Changes to the original context will not be reflected in the
    geometry object.

    Parameters
    ----------
    context :
        a GeoJSON-like dict, which provides a "type" member describing the type
        of the geometry and "coordinates" member providing a list of coordinates,
        or an object which implements __geo_interface__.

    Returns
    -------
    Geometry object

    Example
    -------
    Create a Point from GeoJSON, and then create a copy using __geo_interface__.

    >>> context = {'type': 'Point', 'coordinates': [0, 1]}
    >>> geom = shape(context)
    >>> geom.type == 'Point'
    True
    >>> geom.wkt
    'POINT (0 1)'
    >>> geom2 = shape(geom)
    >>> geom == geom2
    True
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context
    geom_type = ob.get("type").lower()
    if 'coordinates' in ob and _is_coordinates_empty(ob['coordinates']):
        return _empty_shape_for_no_coordinates(geom_type)
    elif geom_type == "point":
        return Point(ob["coordinates"])
    elif geom_type == "linestring":
        return LineString(ob["coordinates"])
    elif geom_type == "linearring":
        return LinearRing(ob["coordinates"])
    elif geom_type == "polygon":
        return Polygon(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return MultiPoint(ob["coordinates"])
    elif geom_type == "multilinestring":
        return MultiLineString(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygon(ob["coordinates"], context_type='geojson')
    elif geom_type == "geometrycollection":
        geoms = [shape(g) for g in ob.get("geometries", [])]
        return GeometryCollection(geoms)
    else:
        raise GeometryTypeError("Unknown geometry type: %s" % geom_type)


def asShape(context):
    """
    Adapts the context to a geometry interface. The coordinates remain
    stored in the context, and changes to them will be reflected in the
    returned geometry object.

    .. deprecated:: 1.8
       The proxy geometries (adapter classes) created by this function are
       deprecated, and this function will be removed in Shapely 2.0.
       Use the `shape` function instead to convert a GeoJSON-like dict
       to a Shapely geometry.

    Parameters
    ----------
    context :
        a GeoJSON-like dict, which provides a "type" member describing the type
        of the geometry and "coordinates" member providing a list of coordinates,
        or an object which implements __geo_interface__.

    Returns
    -------
    Geometry object

    Notes
    -----
    The Adapter classes returned by this function trade performance for
    reduced storage of coordinate values. In general, the shape() function
    should be used instead.

    Example
    -------
    Create a Point and Polygon from GeoJSON, change the coordinates of the Point's
    context and show that the corresponding geometry is changed, as well.

    >>> point_context = {'type': 'Point', 'coordinates': [0.5, 0.5]}
    >>> poly_context = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    >>> point, poly = asShape(point_context), asShape(poly_context)
    >>> poly.intersects(point)
    True
    >>> point_context['coordinates'][0] = 1.5
    >>> poly.intersects(point)
    False
    """
    if hasattr(context, "__geo_interface__"):
        ob = context.__geo_interface__
    else:
        ob = context

    try:
        geom_type = ob.get("type").lower()
    except AttributeError:
        raise ValueError("Context does not provide geo interface")

    if geom_type == "point":
        return asPoint(ob["coordinates"])
    elif geom_type == "linestring":
        return asLineString(ob["coordinates"])
    elif geom_type == "polygon":
        return asPolygon(ob["coordinates"][0], ob["coordinates"][1:])
    elif geom_type == "multipoint":
        return asMultiPoint(ob["coordinates"])
    elif geom_type == "multilinestring":
        return asMultiLineString(ob["coordinates"])
    elif geom_type == "multipolygon":
        return MultiPolygonAdapter(ob["coordinates"], context_type='geojson')
    elif geom_type == "geometrycollection":
        geoms = [asShape(g) for g in ob.get("geometries", [])]
        if len(geoms) == 0:
            # in this case no asShape call already raised the warning
            warnings.warn(
                "The proxy geometries (through the 'asShape()' constructor) "
                "are deprecated and will be removed in Shapely 2.0. Use the "
                "'shape()' function instead.",
                ShapelyDeprecationWarning, stacklevel=2)
        return GeometryCollection(geoms)
    else:
        raise GeometryTypeError("Unknown geometry type: %s" % geom_type)


def mapping(ob):
    """
    Returns a GeoJSON-like mapping from a Geometry or any
    object which implements __geo_interface__

    Parameters
    ----------
    ob :
        An object which implements __geo_interface__.

    Returns
    -------
    dict

    Example
    -------
    >>> pt = Point(0, 0)
    >>> mapping(pt)
    {'type': 'Point', 'coordinates': (0.0, 0.0)}
    """
    return ob.__geo_interface__
