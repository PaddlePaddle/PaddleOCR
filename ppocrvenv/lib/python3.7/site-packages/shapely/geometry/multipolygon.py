"""Collections of polygons and related utilities
"""

from ctypes import c_void_p, cast
import warnings

from shapely.errors import ShapelyDeprecationWarning
from shapely.geos import lgeos
from shapely.geometry.base import BaseMultipartGeometry, geos_geom_from_py
from shapely.geometry import polygon
from shapely.geometry.proxy import CachingGeometryProxy

__all__ = ['MultiPolygon', 'asMultiPolygon']


class MultiPolygon(BaseMultipartGeometry):

    """A collection of one or more polygons

    If component polygons overlap the collection is `invalid` and some
    operations on it may fail.

    Attributes
    ----------
    geoms : sequence
        A sequence of `Polygon` instances
    """

    def __init__(self, polygons=None, context_type='polygons'):
        """
        Parameters
        ----------
        polygons : sequence
            A sequence of (shell, holes) tuples where shell is the sequence
            representation of a linear ring (see linearring.py) and holes is
            a sequence of such linear rings

        Example
        -------
        Construct a collection from a sequence of coordinate tuples

          >>> from shapely.geometry import Polygon
          >>> ob = MultiPolygon( [
          ...     (
          ...     ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
          ...     [((0.1,0.1), (0.1,0.2), (0.2,0.2), (0.2,0.1))]
          ...     )
          ... ] )
          >>> len(ob.geoms)
          1
          >>> type(ob.geoms[0]) == Polygon
          True
        """
        super().__init__()

        if not polygons:
            # allow creation of empty multipolygons, to support unpickling
            return
        elif context_type == 'polygons':
            geom, n = geos_multipolygon_from_polygons(polygons)
        elif context_type == 'geojson':
            geom, n = geos_multipolygon_from_py(polygons)
        self._set_geom(geom)
        self._ndim = n

    def shape_factory(self, *args):
        return polygon.Polygon(*args)

    @property
    def __geo_interface__(self):
        allcoords = []
        for geom in self.geoms:
            coords = []
            coords.append(tuple(geom.exterior.coords))
            for hole in geom.interiors:
                coords.append(tuple(hole.coords))
            allcoords.append(tuple(coords))
        return {
            'type': 'MultiPolygon',
            'coordinates': allcoords
            }

    def svg(self, scale_factor=1., fill_color=None, opacity=None):
        """Returns group of SVG path elements for the MultiPolygon geometry.

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
        return '<g>' + \
            ''.join(p.svg(scale_factor, fill_color, opacity) for p in self.geoms) + \
            '</g>'


class MultiPolygonAdapter(CachingGeometryProxy, MultiPolygon):

    context = None
    _other_owned = False

    def __init__(self, context, context_type='polygons'):
        warnings.warn(
            "The proxy geometries (through the 'asShape()', 'asMultiPolygon()' "
            "or 'MultiPolygonAdapter()' constructors) are deprecated and will be "
            "removed in Shapely 2.0. Use the 'shape()' function or the "
            "standard 'MultiPolygon()' constructor instead.",
            ShapelyDeprecationWarning, stacklevel=3)
        self.context = context
        if context_type == 'geojson':
            self.factory = geos_multipolygon_from_py
        elif context_type == 'polygons':
            self.factory = geos_multipolygon_from_polygons

    @property
    def _ndim(self):
        try:
            # From array protocol
            array = self.context[0][0].__array_interface__
            n = array['shape'][1]
            assert n == 2 or n == 3
            return n
        except AttributeError:
            # Fall back on list
            return len(self.context[0][0][0])


def asMultiPolygon(context):
    """Adapts a sequence of objects to the MultiPolygon interface"""
    return MultiPolygonAdapter(context)


def geos_multipolygon_from_py(ob):
    """ob must provide Python geo interface coordinates."""
    L = len(ob)
    assert L >= 1

    N = len(ob[0][0][0])
    assert N == 2 or N == 3

    subs = (c_void_p * L)()
    for l in range(L):
        geom, ndims = polygon.geos_polygon_from_py(ob[l][0], ob[l][1:])
        subs[l] = cast(geom, c_void_p)

    return (lgeos.GEOSGeom_createCollection(6, subs, L), N)


def geos_multipolygon_from_polygons(arg):
    """Creates a GEOS multipolygon from a sequence of polygon-like objects.

    Parameters
    ----------
    arg : sequence or MultiPolygon

    Returns
    -------
    int
        Pointer to a GEOS multipolygon.

    """
    if isinstance(arg, MultiPolygon):
        return geos_geom_from_py(arg)

    obs = getattr(arg, 'geoms', arg)
    obs = [ob for ob in obs
           if ob and not (isinstance(ob, polygon.Polygon) and ob.is_empty)]
    L = len(obs)

    # Bail immediately if we have no input points.
    if L <= 0:
        return (lgeos.GEOSGeom_createEmptyCollection(6), 3)

    # This function does not accept sequences of MultiPolygons: there is
    # no implicit flattening.
    if isinstance(obs[0], MultiPolygon):
        raise ValueError("Sequences of multi-polygons are not valid arguments")

    exemplar = obs[0]
    try:
        N = len(exemplar[0][0])
    except TypeError:
        N = exemplar._ndim

    assert N == 2 or N == 3

    subs = (c_void_p * L)()

    for i, ob in enumerate(obs):
        if isinstance(ob, polygon.Polygon):
            shell = ob.exterior
            holes = ob.interiors
        else:
            shell = ob[0]
            holes = ob[1]

        geom, ndims = polygon.geos_polygon_from_py(shell, holes)
        subs[i] = cast(geom, c_void_p)

    return (lgeos.GEOSGeom_createCollection(6, subs, L), N)
