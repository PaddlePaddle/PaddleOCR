"""
Support for GEOS prepared geometry operations.
"""

from shapely.geos import lgeos
from shapely.impl import DefaultImplementation, delegated
from pickle import PicklingError


class PreparedGeometry:
    """
    A geometry prepared for efficient comparison to a set of other geometries.

    Example:

      >>> from shapely.geometry import Point, Polygon
      >>> triangle = Polygon(((0.0, 0.0), (1.0, 1.0), (1.0, -1.0)))
      >>> p = prep(triangle)
      >>> p.intersects(Point(0.5, 0.5))
      True
    """

    impl = DefaultImplementation

    def __init__(self, context):
        if isinstance(context, PreparedGeometry):
            self.context = context.context
        else:
            self.context = context
        self.__geom__ = lgeos.GEOSPrepare(self.context._geom)
        self.prepared = True

    def __del__(self):
        if self.__geom__ is not None:
            try:
                lgeos.GEOSPreparedGeom_destroy(self.__geom__)
            except AttributeError:
                pass  # lgeos might be empty on shutdown.

        self.__geom__ = None
        self.context = None
        self.prepared = False

    @property
    def _geom(self):
        return self.__geom__

    @delegated
    def contains(self, other):
        """Returns True if the geometry contains the other, else False"""
        return bool(self.impl['prepared_contains'](self, other))

    @delegated
    def contains_properly(self, other):
        """Returns True if the geometry properly contains the other, else False"""
        return bool(self.impl['prepared_contains_properly'](self, other))

    @delegated
    def covers(self, other):
        """Returns True if the geometry covers the other, else False"""
        return bool(self.impl['prepared_covers'](self, other))

    @delegated
    def crosses(self, other):
        """Returns True if the geometries cross, else False"""
        return bool(self.impl['prepared_crosses'](self, other))

    @delegated
    def disjoint(self, other):
        """Returns True if geometries are disjoint, else False"""
        return bool(self.impl['prepared_disjoint'](self, other))

    @delegated
    def intersects(self, other):
        """Returns True if geometries intersect, else False"""
        return bool(self.impl['prepared_intersects'](self, other))

    @delegated
    def overlaps(self, other):
        """Returns True if geometries overlap, else False"""
        return bool(self.impl['prepared_overlaps'](self, other))

    @delegated
    def touches(self, other):
        """Returns True if geometries touch, else False"""
        return bool(self.impl['prepared_touches'](self, other))

    @delegated
    def within(self, other):
        """Returns True if geometry is within the other, else False"""
        return bool(self.impl['prepared_within'](self, other))

    def __reduce__(self):
        raise PicklingError("Prepared geometries cannot be pickled.")


def prep(ob):
    """Creates and returns a prepared geometric object."""
    return PreparedGeometry(ob)
