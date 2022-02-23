"""Proxy for coordinates stored outside Shapely geometries
"""

from shapely.geometry.base import EMPTY
from shapely.geos import lgeos


class CachingGeometryProxy:

    context = None
    factory = None
    __geom__ = EMPTY
    _gtag = None

    def __init__(self, context):
        self.context = context

    @property
    def _is_empty(self):
        return self.__geom__ in [EMPTY, None]

    def empty(self, val=EMPTY):
        if not self._is_empty and self.__geom__:
            lgeos.GEOSGeom_destroy(self.__geom__)
        self.__geom__ = val

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        gtag = self.gtag()
        if gtag != self._gtag or self._is_empty:
            self.empty()
            if len(self.context) > 0:
                self.__geom__, n = self.factory(self.context)
        self._gtag = gtag
        return self.__geom__
        
    def gtag(self):
        return hash(repr(self.context))

    def __setattr__(self, name, value):
        # to override the custom one in BaseGeometry, so we don't warn
        # for the proxy classes, which are already deprecated itself
        object.__setattr__(self, name, value)


class PolygonProxy(CachingGeometryProxy):

    @property
    def _geom(self):
        """Keeps the GEOS geometry in synch with the context."""
        gtag = self.gtag()
        if gtag != self._gtag or self._is_empty:
            self.empty()
            self.__geom__, n = self.factory(self.context[0], self.context[1])
        self._gtag = gtag
        return self.__geom__
