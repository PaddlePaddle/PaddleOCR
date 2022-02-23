"""
Iterative forms of operations
"""
import warnings

from shapely.errors import InvalidGeometryError, ShapelyDeprecationWarning
from shapely.topology import Delegating


class IterOp(Delegating):

    """A generating non-data descriptor.
    """

    def __call__(self, context, iterator, value=True):
        warnings.warn(
            "The '{0}' function is deprecated and will be removed in "
            "Shapely 2.0".format(self._name),
            ShapelyDeprecationWarning, stacklevel=2)
        if context._geom is None:
            raise InvalidGeometryError("Null geometry supports no operations")
        for item in iterator:
            try:
                this_geom, ob = item
            except TypeError:
                this_geom = item
                ob = this_geom
            if not this_geom._geom:
                raise InvalidGeometryError("Null geometry supports no operations")
            try:
                retval = self.fn(context._geom, this_geom._geom)
            except Exception as err:
                self._check_topology(err, context, this_geom)
            if bool(retval) == value:
                yield ob


# utilities
disjoint = IterOp('disjoint')
touches = IterOp('touches')
intersects = IterOp('intersects')
crosses = IterOp('crosses')
within = IterOp('within')
contains = IterOp('contains')
overlaps = IterOp('overlaps')
equals = IterOp('equals')
