"""
Support for GEOS spatial predicates
"""

from shapely.geos import PredicateError
from shapely.topology import Delegating


class BinaryPredicate(Delegating):

    def __call__(self, this, other, *args):
        self._validate(this)
        self._validate(other, stop_prepared=True)
        try:
            return self.fn(this._geom, other._geom, *args)
        except PredicateError as err:
            # Dig deeper into causes of errors.
            self._check_topology(err, this, other)


class UnaryPredicate(Delegating):

    def __call__(self, this):
        self._validate(this)
        return self.fn(this._geom)
