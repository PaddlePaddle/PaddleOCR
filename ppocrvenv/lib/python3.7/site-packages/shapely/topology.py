"""
Intermediaries supporting GEOS topological operations

These methods all take Shapely geometries and other Python objects and delegate
to GEOS functions via ctypes.

These methods return ctypes objects that should be recast by the caller.
"""

from ctypes import byref, c_double
from shapely.geos import TopologicalError, lgeos
from shapely.errors import InvalidGeometryError


class Validating:

    def _validate(self, ob, stop_prepared=False):
        if ob is None or ob._geom is None:
            raise InvalidGeometryError("Null geometry supports no operations")
        if stop_prepared and hasattr(ob, 'prepared'):
            raise ValueError("Prepared geometries cannot be operated on")


class Delegating(Validating):

    def __init__(self, name):
        self._name = name
        self.fn = lgeos.methods[name]

    def _check_topology(self, err, *geoms):
        """Raise TopologicalError if geoms are invalid.

        Else, raise original error.
        """
        for geom in geoms:
            if not geom.is_valid:
                raise TopologicalError(
                    "The operation '%s' could not be performed. "
                    "Likely cause is invalidity of the geometry %s" % (
                        self.fn.__name__, repr(geom)))
        raise err


class BinaryRealProperty(Delegating):

    def __call__(self, this, other):
        self._validate(this)
        self._validate(other, stop_prepared=True)
        d = c_double()
        retval = self.fn(this._geom, other._geom, byref(d))
        return d.value


class UnaryRealProperty(Delegating):

    def __call__(self, this):
        self._validate(this)
        d = c_double()
        retval = self.fn(this._geom, byref(d))
        return d.value


class BinaryTopologicalOp(Delegating):

    def __call__(self, this, other, *args):
        self._validate(this)
        self._validate(other, stop_prepared=True)
        product = self.fn(this._geom, other._geom, *args)
        if product is None:
            err = TopologicalError(
                    "This operation could not be performed. Reason: unknown")
            self._check_topology(err, this, other)
        return product


class UnaryTopologicalOp(Delegating):

    def __call__(self, this, *args):
        self._validate(this)
        return self.fn(this._geom, *args)
