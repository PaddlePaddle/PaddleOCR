"""Linear referencing
"""

from shapely.errors import GeometryTypeError
from shapely.topology import Delegating


class LinearRefBase(Delegating):
    def _validate_line(self, ob):
        super()._validate(ob)
        if not ob.geom_type in ['LinearRing', 'LineString', 'MultiLineString']:
            raise GeometryTypeError("Only linear types support this operation")

class ProjectOp(LinearRefBase):
    def __call__(self, this, other):
        self._validate_line(this)
        self._validate(other)
        return self.fn(this._geom, other._geom)

class InterpolateOp(LinearRefBase):
    def __call__(self, this, distance):
        self._validate_line(this)
        return self.fn(this._geom, distance)


