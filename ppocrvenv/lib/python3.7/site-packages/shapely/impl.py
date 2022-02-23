"""Implementation of the intermediary layer between Shapely and GEOS

This is layer number 2 from the list below.

1) geometric objects: the Python OO API.
2) implementation map: an abstraction that permits different backends.
3) backend: callable objects that take Shapely geometric objects as arguments
   and, with GEOS as a backend, translate them to C data structures.
4) GEOS library: algorithms implemented in C++.

Shapely 1.2 includes a GEOS backend and it is the default.
"""

from functools import wraps

from shapely.algorithms import cga
from shapely.coords import BoundsOp
from shapely.geos import lgeos
from shapely.linref import ProjectOp, InterpolateOp
from shapely.predicates import BinaryPredicate, UnaryPredicate
from shapely.topology import BinaryRealProperty, BinaryTopologicalOp
from shapely.topology import UnaryRealProperty, UnaryTopologicalOp


class ImplementationError(
        AttributeError, KeyError, NotImplementedError):
    """To be raised when the registered implementation does not
    support the requested method."""


def delegated(func):
    """A delegated method raises AttributeError in the absence of backend
    support."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError:
            raise ImplementationError(
                "Method '%s' not provided by registered "
                "implementation '%s'" % (func.__name__, args[0].impl))
    return wrapper

# Map geometry methods to their GEOS delegates


class BaseImpl:
    """Base class for registrable implementations."""

    def __init__(self, values):
        self.map = dict(values)

    def update(self, values):
        self.map.update(values)

    def __getitem__(self, key):
        try:
            return self.map[key]
        except KeyError:
            raise ImplementationError(
                "Method '%s' not provided by registered "
                "implementation '%s'" % (key, self.map))

    def __contains__(self, key):
        return key in self.map


class GEOSImpl(BaseImpl):
    """GEOS implementation"""

    def __repr__(self):
        return '<GEOSImpl object: GEOS C API version %s>' % (
            lgeos.geos_capi_version,)


IMPL330 = {
    'area': (UnaryRealProperty, 'area'),
    'distance': (BinaryRealProperty, 'distance'),
    'length': (UnaryRealProperty, 'length'),
    #
    'boundary': (UnaryTopologicalOp, 'boundary'),
    'bounds': (BoundsOp, None),
    'centroid': (UnaryTopologicalOp, 'centroid'),
    'representative_point': (UnaryTopologicalOp, 'representative_point'),
    'envelope': (UnaryTopologicalOp, 'envelope'),
    'convex_hull': (UnaryTopologicalOp, 'convex_hull'),
    'buffer': (UnaryTopologicalOp, 'buffer'),
    'simplify': (UnaryTopologicalOp, 'simplify'),
    'topology_preserve_simplify':
        (UnaryTopologicalOp, 'topology_preserve_simplify'),
    'normalize': (UnaryTopologicalOp, 'normalize'),
    #
    'difference': (BinaryTopologicalOp, 'difference'),
    'intersection': (BinaryTopologicalOp, 'intersection'),
    'symmetric_difference': (BinaryTopologicalOp, 'symmetric_difference'),
    'union': (BinaryTopologicalOp, 'union'),
    #
    'has_z': (UnaryPredicate, 'has_z'),
    'is_empty': (UnaryPredicate, 'is_empty'),
    'is_ring': (UnaryPredicate, 'is_ring'),
    'is_simple': (UnaryPredicate, 'is_simple'),
    'is_valid': (UnaryPredicate, 'is_valid'),
    'is_closed': (UnaryPredicate, 'is_closed'),
    #
    'relate': (BinaryPredicate, 'relate'),
    'contains': (BinaryPredicate, 'contains'),
    'crosses': (BinaryPredicate, 'crosses'),
    'disjoint': (BinaryPredicate, 'disjoint'),
    'equals': (BinaryPredicate, 'equals'),
    'intersects': (BinaryPredicate, 'intersects'),
    'overlaps': (BinaryPredicate, 'overlaps'),
    'touches': (BinaryPredicate, 'touches'),
    'within': (BinaryPredicate, 'within'),
    'covers': (BinaryPredicate, 'covers'),
    'covered_by': (BinaryPredicate, 'covered_by'),
    'equals_exact': (BinaryPredicate, 'equals_exact'),
    'relate_pattern': (BinaryPredicate, 'relate_pattern'),
    'prepared_disjoint': (BinaryPredicate, 'prepared_disjoint'),
    'prepared_touches': (BinaryPredicate, 'prepared_touches'),
    'prepared_crosses': (BinaryPredicate, 'prepared_crosses'),
    'prepared_within': (BinaryPredicate, 'prepared_within'),
    'prepared_overlaps': (BinaryPredicate, 'prepared_overlaps'),
    'prepared_intersects': (BinaryPredicate, 'prepared_intersects'),
    'prepared_contains': (BinaryPredicate, 'prepared_contains'),
    'prepared_contains_properly':
        (BinaryPredicate, 'prepared_contains_properly'),
    'prepared_covers': (BinaryPredicate, 'prepared_covers'),

    'parallel_offset': (UnaryTopologicalOp, 'parallel_offset'),
    'project_normalized': (ProjectOp, 'project_normalized'),
    'project': (ProjectOp, 'project'),
    'interpolate_normalized': (InterpolateOp, 'interpolate_normalized'),
    'interpolate': (InterpolateOp, 'interpolate'),
    'buffer_with_style': (UnaryTopologicalOp, 'buffer_with_style'),
    'buffer_with_params': (UnaryTopologicalOp, 'buffer_with_params'),
    'hausdorff_distance': (BinaryRealProperty, 'hausdorff_distance'),

    # First pure Python implementation
    'is_ccw': (cga.is_ccw_impl, 'is_ccw'),
    }

IMPL360 = {
    'minimum_clearance': (UnaryRealProperty, 'minimum_clearance')
}

def impl_items(defs):
    return [(k, v[0](v[1])) for k, v in list(defs.items())]

imp = GEOSImpl(dict(impl_items(IMPL330)))
if lgeos.geos_version >= (3, 6, 0):
    imp.update(impl_items(IMPL360))

DefaultImplementation = imp
