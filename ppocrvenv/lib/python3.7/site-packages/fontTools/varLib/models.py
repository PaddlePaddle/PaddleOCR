"""Variation fonts interpolation models."""

__all__ = [
    "nonNone",
    "allNone",
    "allEqual",
    "allEqualTo",
    "subList",
    "normalizeValue",
    "normalizeLocation",
    "supportScalar",
    "VariationModel",
]

from fontTools.misc.roundTools import noRound
from .errors import VariationModelError


def nonNone(lst):
    return [l for l in lst if l is not None]


def allNone(lst):
    return all(l is None for l in lst)


def allEqualTo(ref, lst, mapper=None):
    if mapper is None:
        return all(ref == item for item in lst)

    mapped = mapper(ref)
    return all(mapped == mapper(item) for item in lst)


def allEqual(lst, mapper=None):
    if not lst:
        return True
    it = iter(lst)
    try:
        first = next(it)
    except StopIteration:
        return True
    return allEqualTo(first, it, mapper=mapper)


def subList(truth, lst):
    assert len(truth) == len(lst)
    return [l for l, t in zip(lst, truth) if t]


def normalizeValue(v, triple):
    """Normalizes value based on a min/default/max triple.
    >>> normalizeValue(400, (100, 400, 900))
    0.0
    >>> normalizeValue(100, (100, 400, 900))
    -1.0
    >>> normalizeValue(650, (100, 400, 900))
    0.5
    """
    lower, default, upper = triple
    if not (lower <= default <= upper):
        raise ValueError(
            f"Invalid axis values, must be minimum, default, maximum: "
            f"{lower:3.3f}, {default:3.3f}, {upper:3.3f}"
        )
    v = max(min(v, upper), lower)
    if v == default:
        v = 0.0
    elif v < default:
        v = (v - default) / (default - lower)
    else:
        v = (v - default) / (upper - default)
    return v


def normalizeLocation(location, axes):
    """Normalizes location based on axis min/default/max values from axes.
    >>> axes = {"wght": (100, 400, 900)}
    >>> normalizeLocation({"wght": 400}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 100}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": 900}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 650}, axes)
    {'wght': 0.5}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': -1.0}
    >>> axes = {"wght": (0, 0, 1000)}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": -1}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 1.0}
    >>> normalizeLocation({"wght": 500}, axes)
    {'wght': 0.5}
    >>> normalizeLocation({"wght": 1001}, axes)
    {'wght': 1.0}
    >>> axes = {"wght": (0, 1000, 1000)}
    >>> normalizeLocation({"wght": 0}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": -1}, axes)
    {'wght': -1.0}
    >>> normalizeLocation({"wght": 500}, axes)
    {'wght': -0.5}
    >>> normalizeLocation({"wght": 1000}, axes)
    {'wght': 0.0}
    >>> normalizeLocation({"wght": 1001}, axes)
    {'wght': 0.0}
    """
    out = {}
    for tag, triple in axes.items():
        v = location.get(tag, triple[1])
        out[tag] = normalizeValue(v, triple)
    return out


def supportScalar(location, support, ot=True):
    """Returns the scalar multiplier at location, for a master
    with support.  If ot is True, then a peak value of zero
    for support of an axis means "axis does not participate".  That
    is how OpenType Variation Font technology works.
    >>> supportScalar({}, {})
    1.0
    >>> supportScalar({'wght':.2}, {})
    1.0
    >>> supportScalar({'wght':.2}, {'wght':(0,2,3)})
    0.1
    >>> supportScalar({'wght':2.5}, {'wght':(0,2,4)})
    0.75
    >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
    0.75
    >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)}, ot=False)
    0.375
    >>> supportScalar({'wght':2.5, 'wdth':0}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
    0.75
    >>> supportScalar({'wght':2.5, 'wdth':.5}, {'wght':(0,2,4), 'wdth':(-1,0,+1)})
    0.75
    """
    scalar = 1.0
    for axis, (lower, peak, upper) in support.items():
        if ot:
            # OpenType-specific case handling
            if peak == 0.0:
                continue
            if lower > peak or peak > upper:
                continue
            if lower < 0.0 and upper > 0.0:
                continue
            v = location.get(axis, 0.0)
        else:
            assert axis in location
            v = location[axis]
        if v == peak:
            continue
        if v <= lower or upper <= v:
            scalar = 0.0
            break
        if v < peak:
            scalar *= (v - lower) / (peak - lower)
        else:  # v > peak
            scalar *= (v - upper) / (peak - upper)
    return scalar


class VariationModel(object):

    """
  Locations must be in normalized space.  Ie. base master
  is at origin (0)::

      >>> from pprint import pprint
      >>> locations = [ \
      {'wght':100}, \
      {'wght':-100}, \
      {'wght':-180}, \
      {'wdth':+.3}, \
      {'wght':+120,'wdth':.3}, \
      {'wght':+120,'wdth':.2}, \
      {}, \
      {'wght':+180,'wdth':.3}, \
      {'wght':+180}, \
      ]
      >>> model = VariationModel(locations, axisOrder=['wght'])
      >>> pprint(model.locations)
      [{},
       {'wght': -100},
       {'wght': -180},
       {'wght': 100},
       {'wght': 180},
       {'wdth': 0.3},
       {'wdth': 0.3, 'wght': 180},
       {'wdth': 0.3, 'wght': 120},
       {'wdth': 0.2, 'wght': 120}]
      >>> pprint(model.deltaWeights)
      [{},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0},
       {0: 1.0, 4: 1.0, 5: 1.0},
       {0: 1.0, 3: 0.75, 4: 0.25, 5: 1.0, 6: 0.6666666666666666},
       {0: 1.0,
        3: 0.75,
        4: 0.25,
        5: 0.6666666666666667,
        6: 0.4444444444444445,
        7: 0.6666666666666667}]
	"""

    def __init__(self, locations, axisOrder=None):
        if len(set(tuple(sorted(l.items())) for l in locations)) != len(locations):
            raise VariationModelError("Locations must be unique.")

        self.origLocations = locations
        self.axisOrder = axisOrder if axisOrder is not None else []

        locations = [{k: v for k, v in loc.items() if v != 0.0} for loc in locations]
        keyFunc = self.getMasterLocationsSortKeyFunc(
            locations, axisOrder=self.axisOrder
        )
        self.locations = sorted(locations, key=keyFunc)

        # Mapping from user's master order to our master order
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]

        self._computeMasterSupports()
        self._subModels = {}

    def getSubModel(self, items):
        if None not in items:
            return self, items
        key = tuple(v is not None for v in items)
        subModel = self._subModels.get(key)
        if subModel is None:
            subModel = VariationModel(subList(key, self.origLocations), self.axisOrder)
            self._subModels[key] = subModel
        return subModel, subList(key, items)

    @staticmethod
    def getMasterLocationsSortKeyFunc(locations, axisOrder=[]):
        if {} not in locations:
            raise VariationModelError("Base master not found.")
        axisPoints = {}
        for loc in locations:
            if len(loc) != 1:
                continue
            axis = next(iter(loc))
            value = loc[axis]
            if axis not in axisPoints:
                axisPoints[axis] = {0.0}
            assert (
                value not in axisPoints[axis]
            ), 'Value "%s" in axisPoints["%s"] -->  %s' % (value, axis, axisPoints)
            axisPoints[axis].add(value)

        def getKey(axisPoints, axisOrder):
            def sign(v):
                return -1 if v < 0 else +1 if v > 0 else 0

            def key(loc):
                rank = len(loc)
                onPointAxes = [
                    axis
                    for axis, value in loc.items()
                    if axis in axisPoints and value in axisPoints[axis]
                ]
                orderedAxes = [axis for axis in axisOrder if axis in loc]
                orderedAxes.extend(
                    [axis for axis in sorted(loc.keys()) if axis not in axisOrder]
                )
                return (
                    rank,  # First, order by increasing rank
                    -len(onPointAxes),  # Next, by decreasing number of onPoint axes
                    tuple(
                        axisOrder.index(axis) if axis in axisOrder else 0x10000
                        for axis in orderedAxes
                    ),  # Next, by known axes
                    tuple(orderedAxes),  # Next, by all axes
                    tuple(
                        sign(loc[axis]) for axis in orderedAxes
                    ),  # Next, by signs of axis values
                    tuple(
                        abs(loc[axis]) for axis in orderedAxes
                    ),  # Next, by absolute value of axis values
                )

            return key

        ret = getKey(axisPoints, axisOrder)
        return ret

    def reorderMasters(self, master_list, mapping):
        # For changing the master data order without
        # recomputing supports and deltaWeights.
        new_list = [master_list[idx] for idx in mapping]
        self.origLocations = [self.origLocations[idx] for idx in mapping]
        locations = [
            {k: v for k, v in loc.items() if v != 0.0} for loc in self.origLocations
        ]
        self.mapping = [self.locations.index(l) for l in locations]
        self.reverseMapping = [locations.index(l) for l in self.locations]
        self._subModels = {}
        return new_list

    def _computeMasterSupports(self):
        self.supports = []
        regions = self._locationsToRegions()
        for i, region in enumerate(regions):
            locAxes = set(region.keys())
            # Walk over previous masters now
            for prev_region in regions[:i]:
                # Master with extra axes do not participte
                if not set(prev_region.keys()).issubset(locAxes):
                    continue
                # If it's NOT in the current box, it does not participate
                relevant = True
                for axis, (lower, peak, upper) in region.items():
                    if axis not in prev_region or not (
                        prev_region[axis][1] == peak
                        or lower < prev_region[axis][1] < upper
                    ):
                        relevant = False
                        break
                if not relevant:
                    continue

                # Split the box for new master; split in whatever direction
                # that has largest range ratio.
                #
                # For symmetry, we actually cut across multiple axes
                # if they have the largest, equal, ratio.
                # https://github.com/fonttools/fonttools/commit/7ee81c8821671157968b097f3e55309a1faa511e#commitcomment-31054804

                bestAxes = {}
                bestRatio = -1
                for axis in prev_region.keys():
                    val = prev_region[axis][1]
                    assert axis in region
                    lower, locV, upper = region[axis]
                    newLower, newUpper = lower, upper
                    if val < locV:
                        newLower = val
                        ratio = (val - locV) / (lower - locV)
                    elif locV < val:
                        newUpper = val
                        ratio = (val - locV) / (upper - locV)
                    else:  # val == locV
                        # Can't split box in this direction.
                        continue
                    if ratio > bestRatio:
                        bestAxes = {}
                        bestRatio = ratio
                    if ratio == bestRatio:
                        bestAxes[axis] = (newLower, locV, newUpper)

                for axis, triple in bestAxes.items():
                    region[axis] = triple
            self.supports.append(region)
        self._computeDeltaWeights()

    def _locationsToRegions(self):
        locations = self.locations
        # Compute min/max across each axis, use it as total range.
        # TODO Take this as input from outside?
        minV = {}
        maxV = {}
        for l in locations:
            for k, v in l.items():
                minV[k] = min(v, minV.get(k, v))
                maxV[k] = max(v, maxV.get(k, v))

        regions = []
        for loc in locations:
            region = {}
            for axis, locV in loc.items():
                if locV > 0:
                    region[axis] = (0, locV, maxV[axis])
                else:
                    region[axis] = (minV[axis], locV, 0)
            regions.append(region)
        return regions

    def _computeDeltaWeights(self):
        self.deltaWeights = []
        for i, loc in enumerate(self.locations):
            deltaWeight = {}
            # Walk over previous masters now, populate deltaWeight
            for j, support in enumerate(self.supports[:i]):
                scalar = supportScalar(loc, support)
                if scalar:
                    deltaWeight[j] = scalar
            self.deltaWeights.append(deltaWeight)

    def getDeltas(self, masterValues, *, round=noRound):
        assert len(masterValues) == len(self.deltaWeights)
        mapping = self.reverseMapping
        out = []
        for i, weights in enumerate(self.deltaWeights):
            delta = masterValues[mapping[i]]
            for j, weight in weights.items():
                if weight == 1:
                    delta -= out[j]
                else:
                    delta -= out[j] * weight
            out.append(round(delta))
        return out

    def getDeltasAndSupports(self, items, *, round=noRound):
        model, items = self.getSubModel(items)
        return model.getDeltas(items, round=round), model.supports

    def getScalars(self, loc):
        return [supportScalar(loc, support) for support in self.supports]

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        v = None
        assert len(deltas) == len(scalars)
        for delta, scalar in zip(deltas, scalars):
            if not scalar:
                continue
            contribution = delta * scalar
            if v is None:
                v = contribution
            else:
                v += contribution
        return v

    def interpolateFromDeltas(self, loc, deltas):
        scalars = self.getScalars(loc)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromMasters(self, loc, masterValues, *, round=noRound):
        deltas = self.getDeltas(masterValues, round=round)
        return self.interpolateFromDeltas(loc, deltas)

    def interpolateFromMastersAndScalars(self, masterValues, scalars, *, round=noRound):
        deltas = self.getDeltas(masterValues, round=round)
        return self.interpolateFromDeltasAndScalars(deltas, scalars)


def piecewiseLinearMap(v, mapping):
    keys = mapping.keys()
    if not keys:
        return v
    if v in keys:
        return mapping[v]
    k = min(keys)
    if v < k:
        return v + mapping[k] - k
    k = max(keys)
    if v > k:
        return v + mapping[k] - k
    # Interpolate
    a = max(k for k in keys if k < v)
    b = min(k for k in keys if k > v)
    va = mapping[a]
    vb = mapping[b]
    return va + (vb - va) * (v - a) / (b - a)


def main(args=None):
    """Normalize locations on a given designspace"""
    from fontTools import configLogger
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.models",
        description=main.__doc__,
    )
    parser.add_argument(
        "--loglevel",
        metavar="LEVEL",
        default="INFO",
        help="Logging level (defaults to INFO)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--designspace", metavar="DESIGNSPACE", type=str)
    group.add_argument(
        "-l",
        "--locations",
        metavar="LOCATION",
        nargs="+",
        help="Master locations as comma-separate coordinates. One must be all zeros.",
    )

    args = parser.parse_args(args)

    configLogger(level=args.loglevel)
    from pprint import pprint

    if args.designspace:
        from fontTools.designspaceLib import DesignSpaceDocument

        doc = DesignSpaceDocument()
        doc.read(args.designspace)
        locs = [s.location for s in doc.sources]
        print("Original locations:")
        pprint(locs)
        doc.normalize()
        print("Normalized locations:")
        locs = [s.location for s in doc.sources]
        pprint(locs)
    else:
        axes = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        locs = [
            dict(zip(axes, (float(v) for v in s.split(",")))) for s in args.locations
        ]

    model = VariationModel(locs)
    print("Sorted locations:")
    pprint(model.locations)
    print("Supports:")
    pprint(model.supports)


if __name__ == "__main__":
    import doctest, sys

    if len(sys.argv) > 1:
        sys.exit(main())

    sys.exit(doctest.testmod().failed)
