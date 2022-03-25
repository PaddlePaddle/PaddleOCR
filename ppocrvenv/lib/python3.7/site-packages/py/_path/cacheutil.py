"""
This module contains multithread-safe cache implementations.

All Caches have

    getorbuild(key, builder)
    delentry(key)

methods and allow configuration when instantiating the cache class.
"""
from time import time as gettime

class BasicCache(object):
    def __init__(self, maxentries=128):
        self.maxentries = maxentries
        self.prunenum = int(maxentries - maxentries/8)
        self._dict = {}

    def clear(self):
        self._dict.clear()

    def _getentry(self, key):
        return self._dict[key]

    def _putentry(self, key, entry):
        self._prunelowestweight()
        self._dict[key] = entry

    def delentry(self, key, raising=False):
        try:
            del self._dict[key]
        except KeyError:
            if raising:
                raise

    def getorbuild(self, key, builder):
        try:
            entry = self._getentry(key)
        except KeyError:
            entry = self._build(key, builder)
            self._putentry(key, entry)
        return entry.value

    def _prunelowestweight(self):
        """ prune out entries with lowest weight. """
        numentries = len(self._dict)
        if numentries >= self.maxentries:
            # evict according to entry's weight
            items = [(entry.weight, key)
                        for key, entry in self._dict.items()]
            items.sort()
            index = numentries - self.prunenum
            if index > 0:
                for weight, key in items[:index]:
                    # in MT situations the element might be gone
                    self.delentry(key, raising=False)

class BuildcostAccessCache(BasicCache):
    """ A BuildTime/Access-counting cache implementation.
        the weight of a value is computed as the product of

            num-accesses-of-a-value * time-to-build-the-value

        The values with the least such weights are evicted
        if the cache maxentries threshold is superceded.
        For implementation flexibility more than one object
        might be evicted at a time.
    """
    # time function to use for measuring build-times

    def _build(self, key, builder):
        start = gettime()
        val = builder()
        end = gettime()
        return WeightedCountingEntry(val, end-start)


class WeightedCountingEntry(object):
    def __init__(self, value, oneweight):
        self._value = value
        self.weight = self._oneweight = oneweight

    def value(self):
        self.weight += self._oneweight
        return self._value
    value = property(value)

class AgingCache(BasicCache):
    """ This cache prunes out cache entries that are too old.
    """
    def __init__(self, maxentries=128, maxseconds=10.0):
        super(AgingCache, self).__init__(maxentries)
        self.maxseconds = maxseconds

    def _getentry(self, key):
        entry = self._dict[key]
        if entry.isexpired():
            self.delentry(key)
            raise KeyError(key)
        return entry

    def _build(self, key, builder):
        val = builder()
        entry = AgingEntry(val, gettime() + self.maxseconds)
        return entry

class AgingEntry(object):
    def __init__(self, value, expirationtime):
        self.value = value
        self.weight = expirationtime

    def isexpired(self):
        t = gettime()
        return t >= self.weight
