"""`functools.lru_cache` compatible memoizing function decorators."""

__all__ = ("fifo_cache", "lfu_cache", "lru_cache", "mru_cache", "rr_cache", "ttl_cache")

import collections
import functools
import math
import random
import time

try:
    from threading import RLock
except ImportError:  # pragma: no cover
    from dummy_threading import RLock

from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys


_CacheInfo = collections.namedtuple(
    "CacheInfo", ["hits", "misses", "maxsize", "currsize"]
)


class _UnboundCache(dict):
    @property
    def maxsize(self):
        return None

    @property
    def currsize(self):
        return len(self)


class _UnboundTTLCache(TTLCache):
    def __init__(self, ttl, timer):
        TTLCache.__init__(self, math.inf, ttl, timer)

    @property
    def maxsize(self):
        return None


def _cache(cache, typed):
    maxsize = cache.maxsize

    def decorator(func):
        key = keys.typedkey if typed else keys.hashkey
        lock = RLock()
        stats = [0, 0]

        def wrapper(*args, **kwargs):
            k = key(*args, **kwargs)
            with lock:
                try:
                    v = cache[k]
                    stats[0] += 1
                    return v
                except KeyError:
                    stats[1] += 1
            v = func(*args, **kwargs)
            # in case of a race, prefer the item already in the cache
            try:
                with lock:
                    return cache.setdefault(k, v)
            except ValueError:
                return v  # value too large

        def cache_info():
            with lock:
                hits, misses = stats
                maxsize = cache.maxsize
                currsize = cache.currsize
            return _CacheInfo(hits, misses, maxsize, currsize)

        def cache_clear():
            with lock:
                try:
                    cache.clear()
                finally:
                    stats[:] = [0, 0]

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper.cache_parameters = lambda: {"maxsize": maxsize, "typed": typed}
        functools.update_wrapper(wrapper, func)
        return wrapper

    return decorator


def fifo_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a First In First Out (FIFO)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(FIFOCache(128), typed)(maxsize)
    else:
        return _cache(FIFOCache(maxsize), typed)


def lfu_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Frequently Used (LFU)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(LFUCache(128), typed)(maxsize)
    else:
        return _cache(LFUCache(maxsize), typed)


def lru_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(LRUCache(128), typed)(maxsize)
    else:
        return _cache(LRUCache(maxsize), typed)


def mru_cache(maxsize=128, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Most Recently Used (MRU)
    algorithm.
    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(MRUCache(128), typed)(maxsize)
    else:
        return _cache(MRUCache(maxsize), typed)


def rr_cache(maxsize=128, choice=random.choice, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Random Replacement (RR)
    algorithm.

    """
    if maxsize is None:
        return _cache(_UnboundCache(), typed)
    elif callable(maxsize):
        return _cache(RRCache(128, choice), typed)(maxsize)
    else:
        return _cache(RRCache(maxsize, choice), typed)


def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False):
    """Decorator to wrap a function with a memoizing callable that saves
    up to `maxsize` results based on a Least Recently Used (LRU)
    algorithm with a per-item time-to-live (TTL) value.
    """
    if maxsize is None:
        return _cache(_UnboundTTLCache(ttl, timer), typed)
    elif callable(maxsize):
        return _cache(TTLCache(128, ttl, timer), typed)(maxsize)
    else:
        return _cache(TTLCache(maxsize, ttl, timer), typed)
