import functools
import os
import threading

import cachetools


# Available cache options.
CACHE_IMPLEMENTATIONS = {
    "LFU": cachetools.LFUCache,
    "LRU": cachetools.LRUCache,
    "TTL": cachetools.TTLCache,
}

# Time to live (seconds) for entries in TTL cache. Defaults to 1 hour.
TTL_CACHE_TIMEOUT = 1 * 60 * 60

# Maximum no. of items to be saved in cache.
DEFAULT_CACHE_MAXSIZE = 128

# Lock to prevent multiple threads from accessing the cache at same time.
cache_access_lock = threading.RLock()

cache_type = os.environ.get("PREMAILER_CACHE", "LFU")
if cache_type not in CACHE_IMPLEMENTATIONS:
    raise ValueError(
        "Unsupported cache implementation. Available options: %s"
        % "/".join(CACHE_IMPLEMENTATIONS.keys())
    )

cache_init_options = {
    "maxsize": int(os.environ.get("PREMAILER_CACHE_MAXSIZE", DEFAULT_CACHE_MAXSIZE))
}
if cache_type == "TTL":
    cache_init_options["ttl"] = int(
        os.environ.get("PREMAILER_CACHE_TTL", TTL_CACHE_TIMEOUT)
    )

cache = CACHE_IMPLEMENTATIONS[cache_type](**cache_init_options)


def function_cache(**options):
    def decorator(func):
        @cachetools.cached(cache, lock=cache_access_lock)
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        return inner

    return decorator
