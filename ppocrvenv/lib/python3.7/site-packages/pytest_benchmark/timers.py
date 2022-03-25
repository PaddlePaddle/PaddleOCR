from time import time as timeout_timer

from .compat import XRANGE

try:
    from __pypy__.time import CLOCK_MONOTONIC
    from __pypy__.time import clock_gettime

    def monotonic():
        return clock_gettime(CLOCK_MONOTONIC)
except ImportError:
    from timeit import default_timer
else:
    default_timer = monotonic


def compute_timer_precision(timer):
    precision = None
    points = 0
    timeout = timeout_timer() + 1.0
    previous = timer()
    while timeout_timer() < timeout or points < 5:
        for _ in XRANGE(10):
            t1 = timer()
            t2 = timer()
            dt = t2 - t1
            if 0 < dt:
                break
        else:
            dt = t2 - previous
            if dt <= 0.0:
                continue
        if precision is not None:
            precision = min(precision, dt)
        else:
            precision = dt
        points += 1
        previous = timer()
    return precision
