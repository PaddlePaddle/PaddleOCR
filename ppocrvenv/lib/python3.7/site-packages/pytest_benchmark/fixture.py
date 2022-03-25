from __future__ import division
from __future__ import print_function

import cProfile
import gc
import pstats
import sys
import time
import traceback
from math import ceil

from .compat import INT
from .compat import XRANGE
from .timers import compute_timer_precision
from .utils import NameWrapper
from .utils import format_time

try:
    import statistics
except (ImportError, SyntaxError):
    statistics_error = traceback.format_exc()
    statistics = None
else:
    statistics_error = None
    from .stats import Metadata


class FixtureAlreadyUsed(Exception):
    pass


class BenchmarkFixture(object):
    _precisions = {}

    def __init__(self, node, disable_gc, timer, min_rounds, min_time, max_time, warmup, warmup_iterations,
                 calibration_precision, add_stats, logger, warner, disabled, cprofile, group=None):
        self.name = node.name
        self.fullname = node._nodeid
        self.disabled = disabled
        if hasattr(node, 'callspec'):
            self.param = node.callspec.id
            self.params = node.callspec.params
        else:
            self.param = None
            self.params = None
        self.group = group
        self.has_error = False
        self.extra_info = {}
        self.skipped = False

        self._disable_gc = disable_gc
        self._timer = timer.target
        self._min_rounds = min_rounds
        self._max_time = float(max_time)
        self._min_time = float(min_time)
        self._add_stats = add_stats
        self._calibration_precision = calibration_precision
        self._warmup = warmup and warmup_iterations
        self._logger = logger
        self._warner = warner
        self._cleanup_callbacks = []
        self._mode = None
        self.cprofile = cprofile
        self.cprofile_stats = None
        self.stats = None

    @property
    def enabled(self):
        return not self.disabled

    def _get_precision(self, timer):
        if timer in self._precisions:
            timer_precision = self._precisions[timer]
        else:
            timer_precision = self._precisions[timer] = compute_timer_precision(timer)
            self._logger.debug("")
            self._logger.debug("Computing precision for %s ... %ss." % (
                NameWrapper(timer), format_time(timer_precision)), blue=True, bold=True)
        return timer_precision

    def _make_runner(self, function_to_benchmark, args, kwargs):
        def runner(loops_range, timer=self._timer):
            gc_enabled = gc.isenabled()
            if self._disable_gc:
                gc.disable()
            tracer = sys.gettrace()
            sys.settrace(None)
            try:
                if loops_range:
                    start = timer()
                    for _ in loops_range:
                        function_to_benchmark(*args, **kwargs)
                    end = timer()
                    return end - start
                else:
                    start = timer()
                    result = function_to_benchmark(*args, **kwargs)
                    end = timer()
                    return end - start, result
            finally:
                sys.settrace(tracer)
                if gc_enabled:
                    gc.enable()

        return runner

    def _make_stats(self, iterations):
        bench_stats = Metadata(self, iterations=iterations, options={
            "disable_gc": self._disable_gc,
            "timer": self._timer,
            "min_rounds": self._min_rounds,
            "max_time": self._max_time,
            "min_time": self._min_time,
            "warmup": self._warmup,
        })
        self._add_stats(bench_stats)
        self.stats = bench_stats
        return bench_stats

    def __call__(self, function_to_benchmark, *args, **kwargs):
        if self._mode:
            self.has_error = True
            raise FixtureAlreadyUsed(
                "Fixture can only be used once. Previously it was used in %s mode." % self._mode)
        try:
            self._mode = 'benchmark(...)'
            return self._raw(function_to_benchmark, *args, **kwargs)
        except Exception:
            self.has_error = True
            raise

    def pedantic(self, target, args=(), kwargs=None, setup=None, rounds=1, warmup_rounds=0, iterations=1):
        if self._mode:
            self.has_error = True
            raise FixtureAlreadyUsed(
                "Fixture can only be used once. Previously it was used in %s mode." % self._mode)
        try:
            self._mode = 'benchmark.pedantic(...)'
            return self._raw_pedantic(target, args=args, kwargs=kwargs, setup=setup, rounds=rounds,
                                      warmup_rounds=warmup_rounds, iterations=iterations)
        except Exception:
            self.has_error = True
            raise

    def _raw(self, function_to_benchmark, *args, **kwargs):
        if self.enabled:
            runner = self._make_runner(function_to_benchmark, args, kwargs)

            duration, iterations, loops_range = self._calibrate_timer(runner)

            # Choose how many time we must repeat the test
            rounds = int(ceil(self._max_time / duration))
            rounds = max(rounds, self._min_rounds)
            rounds = min(rounds, sys.maxsize)

            stats = self._make_stats(iterations)

            self._logger.debug("  Running %s rounds x %s iterations ..." % (rounds, iterations), yellow=True, bold=True)
            run_start = time.time()
            if self._warmup:
                warmup_rounds = min(rounds, max(1, int(self._warmup / iterations)))
                self._logger.debug("  Warmup %s rounds x %s iterations ..." % (warmup_rounds, iterations))
                for _ in XRANGE(warmup_rounds):
                    runner(loops_range)
            for _ in XRANGE(rounds):
                stats.update(runner(loops_range))
            self._logger.debug("  Ran for %ss." % format_time(time.time() - run_start), yellow=True, bold=True)
        if self.enabled and self.cprofile:
            profile = cProfile.Profile()
            function_result = profile.runcall(function_to_benchmark, *args, **kwargs)
            self.stats.cprofile_stats = pstats.Stats(profile)
        else:
            function_result = function_to_benchmark(*args, **kwargs)
        return function_result

    def _raw_pedantic(self, target, args=(), kwargs=None, setup=None, rounds=1, warmup_rounds=0, iterations=1):
        if kwargs is None:
            kwargs = {}

        has_args = bool(args or kwargs)

        if not isinstance(iterations, INT) or iterations < 1:
            raise ValueError("Must have positive int for `iterations`.")

        if not isinstance(rounds, INT) or rounds < 1:
            raise ValueError("Must have positive int for `rounds`.")

        if not isinstance(warmup_rounds, INT) or warmup_rounds < 0:
            raise ValueError("Must have positive int for `warmup_rounds`.")

        if iterations > 1 and setup:
            raise ValueError("Can't use more than 1 `iterations` with a `setup` function.")

        def make_arguments(args=args, kwargs=kwargs):
            if setup:
                maybe_args = setup()
                if maybe_args:
                    if has_args:
                        raise TypeError("Can't use `args` or `kwargs` if `setup` returns the arguments.")
                    args, kwargs = maybe_args
            return args, kwargs

        if self.disabled:
            args, kwargs = make_arguments()
            return target(*args, **kwargs)

        stats = self._make_stats(iterations)
        loops_range = XRANGE(iterations) if iterations > 1 else None
        for _ in XRANGE(warmup_rounds):
            args, kwargs = make_arguments()

            runner = self._make_runner(target, args, kwargs)
            runner(loops_range)

        for _ in XRANGE(rounds):
            args, kwargs = make_arguments()

            runner = self._make_runner(target, args, kwargs)
            if loops_range:
                duration = runner(loops_range)
            else:
                duration, result = runner(loops_range)
            stats.update(duration)

        if loops_range:
            args, kwargs = make_arguments()
            result = target(*args, **kwargs)

        if self.cprofile:
            profile = cProfile.Profile()
            args, kwargs = make_arguments()
            profile.runcall(target, *args, **kwargs)
            self.stats.cprofile_stats = pstats.Stats(profile)

        return result

    def weave(self, target, **kwargs):
        try:
            import aspectlib
        except ImportError as exc:
            raise ImportError(exc.args, "Please install aspectlib or pytest-benchmark[aspect]")

        def aspect(function):
            def wrapper(*args, **kwargs):
                return self(function, *args, **kwargs)

            return wrapper

        self._cleanup_callbacks.append(aspectlib.weave(target, aspect, **kwargs).rollback)

    patch = weave

    def _cleanup(self):
        while self._cleanup_callbacks:
            callback = self._cleanup_callbacks.pop()
            callback()
        if not self._mode and not self.skipped:
            self._logger.warn("Benchmark fixture was not used at all in this test!",
                              warner=self._warner, suspend=True)

    def _calibrate_timer(self, runner):
        timer_precision = self._get_precision(self._timer)
        min_time = max(self._min_time, timer_precision * self._calibration_precision)
        min_time_estimate = min_time * 5 / self._calibration_precision
        self._logger.debug("")
        self._logger.debug("  Calibrating to target round %ss; will estimate when reaching %ss "
                           "(using: %s, precision: %ss)." % (
                               format_time(min_time),
                               format_time(min_time_estimate),
                               NameWrapper(self._timer),
                               format_time(timer_precision)
                           ), yellow=True, bold=True)

        loops = 1
        while True:
            loops_range = XRANGE(loops)
            duration = runner(loops_range)
            if self._warmup:
                warmup_start = time.time()
                warmup_iterations = 0
                warmup_rounds = 0
                while time.time() - warmup_start < self._max_time and warmup_iterations < self._warmup:
                    duration = min(duration, runner(loops_range))
                    warmup_rounds += 1
                    warmup_iterations += loops
                self._logger.debug("    Warmup: %ss (%s x %s iterations)." % (
                    format_time(time.time() - warmup_start),
                    warmup_rounds, loops
                ))

            self._logger.debug("    Measured %s iterations: %ss." % (loops, format_time(duration)), yellow=True)
            if duration >= min_time:
                break

            if duration >= min_time_estimate:
                # coarse estimation of the number of loops
                loops = int(ceil(min_time * loops / duration))
                self._logger.debug("    Estimating %s iterations." % loops, green=True)
                if loops == 1:
                    # If we got a single loop then bail early - nothing to calibrate if the the
                    # test function is 100 times slower than the timer resolution.
                    loops_range = XRANGE(loops)
                    break
            else:
                loops *= 10
        return duration, loops, loops_range
