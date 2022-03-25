from __future__ import division
from __future__ import print_function

import operator
import statistics
from bisect import bisect_left
from bisect import bisect_right

from .utils import cached_property
from .utils import funcname
from .utils import get_cprofile_functions


class Stats(object):
    fields = (
        "min", "max", "mean", "stddev", "rounds", "median", "iqr", "q1", "q3", "iqr_outliers", "stddev_outliers",
        "outliers", "ld15iqr", "hd15iqr", "ops", "total"
    )

    def __init__(self):
        self.data = []

    def __bool__(self):
        return bool(self.data)

    def __nonzero__(self):
        return bool(self.data)

    def as_dict(self):
        return dict(
            (field, getattr(self, field))
            for field in self.fields
        )

    def update(self, duration):
        self.data.append(duration)

    @cached_property
    def sorted_data(self):
        return sorted(self.data)

    @cached_property
    def total(self):
        return sum(self.data)

    @cached_property
    def min(self):
        return min(self.data)

    @cached_property
    def max(self):
        return max(self.data)

    @cached_property
    def mean(self):
        return statistics.mean(self.data)

    @cached_property
    def stddev(self):
        if len(self.data) > 1:
            return statistics.stdev(self.data)
        else:
            return 0

    @property
    def stddev_outliers(self):
        """
        Count of StdDev outliers: what's beyond (Mean - StdDev, Mean - StdDev)
        """
        count = 0
        q0 = self.mean - self.stddev
        q4 = self.mean + self.stddev
        for val in self.data:
            if val < q0 or val > q4:
                count += 1
        return count

    @cached_property
    def rounds(self):
        return len(self.data)

    @cached_property
    def median(self):
        return statistics.median(self.data)

    @cached_property
    def ld15iqr(self):
        """
        Tukey-style Lowest Datum within 1.5 IQR under Q1.
        """
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.sorted_data[bisect_left(self.sorted_data, self.q1 - 1.5 * self.iqr)]

    @cached_property
    def hd15iqr(self):
        """
        Tukey-style Highest Datum within 1.5 IQR over Q3.
        """
        if len(self.data) == 1:
            return self.data[0]
        else:
            pos = bisect_right(self.sorted_data, self.q3 + 1.5 * self.iqr)
            if pos == len(self.data):
                return self.sorted_data[-1]
            else:
                return self.sorted_data[pos]

    @cached_property
    def q1(self):
        rounds = self.rounds
        data = self.sorted_data

        # See: https://en.wikipedia.org/wiki/Quartile#Computing_methods
        if rounds == 1:
            return data[0]
        elif rounds % 2:  # Method 3
            n, q = rounds // 4, rounds % 4
            if q == 1:
                return 0.25 * data[n - 1] + 0.75 * data[n]
            else:
                return 0.75 * data[n] + 0.25 * data[n + 1]
        else:  # Method 2
            return statistics.median(data[:rounds // 2])

    @cached_property
    def q3(self):
        rounds = self.rounds
        data = self.sorted_data

        # See: https://en.wikipedia.org/wiki/Quartile#Computing_methods
        if rounds == 1:
            return data[0]
        elif rounds % 2:  # Method 3
            n, q = rounds // 4, rounds % 4
            if q == 1:
                return 0.75 * data[3 * n] + 0.25 * data[3 * n + 1]
            else:
                return 0.25 * data[3 * n + 1] + 0.75 * data[3 * n + 2]
        else:  # Method 2
            return statistics.median(data[rounds // 2:])

    @cached_property
    def iqr(self):
        return self.q3 - self.q1

    @property
    def iqr_outliers(self):
        """
        Count of Tukey outliers: what's beyond (Q1 - 1.5IQR, Q3 + 1.5IQR)
        """
        count = 0
        q0 = self.q1 - 1.5 * self.iqr
        q4 = self.q3 + 1.5 * self.iqr
        for val in self.data:
            if val < q0 or val > q4:
                count += 1
        return count

    @cached_property
    def outliers(self):
        return "%s;%s" % (self.stddev_outliers, self.iqr_outliers)

    @cached_property
    def ops(self):
        if self.total:
            return self.rounds / self.total
        return 0


class Metadata(object):
    def __init__(self, fixture, iterations, options):
        self.name = fixture.name
        self.fullname = fixture.fullname
        self.group = fixture.group
        self.param = fixture.param
        self.params = fixture.params
        self.extra_info = fixture.extra_info
        self.cprofile_stats = fixture.cprofile_stats

        self.iterations = iterations
        self.stats = Stats()
        self.options = options
        self.fixture = fixture

    def __bool__(self):
        return bool(self.stats)

    def __nonzero__(self):
        return bool(self.stats)

    def get(self, key, default=None):
        try:
            return getattr(self.stats, key)
        except AttributeError:
            return getattr(self, key, default)

    def __getitem__(self, key):
        try:
            return getattr(self.stats, key)
        except AttributeError:
            return getattr(self, key)

    @property
    def has_error(self):
        return self.fixture.has_error

    def as_dict(self, include_data=True, flat=False, stats=True, cprofile=None):
        result = {
            "group": self.group,
            "name": self.name,
            "fullname": self.fullname,
            "params": self.params,
            "param": self.param,
            "extra_info": self.extra_info,
            "options": dict(
                (k, funcname(v) if callable(v) else v) for k, v in self.options.items()
            )
        }
        if self.cprofile_stats:
            cprofile_list = result["cprofile"] = []
            cprofile_functions = get_cprofile_functions(self.cprofile_stats)
            stats_columns = ["cumtime", "tottime", "ncalls", "ncalls_recursion",
                             "tottime_per", "cumtime_per", "function_name"]
            # move column first
            if cprofile is not None:
                stats_columns.remove(cprofile)
                stats_columns.insert(0, cprofile)
            for column in stats_columns:
                cprofile_functions.sort(key=operator.itemgetter(column), reverse=True)
                for cprofile_function in cprofile_functions[:25]:
                    if cprofile_function not in cprofile_list:
                        cprofile_list.append(cprofile_function)
                # if we want only one column or we already have all available functions
                if cprofile is None or len(cprofile_functions) == len(cprofile_list):
                    break
        if stats:
            stats = self.stats.as_dict()
            if include_data:
                stats["data"] = self.stats.data
            stats["iterations"] = self.iterations
            if flat:
                result.update(stats)
            else:
                result["stats"] = stats
        return result

    def update(self, duration):
        self.stats.update(duration / self.iterations)


def normalize_stats(stats):
    if 'ops' not in stats:
        # fill field added in 3.1.0
        stats['ops'] = 1 / stats['mean']
    return stats
