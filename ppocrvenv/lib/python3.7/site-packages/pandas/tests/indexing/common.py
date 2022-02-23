""" common utilities """
import itertools

import numpy as np

from pandas import (
    DataFrame,
    Float64Index,
    MultiIndex,
    Series,
    UInt64Index,
    date_range,
)
import pandas._testing as tm


def _mklbl(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def _axify(obj, key, axis):
    # create a tuple accessor
    axes = [slice(None)] * obj.ndim
    axes[axis] = key
    return tuple(axes)


class Base:
    """indexing comprehensive base class"""

    _kinds = {"series", "frame"}
    _typs = {
        "ints",
        "uints",
        "labels",
        "mixed",
        "ts",
        "floats",
        "empty",
        "ts_rev",
        "multi",
    }

    def setup_method(self, method):

        self.series_ints = Series(np.random.rand(4), index=np.arange(0, 8, 2))
        self.frame_ints = DataFrame(
            np.random.randn(4, 4), index=np.arange(0, 8, 2), columns=np.arange(0, 12, 3)
        )

        self.series_uints = Series(
            np.random.rand(4), index=UInt64Index(np.arange(0, 8, 2))
        )
        self.frame_uints = DataFrame(
            np.random.randn(4, 4),
            index=UInt64Index(range(0, 8, 2)),
            columns=UInt64Index(range(0, 12, 3)),
        )

        self.series_floats = Series(
            np.random.rand(4), index=Float64Index(range(0, 8, 2))
        )
        self.frame_floats = DataFrame(
            np.random.randn(4, 4),
            index=Float64Index(range(0, 8, 2)),
            columns=Float64Index(range(0, 12, 3)),
        )

        m_idces = [
            MultiIndex.from_product([[1, 2], [3, 4]]),
            MultiIndex.from_product([[5, 6], [7, 8]]),
            MultiIndex.from_product([[9, 10], [11, 12]]),
        ]

        self.series_multi = Series(np.random.rand(4), index=m_idces[0])
        self.frame_multi = DataFrame(
            np.random.randn(4, 4), index=m_idces[0], columns=m_idces[1]
        )

        self.series_labels = Series(np.random.randn(4), index=list("abcd"))
        self.frame_labels = DataFrame(
            np.random.randn(4, 4), index=list("abcd"), columns=list("ABCD")
        )

        self.series_mixed = Series(np.random.randn(4), index=[2, 4, "null", 8])
        self.frame_mixed = DataFrame(np.random.randn(4, 4), index=[2, 4, "null", 8])

        self.series_ts = Series(
            np.random.randn(4), index=date_range("20130101", periods=4)
        )
        self.frame_ts = DataFrame(
            np.random.randn(4, 4), index=date_range("20130101", periods=4)
        )

        dates_rev = date_range("20130101", periods=4).sort_values(ascending=False)
        self.series_ts_rev = Series(np.random.randn(4), index=dates_rev)
        self.frame_ts_rev = DataFrame(np.random.randn(4, 4), index=dates_rev)

        self.frame_empty = DataFrame()
        self.series_empty = Series(dtype=object)

        # form agglomerates
        for kind in self._kinds:
            d = {}
            for typ in self._typs:
                d[typ] = getattr(self, f"{kind}_{typ}")

            setattr(self, kind, d)

    def generate_indices(self, f, values=False):
        """
        generate the indices
        if values is True , use the axis values
        is False, use the range
        """
        axes = f.axes
        if values:
            axes = (list(range(len(ax))) for ax in axes)

        return itertools.product(*axes)

    def get_value(self, name, f, i, values=False):
        """return the value for the location i"""
        # check against values
        if values:
            return f.values[i]

        elif name == "iat":
            return f.iloc[i]
        else:
            assert name == "at"
            return f.loc[i]

    def check_values(self, f, func, values=False):

        if f is None:
            return
        axes = f.axes
        indices = itertools.product(*axes)

        for i in indices:
            result = getattr(f, func)[i]

            # check against values
            if values:
                expected = f.values[i]
            else:
                expected = f
                for a in reversed(i):
                    expected = expected.__getitem__(a)

            tm.assert_almost_equal(result, expected)

    def check_result(self, method, key, typs=None, axes=None, fails=None):
        def _eq(axis, obj, key):
            """compare equal for these 2 keys"""
            axified = _axify(obj, key, axis)
            try:
                getattr(obj, method).__getitem__(axified)

            except (IndexError, TypeError, KeyError) as detail:

                # if we are in fails, the ok, otherwise raise it
                if fails is not None:
                    if isinstance(detail, fails):
                        return
                raise

        if typs is None:
            typs = self._typs

        if axes is None:
            axes = [0, 1]
        else:
            assert axes in [0, 1]
            axes = [axes]

        # check
        for kind in self._kinds:

            d = getattr(self, kind)
            for ax in axes:
                for typ in typs:
                    assert typ in self._typs

                    obj = d[typ]
                    if ax < obj.ndim:
                        _eq(axis=ax, obj=obj, key=key)
