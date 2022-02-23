from warnings import catch_warnings

import numpy as np
import pytest

from pandas.core.dtypes import generic as gt

import pandas as pd
import pandas._testing as tm


class TestABCClasses:
    tuples = [[1, 2, 2], ["red", "blue", "red"]]
    multi_index = pd.MultiIndex.from_arrays(tuples, names=("number", "color"))
    datetime_index = pd.to_datetime(["2000/1/1", "2010/1/1"])
    timedelta_index = pd.to_timedelta(np.arange(5), unit="s")
    period_index = pd.period_range("2000/1/1", "2010/1/1/", freq="M")
    categorical = pd.Categorical([1, 2, 3], categories=[2, 3, 1])
    categorical_df = pd.DataFrame({"values": [1, 2, 3]}, index=categorical)
    df = pd.DataFrame({"names": ["a", "b", "c"]}, index=multi_index)
    sparse_array = pd.arrays.SparseArray(np.random.randn(10))
    datetime_array = pd.core.arrays.DatetimeArray(datetime_index)
    timedelta_array = pd.core.arrays.TimedeltaArray(timedelta_index)

    abc_pairs = [
        ("ABCInt64Index", pd.Int64Index([1, 2, 3])),
        ("ABCUInt64Index", pd.UInt64Index([1, 2, 3])),
        ("ABCFloat64Index", pd.Float64Index([1, 2, 3])),
        ("ABCMultiIndex", multi_index),
        ("ABCDatetimeIndex", datetime_index),
        ("ABCRangeIndex", pd.RangeIndex(3)),
        ("ABCTimedeltaIndex", timedelta_index),
        ("ABCIntervalIndex", pd.interval_range(start=0, end=3)),
        ("ABCPeriodArray", pd.arrays.PeriodArray([2000, 2001, 2002], freq="D")),
        ("ABCPandasArray", pd.arrays.PandasArray(np.array([0, 1, 2]))),
        ("ABCPeriodIndex", period_index),
        ("ABCCategoricalIndex", categorical_df.index),
        ("ABCSeries", pd.Series([1, 2, 3])),
        ("ABCDataFrame", df),
        ("ABCCategorical", categorical),
        ("ABCDatetimeArray", datetime_array),
        ("ABCTimedeltaArray", timedelta_array),
    ]

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs(self, abctype1, abctype2, inst, _):
        # GH 38588
        if abctype1 == abctype2:
            assert isinstance(inst, getattr(gt, abctype2))
        else:
            assert not isinstance(inst, getattr(gt, abctype2))

    abc_subclasses = {
        "ABCIndex": [
            abctype
            for abctype, _ in abc_pairs
            if "Index" in abctype and abctype != "ABCIndex"
        ],
        "ABCNDFrame": ["ABCSeries", "ABCDataFrame"],
        "ABCExtensionArray": [
            "ABCCategorical",
            "ABCDatetimeArray",
            "ABCPeriodArray",
            "ABCTimedeltaArray",
        ],
    }

    @pytest.mark.parametrize("parent, subs", abc_subclasses.items())
    @pytest.mark.parametrize("abctype, inst", abc_pairs)
    def test_abc_hierarchy(self, parent, subs, abctype, inst):
        # GH 38588
        if abctype in subs:
            assert isinstance(inst, getattr(gt, parent))
        else:
            assert not isinstance(inst, getattr(gt, parent))

    @pytest.mark.parametrize("abctype", [e for e in gt.__dict__ if e.startswith("ABC")])
    def test_abc_coverage(self, abctype):
        # GH 38588
        assert (
            abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses
        )


def test_setattr_warnings():
    # GH7175 - GOTCHA: You can't use dot notation to add a column...
    d = {
        "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
    }
    df = pd.DataFrame(d)

    with catch_warnings(record=True) as w:
        #  successfully add new column
        #  this should not raise a warning
        df["three"] = df.two + 1
        assert len(w) == 0
        assert df.three.sum() > df.two.sum()

    with catch_warnings(record=True) as w:
        #  successfully modify column in place
        #  this should not raise a warning
        df.one += 1
        assert len(w) == 0
        assert df.one.iloc[0] == 2

    with catch_warnings(record=True) as w:
        #  successfully add an attribute to a series
        #  this should not raise a warning
        df.two.not_an_index = [1, 2]
        assert len(w) == 0

    with tm.assert_produces_warning(UserWarning):
        #  warn when setting column to nonexistent name
        df.four = df.two + 2
        assert df.four.sum() > df.two.sum()
