from copy import deepcopy
from operator import methodcaller

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.tests.generic.test_generic import Generic


class TestDataFrame(Generic):
    _typ = DataFrame
    _comparator = lambda self, x, y: tm.assert_frame_equal(x, y)

    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name(self, func):
        df = DataFrame([[1, 2], [3, 4]])

        result = methodcaller(func, "foo")(df)
        assert df.index.name is None
        assert result.index.name == "foo"

        result = methodcaller(func, "cols", axis=1)(df)
        assert df.columns.name is None
        assert result.columns.name == "cols"

    @pytest.mark.parametrize("func", ["_set_axis_name", "rename_axis"])
    def test_set_axis_name_mi(self, func):
        df = DataFrame(
            np.empty((3, 3)),
            index=MultiIndex.from_tuples([("A", x) for x in list("aBc")]),
            columns=MultiIndex.from_tuples([("C", x) for x in list("xyz")]),
        )

        level_names = ["L1", "L2"]

        result = methodcaller(func, level_names)(df)
        assert result.index.names == level_names
        assert result.columns.names == [None, None]

        result = methodcaller(func, level_names, axis=1)(df)
        assert result.columns.names == ["L1", "L2"]
        assert result.index.names == [None, None]

    def test_nonzero_single_element(self):

        # allow single item via bool method
        df = DataFrame([[True]])
        assert df.bool()

        df = DataFrame([[False]])
        assert not df.bool()

        df = DataFrame([[False, False]])
        msg = "The truth value of a DataFrame is ambiguous"
        with pytest.raises(ValueError, match=msg):
            df.bool()
        with pytest.raises(ValueError, match=msg):
            bool(df)

    def test_metadata_propagation_indiv_groupby(self):
        # groupby
        df = DataFrame(
            {
                "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
                "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "C": np.random.randn(8),
                "D": np.random.randn(8),
            }
        )
        result = df.groupby("A").sum()
        self.check_metadata(df, result)

    def test_metadata_propagation_indiv_resample(self):
        # resample
        df = DataFrame(
            np.random.randn(1000, 2),
            index=date_range("20130101", periods=1000, freq="s"),
        )
        result = df.resample("1T")
        self.check_metadata(df, result)

    def test_metadata_propagation_indiv(self):
        # merging with override
        # GH 6923
        _metadata = DataFrame._metadata
        _finalize = DataFrame.__finalize__

        np.random.seed(10)
        df1 = DataFrame(np.random.randint(0, 4, (3, 2)), columns=["a", "b"])
        df2 = DataFrame(np.random.randint(0, 4, (3, 2)), columns=["c", "d"])
        DataFrame._metadata = ["filename"]
        df1.filename = "fname1.csv"
        df2.filename = "fname2.csv"

        def finalize(self, other, method=None, **kwargs):

            for name in self._metadata:
                if method == "merge":
                    left, right = other.left, other.right
                    value = getattr(left, name, "") + "|" + getattr(right, name, "")
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, ""))

            return self

        DataFrame.__finalize__ = finalize
        result = df1.merge(df2, left_on=["a"], right_on=["c"], how="inner")
        assert result.filename == "fname1.csv|fname2.csv"

        # concat
        # GH 6927
        DataFrame._metadata = ["filename"]
        df1 = DataFrame(np.random.randint(0, 4, (3, 2)), columns=list("ab"))
        df1.filename = "foo"

        def finalize(self, other, method=None, **kwargs):
            for name in self._metadata:
                if method == "concat":
                    value = "+".join(
                        getattr(o, name) for o in other.objs if getattr(o, name, None)
                    )
                    object.__setattr__(self, name, value)
                else:
                    object.__setattr__(self, name, getattr(other, name, None))

            return self

        DataFrame.__finalize__ = finalize

        result = pd.concat([df1, df1])
        assert result.filename == "foo+foo"

        # reset
        DataFrame._metadata = _metadata
        DataFrame.__finalize__ = _finalize  # FIXME: use monkeypatch

    def test_set_attribute(self):
        # Test for consistent setattr behavior when an attribute and a column
        # have the same name (Issue #8994)
        df = DataFrame({"x": [1, 2, 3]})

        df.y = 2
        df["y"] = [2, 4, 6]
        df.y = 5

        assert df.y == 5
        tm.assert_series_equal(df["y"], Series([2, 4, 6], name="y"))

    def test_deepcopy_empty(self):
        # This test covers empty frame copying with non-empty column sets
        # as reported in issue GH15370
        empty_frame = DataFrame(data=[], index=[], columns=["A"])
        empty_frame_copy = deepcopy(empty_frame)

        self._compare(empty_frame_copy, empty_frame)


# formerly in Generic but only test DataFrame
class TestDataFrame2:
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).rename_axis(
                mapper={"a": "x", "b": "y"}, axis=1, inplace=value
            )

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).drop("a", axis=1, inplace=value)

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).fillna(value=0, inplace=value)

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).replace(to_replace=1, value=7, inplace=value)

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).interpolate(inplace=value)

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df)._where(cond=df.a > 2, inplace=value)

        with pytest.raises(ValueError, match=msg):
            super(DataFrame, df).mask(cond=df.a > 2, inplace=value)

    def test_unexpected_keyword(self):
        # GH8597
        df = DataFrame(np.random.randn(5, 2), columns=["jim", "joe"])
        ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
        ts = df["joe"].copy()
        ts[2] = np.nan

        msg = "unexpected keyword"
        with pytest.raises(TypeError, match=msg):
            df.drop("joe", axis=1, in_place=True)

        with pytest.raises(TypeError, match=msg):
            df.reindex([1, 0], inplace=True)

        with pytest.raises(TypeError, match=msg):
            ca.fillna(0, inplace=True)

        with pytest.raises(TypeError, match=msg):
            ts.fillna(0, in_place=True)
