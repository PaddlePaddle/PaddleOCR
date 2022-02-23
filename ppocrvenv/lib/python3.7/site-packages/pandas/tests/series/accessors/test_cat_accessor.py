import warnings

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties


class TestCatAccessor:
    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.cat.set_categories([1, 2, 3]),
            lambda x: x.cat.reorder_categories([2, 3, 1], ordered=True),
            lambda x: x.cat.rename_categories([1, 2, 3]),
            lambda x: x.cat.remove_unused_categories(),
            lambda x: x.cat.remove_categories([2]),
            lambda x: x.cat.add_categories([4]),
            lambda x: x.cat.as_ordered(),
            lambda x: x.cat.as_unordered(),
        ],
    )
    def test_getname_categorical_accessor(self, method):
        # GH#17509
        ser = Series([1, 2, 3], name="A").astype("category")
        expected = "A"
        result = method(ser).name
        assert result == expected

    def test_cat_accessor(self):
        ser = Series(Categorical(["a", "b", np.nan, "a"]))
        tm.assert_index_equal(ser.cat.categories, Index(["a", "b"]))
        assert not ser.cat.ordered, False

        exp = Categorical(["a", "b", np.nan, "a"], categories=["b", "a"])

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            # issue #37643 inplace kwarg deprecated
            return_value = ser.cat.set_categories(["b", "a"], inplace=True)

        assert return_value is None
        tm.assert_categorical_equal(ser.values, exp)

        res = ser.cat.set_categories(["b", "a"])
        tm.assert_categorical_equal(res.values, exp)

        ser[:] = "a"
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, Index(["a"]))

    def test_cat_accessor_api(self):
        # GH#9322

        assert Series.cat is CategoricalAccessor
        ser = Series(list("aabbcde")).astype("category")
        assert isinstance(ser.cat, CategoricalAccessor)

        invalid = Series([1])
        with pytest.raises(AttributeError, match="only use .cat accessor"):
            invalid.cat
        assert not hasattr(invalid, "cat")

    def test_cat_accessor_no_new_attributes(self):
        # https://github.com/pandas-dev/pandas/issues/10673
        cat = Series(list("aabbcde")).astype("category")
        with pytest.raises(AttributeError, match="You cannot add any new attribute"):
            cat.cat.xlabel = "a"

    def test_cat_accessor_updates_on_inplace(self):
        ser = Series(list("abc")).astype("category")
        return_value = ser.drop(0, inplace=True)
        assert return_value is None

        with tm.assert_produces_warning(FutureWarning, check_stacklevel=False):
            return_value = ser.cat.remove_unused_categories(inplace=True)

        assert return_value is None
        assert len(ser.cat.categories) == 2

    def test_categorical_delegations(self):

        # invalid accessor
        msg = r"Can only use \.cat accessor with a 'category' dtype"
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat
        with pytest.raises(AttributeError, match=msg):
            Series([1, 2, 3]).cat()
        with pytest.raises(AttributeError, match=msg):
            Series(["a", "b", "c"]).cat
        with pytest.raises(AttributeError, match=msg):
            Series(np.arange(5.0)).cat
        with pytest.raises(AttributeError, match=msg):
            Series([Timestamp("20130101")]).cat

        # Series should delegate calls to '.categories', '.codes', '.ordered'
        # and the methods '.set_categories()' 'drop_unused_categories()' to the
        # categorical
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["a", "b", "c"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        ser.cat.categories = [1, 2, 3]
        exp_categories = Index([1, 2, 3])
        tm.assert_index_equal(ser.cat.categories, exp_categories)

        exp_codes = Series([0, 1, 2, 0], dtype="int8")
        tm.assert_series_equal(ser.cat.codes, exp_codes)

        assert ser.cat.ordered
        ser = ser.cat.as_unordered()
        assert not ser.cat.ordered
        return_value = ser.cat.as_ordered(inplace=True)
        assert return_value is None
        assert ser.cat.ordered

        # reorder
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        exp_categories = Index(["c", "b", "a"])
        exp_values = np.array(["a", "b", "c", "a"], dtype=np.object_)
        ser = ser.cat.set_categories(["c", "b", "a"])
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # remove unused categories
        ser = Series(Categorical(["a", "b", "b", "a"], categories=["a", "b", "c"]))
        exp_categories = Index(["a", "b"])
        exp_values = np.array(["a", "b", "b", "a"], dtype=np.object_)
        ser = ser.cat.remove_unused_categories()
        tm.assert_index_equal(ser.cat.categories, exp_categories)
        tm.assert_numpy_array_equal(ser.values.__array__(), exp_values)
        tm.assert_numpy_array_equal(ser.__array__(), exp_values)

        # This method is likely to be confused, so test that it raises an error
        # on wrong inputs:
        msg = "'Series' object has no attribute 'set_categories'"
        with pytest.raises(AttributeError, match=msg):
            ser.set_categories([4, 3, 2, 1])

        # right: ser.cat.set_categories([4,3,2,1])

        # GH#18862 (let Series.cat.rename_categories take callables)
        ser = Series(Categorical(["a", "b", "c", "a"], ordered=True))
        result = ser.cat.rename_categories(lambda x: x.upper())
        expected = Series(
            Categorical(["A", "B", "C", "A"], categories=["A", "B", "C"], ordered=True)
        )
        tm.assert_series_equal(result, expected)

    def test_dt_accessor_api_for_categorical(self):
        # https://github.com/pandas-dev/pandas/issues/10661

        s_dr = Series(date_range("1/1/2015", periods=5, tz="MET"))
        c_dr = s_dr.astype("category")

        s_pr = Series(period_range("1/1/2015", freq="D", periods=5))
        c_pr = s_pr.astype("category")

        s_tdr = Series(timedelta_range("1 days", "10 days"))
        c_tdr = s_tdr.astype("category")

        # only testing field (like .day)
        # and bool (is_month_start)
        get_ops = lambda x: x._datetimelike_ops

        test_data = [
            ("Datetime", get_ops(DatetimeArray), s_dr, c_dr),
            ("Period", get_ops(PeriodArray), s_pr, c_pr),
            ("Timedelta", get_ops(TimedeltaArray), s_tdr, c_tdr),
        ]

        assert isinstance(c_dr.dt, Properties)

        special_func_defs = [
            ("strftime", ("%Y-%m-%d",), {}),
            ("tz_convert", ("EST",), {}),
            ("round", ("D",), {}),
            ("floor", ("D",), {}),
            ("ceil", ("D",), {}),
            ("asfreq", ("D",), {}),
            # FIXME: don't leave commented-out
            # ('tz_localize', ("UTC",), {}),
        ]
        _special_func_names = [f[0] for f in special_func_defs]

        # the series is already localized
        _ignore_names = ["tz_localize", "components"]

        for name, attr_names, s, c in test_data:
            func_names = [
                f
                for f in dir(s.dt)
                if not (
                    f.startswith("_")
                    or f in attr_names
                    or f in _special_func_names
                    or f in _ignore_names
                )
            ]

            func_defs = [(f, (), {}) for f in func_names]
            for f_def in special_func_defs:
                if f_def[0] in dir(s.dt):
                    func_defs.append(f_def)

            for func, args, kwargs in func_defs:
                with warnings.catch_warnings():
                    if func == "to_period":
                        # dropping TZ
                        warnings.simplefilter("ignore", UserWarning)
                    res = getattr(c.dt, func)(*args, **kwargs)
                    exp = getattr(s.dt, func)(*args, **kwargs)

                tm.assert_equal(res, exp)

            for attr in attr_names:
                if attr in ["week", "weekofyear"]:
                    # GH#33595 Deprecate week and weekofyear
                    continue
                res = getattr(c.dt, attr)
                exp = getattr(s.dt, attr)

            if isinstance(res, DataFrame):
                tm.assert_frame_equal(res, exp)
            elif isinstance(res, Series):
                tm.assert_series_equal(res, exp)
            else:
                tm.assert_almost_equal(res, exp)

        invalid = Series([1, 2, 3]).astype("category")
        msg = "Can only use .dt accessor with datetimelike"

        with pytest.raises(AttributeError, match=msg):
            invalid.dt
        assert not hasattr(invalid, "str")

    def test_reorder_categories_updates_dtype(self):
        # GH#43232
        ser = Series(["a", "b", "c"], dtype="category")
        orig_dtype = ser.dtype

        # Need to construct this before calling reorder_categories inplace
        expected = ser.cat.reorder_categories(["c", "b", "a"])

        with tm.assert_produces_warning(FutureWarning, match="`inplace` parameter"):
            ser.cat.reorder_categories(["c", "b", "a"], inplace=True)

        assert not orig_dtype.categories.equals(ser.dtype.categories)
        assert not orig_dtype.categories.equals(expected.dtype.categories)
        assert ser.dtype == expected.dtype
        assert ser.dtype.categories.equals(expected.dtype.categories)

        tm.assert_series_equal(ser, expected)

    def test_set_categories_setitem(self):
        # GH#43334

        df = DataFrame({"Survived": [1, 0, 1], "Sex": [0, 1, 1]}, dtype="category")

        # change the dtype in-place
        df["Survived"].cat.categories = ["No", "Yes"]
        df["Sex"].cat.categories = ["female", "male"]

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]

        df["Sex"] = Categorical(df["Sex"], categories=["female", "male"], ordered=False)
        df["Survived"] = Categorical(
            df["Survived"], categories=["No", "Yes"], ordered=False
        )

        # values should not be coerced to NaN
        assert list(df["Sex"]) == ["female", "male", "male"]
        assert list(df["Survived"]) == ["Yes", "No", "Yes"]
