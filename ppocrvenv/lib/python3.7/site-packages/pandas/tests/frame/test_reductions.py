from datetime import timedelta
from decimal import Decimal
import re

from dateutil.tz import tzlocal
import numpy as np
import pytest

from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_categorical_dtype

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
import pandas.core.algorithms as algorithms
import pandas.core.nanops as nanops


def assert_stat_op_calc(
    opname,
    alternative,
    frame,
    has_skipna=True,
    check_dtype=True,
    check_dates=False,
    rtol=1e-5,
    atol=1e-8,
    skipna_alternative=None,
):
    """
    Check that operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    check_dtype : bool, default True
        Whether the dtypes of the result of "frame.opname()" and
        "alternative(frame)" should be checked.
    check_dates : bool, default false
        Whether opname should be tested on a Datetime Series
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    skipna_alternative : function, default None
        NaN-safe version of alternative
    """
    f = getattr(frame, opname)

    if check_dates:
        expected_warning = FutureWarning if opname in ["mean", "median"] else None
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(expected_warning):
            result = getattr(df, opname)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(expected_warning):
            result = getattr(df, opname)()
        assert isinstance(result, Series)
        assert len(result)

    if has_skipna:

        def wrapper(x):
            return alternative(x.values)

        skipna_wrapper = tm._make_skipna_wrapper(alternative, skipna_alternative)
        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)
        tm.assert_series_equal(
            result0, frame.apply(wrapper), check_dtype=check_dtype, rtol=rtol, atol=atol
        )
        # FIXME: HACK: win32
        tm.assert_series_equal(
            result1,
            frame.apply(wrapper, axis=1),
            check_dtype=False,
            rtol=rtol,
            atol=atol,
        )
    else:
        skipna_wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)
    tm.assert_series_equal(
        result0,
        frame.apply(skipna_wrapper),
        check_dtype=check_dtype,
        rtol=rtol,
        atol=atol,
    )

    if opname in ["sum", "prod"]:
        expected = frame.apply(skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1, expected, check_dtype=False, rtol=rtol, atol=atol
        )

    # check dtypes
    if check_dtype:
        lcd_dtype = frame.values.dtype
        assert lcd_dtype == result0.dtype
        assert lcd_dtype == result1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.NaN
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname in ["sum", "prod"]:
            unit = 1 if opname == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0.index, dtype=r0.dtype)
            tm.assert_series_equal(r0, expected)
            expected = Series(unit, index=r1.index, dtype=r1.dtype)
            tm.assert_series_equal(r1, expected)


def assert_stat_op_api(opname, float_frame, float_string_frame, has_numeric_only=True):
    """
    Check that API for operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    float_frame : DataFrame
        DataFrame with columns of type float
    float_string_frame : DataFrame
        DataFrame with both float and string columns
    has_numeric_only : bool, default False
        Whether the method "opname" has the kwarg "numeric_only"
    """
    # make sure works on mixed-type frame
    getattr(float_string_frame, opname)(axis=0)
    getattr(float_string_frame, opname)(axis=1)

    if has_numeric_only:
        getattr(float_string_frame, opname)(axis=0, numeric_only=True)
        getattr(float_string_frame, opname)(axis=1, numeric_only=True)
        getattr(float_frame, opname)(axis=0, numeric_only=False)
        getattr(float_frame, opname)(axis=1, numeric_only=False)


def assert_bool_op_calc(opname, alternative, frame, has_skipna=True):
    """
    Check that bool operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    alternative : function
        Function that opname is tested against; i.e. "frame.opname()" should
        equal "alternative(frame)".
    frame : DataFrame
        The object that the tests are executed on
    has_skipna : bool, default True
        Whether the method "opname" has the kwarg "skip_na"
    """
    f = getattr(frame, opname)

    if has_skipna:

        def skipna_wrapper(x):
            nona = x.dropna().values
            return alternative(nona)

        def wrapper(x):
            return alternative(x.values)

        result0 = f(axis=0, skipna=False)
        result1 = f(axis=1, skipna=False)

        tm.assert_series_equal(result0, frame.apply(wrapper))
        tm.assert_series_equal(
            result1, frame.apply(wrapper, axis=1), check_dtype=False
        )  # FIXME: HACK: win32
    else:
        skipna_wrapper = alternative
        wrapper = alternative

    result0 = f(axis=0)
    result1 = f(axis=1)

    tm.assert_series_equal(result0, frame.apply(skipna_wrapper))
    tm.assert_series_equal(
        result1, frame.apply(skipna_wrapper, axis=1), check_dtype=False
    )

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        f(axis=2)

    # all NA case
    if has_skipna:
        all_na = frame * np.NaN
        r0 = getattr(all_na, opname)(axis=0)
        r1 = getattr(all_na, opname)(axis=1)
        if opname == "any":
            assert not r0.any()
            assert not r1.any()
        else:
            assert r0.all()
            assert r1.all()


def assert_bool_op_api(
    opname, bool_frame_with_na, float_string_frame, has_bool_only=False
):
    """
    Check that API for boolean operator opname works as advertised on frame

    Parameters
    ----------
    opname : str
        Name of the operator to test on frame
    float_frame : DataFrame
        DataFrame with columns of type float
    float_string_frame : DataFrame
        DataFrame with both float and string columns
    has_bool_only : bool, default False
        Whether the method "opname" has the kwarg "bool_only"
    """
    # make sure op works on mixed-type frame
    mixed = float_string_frame
    mixed["_bool_"] = np.random.randn(len(mixed)) > 0.5

    getattr(mixed, opname)(axis=0)
    getattr(mixed, opname)(axis=1)

    if has_bool_only:
        getattr(mixed, opname)(axis=0, bool_only=True)
        getattr(mixed, opname)(axis=1, bool_only=True)
        getattr(bool_frame_with_na, opname)(axis=0, bool_only=False)
        getattr(bool_frame_with_na, opname)(axis=1, bool_only=False)


class TestDataFrameAnalytics:

    # ---------------------------------------------------------------------
    # Reductions

    @pytest.mark.filterwarnings("ignore:Dropping of nuisance:FutureWarning")
    def test_stat_op_api(self, float_frame, float_string_frame):
        assert_stat_op_api("count", float_frame, float_string_frame)
        assert_stat_op_api("sum", float_frame, float_string_frame)

        assert_stat_op_api(
            "nunique", float_frame, float_string_frame, has_numeric_only=False
        )
        assert_stat_op_api("mean", float_frame, float_string_frame)
        assert_stat_op_api("product", float_frame, float_string_frame)
        assert_stat_op_api("median", float_frame, float_string_frame)
        assert_stat_op_api("min", float_frame, float_string_frame)
        assert_stat_op_api("max", float_frame, float_string_frame)
        assert_stat_op_api(
            "mad", float_frame, float_string_frame, has_numeric_only=False
        )
        assert_stat_op_api("var", float_frame, float_string_frame)
        assert_stat_op_api("std", float_frame, float_string_frame)
        assert_stat_op_api("sem", float_frame, float_string_frame)
        assert_stat_op_api("median", float_frame, float_string_frame)

        try:
            from scipy.stats import (  # noqa:F401
                kurtosis,
                skew,
            )

            assert_stat_op_api("skew", float_frame, float_string_frame)
            assert_stat_op_api("kurt", float_frame, float_string_frame)
        except ImportError:
            pass

    def test_stat_op_calc(self, float_frame_with_na, mixed_float_frame):
        def count(s):
            return notna(s).sum()

        def nunique(s):
            return len(algorithms.unique1d(s.dropna()))

        def mad(x):
            return np.abs(x - x.mean()).mean()

        def var(x):
            return np.var(x, ddof=1)

        def std(x):
            return np.std(x, ddof=1)

        def sem(x):
            return np.std(x, ddof=1) / np.sqrt(len(x))

        def skewness(x):
            from scipy.stats import skew  # noqa:F811

            if len(x) < 3:
                return np.nan
            return skew(x, bias=False)

        def kurt(x):
            from scipy.stats import kurtosis  # noqa:F811

            if len(x) < 4:
                return np.nan
            return kurtosis(x, bias=False)

        assert_stat_op_calc(
            "nunique",
            nunique,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

        # GH#32571 check_less_precise is needed on apparently-random
        #  py37-npdev builds and OSX-PY36-min_version builds
        # mixed types (with upcasting happening)
        assert_stat_op_calc(
            "sum",
            np.sum,
            mixed_float_frame.astype("float32"),
            check_dtype=False,
            rtol=1e-3,
        )

        assert_stat_op_calc(
            "sum", np.sum, float_frame_with_na, skipna_alternative=np.nansum
        )
        assert_stat_op_calc("mean", np.mean, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "product", np.prod, float_frame_with_na, skipna_alternative=np.nanprod
        )

        assert_stat_op_calc("mad", mad, float_frame_with_na)
        assert_stat_op_calc("var", var, float_frame_with_na)
        assert_stat_op_calc("std", std, float_frame_with_na)
        assert_stat_op_calc("sem", sem, float_frame_with_na)

        assert_stat_op_calc(
            "count",
            count,
            float_frame_with_na,
            has_skipna=False,
            check_dtype=False,
            check_dates=True,
        )

        try:
            from scipy import (  # noqa:F401
                kurtosis,
                skew,
            )

            assert_stat_op_calc("skew", skewness, float_frame_with_na)
            assert_stat_op_calc("kurt", kurt, float_frame_with_na)
        except ImportError:
            pass

    # TODO: Ensure warning isn't emitted in the first place
    # ignore mean of empty slice and all-NaN
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_median(self, float_frame_with_na, int_frame):
        def wrapper(x):
            if isna(x).any():
                return np.nan
            return np.median(x)

        assert_stat_op_calc("median", wrapper, float_frame_with_na, check_dates=True)
        assert_stat_op_calc(
            "median", wrapper, int_frame, check_dtype=False, check_dates=True
        )

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "prod", "var", "std", "skew", "min", "max"]
    )
    def test_stat_operators_attempt_obj_array(self, method):
        # GH#676
        data = {
            "a": [
                -0.00049987540199591344,
                -0.0016467257772919831,
                0.00067695870775883013,
            ],
            "b": [-0, -0, 0.0],
            "c": [
                0.00031111847529610595,
                0.0014902627951905339,
                -0.00094099200035979691,
            ],
        }
        df1 = DataFrame(data, index=["foo", "bar", "baz"], dtype="O")

        df2 = DataFrame({0: [np.nan, 2], 1: [np.nan, 3], 2: [np.nan, 4]}, dtype=object)

        for df in [df1, df2]:
            assert df.values.dtype == np.object_
            result = getattr(df, method)(1)
            expected = getattr(df.astype("f8"), method)(1)

            if method in ["sum", "prod"]:
                tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", ["mean", "std", "var", "skew", "kurt", "sem"])
    def test_mixed_ops(self, op):
        # GH#16116
        df = DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.0, 2.0, 3.0, 4.0],
                "str": ["a", "b", "c", "d"],
            }
        )
        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = getattr(df, op)()
        assert len(result) == 2

        with pd.option_context("use_bottleneck", False):
            with tm.assert_produces_warning(
                FutureWarning, match="Select only valid columns"
            ):
                result = getattr(df, op)()
            assert len(result) == 2

    def test_reduce_mixed_frame(self):
        # GH 6806
        df = DataFrame(
            {
                "bool_data": [True, True, False, False, False],
                "int_data": [10, 20, 30, 40, 50],
                "string_data": ["a", "b", "c", "d", "e"],
            }
        )
        df.reindex(columns=["bool_data", "int_data", "string_data"])
        test = df.sum(axis=0)
        tm.assert_numpy_array_equal(
            test.values, np.array([2, 150, "abcde"], dtype=object)
        )
        alt = df.T.sum(axis=1)
        tm.assert_series_equal(test, alt)

    def test_nunique(self):
        df = DataFrame({"A": [1, 1, 1], "B": [1, 2, 3], "C": [1, np.nan, 3]})
        tm.assert_series_equal(df.nunique(), Series({"A": 1, "B": 3, "C": 2}))
        tm.assert_series_equal(
            df.nunique(dropna=False), Series({"A": 1, "B": 3, "C": 3})
        )
        tm.assert_series_equal(df.nunique(axis=1), Series({0: 1, 1: 2, 2: 2}))
        tm.assert_series_equal(
            df.nunique(axis=1, dropna=False), Series({0: 1, 1: 3, 2: 2})
        )

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_mixed_datetime_numeric(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        df = DataFrame({"A": [1, 1], "B": [Timestamp("2000", tz=tz)] * 2})
        with tm.assert_produces_warning(FutureWarning):
            result = df.mean()
        expected = Series([1.0], index=["A"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_mean_excludes_datetimes(self, tz):
        # https://github.com/pandas-dev/pandas/issues/24752
        # Our long-term desired behavior is unclear, but the behavior in
        # 0.24.0rc1 was buggy.
        df = DataFrame({"A": [Timestamp("2000", tz=tz)] * 2})
        with tm.assert_produces_warning(FutureWarning):
            result = df.mean()

        expected = Series(dtype=np.float64)
        tm.assert_series_equal(result, expected)

    def test_mean_mixed_string_decimal(self):
        # GH 11670
        # possible bug when calculating mean of DataFrame?

        d = [
            {"A": 2, "B": None, "C": Decimal("628.00")},
            {"A": 1, "B": None, "C": Decimal("383.00")},
            {"A": 3, "B": None, "C": Decimal("651.00")},
            {"A": 2, "B": None, "C": Decimal("575.00")},
            {"A": 4, "B": None, "C": Decimal("1114.00")},
            {"A": 1, "B": "TEST", "C": Decimal("241.00")},
            {"A": 2, "B": None, "C": Decimal("572.00")},
            {"A": 4, "B": None, "C": Decimal("609.00")},
            {"A": 3, "B": None, "C": Decimal("820.00")},
            {"A": 5, "B": None, "C": Decimal("1223.00")},
        ]

        df = DataFrame(d)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = df.mean()
        expected = Series([2.7, 681.6], index=["A", "C"])
        tm.assert_series_equal(result, expected)

    def test_var_std(self, datetime_frame):
        result = datetime_frame.std(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4))
        tm.assert_almost_equal(result, expected)

        result = datetime_frame.var(ddof=4)
        expected = datetime_frame.apply(lambda x: x.var(ddof=4))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.random((1, 1000)), 1000, 0)
        result = nanops.nanvar(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nanvar(arr, axis=0)
            assert not (result < 0).any()

    @pytest.mark.parametrize("meth", ["sem", "var", "std"])
    def test_numeric_only_flag(self, meth):
        # GH 9201
        df1 = DataFrame(np.random.randn(5, 3), columns=["foo", "bar", "baz"])
        # set one entry to a number in str format
        df1.loc[0, "foo"] = "100"

        df2 = DataFrame(np.random.randn(5, 3), columns=["foo", "bar", "baz"])
        # set one entry to a non-number str
        df2.loc[0, "foo"] = "a"

        result = getattr(df1, meth)(axis=1, numeric_only=True)
        expected = getattr(df1[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        result = getattr(df2, meth)(axis=1, numeric_only=True)
        expected = getattr(df2[["bar", "baz"]], meth)(axis=1)
        tm.assert_series_equal(expected, result)

        # df1 has all numbers, df2 has a letter inside
        msg = r"unsupported operand type\(s\) for -: 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            getattr(df1, meth)(axis=1, numeric_only=False)
        msg = "could not convert string to float: 'a'"
        with pytest.raises(TypeError, match=msg):
            getattr(df2, meth)(axis=1, numeric_only=False)

    def test_sem(self, datetime_frame):
        result = datetime_frame.sem(ddof=4)
        expected = datetime_frame.apply(lambda x: x.std(ddof=4) / np.sqrt(len(x)))
        tm.assert_almost_equal(result, expected)

        arr = np.repeat(np.random.random((1, 1000)), 1000, 0)
        result = nanops.nansem(arr, axis=0)
        assert not (result < 0).any()

        with pd.option_context("use_bottleneck", False):
            result = nanops.nansem(arr, axis=0)
            assert not (result < 0).any()

    @td.skip_if_no_scipy
    def test_kurt(self):
        index = MultiIndex(
            levels=[["bar"], ["one", "two", "three"], [0, 1]],
            codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
        )
        df = DataFrame(np.random.randn(6, 3), index=index)

        kurt = df.kurt()
        with tm.assert_produces_warning(FutureWarning):
            kurt2 = df.kurt(level=0).xs("bar")
        tm.assert_series_equal(kurt, kurt2, check_names=False)
        assert kurt.name is None
        assert kurt2.name == "bar"

    @pytest.mark.parametrize(
        "dropna, expected",
        [
            (
                True,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [1.0],
                    "D": ["a"],
                    "E": Categorical(["a"], categories=["a"]),
                    "F": to_datetime(["2000-1-2"]),
                    "G": to_timedelta(["1 days"]),
                },
            ),
            (
                False,
                {
                    "A": [12],
                    "B": [10.0],
                    "C": [np.nan],
                    "D": np.array([np.nan], dtype=object),
                    "E": Categorical([np.nan], categories=["a"]),
                    "F": [pd.NaT],
                    "G": to_timedelta([pd.NaT]),
                },
            ),
            (
                True,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical(["a", np.nan, np.nan, np.nan], categories=["a"]),
                    "L": to_datetime(["2000-1-2", "NaT", "NaT", "NaT"]),
                    "M": to_timedelta(["1 days", "nan", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
            (
                False,
                {
                    "H": [8, 9, np.nan, np.nan],
                    "I": [8, 9, np.nan, np.nan],
                    "J": [1, np.nan, np.nan, np.nan],
                    "K": Categorical([np.nan, "a", np.nan, np.nan], categories=["a"]),
                    "L": to_datetime(["NaT", "2000-1-2", "NaT", "NaT"]),
                    "M": to_timedelta(["nan", "1 days", "nan", "nan"]),
                    "N": [0, 1, 2, 3],
                },
            ),
        ],
    )
    def test_mode_dropna(self, dropna, expected):

        df = DataFrame(
            {
                "A": [12, 12, 19, 11],
                "B": [10, 10, np.nan, 3],
                "C": [1, np.nan, np.nan, np.nan],
                "D": [np.nan, np.nan, "a", np.nan],
                "E": Categorical([np.nan, np.nan, "a", np.nan]),
                "F": to_datetime(["NaT", "2000-1-2", "NaT", "NaT"]),
                "G": to_timedelta(["1 days", "nan", "nan", "nan"]),
                "H": [8, 8, 9, 9],
                "I": [9, 9, 8, 8],
                "J": [1, 1, np.nan, np.nan],
                "K": Categorical(["a", np.nan, "a", np.nan]),
                "L": to_datetime(["2000-1-2", "2000-1-2", "NaT", "NaT"]),
                "M": to_timedelta(["1 days", "nan", "1 days", "nan"]),
                "N": np.arange(4, dtype="int64"),
            }
        )

        result = df[sorted(expected.keys())].mode(dropna=dropna)
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)

    def test_mode_sortwarning(self):
        # Check for the warning that is raised when the mode
        # results cannot be sorted

        df = DataFrame({"A": [np.nan, np.nan, "a", "a"]})
        expected = DataFrame({"A": ["a", np.nan]})

        with tm.assert_produces_warning(UserWarning):
            result = df.mode(dropna=False)
            result = result.sort_values(by="A").reset_index(drop=True)

        tm.assert_frame_equal(result, expected)

    def test_mode_empty_df(self):
        df = DataFrame([], columns=["a", "b"])
        result = df.mode()
        expected = DataFrame([], columns=["a", "b"], index=Index([], dtype=int))
        tm.assert_frame_equal(result, expected)

    def test_operators_timedelta64(self):
        df = DataFrame(
            {
                "A": date_range("2012-1-1", periods=3, freq="D"),
                "B": date_range("2012-1-2", periods=3, freq="D"),
                "C": Timestamp("20120101") - timedelta(minutes=5, seconds=5),
            }
        )

        diffs = DataFrame({"A": df["A"] - df["C"], "B": df["A"] - df["B"]})

        # min
        result = diffs.min()
        assert result[0] == diffs.loc[0, "A"]
        assert result[1] == diffs.loc[0, "B"]

        result = diffs.min(axis=1)
        assert (result == diffs.loc[0, "B"]).all()

        # max
        result = diffs.max()
        assert result[0] == diffs.loc[2, "A"]
        assert result[1] == diffs.loc[2, "B"]

        result = diffs.max(axis=1)
        assert (result == diffs["A"]).all()

        # abs
        result = diffs.abs()
        result2 = abs(diffs)
        expected = DataFrame({"A": df["A"] - df["C"], "B": df["B"] - df["A"]})
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # mixed frame
        mixed = diffs.copy()
        mixed["C"] = "foo"
        mixed["D"] = 1
        mixed["E"] = 1.0
        mixed["F"] = Timestamp("20130101")

        # results in an object array
        result = mixed.min()
        expected = Series(
            [
                pd.Timedelta(timedelta(seconds=5 * 60 + 5)),
                pd.Timedelta(timedelta(days=-1)),
                "foo",
                1,
                1.0,
                Timestamp("20130101"),
            ],
            index=mixed.columns,
        )
        tm.assert_series_equal(result, expected)

        # excludes numeric
        with tm.assert_produces_warning(FutureWarning, match="Select only valid"):
            result = mixed.min(axis=1)
        expected = Series([1, 1, 1.0], index=[0, 1, 2])
        tm.assert_series_equal(result, expected)

        # works when only those columns are selected
        result = mixed[["A", "B"]].min(1)
        expected = Series([timedelta(days=-1)] * 3)
        tm.assert_series_equal(result, expected)

        result = mixed[["A", "B"]].min()
        expected = Series(
            [timedelta(seconds=5 * 60 + 5), timedelta(days=-1)], index=["A", "B"]
        )
        tm.assert_series_equal(result, expected)

        # GH 3106
        df = DataFrame(
            {
                "time": date_range("20130102", periods=5),
                "time2": date_range("20130105", periods=5),
            }
        )
        df["off1"] = df["time2"] - df["time"]
        assert df["off1"].dtype == "timedelta64[ns]"

        df["off2"] = df["time"] - df["time2"]
        df._consolidate_inplace()
        assert df["off1"].dtype == "timedelta64[ns]"
        assert df["off2"].dtype == "timedelta64[ns]"

    def test_std_timedelta64_skipna_false(self):
        # GH#37392
        tdi = pd.timedelta_range("1 Day", periods=10)
        df = DataFrame({"A": tdi, "B": tdi})
        df.iloc[-2, -1] = pd.NaT

        result = df.std(skipna=False)
        expected = Series(
            [df["A"].std(), pd.NaT], index=["A", "B"], dtype="timedelta64[ns]"
        )
        tm.assert_series_equal(result, expected)

        result = df.std(axis=1, skipna=False)
        expected = Series([pd.Timedelta(0)] * 8 + [pd.NaT, pd.Timedelta(0)])
        tm.assert_series_equal(result, expected)

    def test_sum_corner(self):
        empty_frame = DataFrame()

        axis0 = empty_frame.sum(0)
        axis1 = empty_frame.sum(1)
        assert isinstance(axis0, Series)
        assert isinstance(axis1, Series)
        assert len(axis0) == 0
        assert len(axis1) == 0

    @pytest.mark.parametrize("method, unit", [("sum", 0), ("prod", 1)])
    @pytest.mark.parametrize("numeric_only", [None, True, False])
    def test_sum_prod_nanops(self, method, unit, numeric_only):
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [unit, unit], "b": [unit, np.nan], "c": [np.nan, np.nan]})
        # The default
        result = getattr(df, method)(numeric_only=numeric_only)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = getattr(df, method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, unit, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = getattr(df, method)(numeric_only=numeric_only, min_count=0)
        expected = Series([unit, unit, unit], index=idx, dtype="float64")
        tm.assert_series_equal(result, expected)

        result = getattr(df.iloc[1:], method)(numeric_only=numeric_only, min_count=1)
        expected = Series([unit, np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

        # min_count > 1
        df = DataFrame({"A": [unit] * 10, "B": [unit] * 5 + [np.nan] * 5})
        result = getattr(df, method)(numeric_only=numeric_only, min_count=5)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

        result = getattr(df, method)(numeric_only=numeric_only, min_count=6)
        expected = Series(result, index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_timedelta(self):
        # prod isn't defined on timedeltas
        idx = ["a", "b", "c"]
        df = DataFrame({"a": [0, 0], "b": [0, np.nan], "c": [np.nan, np.nan]})

        df2 = df.apply(to_timedelta)

        # 0 by default
        result = df2.sum()
        expected = Series([0, 0, 0], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

        # min_count=0
        result = df2.sum(min_count=0)
        tm.assert_series_equal(result, expected)

        # min_count=1
        result = df2.sum(min_count=1)
        expected = Series([0, 0, np.nan], dtype="m8[ns]", index=idx)
        tm.assert_series_equal(result, expected)

    def test_sum_nanops_min_count(self):
        # https://github.com/pandas-dev/pandas/issues/39738
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = df.sum(min_count=10)
        expected = Series([np.nan, np.nan], index=["x", "y"])
        tm.assert_series_equal(result, expected)

    def test_sum_object(self, float_frame):
        values = float_frame.values.astype(int)
        frame = DataFrame(values, index=float_frame.index, columns=float_frame.columns)
        deltas = frame * timedelta(1)
        deltas.sum()

    def test_sum_bool(self, float_frame):
        # ensure this works, bug report
        bools = np.isnan(float_frame)
        bools.sum(1)
        bools.sum(0)

    def test_sum_mixed_datetime(self):
        # GH#30886
        df = DataFrame({"A": date_range("2000", periods=4), "B": [1, 2, 3, 4]}).reindex(
            [2, 3, 4]
        )
        with tm.assert_produces_warning(FutureWarning, match="Select only valid"):
            result = df.sum()

        expected = Series({"B": 7.0})
        tm.assert_series_equal(result, expected)

    def test_mean_corner(self, float_frame, float_string_frame):
        # unit test when have object data
        with tm.assert_produces_warning(FutureWarning, match="Select only valid"):
            the_mean = float_string_frame.mean(axis=0)
        the_sum = float_string_frame.sum(axis=0, numeric_only=True)
        tm.assert_index_equal(the_sum.index, the_mean.index)
        assert len(the_mean.index) < len(float_string_frame.columns)

        # xs sum mixed type, just want to know it works...
        with tm.assert_produces_warning(FutureWarning, match="Select only valid"):
            the_mean = float_string_frame.mean(axis=1)
        the_sum = float_string_frame.sum(axis=1, numeric_only=True)
        tm.assert_index_equal(the_sum.index, the_mean.index)

        # take mean of boolean column
        float_frame["bool"] = float_frame["A"] > 0
        means = float_frame.mean(0)
        assert means["bool"] == float_frame["bool"].values.mean()

    def test_mean_datetimelike(self):
        # GH#24757 check that datetimelike are excluded by default, handled
        #  correctly with numeric_only=True

        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
                "D": pd.period_range("2016", periods=3, freq="A"),
            }
        )
        result = df.mean(numeric_only=True)
        expected = Series({"A": 1.0})
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(FutureWarning):
            # in the future datetime columns will be included
            result = df.mean()
        expected = Series({"A": 1.0, "C": df.loc[1, "C"]})
        tm.assert_series_equal(result, expected)

    def test_mean_datetimelike_numeric_only_false(self):
        df = DataFrame(
            {
                "A": np.arange(3),
                "B": date_range("2016-01-01", periods=3),
                "C": pd.timedelta_range("1D", periods=3),
            }
        )

        # datetime(tz) and timedelta work
        result = df.mean(numeric_only=False)
        expected = Series({"A": 1, "B": df.loc[1, "B"], "C": df.loc[1, "C"]})
        tm.assert_series_equal(result, expected)

        # mean of period is not allowed
        df["D"] = pd.period_range("2016", periods=3, freq="A")

        with pytest.raises(TypeError, match="mean is not implemented for Period"):
            df.mean(numeric_only=False)

    def test_mean_extensionarray_numeric_only_true(self):
        # https://github.com/pandas-dev/pandas/issues/33256
        arr = np.random.randint(1000, size=(10, 5))
        df = DataFrame(arr, dtype="Int64")
        result = df.mean(numeric_only=True)
        expected = DataFrame(arr).mean()
        tm.assert_series_equal(result, expected)

    def test_stats_mixed_type(self, float_string_frame):
        # don't blow up
        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            float_string_frame.std(1)
            float_string_frame.var(1)
            float_string_frame.mean(1)
            float_string_frame.skew(1)

    def test_sum_bools(self):
        df = DataFrame(index=range(1), columns=range(10))
        bools = isna(df)
        assert bools.sum(axis=1)[0] == 10

    # ----------------------------------------------------------------------
    # Index of max / min

    def test_idxmin(self, float_frame, int_frame):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for skipna in [True, False]:
            for axis in [0, 1]:
                for df in [frame, int_frame]:
                    result = df.idxmin(axis=axis, skipna=skipna)
                    expected = df.apply(Series.idxmin, axis=axis, skipna=skipna)
                    tm.assert_series_equal(result, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmin(axis=2)

    def test_idxmax(self, float_frame, int_frame):
        frame = float_frame
        frame.iloc[5:10] = np.nan
        frame.iloc[15:20, -2:] = np.nan
        for skipna in [True, False]:
            for axis in [0, 1]:
                for df in [frame, int_frame]:
                    result = df.idxmax(axis=axis, skipna=skipna)
                    expected = df.apply(Series.idxmax, axis=axis, skipna=skipna)
                    tm.assert_series_equal(result, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.idxmax(axis=2)

    def test_idxmax_mixed_dtype(self):
        # don't cast to object, which would raise in nanops
        dti = date_range("2016-01-01", periods=3)

        df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti})

        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 0], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        # with NaTs
        df.loc[0, 3] = pd.NaT
        result = df.idxmax()
        expected = Series([1, 0, 2], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1], index=[1, 2, 3])
        tm.assert_series_equal(result, expected)

        # with multi-column dt64 block
        df[4] = dti[::-1]
        df._consolidate_inplace()

        result = df.idxmax()
        expected = Series([1, 0, 2, 0], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

        result = df.idxmin()
        expected = Series([0, 2, 1, 2], index=[1, 2, 3, 4])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "op, expected_value",
        [("idxmax", [0, 4]), ("idxmin", [0, 5])],
    )
    def test_idxmax_idxmin_convert_dtypes(self, op, expected_value):
        # GH 40346
        df = DataFrame(
            {
                "ID": [100, 100, 100, 200, 200, 200],
                "value": [0, 0, 0, 1, 2, 0],
            },
            dtype="Int64",
        )
        df = df.groupby("ID")

        result = getattr(df, op)()
        expected = DataFrame(
            {"value": expected_value},
            index=Index([100, 200], dtype="object", name="ID"),
        )
        tm.assert_frame_equal(result, expected)

    def test_idxmax_dt64_multicolumn_axis1(self):
        dti = date_range("2016-01-01", periods=3)
        df = DataFrame({3: dti, 4: dti[::-1]})
        df.iloc[0, 0] = pd.NaT

        df._consolidate_inplace()

        result = df.idxmax(axis=1)
        expected = Series([4, 3, 3])
        tm.assert_series_equal(result, expected)

        result = df.idxmin(axis=1)
        expected = Series([4, 3, 4])
        tm.assert_series_equal(result, expected)

    # ----------------------------------------------------------------------
    # Logical reductions

    @pytest.mark.parametrize("opname", ["any", "all"])
    def test_any_all(self, opname, bool_frame_with_na, float_string_frame):
        assert_bool_op_api(
            opname, bool_frame_with_na, float_string_frame, has_bool_only=True
        )

    @pytest.mark.parametrize("opname", ["any", "all"])
    def test_any_all_bool_frame(self, opname, bool_frame_with_na):
        # GH#12863: numpy gives back non-boolean data for object type
        # so fill NaNs to compare with pandas behavior
        df = bool_frame_with_na.fillna(True)
        assert_bool_op_calc(opname, getattr(np, opname), df, has_skipna=True)

    def test_any_all_extra(self):
        df = DataFrame(
            {
                "A": [True, False, False],
                "B": [True, True, False],
                "C": [True, True, True],
            },
            index=["a", "b", "c"],
        )
        result = df[["A", "B"]].any(1)
        expected = Series([True, True, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df[["A", "B"]].any(1, bool_only=True)
        tm.assert_series_equal(result, expected)

        result = df.all(1)
        expected = Series([True, False, False], index=["a", "b", "c"])
        tm.assert_series_equal(result, expected)

        result = df.all(1, bool_only=True)
        tm.assert_series_equal(result, expected)

        # Axis is None
        result = df.all(axis=None).item()
        assert result is False

        result = df.any(axis=None).item()
        assert result is True

        result = df[["C"]].all(axis=None).item()
        assert result is True

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("bool_agg_func", ["any", "all"])
    @pytest.mark.parametrize("skipna", [True, False])
    def test_any_all_object_dtype(self, axis, bool_agg_func, skipna):
        # GH#35450
        df = DataFrame(
            data=[
                [1, np.nan, np.nan, True],
                [np.nan, 2, np.nan, True],
                [np.nan, np.nan, np.nan, True],
                [np.nan, np.nan, "5", np.nan],
            ]
        )
        result = getattr(df, bool_agg_func)(axis=axis, skipna=skipna)
        expected = Series([True, True, True, True])
        tm.assert_series_equal(result, expected)

    def test_any_datetime(self):

        # GH 23070
        float_data = [1, np.nan, 3, np.nan]
        datetime_data = [
            Timestamp("1960-02-15"),
            Timestamp("1960-02-16"),
            pd.NaT,
            pd.NaT,
        ]
        df = DataFrame({"A": float_data, "B": datetime_data})

        result = df.any(1)
        expected = Series([True, True, True, False])
        tm.assert_series_equal(result, expected)

    def test_any_all_bool_only(self):

        # GH 25101
        df = DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [None, None, None]}
        )

        result = df.all(bool_only=True)
        expected = Series(dtype=np.bool_)
        tm.assert_series_equal(result, expected)

        df = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [None, None, None],
                "col4": [False, False, True],
            }
        )

        result = df.all(bool_only=True)
        expected = Series({"col4": False})
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func, data, expected",
        [
            (np.any, {}, False),
            (np.all, {}, True),
            (np.any, {"A": []}, False),
            (np.all, {"A": []}, True),
            (np.any, {"A": [False, False]}, False),
            (np.all, {"A": [False, False]}, False),
            (np.any, {"A": [True, False]}, True),
            (np.all, {"A": [True, False]}, False),
            (np.any, {"A": [True, True]}, True),
            (np.all, {"A": [True, True]}, True),
            (np.any, {"A": [False], "B": [False]}, False),
            (np.all, {"A": [False], "B": [False]}, False),
            (np.any, {"A": [False, False], "B": [False, True]}, True),
            (np.all, {"A": [False, False], "B": [False, True]}, False),
            # other types
            (np.all, {"A": Series([0.0, 1.0], dtype="float")}, False),
            (np.any, {"A": Series([0.0, 1.0], dtype="float")}, True),
            (np.all, {"A": Series([0, 1], dtype=int)}, False),
            (np.any, {"A": Series([0, 1], dtype=int)}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns]")}, False),
            pytest.param(np.all, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([0, 1], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="M8[ns, UTC]")}, True),
            pytest.param(np.all, {"A": Series([0, 1], dtype="m8[ns]")}, False),
            pytest.param(np.any, {"A": Series([0, 1], dtype="m8[ns]")}, True),
            pytest.param(np.all, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            pytest.param(np.any, {"A": Series([1, 2], dtype="m8[ns]")}, True),
            # np.all on Categorical raises, so the reduction drops the
            #  column, so all is being done on an empty Series, so is True
            (np.all, {"A": Series([0, 1], dtype="category")}, True),
            (np.any, {"A": Series([0, 1], dtype="category")}, False),
            (np.all, {"A": Series([1, 2], dtype="category")}, True),
            (np.any, {"A": Series([1, 2], dtype="category")}, False),
            # Mix GH#21484
            pytest.param(
                np.all,
                {
                    "A": Series([10, 20], dtype="M8[ns]"),
                    "B": Series([10, 20], dtype="m8[ns]"),
                },
                True,
            ),
        ],
    )
    def test_any_all_np_func(self, func, data, expected):
        # GH 19976
        data = DataFrame(data)

        warn = None
        if any(is_categorical_dtype(x) for x in data.dtypes):
            warn = FutureWarning

        with tm.assert_produces_warning(
            warn, match="Select only valid columns", check_stacklevel=False
        ):
            result = func(data)
        assert isinstance(result, np.bool_)
        assert result.item() is expected

        # method version
        with tm.assert_produces_warning(
            warn, match="Select only valid columns", check_stacklevel=False
        ):
            result = getattr(DataFrame(data), func.__name__)(axis=None)
        assert isinstance(result, np.bool_)
        assert result.item() is expected

    def test_any_all_object(self):
        # GH 19976
        result = np.all(DataFrame(columns=["a", "b"])).item()
        assert result is True

        result = np.any(DataFrame(columns=["a", "b"])).item()
        assert result is False

    def test_any_all_object_bool_only(self):
        df = DataFrame({"A": ["foo", 2], "B": [True, False]}).astype(object)
        df._consolidate_inplace()
        df["C"] = Series([True, True])

        # Categorical of bools is _not_ considered booly
        df["D"] = df["C"].astype("category")

        # The underlying bug is in DataFrame._get_bool_data, so we check
        #  that while we're here
        res = df._get_bool_data()
        expected = df[["B", "C"]]
        tm.assert_frame_equal(res, expected)

        res = df.all(bool_only=True, axis=0)
        expected = Series([False, True], index=["B", "C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["B", "C"]].all(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert not df.all(bool_only=True, axis=None)

        res = df.any(bool_only=True, axis=0)
        expected = Series([True, True], index=["B", "C"])
        tm.assert_series_equal(res, expected)

        # operating on a subset of columns should not produce a _larger_ Series
        res = df[["B", "C"]].any(bool_only=True, axis=0)
        tm.assert_series_equal(res, expected)

        assert df.any(bool_only=True, axis=None)

    @pytest.mark.parametrize("method", ["any", "all"])
    def test_any_all_level_axis_none_raises(self, method):
        df = DataFrame(
            {"A": 1},
            index=MultiIndex.from_product(
                [["A", "B"], ["a", "b"]], names=["out", "in"]
            ),
        )
        xpr = "Must specify 'axis' when aggregating by level."
        with pytest.raises(ValueError, match=xpr):
            with tm.assert_produces_warning(FutureWarning):
                getattr(df, method)(axis=None, level="out")

    # ---------------------------------------------------------------------
    # Unsorted

    def test_series_broadcasting(self):
        # smoke test for numpy warnings
        # GH 16378, GH 16306
        df = DataFrame([1.0, 1.0, 1.0])
        df_nan = DataFrame({"A": [np.nan, 2.0, np.nan]})
        s = Series([1, 1, 1])
        s_nan = Series([np.nan, np.nan, 1])

        with tm.assert_produces_warning(None):
            df_nan.clip(lower=s, axis=0)
            for op in ["lt", "le", "gt", "ge", "eq", "ne"]:
                getattr(df, op)(s_nan, axis=0)


class TestDataFrameReductions:
    def test_min_max_dt64_with_NaT(self):
        # Both NaT and Timestamp are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT, Timestamp("2012-05-01")]})

        res = df.min()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([Timestamp("2012-05-01")], index=["foo"])
        tm.assert_series_equal(res, exp)

        # GH12941, only NaTs are in DataFrame.
        df = DataFrame({"foo": [pd.NaT, pd.NaT]})

        res = df.min()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

        res = df.max()
        exp = Series([pd.NaT], index=["foo"])
        tm.assert_series_equal(res, exp)

    def test_min_max_dt64_with_NaT_skipna_false(self, request, tz_naive_fixture):
        # GH#36907
        tz = tz_naive_fixture
        if isinstance(tz, tzlocal) and is_platform_windows():
            pytest.skip(
                "GH#37659 OSError raised within tzlocal bc Windows "
                "chokes in times before 1970-01-01"
            )

        df = DataFrame(
            {
                "a": [
                    Timestamp("2020-01-01 08:00:00", tz=tz),
                    Timestamp("1920-02-01 09:00:00", tz=tz),
                ],
                "b": [Timestamp("2020-02-01 08:00:00", tz=tz), pd.NaT],
            }
        )
        res = df.min(axis=1, skipna=False)
        expected = Series([df.loc[0, "a"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

        res = df.max(axis=1, skipna=False)
        expected = Series([df.loc[0, "b"], pd.NaT])
        assert expected.dtype == df["a"].dtype

        tm.assert_series_equal(res, expected)

    def test_min_max_dt64_api_consistency_with_NaT(self):
        # Calling the following sum functions returned an error for dataframes but
        # returned NaT for series. These tests check that the API is consistent in
        # min/max calls on empty Series/DataFrames. See GH:33704 for more
        # information
        df = DataFrame({"x": to_datetime([])})
        expected_dt_series = Series(to_datetime([]))
        # check axis 0
        assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
        assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)

        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_dt_series)
        tm.assert_series_equal(df.max(axis=1), expected_dt_series)

    def test_min_max_dt64_api_consistency_empty_df(self):
        # check DataFrame/Series api consistency when calling min/max on an empty
        # DataFrame/Series.
        df = DataFrame({"x": []})
        expected_float_series = Series([], dtype=float)
        # check axis 0
        assert np.isnan(df.min(axis=0).x) == np.isnan(expected_float_series.min())
        assert np.isnan(df.max(axis=0).x) == np.isnan(expected_float_series.max())
        # check axis 1
        tm.assert_series_equal(df.min(axis=1), expected_float_series)
        tm.assert_series_equal(df.min(axis=1), expected_float_series)

    @pytest.mark.parametrize(
        "initial",
        ["2018-10-08 13:36:45+00:00", "2018-10-08 13:36:45+03:00"],  # Non-UTC timezone
    )
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_preserve_timezone(self, initial: str, method):
        # GH 28552
        initial_dt = to_datetime(initial)
        expected = Series([initial_dt])
        df = DataFrame([expected])
        result = getattr(df, method)(axis=1)
        tm.assert_series_equal(result, expected)

    def test_frame_any_all_with_level(self):
        df = DataFrame(
            {"data": [False, False, True, False, True, False, True]},
            index=[
                ["one", "one", "two", "one", "two", "two", "two"],
                [0, 1, 0, 2, 1, 2, 3],
            ],
        )

        with tm.assert_produces_warning(FutureWarning, match="Using the level"):
            result = df.any(level=0)
        ex = DataFrame({"data": [False, True]}, index=["one", "two"])
        tm.assert_frame_equal(result, ex)

        with tm.assert_produces_warning(FutureWarning, match="Using the level"):
            result = df.all(level=0)
        ex = DataFrame({"data": [False, False]}, index=["one", "two"])
        tm.assert_frame_equal(result, ex)

    def test_frame_any_with_timedelta(self):
        # GH#17667
        df = DataFrame(
            {
                "a": Series([0, 0]),
                "t": Series([to_timedelta(0, "s"), to_timedelta(1, "ms")]),
            }
        )

        result = df.any(axis=0)
        expected = Series(data=[False, True], index=["a", "t"])
        tm.assert_series_equal(result, expected)

        result = df.any(axis=1)
        expected = Series(data=[False, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "func",
        [
            "any",
            "all",
            "count",
            "sum",
            "prod",
            "max",
            "min",
            "mean",
            "median",
            "skew",
            "kurt",
            "sem",
            "var",
            "std",
            "mad",
        ],
    )
    def test_reductions_deprecation_level_argument(self, frame_or_series, func):
        # GH#39983
        obj = frame_or_series(
            [1, 2, 3], index=MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])
        )
        with tm.assert_produces_warning(FutureWarning, match="level"):
            getattr(obj, func)(level=0)


class TestNuisanceColumns:
    @pytest.mark.parametrize("method", ["any", "all"])
    def test_any_all_categorical_dtype_nuisance_column(self, method):
        # GH#36076 DataFrame should match Series behavior
        ser = Series([0, 1], dtype="category", name="A")
        df = ser.to_frame()

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not implement reduction"):
            getattr(ser, method)()

        with pytest.raises(TypeError, match="does not implement reduction"):
            getattr(np, method)(ser)

        with pytest.raises(TypeError, match="does not implement reduction"):
            getattr(df, method)(bool_only=False)

        # With bool_only=None, operating on this column raises and is ignored,
        #  so we expect an empty result.
        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = getattr(df, method)(bool_only=None)
        expected = Series([], index=Index([]), dtype=bool)
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns", check_stacklevel=False
        ):
            result = getattr(np, method)(df, axis=0)
        tm.assert_series_equal(result, expected)

    def test_median_categorical_dtype_nuisance_column(self):
        # GH#21020 DataFrame.median should match Series.median
        df = DataFrame({"A": Categorical([1, 2, 2, 2, 3])})
        ser = df["A"]

        # Double-check the Series behavior is to raise
        with pytest.raises(TypeError, match="does not implement reduction"):
            ser.median()

        with pytest.raises(TypeError, match="does not implement reduction"):
            df.median(numeric_only=False)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = df.median()
        expected = Series([], index=Index([]), dtype=np.float64)
        tm.assert_series_equal(result, expected)

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(int)

        with pytest.raises(TypeError, match="does not implement reduction"):
            df.median(numeric_only=False)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = df.median()
        expected = Series([2.0], index=["B"])
        tm.assert_series_equal(result, expected)

        # TODO: np.median(df, axis=0) gives np.array([2.0, 2.0]) instead
        #  of expected.values

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method):
        # GH#28949 DataFrame.min should behave like Series.min
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        ser = Series(cat)
        df = ser.to_frame("A")

        # Double-check the Series behavior
        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(ser, method)()

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(np, method)(ser)

        with pytest.raises(TypeError, match="is not ordered for operation"):
            getattr(df, method)(numeric_only=False)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = getattr(df, method)()
        expected = Series([], index=Index([]), dtype=np.float64)
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns", check_stacklevel=False
        ):
            result = getattr(np, method)(df)
        tm.assert_series_equal(result, expected)

        # same thing, but with an additional non-categorical column
        df["B"] = df["A"].astype(object)
        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = getattr(df, method)()
        if method == "min":
            expected = Series(["a"], index=["B"])
        else:
            expected = Series(["c"], index=["B"])
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns", check_stacklevel=False
        ):
            result = getattr(np, method)(df)
        tm.assert_series_equal(result, expected)

    def test_reduction_object_block_splits_nuisance_columns(self):
        # GH#37827
        df = DataFrame({"A": [0, 1, 2], "B": ["a", "b", "c"]}, dtype=object)

        # We should only exclude "B", not "A"
        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = df.mean()
        expected = Series([1.0], index=["A"])
        tm.assert_series_equal(result, expected)

        # Same behavior but heterogeneous dtype
        df["C"] = df["A"].astype(int) + 4

        with tm.assert_produces_warning(
            FutureWarning, match="Select only valid columns"
        ):
            result = df.mean()
        expected = Series([1.0, 5.0], index=["A", "C"])
        tm.assert_series_equal(result, expected)


def test_sum_timedelta64_skipna_false():
    # GH#17235
    arr = np.arange(8).astype(np.int64).view("m8[s]").reshape(4, 2)
    arr[-1, -1] = "Nat"

    df = DataFrame(arr)

    result = df.sum(skipna=False)
    expected = Series([pd.Timedelta(seconds=12), pd.NaT])
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=0, skipna=False)
    tm.assert_series_equal(result, expected)

    result = df.sum(axis=1, skipna=False)
    expected = Series(
        [
            pd.Timedelta(seconds=1),
            pd.Timedelta(seconds=5),
            pd.Timedelta(seconds=9),
            pd.NaT,
        ]
    )
    tm.assert_series_equal(result, expected)


def test_mixed_frame_with_integer_sum():
    # https://github.com/pandas-dev/pandas/issues/34520
    df = DataFrame([["a", 1]], columns=list("ab"))
    df = df.astype({"b": "Int64"})
    result = df.sum()
    expected = Series(["a", 1], index=["a", "b"])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False, None])
@pytest.mark.parametrize("method", ["min", "max"])
def test_minmax_extensionarray(method, numeric_only):
    # https://github.com/pandas-dev/pandas/issues/32651
    int64_info = np.iinfo("int64")
    ser = Series([int64_info.max, None, int64_info.min], dtype=pd.Int64Dtype())
    df = DataFrame({"Int64": ser})
    result = getattr(df, method)(numeric_only=numeric_only)
    expected = Series(
        [getattr(int64_info, method)], index=Index(["Int64"], dtype="object")
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("meth", ["max", "min", "sum", "mean", "median"])
def test_groupby_regular_arithmetic_equivalent(meth):
    # GH#40660
    df = DataFrame(
        {"a": [pd.Timedelta(hours=6), pd.Timedelta(hours=7)], "b": [12.1, 13.3]}
    )
    expected = df.copy()

    with tm.assert_produces_warning(FutureWarning):
        result = getattr(df, meth)(level=0)
    tm.assert_frame_equal(result, expected)

    result = getattr(df.groupby(level=0), meth)(numeric_only=False)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ts_value", [Timestamp("2000-01-01"), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value):
    # GH 13912
    df = DataFrame({"a": [1], "b": [1.1], "c": ["foo"], "d": [ts_value]})
    with tm.assert_produces_warning(FutureWarning, match="Dropping of nuisance"):
        result = df.sum()
    expected = Series([1, 1.1, "foo"], index=list("abc"))
    tm.assert_series_equal(result, expected)


def test_prod_sum_min_count_mixed_object():
    # https://github.com/pandas-dev/pandas/issues/41074
    df = DataFrame([1, "a", True])

    result = df.prod(axis=0, min_count=1, numeric_only=False)
    expected = Series(["a"])
    tm.assert_series_equal(result, expected)

    msg = re.escape("unsupported operand type(s) for +: 'int' and 'str'")
    with pytest.raises(TypeError, match=msg):
        df.sum(axis=0, min_count=1, numeric_only=False)
