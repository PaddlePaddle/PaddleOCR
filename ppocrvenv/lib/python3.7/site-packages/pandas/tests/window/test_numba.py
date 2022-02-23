import numpy as np
import pytest

from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    option_context,
    to_datetime,
)
import pandas._testing as tm
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE


@td.skip_if_no("numba", "0.46.0")
@pytest.mark.filterwarnings("ignore:\\nThe keyword argument")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestEngine:
    @pytest.mark.parametrize("jit", [True, False])
    def test_numba_vs_cython_apply(self, jit, nogil, parallel, nopython, center):
        def f(x, *args):
            arg_sum = 0
            for arg in args:
                arg_sum += arg
            return np.mean(x) + arg_sum

        if jit:
            import numba

            f = numba.jit(f)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        args = (2,)

        s = Series(range(10))
        result = s.rolling(2, center=center).apply(
            f, args=args, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = s.rolling(2, center=center).apply(
            f, engine="cython", args=args, raw=True
        )
        tm.assert_series_equal(result, expected)

    def test_numba_vs_cython_rolling_methods(
        self, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):

        method = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(5))
        roll = df.rolling(2)
        result = getattr(roll, method)(engine="numba", engine_kwargs=engine_kwargs)
        expected = getattr(roll, method)(engine="cython")

        # Check the cache
        assert (getattr(np, f"nan{method}"), "Rolling_apply_single") in NUMBA_FUNC_CACHE

        tm.assert_frame_equal(result, expected)

    def test_numba_vs_cython_expanding_methods(
        self, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):

        method = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(5))
        expand = df.expanding()
        result = getattr(expand, method)(engine="numba", engine_kwargs=engine_kwargs)
        expected = getattr(expand, method)(engine="cython")

        # Check the cache
        assert (
            getattr(np, f"nan{method}"),
            "Expanding_apply_single",
        ) in NUMBA_FUNC_CACHE

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("jit", [True, False])
    def test_cache_apply(self, jit, nogil, parallel, nopython):
        # Test that the functions are cached correctly if we switch functions
        def func_1(x):
            return np.mean(x) + 4

        def func_2(x):
            return np.std(x) * 5

        if jit:
            import numba

            func_1 = numba.jit(func_1)
            func_2 = numba.jit(func_2)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        roll = Series(range(10)).rolling(2)
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)

        # func_1 should be in the cache now
        assert (func_1, "Rolling_apply_single") in NUMBA_FUNC_CACHE

        result = roll.apply(
            func_2, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_2, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)
        # This run should use the cached func_1
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        expected = roll.apply(func_1, engine="cython", raw=True)
        tm.assert_series_equal(result, expected)


@td.skip_if_no("numba", "0.46.0")
class TestEWMMean:
    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    def test_invalid_engine(self, grouper):
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="engine must be either"):
            grouper(df).ewm(com=1.0).mean(engine="foo")

    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    def test_invalid_engine_kwargs(self, grouper):
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        with pytest.raises(ValueError, match="cython engine does not"):
            grouper(df).ewm(com=1.0).mean(
                engine="cython", engine_kwargs={"nopython": True}
            )

    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    def test_cython_vs_numba(
        self, grouper, nogil, parallel, nopython, ignore_na, adjust
    ):
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        ewm = grouper(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        result = ewm.mean(engine="numba", engine_kwargs=engine_kwargs)
        expected = ewm.mean(engine="cython")

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    def test_cython_vs_numba_times(self, grouper, nogil, parallel, nopython, ignore_na):
        # GH 40951
        halflife = "23 days"
        times = to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-10",
                "2020-02-23",
                "2020-01-03",
            ]
        )
        df = DataFrame({"A": ["a", "b", "a", "b", "b", "a"], "B": [0, 0, 1, 1, 2, 2]})
        ewm = grouper(df).ewm(
            halflife=halflife, adjust=True, ignore_na=ignore_na, times=times
        )

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        result = ewm.mean(engine="numba", engine_kwargs=engine_kwargs)
        expected = ewm.mean(engine="cython")

        tm.assert_frame_equal(result, expected)


@td.skip_if_no("numba", "0.46.0")
def test_use_global_config():
    def f(x):
        return np.mean(x) + 2

    s = Series(range(10))
    with option_context("compute.use_numba", True):
        result = s.rolling(2).apply(f, engine=None, raw=True)
    expected = s.rolling(2).apply(f, engine="numba", raw=True)
    tm.assert_series_equal(expected, result)


@td.skip_if_no("numba", "0.46.0")
def test_invalid_kwargs_nopython():
    with pytest.raises(NumbaUtilError, match="numba does not support kwargs with"):
        Series(range(1)).rolling(1).apply(
            lambda x: x, kwargs={"a": 1}, engine="numba", raw=True
        )


@td.skip_if_no("numba", "0.46.0")
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:\\nThe keyword argument")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestTableMethod:
    def test_table_series_valueerror(self):
        def f(x):
            return np.sum(x, axis=0) + 1

        with pytest.raises(
            ValueError, match="method='table' not applicable for Series objects."
        ):
            Series(range(1)).rolling(1, method="table").apply(
                f, engine="numba", raw=True
            )

    def test_table_method_rolling_methods(
        self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        method = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))

        result = getattr(
            df.rolling(2, method="table", axis=axis, min_periods=0), method
        )(engine_kwargs=engine_kwargs, engine="numba")
        expected = getattr(
            df.rolling(2, method="single", axis=axis, min_periods=0), method
        )(engine_kwargs=engine_kwargs, engine="numba")
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_apply(self, axis, nogil, parallel, nopython):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.rolling(2, method="table", axis=axis, min_periods=0).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = df.rolling(2, method="single", axis=axis, min_periods=0).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_rolling_weighted_mean(self):
        def weighted_mean(x):
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr

        df = DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
        result = df.rolling(2, method="table", min_periods=0).apply(
            weighted_mean, raw=True, engine="numba"
        )
        expected = DataFrame(
            [
                [1.0, 2.0, 1.0],
                [1.8, 2.0, 1.0],
                [3.333333, 2.333333, 1.0],
                [1.555556, 7, 1.0],
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_apply(self, axis, nogil, parallel, nopython):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        def f(x):
            return np.sum(x, axis=0) + 1

        df = DataFrame(np.eye(3))
        result = df.expanding(method="table", axis=axis).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = df.expanding(method="single", axis=axis).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)

    def test_table_method_expanding_methods(
        self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        method = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        df = DataFrame(np.eye(3))

        result = getattr(df.expanding(method="table", axis=axis), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        expected = getattr(df.expanding(method="single", axis=axis), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        tm.assert_frame_equal(result, expected)
