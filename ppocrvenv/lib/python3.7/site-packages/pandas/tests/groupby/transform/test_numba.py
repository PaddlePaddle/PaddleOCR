import pytest

from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Series,
    option_context,
)
import pandas._testing as tm
from pandas.core.util.numba_ import NUMBA_FUNC_CACHE


@td.skip_if_no("numba", "0.46.0")
def test_correct_function_signature():
    def incorrect_function(x):
        return x + 1

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key").transform(incorrect_function, engine="numba")

    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key")["data"].transform(incorrect_function, engine="numba")


@td.skip_if_no("numba", "0.46.0")
def test_check_nopython_kwargs():
    def incorrect_function(x, **kwargs):
        return x + 1

    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )
    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key").transform(incorrect_function, engine="numba", a=1)

    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key")["data"].transform(incorrect_function, engine="numba", a=1)


@td.skip_if_no("numba", "0.46.0")
@pytest.mark.filterwarnings("ignore:\\nThe keyword argument")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("pandas_obj", ["Series", "DataFrame"])
def test_numba_vs_cython(jit, pandas_obj, nogil, parallel, nopython):
    def func(values, index):
        return values + 1

    if jit:
        # Test accepted jitted functions
        import numba

        func = numba.jit(func)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0)
    if pandas_obj == "Series":
        grouped = grouped[1]

    result = grouped.transform(func, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.transform(lambda x: x + 1, engine="cython")

    tm.assert_equal(result, expected)


@td.skip_if_no("numba", "0.46.0")
@pytest.mark.filterwarnings("ignore:\\nThe keyword argument")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("pandas_obj", ["Series", "DataFrame"])
def test_cache(jit, pandas_obj, nogil, parallel, nopython):
    # Test that the functions are cached correctly if we switch functions
    def func_1(values, index):
        return values + 1

    def func_2(values, index):
        return values * 5

    if jit:
        import numba

        func_1 = numba.jit(func_1)
        func_2 = numba.jit(func_2)

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0)
    if pandas_obj == "Series":
        grouped = grouped[1]

    result = grouped.transform(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.transform(lambda x: x + 1, engine="cython")
    tm.assert_equal(result, expected)
    # func_1 should be in the cache now
    assert (func_1, "groupby_transform") in NUMBA_FUNC_CACHE

    # Add func_2 to the cache
    result = grouped.transform(func_2, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.transform(lambda x: x * 5, engine="cython")
    tm.assert_equal(result, expected)
    assert (func_2, "groupby_transform") in NUMBA_FUNC_CACHE

    # Retest func_1 which should use the cache
    result = grouped.transform(func_1, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.transform(lambda x: x + 1, engine="cython")
    tm.assert_equal(result, expected)


@td.skip_if_no("numba", "0.46.0")
def test_use_global_config():
    def func_1(values, index):
        return values + 1

    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    grouped = data.groupby(0)
    expected = grouped.transform(func_1, engine="numba")
    with option_context("compute.use_numba", True):
        result = grouped.transform(func_1, engine=None)
    tm.assert_frame_equal(expected, result)


@td.skip_if_no("numba", "0.46.0")
@pytest.mark.parametrize(
    "agg_func", [["min", "max"], "min", {"B": ["min", "max"], "C": "sum"}]
)
def test_multifunc_notimplimented(agg_func):
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    grouped = data.groupby(0)
    with pytest.raises(NotImplementedError, match="Numba engine can"):
        grouped.transform(agg_func, engine="numba")

    with pytest.raises(NotImplementedError, match="Numba engine can"):
        grouped[1].transform(agg_func, engine="numba")


@td.skip_if_no("numba", "0.46.0")
def test_args_not_cached():
    # GH 41647
    def sum_last(values, index, n):
        return values[-n:].sum()

    df = DataFrame({"id": [0, 0, 1, 1], "x": [1, 1, 1, 1]})
    grouped_x = df.groupby("id")["x"]
    result = grouped_x.transform(sum_last, 1, engine="numba")
    expected = Series([1.0] * 4, name="x")
    tm.assert_series_equal(result, expected)

    result = grouped_x.transform(sum_last, 2, engine="numba")
    expected = Series([2.0] * 4, name="x")
    tm.assert_series_equal(result, expected)


@td.skip_if_no("numba", "0.46.0")
def test_index_data_correctly_passed():
    # GH 43133
    def f(values, index):
        return index - 1

    df = DataFrame({"group": ["A", "A", "B"], "v": [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby("group").transform(f, engine="numba")
    expected = DataFrame([-4.0, -3.0, -2.0], columns=["v"], index=[-1, -2, -3])
    tm.assert_frame_equal(result, expected)
