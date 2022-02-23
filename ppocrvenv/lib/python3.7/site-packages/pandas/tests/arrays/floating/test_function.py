import numpy as np
import pytest

from pandas.compat import IS64

import pandas as pd
import pandas._testing as tm


@pytest.mark.parametrize("ufunc", [np.abs, np.sign])
# np.sign emits a warning with nans, <https://github.com/numpy/numpy/issues/15127>
@pytest.mark.filterwarnings("ignore:invalid value encountered in sign")
def test_ufuncs_single(ufunc):
    a = pd.array([1, 2, -3, np.nan], dtype="Float64")
    result = ufunc(a)
    expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.log, np.exp, np.sin, np.cos, np.sqrt])
def test_ufuncs_single_float(ufunc):
    a = pd.array([1.0, 0.2, 3.0, np.nan], dtype="Float64")
    with np.errstate(invalid="ignore"):
        result = ufunc(a)
        expected = pd.array(ufunc(a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    s = pd.Series(a)
    with np.errstate(invalid="ignore"):
        result = ufunc(s)
        expected = pd.Series(ufunc(s.astype(float)), dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract])
def test_ufuncs_binary_float(ufunc):
    # two FloatingArrays
    a = pd.array([1, 0.2, -3, np.nan], dtype="Float64")
    result = ufunc(a, a)
    expected = pd.array(ufunc(a.astype(float), a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with numpy array
    arr = np.array([1, 2, 3, 4])
    result = ufunc(a, arr)
    expected = pd.array(ufunc(a.astype(float), arr), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(arr, a)
    expected = pd.array(ufunc(arr, a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    # FloatingArray with scalar
    result = ufunc(a, 1)
    expected = pd.array(ufunc(a.astype(float), 1), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)

    result = ufunc(1, a)
    expected = pd.array(ufunc(1, a.astype(float)), dtype="Float64")
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize("values", [[0, 1], [0, None]])
def test_ufunc_reduce_raises(values):
    a = pd.array(values, dtype="Float64")
    msg = r"The 'reduce' method is not supported."
    with pytest.raises(NotImplementedError, match=msg):
        np.add.reduce(a)


@pytest.mark.skipif(not IS64, reason="GH 36579: fail on 32-bit system")
@pytest.mark.parametrize(
    "pandasmethname, kwargs",
    [
        ("var", {"ddof": 0}),
        ("var", {"ddof": 1}),
        ("kurtosis", {}),
        ("skew", {}),
        ("sem", {}),
    ],
)
def test_stat_method(pandasmethname, kwargs):
    s = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, np.nan, np.nan], dtype="Float64")
    pandasmeth = getattr(s, pandasmethname)
    result = pandasmeth(**kwargs)
    s2 = pd.Series(data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype="float64")
    pandasmeth = getattr(s2, pandasmethname)
    expected = pandasmeth(**kwargs)
    assert expected == result


def test_value_counts_na():
    arr = pd.array([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = arr.value_counts(dropna=False)
    expected = pd.Series([2, 1, 1], index=[0.1, 0.2, pd.NA], dtype="Int64")
    tm.assert_series_equal(result, expected)

    result = arr.value_counts(dropna=True)
    expected = pd.Series([2, 1], index=[0.1, 0.2], dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_value_counts_empty():
    s = pd.Series([], dtype="Float64")
    result = s.value_counts()
    idx = pd.Index([], dtype="object")
    expected = pd.Series([], index=idx, dtype="Int64")
    tm.assert_series_equal(result, expected)


def test_value_counts_with_normalize():
    s = pd.Series([0.1, 0.2, 0.1, pd.NA], dtype="Float64")
    result = s.value_counts(normalize=True)
    expected = pd.Series([2, 1], index=[0.1, 0.2], dtype="Float64") / 3
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 4])
def test_floating_array_sum(skipna, min_count, dtype):
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6.0
    else:
        assert result is pd.NA


@pytest.mark.parametrize(
    "values, expected", [([1, 2, 3], 6.0), ([1, 2, 3, None], 6.0), ([None], 0.0)]
)
def test_floating_array_numpy_sum(values, expected):
    arr = pd.array(values, dtype="Float64")
    result = np.sum(arr)
    assert result == expected


@pytest.mark.parametrize("op", ["sum", "min", "max", "prod"])
def test_preserve_dtypes(op):
    df = pd.DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": pd.array([0.1, None, 3.0], dtype="Float64"),
        }
    )

    # op
    result = getattr(df.C, op)()
    assert isinstance(result, np.float64)

    # groupby
    result = getattr(df.groupby("A"), op)()

    expected = pd.DataFrame(
        {"B": np.array([1.0, 3.0]), "C": pd.array([0.1, 3], dtype="Float64")},
        index=pd.Index(["a", "b"], name="A"),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("method", ["min", "max"])
def test_floating_array_min_max(skipna, method, dtype):
    arr = pd.array([0.0, 1.0, None], dtype=dtype)
    func = getattr(arr, method)
    result = func(skipna=skipna)
    if skipna:
        assert result == (0 if method == "min" else 1)
    else:
        assert result is pd.NA


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("min_count", [0, 9])
def test_floating_array_prod(skipna, min_count, dtype):
    arr = pd.array([1.0, 2.0, None], dtype=dtype)
    result = arr.prod(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 2
    else:
        assert result is pd.NA
