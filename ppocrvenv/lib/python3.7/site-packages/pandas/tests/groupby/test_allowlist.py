"""
test methods relating to generic function evaluation
the so-called white/black lists
"""

from string import ascii_lowercase

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.groupby.base import (
    groupby_other_methods,
    reduction_kernels,
    transformation_kernels,
)

AGG_FUNCTIONS = [
    "sum",
    "prod",
    "min",
    "max",
    "median",
    "mean",
    "skew",
    "mad",
    "std",
    "var",
    "sem",
]
AGG_FUNCTIONS_WITH_SKIPNA = ["skew", "mad"]

df_allowlist = [
    "quantile",
    "fillna",
    "mad",
    "take",
    "idxmax",
    "idxmin",
    "tshift",
    "skew",
    "plot",
    "hist",
    "dtypes",
    "corrwith",
    "corr",
    "cov",
    "diff",
]


@pytest.fixture(params=df_allowlist)
def df_allowlist_fixture(request):
    return request.param


s_allowlist = [
    "quantile",
    "fillna",
    "mad",
    "take",
    "idxmax",
    "idxmin",
    "tshift",
    "skew",
    "plot",
    "hist",
    "dtype",
    "corr",
    "cov",
    "diff",
    "unique",
    "nlargest",
    "nsmallest",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]


@pytest.fixture(params=s_allowlist)
def s_allowlist_fixture(request):
    return request.param


@pytest.fixture
def mframe():
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    return DataFrame(np.random.randn(10, 3), index=index, columns=["A", "B", "C"])


@pytest.fixture
def df():
    return DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.randn(8),
            "D": np.random.randn(8),
        }
    )


@pytest.fixture
def df_letters():
    letters = np.array(list(ascii_lowercase))
    N = 10
    random_letters = letters.take(np.random.randint(0, 26, N))
    df = DataFrame(
        {
            "floats": N / 10 * Series(np.random.random(N)),
            "letters": Series(random_letters),
        }
    )
    return df


@pytest.mark.parametrize("allowlist", [df_allowlist, s_allowlist])
def test_groupby_allowlist(df_letters, allowlist):
    df = df_letters
    if allowlist == df_allowlist:
        # dataframe
        obj = df_letters
    else:
        obj = df_letters["floats"]

    gb = obj.groupby(df.letters)

    assert set(allowlist) == set(gb._apply_allowlist)


def check_allowlist(obj, df, m):
    # check the obj for a particular allowlist m

    gb = obj.groupby(df.letters)

    f = getattr(type(gb), m)

    # name
    try:
        n = f.__name__
    except AttributeError:
        return
    assert n == m

    # qualname
    try:
        n = f.__qualname__
    except AttributeError:
        return
    assert n.endswith(m)


def test_groupby_series_allowlist(df_letters, s_allowlist_fixture):
    m = s_allowlist_fixture
    df = df_letters
    check_allowlist(df.letters, df, m)


def test_groupby_frame_allowlist(df_letters, df_allowlist_fixture):
    m = df_allowlist_fixture
    df = df_letters
    check_allowlist(df, df, m)


@pytest.fixture
def raw_frame():
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    raw_frame = DataFrame(
        np.random.randn(10, 3), index=index, columns=Index(["A", "B", "C"], name="exp")
    )
    raw_frame.iloc[1, [1, 2]] = np.nan
    raw_frame.iloc[7, [0, 1]] = np.nan
    return raw_frame


@pytest.mark.parametrize("op", AGG_FUNCTIONS)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_regression_allowlist_methods(raw_frame, op, level, axis, skipna, sort):
    # GH6944
    # GH 17537
    # explicitly test the allowlist methods

    if axis == 0:
        frame = raw_frame
    else:
        frame = raw_frame.T

    if op in AGG_FUNCTIONS_WITH_SKIPNA:
        grouped = frame.groupby(level=level, axis=axis, sort=sort)
        result = getattr(grouped, op)(skipna=skipna)
        with tm.assert_produces_warning(FutureWarning):
            expected = getattr(frame, op)(level=level, axis=axis, skipna=skipna)
        if sort:
            expected = expected.sort_index(axis=axis, level=level)
        tm.assert_frame_equal(result, expected)
    else:
        grouped = frame.groupby(level=level, axis=axis, sort=sort)
        result = getattr(grouped, op)()
        with tm.assert_produces_warning(FutureWarning):
            expected = getattr(frame, op)(level=level, axis=axis)
        if sort:
            expected = expected.sort_index(axis=axis, level=level)
        tm.assert_frame_equal(result, expected)


def test_groupby_blocklist(df_letters):
    df = df_letters
    s = df_letters.floats

    blocklist = [
        "eval",
        "query",
        "abs",
        "where",
        "mask",
        "align",
        "groupby",
        "clip",
        "astype",
        "at",
        "combine",
        "consolidate",
        "convert_objects",
    ]
    to_methods = [method for method in dir(df) if method.startswith("to_")]

    blocklist.extend(to_methods)

    for bl in blocklist:
        for obj in (df, s):
            gb = obj.groupby(df.letters)

            # e.g., to_csv
            defined_but_not_allowed = (
                f"(?:^Cannot.+{repr(bl)}.+'{type(gb).__name__}'.+try "
                f"using the 'apply' method$)"
            )

            # e.g., query, eval
            not_defined = (
                f"(?:^'{type(gb).__name__}' object has no attribute {repr(bl)}$)"
            )

            msg = f"{defined_but_not_allowed}|{not_defined}"

            with pytest.raises(AttributeError, match=msg):
                getattr(gb, bl)


def test_tab_completion(mframe):
    grp = mframe.groupby(level="second")
    results = {v for v in dir(grp) if not v.startswith("_")}
    expected = {
        "A",
        "B",
        "C",
        "agg",
        "aggregate",
        "apply",
        "boxplot",
        "filter",
        "first",
        "get_group",
        "groups",
        "hist",
        "indices",
        "last",
        "max",
        "mean",
        "median",
        "min",
        "ngroups",
        "nth",
        "ohlc",
        "plot",
        "prod",
        "size",
        "std",
        "sum",
        "transform",
        "var",
        "sem",
        "count",
        "nunique",
        "head",
        "describe",
        "cummax",
        "quantile",
        "rank",
        "cumprod",
        "tail",
        "resample",
        "cummin",
        "fillna",
        "cumsum",
        "cumcount",
        "ngroup",
        "all",
        "shift",
        "skew",
        "take",
        "tshift",
        "pct_change",
        "any",
        "mad",
        "corr",
        "corrwith",
        "cov",
        "dtypes",
        "ndim",
        "diff",
        "idxmax",
        "idxmin",
        "ffill",
        "bfill",
        "pad",
        "backfill",
        "rolling",
        "expanding",
        "pipe",
        "sample",
        "ewm",
    }
    assert results == expected


def test_groupby_function_rename(mframe):
    grp = mframe.groupby(level="second")
    for name in ["sum", "prod", "min", "max", "first", "last"]:
        f = getattr(grp, name)
        assert f.__name__ == name


@pytest.mark.parametrize(
    "method",
    [
        "count",
        "corr",
        "cummax",
        "cummin",
        "cumprod",
        # TODO(ArrayManager) quantile
        pytest.param("describe", marks=td.skip_array_manager_not_yet_implemented),
        "rank",
        "quantile",
        "diff",
        "shift",
        "all",
        "any",
        "idxmin",
        "idxmax",
        "ffill",
        "bfill",
        "pct_change",
    ],
)
def test_groupby_selection_with_methods(df, method):
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])
    # TODO check groupby with > 1 col ?

    res = getattr(g, method)()
    exp = getattr(g_exp, method)()

    # should always be frames!
    tm.assert_frame_equal(res, exp)


@pytest.mark.filterwarnings("ignore:tshift is deprecated:FutureWarning")
def test_groupby_selection_tshift_raises(df):
    rng = date_range("2014", periods=len(df))
    df.index = rng

    g = df.groupby(["A"])[["C"]]

    # check that the index cache is cleared
    with pytest.raises(ValueError, match="Freq was not set in the index"):
        # GH#35937
        g.tshift()


def test_groupby_selection_other_methods(df):
    # some methods which require DatetimeIndex
    rng = date_range("2014", periods=len(df))
    df.index = rng

    g = df.groupby(["A"])[["C"]]
    g_exp = df[["C"]].groupby(df["A"])

    # methods which aren't just .foo()
    tm.assert_frame_equal(g.fillna(0), g_exp.fillna(0))
    tm.assert_frame_equal(g.dtypes, g_exp.dtypes)
    tm.assert_frame_equal(g.apply(lambda x: x.sum()), g_exp.apply(lambda x: x.sum()))

    tm.assert_frame_equal(g.resample("D").mean(), g_exp.resample("D").mean())
    tm.assert_frame_equal(g.resample("D").ohlc(), g_exp.resample("D").ohlc())

    tm.assert_frame_equal(
        g.filter(lambda x: len(x) == 3), g_exp.filter(lambda x: len(x) == 3)
    )


def test_all_methods_categorized(mframe):
    grp = mframe.groupby(mframe.iloc[:, 0])
    names = {_ for _ in dir(grp) if not _.startswith("_")} - set(mframe.columns)
    new_names = set(names)
    new_names -= reduction_kernels
    new_names -= transformation_kernels
    new_names -= groupby_other_methods

    assert not (reduction_kernels & transformation_kernels)
    assert not (reduction_kernels & groupby_other_methods)
    assert not (transformation_kernels & groupby_other_methods)

    # new public method?
    if new_names:
        msg = f"""
There are uncatgeorized methods defined on the Grouper class:
{new_names}.

Was a new method recently added?

Every public method On Grouper must appear in exactly one the
following three lists defined in pandas.core.groupby.base:
- `reduction_kernels`
- `transformation_kernels`
- `groupby_other_methods`
see the comments in pandas/core/groupby/base.py for guidance on
how to fix this test.
        """
        raise AssertionError(msg)

    # removed a public method?
    all_categorized = reduction_kernels | transformation_kernels | groupby_other_methods
    print(names)
    print(all_categorized)
    if not (names == all_categorized):
        msg = f"""
Some methods which are supposed to be on the Grouper class
are missing:
{all_categorized - names}.

They're still defined in one of the lists that live in pandas/core/groupby/base.py.
If you removed a method, you should update them
"""
        raise AssertionError(msg)
