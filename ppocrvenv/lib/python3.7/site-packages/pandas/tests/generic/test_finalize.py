"""
An exhaustive list of pandas methods exercising NDFrame.__finalize__.
"""
import operator
import re

import numpy as np
import pytest

import pandas as pd

# TODO:
# * Binary methods (mul, div, etc.)
# * Binary outputs (align, etc.)
# * top-level methods (concat, merge, get_dummies, etc.)
# * window
# * cumulative reductions

not_implemented_mark = pytest.mark.xfail(reason="not implemented")

mi = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=["A", "B"])

frame_data = ({"A": [1]},)
frame_mi_data = ({"A": [1, 2, 3, 4]}, mi)


# Tuple of
# - Callable: Constructor (Series, DataFrame)
# - Tuple: Constructor args
# - Callable: pass the constructed value with attrs set to this.

_all_methods = [
    (
        pd.Series,
        (np.array([0], dtype="float64")),
        operator.methodcaller("view", "int64"),
    ),
    (pd.Series, ([0],), operator.methodcaller("take", [])),
    (pd.Series, ([0],), operator.methodcaller("__getitem__", [True])),
    (pd.Series, ([0],), operator.methodcaller("repeat", 2)),
    pytest.param(
        (pd.Series, ([0],), operator.methodcaller("reset_index")),
        marks=pytest.mark.xfail,
    ),
    (pd.Series, ([0],), operator.methodcaller("reset_index", drop=True)),
    pytest.param(
        (pd.Series, ([0],), operator.methodcaller("to_frame")), marks=pytest.mark.xfail
    ),
    (pd.Series, ([0, 0],), operator.methodcaller("drop_duplicates")),
    (pd.Series, ([0, 0],), operator.methodcaller("duplicated")),
    (pd.Series, ([0, 0],), operator.methodcaller("round")),
    (pd.Series, ([0, 0],), operator.methodcaller("rename", lambda x: x + 1)),
    (pd.Series, ([0, 0],), operator.methodcaller("rename", "name")),
    (pd.Series, ([0, 0],), operator.methodcaller("set_axis", ["a", "b"])),
    (pd.Series, ([0, 0],), operator.methodcaller("reindex", [1, 0])),
    (pd.Series, ([0, 0],), operator.methodcaller("drop", [0])),
    (pd.Series, (pd.array([0, pd.NA]),), operator.methodcaller("fillna", 0)),
    (pd.Series, ([0, 0],), operator.methodcaller("replace", {0: 1})),
    (pd.Series, ([0, 0],), operator.methodcaller("shift")),
    (pd.Series, ([0, 0],), operator.methodcaller("isin", [0, 1])),
    (pd.Series, ([0, 0],), operator.methodcaller("between", 0, 2)),
    (pd.Series, ([0, 0],), operator.methodcaller("isna")),
    (pd.Series, ([0, 0],), operator.methodcaller("isnull")),
    (pd.Series, ([0, 0],), operator.methodcaller("notna")),
    (pd.Series, ([0, 0],), operator.methodcaller("notnull")),
    (pd.Series, ([1],), operator.methodcaller("add", pd.Series([1]))),
    # TODO: mul, div, etc.
    (
        pd.Series,
        ([0], pd.period_range("2000", periods=1)),
        operator.methodcaller("to_timestamp"),
    ),
    (
        pd.Series,
        ([0], pd.date_range("2000", periods=1)),
        operator.methodcaller("to_period"),
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("dot", pd.DataFrame(index=["A"])),
        ),
        marks=pytest.mark.xfail(reason="Implement binary finalize"),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("transpose")),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("__getitem__", np.array([True]))),
    (pd.DataFrame, ({("A", "a"): [1]},), operator.methodcaller("__getitem__", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("query", "A == 1")),
    (pd.DataFrame, frame_data, operator.methodcaller("eval", "A + 1", engine="python")),
    (pd.DataFrame, frame_data, operator.methodcaller("select_dtypes", include="int")),
    (pd.DataFrame, frame_data, operator.methodcaller("assign", b=1)),
    (pd.DataFrame, frame_data, operator.methodcaller("set_axis", ["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("reindex", [0, 1])),
    (pd.DataFrame, frame_data, operator.methodcaller("drop", columns=["A"])),
    (pd.DataFrame, frame_data, operator.methodcaller("drop", index=[0])),
    (pd.DataFrame, frame_data, operator.methodcaller("rename", columns={"A": "a"})),
    (pd.DataFrame, frame_data, operator.methodcaller("rename", index=lambda x: x)),
    (pd.DataFrame, frame_data, operator.methodcaller("fillna", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("fillna", method="ffill")),
    (pd.DataFrame, frame_data, operator.methodcaller("set_index", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("reset_index")),
    (pd.DataFrame, frame_data, operator.methodcaller("isna")),
    (pd.DataFrame, frame_data, operator.methodcaller("isnull")),
    (pd.DataFrame, frame_data, operator.methodcaller("notna")),
    (pd.DataFrame, frame_data, operator.methodcaller("notnull")),
    (pd.DataFrame, frame_data, operator.methodcaller("dropna")),
    (pd.DataFrame, frame_data, operator.methodcaller("drop_duplicates")),
    (pd.DataFrame, frame_data, operator.methodcaller("duplicated")),
    (pd.DataFrame, frame_data, operator.methodcaller("sort_values", by="A")),
    (pd.DataFrame, frame_data, operator.methodcaller("sort_index")),
    (pd.DataFrame, frame_data, operator.methodcaller("nlargest", 1, "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("nsmallest", 1, "A")),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("swaplevel")),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("add", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    # TODO: div, mul, etc.
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("combine", pd.DataFrame(*frame_data), operator.add),
        ),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("combine_first", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("update", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("pivot", columns="A")),
    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A"),
    ),
    (
        pd.DataFrame,
        ({"A": [1], "B": [1]},),
        operator.methodcaller("pivot_table", columns="A", aggfunc=["mean", "sum"]),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("stack")),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("explode", "A")),
        marks=not_implemented_mark,
    ),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("unstack")),
    pytest.param(
        (
            pd.DataFrame,
            ({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]},),
            operator.methodcaller("melt", id_vars=["A"], value_vars=["B"]),
        ),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("applymap", lambda x: x))
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("append", pd.DataFrame({"A": [1]})),
        )
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("append", pd.DataFrame({"B": [1]})),
        ),
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("merge", pd.DataFrame({"A": [1]})),
        ),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("round", 2)),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("corr")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("cov")),
        marks=[
            not_implemented_mark,
            pytest.mark.filterwarnings("ignore::RuntimeWarning"),
        ],
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_data,
            operator.methodcaller("corrwith", pd.DataFrame(*frame_data)),
        ),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("count")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_mi_data, operator.methodcaller("count", level="A")),
        marks=[
            not_implemented_mark,
            pytest.mark.filterwarnings("ignore:Using the level keyword:FutureWarning"),
        ],
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("nunique")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("idxmin")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("idxmax")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("mode")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("quantile")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("quantile", q=[0.25, 0.75])),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("quantile")),
        marks=not_implemented_mark,
    ),
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Period("2000", "D")]),
        operator.methodcaller("to_timestamp"),
    ),
    (
        pd.DataFrame,
        ({"A": [1]}, [pd.Timestamp("2000")]),
        operator.methodcaller("to_period", freq="D"),
    ),
    pytest.param(
        (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", [1])),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_mi_data, operator.methodcaller("isin", pd.Series([1]))),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (
            pd.DataFrame,
            frame_mi_data,
            operator.methodcaller("isin", pd.DataFrame({"A": [1]})),
        ),
        marks=not_implemented_mark,
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("swapaxes", 0, 1)),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("droplevel", "A")),
    (pd.DataFrame, frame_data, operator.methodcaller("pop", "A")),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("squeeze")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.Series, ([1, 2],), operator.methodcaller("squeeze"))
        # marks=not_implemented_mark,
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("rename_axis", index="a")),
    (pd.DataFrame, frame_data, operator.methodcaller("rename_axis", columns="a")),
    # Unary ops
    (pd.DataFrame, frame_data, operator.neg),
    (pd.Series, [1], operator.neg),
    (pd.DataFrame, frame_data, operator.pos),
    (pd.Series, [1], operator.pos),
    (pd.DataFrame, frame_data, operator.inv),
    (pd.Series, [1], operator.inv),
    (pd.DataFrame, frame_data, abs),
    (pd.Series, [1], abs),
    pytest.param((pd.DataFrame, frame_data, round), marks=not_implemented_mark),
    (pd.Series, [1], round),
    (pd.DataFrame, frame_data, operator.methodcaller("take", [0, 0])),
    (pd.DataFrame, frame_mi_data, operator.methodcaller("xs", "a")),
    (pd.Series, (1, mi), operator.methodcaller("xs", "a")),
    (pd.DataFrame, frame_data, operator.methodcaller("get", "A")),
    (
        pd.DataFrame,
        frame_data,
        operator.methodcaller("reindex_like", pd.DataFrame({"A": [1, 2, 3]})),
    ),
    (
        pd.Series,
        frame_data,
        operator.methodcaller("reindex_like", pd.Series([0, 1, 2])),
    ),
    (pd.DataFrame, frame_data, operator.methodcaller("add_prefix", "_")),
    (pd.DataFrame, frame_data, operator.methodcaller("add_suffix", "_")),
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_prefix", "_")),
    (pd.Series, (1, ["a", "b"]), operator.methodcaller("add_suffix", "_")),
    (pd.Series, ([3, 2],), operator.methodcaller("sort_values")),
    (pd.Series, ([1] * 10,), operator.methodcaller("head")),
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("head")),
    (pd.Series, ([1] * 10,), operator.methodcaller("tail")),
    (pd.DataFrame, ({"A": [1] * 10},), operator.methodcaller("tail")),
    (pd.Series, ([1, 2],), operator.methodcaller("sample", n=2, replace=True)),
    (pd.DataFrame, (frame_data,), operator.methodcaller("sample", n=2, replace=True)),
    (pd.Series, ([1, 2],), operator.methodcaller("astype", float)),
    (pd.DataFrame, frame_data, operator.methodcaller("astype", float)),
    (pd.Series, ([1, 2],), operator.methodcaller("copy")),
    (pd.DataFrame, frame_data, operator.methodcaller("copy")),
    (pd.Series, ([1, 2], None, object), operator.methodcaller("infer_objects")),
    (
        pd.DataFrame,
        ({"A": np.array([1, 2], dtype=object)},),
        operator.methodcaller("infer_objects"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("convert_dtypes")),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("convert_dtypes")),
        marks=not_implemented_mark,
    ),
    (pd.Series, ([1, None, 3],), operator.methodcaller("interpolate")),
    (pd.DataFrame, ({"A": [1, None, 3]},), operator.methodcaller("interpolate")),
    (pd.Series, ([1, 2],), operator.methodcaller("clip", lower=1)),
    (pd.DataFrame, frame_data, operator.methodcaller("clip", lower=1)),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("asfreq", "H"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("asfreq", "H"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("at_time", "12:00"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("between_time", "12:00", "13:00"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("first", "3D"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("first", "3D"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("last", "3D"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("last", "3D"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("rank")),
    (pd.DataFrame, frame_data, operator.methodcaller("rank")),
    (pd.Series, ([1, 2],), operator.methodcaller("where", np.array([True, False]))),
    (pd.DataFrame, frame_data, operator.methodcaller("where", np.array([[True]]))),
    (pd.Series, ([1, 2],), operator.methodcaller("mask", np.array([True, False]))),
    (pd.DataFrame, frame_data, operator.methodcaller("mask", np.array([[True]]))),
    pytest.param(
        (
            pd.Series,
            (1, pd.date_range("2000", periods=4)),
            operator.methodcaller("tshift"),
        ),
        marks=pytest.mark.filterwarnings("ignore::FutureWarning"),
    ),
    pytest.param(
        (
            pd.DataFrame,
            ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
            operator.methodcaller("tshift"),
        ),
        marks=pytest.mark.filterwarnings("ignore::FutureWarning"),
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("truncate", before=0)),
    (pd.DataFrame, frame_data, operator.methodcaller("truncate", before=0)),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4, tz="UTC")),
        operator.methodcaller("tz_convert", "CET"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4, tz="UTC")),
        operator.methodcaller("tz_convert", "CET"),
    ),
    (
        pd.Series,
        (1, pd.date_range("2000", periods=4)),
        operator.methodcaller("tz_localize", "CET"),
    ),
    (
        pd.DataFrame,
        ({"A": [1, 1, 1, 1]}, pd.date_range("2000", periods=4)),
        operator.methodcaller("tz_localize", "CET"),
    ),
    pytest.param(
        (pd.Series, ([1, 2],), operator.methodcaller("describe")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("describe")),
        marks=not_implemented_mark,
    ),
    (pd.Series, ([1, 2],), operator.methodcaller("pct_change")),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("pct_change")),
        marks=not_implemented_mark,
    ),
    (pd.Series, ([1],), operator.methodcaller("transform", lambda x: x - x.min())),
    pytest.param(
        (
            pd.DataFrame,
            frame_mi_data,
            operator.methodcaller("transform", lambda x: x - x.min()),
        ),
        marks=not_implemented_mark,
    ),
    (pd.Series, ([1],), operator.methodcaller("apply", lambda x: x)),
    pytest.param(
        (pd.DataFrame, frame_mi_data, operator.methodcaller("apply", lambda x: x)),
        marks=not_implemented_mark,
    ),
    # Cumulative reductions
    (pd.Series, ([1],), operator.methodcaller("cumsum")),
    (pd.DataFrame, frame_data, operator.methodcaller("cumsum")),
    # Reductions
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("any")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("sum")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("std")),
        marks=not_implemented_mark,
    ),
    pytest.param(
        (pd.DataFrame, frame_data, operator.methodcaller("mean")),
        marks=not_implemented_mark,
    ),
]


def idfn(x):
    xpr = re.compile(r"'(.*)?'")
    m = xpr.search(str(x))
    if m:
        return m.group(1)
    else:
        return str(x)


@pytest.fixture(params=_all_methods, ids=lambda x: idfn(x[-1]))
def ndframe_method(request):
    """
    An NDFrame method returning an NDFrame.
    """
    return request.param


def test_finalize_called(ndframe_method):
    cls, init_args, method = ndframe_method
    ndframe = cls(*init_args)

    ndframe.attrs = {"a": 1}
    result = method(ndframe)

    assert result.attrs == {"a": 1}


@not_implemented_mark
def test_finalize_called_eval_numexpr():
    pytest.importorskip("numexpr")
    df = pd.DataFrame({"A": [1, 2]})
    df.attrs["A"] = 1
    result = df.eval("A + 1", engine="numexpr")
    assert result.attrs == {"A": 1}


# ----------------------------------------------------------------------------
# Binary operations


@pytest.mark.parametrize("annotate", ["left", "right", "both"])
@pytest.mark.parametrize(
    "args",
    [
        (1, pd.Series([1])),
        (1, pd.DataFrame({"A": [1]})),
        (pd.Series([1]), 1),
        (pd.DataFrame({"A": [1]}), 1),
        (pd.Series([1]), pd.Series([1])),
        (pd.DataFrame({"A": [1]}), pd.DataFrame({"A": [1]})),
        (pd.Series([1]), pd.DataFrame({"A": [1]})),
        (pd.DataFrame({"A": [1]}), pd.Series([1])),
    ],
)
def test_binops(request, args, annotate, all_arithmetic_functions):
    # This generates 326 tests... Is that needed?
    left, right = args
    if annotate == "both" and isinstance(left, int) or isinstance(right, int):
        return

    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        request.node.add_marker(pytest.mark.xfail(reason="not implemented"))

    if annotate in {"left", "both"} and not isinstance(left, int):
        left.attrs = {"a": 1}
    if annotate in {"left", "both"} and not isinstance(right, int):
        right.attrs = {"a": 1}

    result = all_arithmetic_functions(left, right)
    assert result.attrs == {"a": 1}


# ----------------------------------------------------------------------------
# Accessors


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("capitalize"),
        operator.methodcaller("casefold"),
        operator.methodcaller("cat", ["a"]),
        operator.methodcaller("contains", "a"),
        operator.methodcaller("count", "a"),
        operator.methodcaller("encode", "utf-8"),
        operator.methodcaller("endswith", "a"),
        operator.methodcaller("extract", r"(\w)(\d)"),
        operator.methodcaller("extract", r"(\w)(\d)", expand=False),
        operator.methodcaller("find", "a"),
        operator.methodcaller("findall", "a"),
        operator.methodcaller("get", 0),
        operator.methodcaller("index", "a"),
        operator.methodcaller("len"),
        operator.methodcaller("ljust", 4),
        operator.methodcaller("lower"),
        operator.methodcaller("lstrip"),
        operator.methodcaller("match", r"\w"),
        operator.methodcaller("normalize", "NFC"),
        operator.methodcaller("pad", 4),
        operator.methodcaller("partition", "a"),
        operator.methodcaller("repeat", 2),
        operator.methodcaller("replace", "a", "b"),
        operator.methodcaller("rfind", "a"),
        operator.methodcaller("rindex", "a"),
        operator.methodcaller("rjust", 4),
        operator.methodcaller("rpartition", "a"),
        operator.methodcaller("rstrip"),
        operator.methodcaller("slice", 4),
        operator.methodcaller("slice_replace", 1, repl="a"),
        operator.methodcaller("startswith", "a"),
        operator.methodcaller("strip"),
        operator.methodcaller("swapcase"),
        operator.methodcaller("translate", {"a": "b"}),
        operator.methodcaller("upper"),
        operator.methodcaller("wrap", 4),
        operator.methodcaller("zfill", 4),
        operator.methodcaller("isalnum"),
        operator.methodcaller("isalpha"),
        operator.methodcaller("isdigit"),
        operator.methodcaller("isspace"),
        operator.methodcaller("islower"),
        operator.methodcaller("isupper"),
        operator.methodcaller("istitle"),
        operator.methodcaller("isnumeric"),
        operator.methodcaller("isdecimal"),
        operator.methodcaller("get_dummies"),
    ],
    ids=idfn,
)
def test_string_method(method):
    s = pd.Series(["a1"])
    s.attrs = {"a": 1}
    result = method(s.str)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("to_period"),
        operator.methodcaller("tz_localize", "CET"),
        operator.methodcaller("normalize"),
        operator.methodcaller("strftime", "%Y"),
        operator.methodcaller("round", "H"),
        operator.methodcaller("floor", "H"),
        operator.methodcaller("ceil", "H"),
        operator.methodcaller("month_name"),
        operator.methodcaller("day_name"),
    ],
    ids=idfn,
)
def test_datetime_method(method):
    s = pd.Series(pd.date_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = method(s.dt)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "attr",
    [
        "date",
        "time",
        "timetz",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
        "daysinmonth",
        "days_in_month",
    ],
)
def test_datetime_property(attr):
    s = pd.Series(pd.date_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = getattr(s.dt, attr)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "attr", ["days", "seconds", "microseconds", "nanoseconds", "components"]
)
def test_timedelta_property(attr):
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = getattr(s.dt, attr)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize("method", [operator.methodcaller("total_seconds")])
def test_timedelta_methods(method):
    s = pd.Series(pd.timedelta_range("2000", periods=4))
    s.attrs = {"a": 1}
    result = method(s.dt)
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("add_categories", ["c"]),
        operator.methodcaller("as_ordered"),
        operator.methodcaller("as_unordered"),
        lambda x: getattr(x, "codes"),
        operator.methodcaller("remove_categories", "a"),
        operator.methodcaller("remove_unused_categories"),
        operator.methodcaller("rename_categories", {"a": "A", "b": "B"}),
        operator.methodcaller("reorder_categories", ["b", "a"]),
        operator.methodcaller("set_categories", ["A", "B"]),
    ],
)
@not_implemented_mark
def test_categorical_accessor(method):
    s = pd.Series(["a", "b"], dtype="category")
    s.attrs = {"a": 1}
    result = method(s.cat)
    assert result.attrs == {"a": 1}


# ----------------------------------------------------------------------------
# Groupby


@pytest.mark.parametrize(
    "obj", [pd.Series([0, 0]), pd.DataFrame({"A": [0, 1], "B": [1, 2]})]
)
@pytest.mark.parametrize(
    "method",
    [
        operator.methodcaller("sum"),
        lambda x: x.agg("sum"),
        lambda x: x.agg("mean"),
        lambda x: x.agg("median"),
    ],
)
def test_groupby_finalize(obj, method):
    obj.attrs = {"a": 1}
    result = method(obj.groupby([0, 0]))
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize(
    "obj", [pd.Series([0, 0]), pd.DataFrame({"A": [0, 1], "B": [1, 2]})]
)
@pytest.mark.parametrize(
    "method",
    [
        lambda x: x.agg(["sum", "count"]),
        lambda x: x.transform(lambda y: y),
        lambda x: x.apply(lambda y: y),
        lambda x: x.agg("std"),
        lambda x: x.agg("var"),
        lambda x: x.agg("sem"),
        lambda x: x.agg("size"),
        lambda x: x.agg("ohlc"),
        lambda x: x.agg("describe"),
    ],
)
@not_implemented_mark
def test_groupby_finalize_not_implemented(obj, method):
    obj.attrs = {"a": 1}
    result = method(obj.groupby([0, 0]))
    assert result.attrs == {"a": 1}


def test_finalize_frame_series_name():
    # https://github.com/pandas-dev/pandas/pull/37186/files#r506978889
    # ensure we don't copy the column `name` to the Series.
    df = pd.DataFrame({"name": [1, 2]})
    result = pd.Series([1, 2]).__finalize__(df)
    assert result.name is None
