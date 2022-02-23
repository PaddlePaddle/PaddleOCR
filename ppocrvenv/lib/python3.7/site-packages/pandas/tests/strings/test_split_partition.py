from datetime import datetime

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
)


def test_split(any_string_dtype):
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = values.str.split("_")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    tm.assert_series_equal(result, exp)

    # more than one char
    values = Series(["a__b__c", "c__d__e", np.nan, "f__g__h"], dtype=any_string_dtype)
    result = values.str.split("__")
    tm.assert_series_equal(result, exp)

    result = values.str.split("__", expand=False)
    tm.assert_series_equal(result, exp)

    # regex split
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.split("[,_]")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    tm.assert_series_equal(result, exp)


def test_split_object_mixed():
    mixed = Series(["a_b_c", np.nan, "d_e_f", True, datetime.today(), None, 1, 2.0])
    result = mixed.str.split("_")
    exp = Series(
        [
            ["a", "b", "c"],
            np.nan,
            ["d", "e", "f"],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)

    result = mixed.str.split("_", expand=False)
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)


@pytest.mark.parametrize("method", ["split", "rsplit"])
def test_split_n(any_string_dtype, method):
    s = Series(["a b", pd.NA, "b c"], dtype=any_string_dtype)
    expected = Series([["a", "b"], pd.NA, ["b", "c"]])

    result = getattr(s.str, method)(" ", n=None)
    tm.assert_series_equal(result, expected)

    result = getattr(s.str, method)(" ", n=0)
    tm.assert_series_equal(result, expected)


def test_rsplit(any_string_dtype):
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = values.str.rsplit("_")
    exp = Series([["a", "b", "c"], ["c", "d", "e"], np.nan, ["f", "g", "h"]])
    tm.assert_series_equal(result, exp)

    # more than one char
    values = Series(["a__b__c", "c__d__e", np.nan, "f__g__h"], dtype=any_string_dtype)
    result = values.str.rsplit("__")
    tm.assert_series_equal(result, exp)

    result = values.str.rsplit("__", expand=False)
    tm.assert_series_equal(result, exp)

    # regex split is not supported by rsplit
    values = Series(["a,b_c", "c_d,e", np.nan, "f,g,h"], dtype=any_string_dtype)
    result = values.str.rsplit("[,_]")
    exp = Series([["a,b_c"], ["c_d,e"], np.nan, ["f,g,h"]])
    tm.assert_series_equal(result, exp)

    # setting max number of splits, make sure it's from reverse
    values = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = values.str.rsplit("_", n=1)
    exp = Series([["a_b", "c"], ["c_d", "e"], np.nan, ["f_g", "h"]])
    tm.assert_series_equal(result, exp)


def test_rsplit_object_mixed():
    # mixed
    mixed = Series(["a_b_c", np.nan, "d_e_f", True, datetime.today(), None, 1, 2.0])
    result = mixed.str.rsplit("_")
    exp = Series(
        [
            ["a", "b", "c"],
            np.nan,
            ["d", "e", "f"],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)

    result = mixed.str.rsplit("_", expand=False)
    assert isinstance(result, Series)
    tm.assert_almost_equal(result, exp)


def test_split_blank_string(any_string_dtype):
    # expand blank split GH 20067
    values = Series([""], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame([[]], dtype=any_string_dtype)  # NOTE: this is NOT an empty df
    tm.assert_frame_equal(result, exp)

    values = Series(["a b c", "a b", "", " "], name="test", dtype=any_string_dtype)
    result = values.str.split(expand=True)
    exp = DataFrame(
        [
            ["a", "b", "c"],
            ["a", "b", np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_split_noargs(any_string_dtype):
    # #1859
    s = Series(["Wes McKinney", "Travis  Oliphant"], dtype=any_string_dtype)
    result = s.str.split()
    expected = ["Travis", "Oliphant"]
    assert result[1] == expected
    result = s.str.rsplit()
    assert result[1] == expected


@pytest.mark.parametrize(
    "data, pat",
    [
        (["bd asdf jfg", "kjasdflqw asdfnfk"], None),
        (["bd asdf jfg", "kjasdflqw asdfnfk"], "asdf"),
        (["bd_asdf_jfg", "kjasdflqw_asdfnfk"], "_"),
    ],
)
def test_split_maxsplit(data, pat, any_string_dtype):
    # re.split 0, str.split -1
    s = Series(data, dtype=any_string_dtype)

    result = s.str.split(pat=pat, n=-1)
    xp = s.str.split(pat=pat)
    tm.assert_series_equal(result, xp)

    result = s.str.split(pat=pat, n=0)
    tm.assert_series_equal(result, xp)


@pytest.mark.parametrize(
    "data, pat, expected",
    [
        (
            ["split once", "split once too!"],
            None,
            Series({0: ["split", "once"], 1: ["split", "once too!"]}),
        ),
        (
            ["split_once", "split_once_too!"],
            "_",
            Series({0: ["split", "once"], 1: ["split", "once_too!"]}),
        ),
    ],
)
def test_split_no_pat_with_nonzero_n(data, pat, expected, any_string_dtype):
    s = Series(data, dtype=any_string_dtype)
    result = s.str.split(pat=pat, n=1)
    tm.assert_series_equal(expected, result, check_index_type=False)


def test_split_to_dataframe(any_string_dtype):
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)})
    tm.assert_frame_equal(result, exp)

    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    s = Series(
        ["some_unequal_splits", "one_of_these_things_is_not"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {
            0: ["some", "one"],
            1: ["unequal", "of"],
            2: ["splits", "these"],
            3: [np.nan, "things"],
            4: [np.nan, "is"],
            5: [np.nan, "not"],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.split("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    with pytest.raises(ValueError, match="expand must be"):
        s.str.split("_", expand="not_a_boolean")


def test_split_to_multiindex_expand():
    # https://github.com/pandas-dev/pandas/issues/23677

    idx = Index(["nosplit", "alsonosplit", np.nan])
    result = idx.str.split("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1

    idx = Index(["some_equal_splits", "with_no_nans", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "equal", "splits"),
            ("with", "no", "nans"),
            [np.nan, np.nan, np.nan],
            [None, None, None],
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3

    idx = Index(["some_unequal_splits", "one_of_these_things_is_not", np.nan, None])
    result = idx.str.split("_", expand=True)
    exp = MultiIndex.from_tuples(
        [
            ("some", "unequal", "splits", np.nan, np.nan, np.nan),
            ("one", "of", "these", "things", "is", "not"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
            (None, None, None, None, None, None),
        ]
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 6

    with pytest.raises(ValueError, match="expand must be"):
        idx.str.split("_", expand="not_a_boolean")


def test_rsplit_to_dataframe_expand(any_string_dtype):
    s = Series(["nosplit", "alsonosplit"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame({0: Series(["nosplit", "alsonosplit"])}, dtype=any_string_dtype)
    tm.assert_frame_equal(result, exp)

    s = Series(["some_equal_splits", "with_no_nans"], dtype=any_string_dtype)
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=2)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["equal", "no"], 2: ["splits", "nans"]},
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)

    result = s.str.rsplit("_", expand=True, n=1)
    exp = DataFrame(
        {0: ["some_equal", "with_no"], 1: ["splits", "nans"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)

    s = Series(
        ["some_splits", "with_index"], index=["preserve", "me"], dtype=any_string_dtype
    )
    result = s.str.rsplit("_", expand=True)
    exp = DataFrame(
        {0: ["some", "with"], 1: ["splits", "index"]},
        index=["preserve", "me"],
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, exp)


def test_rsplit_to_multiindex_expand():
    idx = Index(["nosplit", "alsonosplit"])
    result = idx.str.rsplit("_", expand=True)
    exp = idx
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1

    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True)
    exp = MultiIndex.from_tuples([("some", "equal", "splits"), ("with", "no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 3

    idx = Index(["some_equal_splits", "with_no_nans"])
    result = idx.str.rsplit("_", expand=True, n=1)
    exp = MultiIndex.from_tuples([("some_equal", "splits"), ("with_no", "nans")])
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 2


def test_split_nan_expand(any_string_dtype):
    # gh-18450
    s = Series(["foo,bar,baz", np.nan], dtype=any_string_dtype)
    result = s.str.split(",", expand=True)
    exp = DataFrame(
        [["foo", "bar", "baz"], [np.nan, np.nan, np.nan]], dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, exp)

    # check that these are actually np.nan/pd.NA and not None
    # TODO see GH 18463
    # tm.assert_frame_equal does not differentiate
    if any_string_dtype == "object":
        assert all(np.isnan(x) for x in result.iloc[1])
    else:
        assert all(x is pd.NA for x in result.iloc[1])


def test_split_with_name(any_string_dtype):
    # GH 12617

    # should preserve name
    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    res = s.str.split(",")
    exp = Series([["a", "b"], ["c", "d"]], name="xxx")
    tm.assert_series_equal(res, exp)

    res = s.str.split(",", expand=True)
    exp = DataFrame([["a", "b"], ["c", "d"]], dtype=any_string_dtype)
    tm.assert_frame_equal(res, exp)

    idx = Index(["a,b", "c,d"], name="xxx")
    res = idx.str.split(",")
    exp = Index([["a", "b"], ["c", "d"]], name="xxx")
    assert res.nlevels == 1
    tm.assert_index_equal(res, exp)

    res = idx.str.split(",", expand=True)
    exp = MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    assert res.nlevels == 2
    tm.assert_index_equal(res, exp)


def test_partition_series(any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/23558

    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)

    result = s.str.partition("_", expand=False)
    expected = Series(
        [("a", "_", "b_c"), ("c", "_", "d_e"), np.nan, ("f", "_", "g_h"), None]
    )
    tm.assert_series_equal(result, expected)

    result = s.str.rpartition("_", expand=False)
    expected = Series(
        [("a_b", "_", "c"), ("c_d", "_", "e"), np.nan, ("f_g", "_", "h"), None]
    )
    tm.assert_series_equal(result, expected)

    # more than one char
    s = Series(["a__b__c", "c__d__e", np.nan, "f__g__h", None])
    result = s.str.partition("__", expand=False)
    expected = Series(
        [
            ("a", "__", "b__c"),
            ("c", "__", "d__e"),
            np.nan,
            ("f", "__", "g__h"),
            None,
        ],
    )
    tm.assert_series_equal(result, expected)

    result = s.str.rpartition("__", expand=False)
    expected = Series(
        [
            ("a__b", "__", "c"),
            ("c__d", "__", "e"),
            np.nan,
            ("f__g", "__", "h"),
            None,
        ],
    )
    tm.assert_series_equal(result, expected)

    # None
    s = Series(["a b c", "c d e", np.nan, "f g h", None], dtype=any_string_dtype)
    result = s.str.partition(expand=False)
    expected = Series(
        [("a", " ", "b c"), ("c", " ", "d e"), np.nan, ("f", " ", "g h"), None]
    )
    tm.assert_series_equal(result, expected)

    result = s.str.rpartition(expand=False)
    expected = Series(
        [("a b", " ", "c"), ("c d", " ", "e"), np.nan, ("f g", " ", "h"), None]
    )
    tm.assert_series_equal(result, expected)

    # Not split
    s = Series(["abc", "cde", np.nan, "fgh", None], dtype=any_string_dtype)
    result = s.str.partition("_", expand=False)
    expected = Series([("abc", "", ""), ("cde", "", ""), np.nan, ("fgh", "", ""), None])
    tm.assert_series_equal(result, expected)

    result = s.str.rpartition("_", expand=False)
    expected = Series([("", "", "abc"), ("", "", "cde"), np.nan, ("", "", "fgh"), None])
    tm.assert_series_equal(result, expected)

    # unicode
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    result = s.str.partition("_", expand=False)
    expected = Series([("a", "_", "b_c"), ("c", "_", "d_e"), np.nan, ("f", "_", "g_h")])
    tm.assert_series_equal(result, expected)

    result = s.str.rpartition("_", expand=False)
    expected = Series([("a_b", "_", "c"), ("c_d", "_", "e"), np.nan, ("f_g", "_", "h")])
    tm.assert_series_equal(result, expected)

    # compare to standard lib
    s = Series(["A_B_C", "B_C_D", "E_F_G", "EFGHEF"], dtype=any_string_dtype)
    result = s.str.partition("_", expand=False).tolist()
    assert result == [v.partition("_") for v in s]
    result = s.str.rpartition("_", expand=False).tolist()
    assert result == [v.rpartition("_") for v in s]


def test_partition_index():
    # https://github.com/pandas-dev/pandas/issues/23558

    values = Index(["a_b_c", "c_d_e", "f_g_h", np.nan, None])

    result = values.str.partition("_", expand=False)
    exp = Index(
        np.array(
            [("a", "_", "b_c"), ("c", "_", "d_e"), ("f", "_", "g_h"), np.nan, None],
            dtype=object,
        )
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1

    result = values.str.rpartition("_", expand=False)
    exp = Index(
        np.array(
            [("a_b", "_", "c"), ("c_d", "_", "e"), ("f_g", "_", "h"), np.nan, None],
            dtype=object,
        )
    )
    tm.assert_index_equal(result, exp)
    assert result.nlevels == 1

    result = values.str.partition("_")
    exp = Index(
        [
            ("a", "_", "b_c"),
            ("c", "_", "d_e"),
            ("f", "_", "g_h"),
            (np.nan, np.nan, np.nan),
            (None, None, None),
        ]
    )
    tm.assert_index_equal(result, exp)
    assert isinstance(result, MultiIndex)
    assert result.nlevels == 3

    result = values.str.rpartition("_")
    exp = Index(
        [
            ("a_b", "_", "c"),
            ("c_d", "_", "e"),
            ("f_g", "_", "h"),
            (np.nan, np.nan, np.nan),
            (None, None, None),
        ]
    )
    tm.assert_index_equal(result, exp)
    assert isinstance(result, MultiIndex)
    assert result.nlevels == 3


def test_partition_to_dataframe(any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/23558

    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = s.str.partition("_")
    expected = DataFrame(
        {
            0: ["a", "c", np.nan, "f", None],
            1: ["_", "_", np.nan, "_", None],
            2: ["b_c", "d_e", np.nan, "g_h", None],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    result = s.str.rpartition("_")
    expected = DataFrame(
        {
            0: ["a_b", "c_d", np.nan, "f_g", None],
            1: ["_", "_", np.nan, "_", None],
            2: ["c", "e", np.nan, "h", None],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h", None], dtype=any_string_dtype)
    result = s.str.partition("_", expand=True)
    expected = DataFrame(
        {
            0: ["a", "c", np.nan, "f", None],
            1: ["_", "_", np.nan, "_", None],
            2: ["b_c", "d_e", np.nan, "g_h", None],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)

    result = s.str.rpartition("_", expand=True)
    expected = DataFrame(
        {
            0: ["a_b", "c_d", np.nan, "f_g", None],
            1: ["_", "_", np.nan, "_", None],
            2: ["c", "e", np.nan, "h", None],
        },
        dtype=any_string_dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_partition_with_name(any_string_dtype):
    # GH 12617

    s = Series(["a,b", "c,d"], name="xxx", dtype=any_string_dtype)
    result = s.str.partition(",")
    expected = DataFrame(
        {0: ["a", "c"], 1: [",", ","], 2: ["b", "d"]}, dtype=any_string_dtype
    )
    tm.assert_frame_equal(result, expected)

    # should preserve name
    result = s.str.partition(",", expand=False)
    expected = Series([("a", ",", "b"), ("c", ",", "d")], name="xxx")
    tm.assert_series_equal(result, expected)


def test_partition_index_with_name():
    idx = Index(["a,b", "c,d"], name="xxx")
    result = idx.str.partition(",")
    expected = MultiIndex.from_tuples([("a", ",", "b"), ("c", ",", "d")])
    assert result.nlevels == 3
    tm.assert_index_equal(result, expected)

    # should preserve name
    result = idx.str.partition(",", expand=False)
    expected = Index(np.array([("a", ",", "b"), ("c", ",", "d")]), name="xxx")
    assert result.nlevels == 1
    tm.assert_index_equal(result, expected)


def test_partition_sep_kwarg(any_string_dtype):
    # GH 22676; depr kwarg "pat" in favor of "sep"
    s = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)

    expected = s.str.partition(sep="_")
    result = s.str.partition("_")
    tm.assert_frame_equal(result, expected)

    expected = s.str.rpartition(sep="_")
    result = s.str.rpartition("_")
    tm.assert_frame_equal(result, expected)


def test_get():
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"])
    result = ser.str.split("_").str.get(1)
    expected = Series(["b", "d", np.nan, "g"])
    tm.assert_series_equal(result, expected)


def test_get_mixed_object():
    ser = Series(["a_b_c", np.nan, "c_d_e", True, datetime.today(), None, 1, 2.0])
    result = ser.str.split("_").str.get(1)
    expected = Series(["b", np.nan, "d", np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_get_bounds():
    ser = Series(["1_2_3_4_5", "6_7_8_9_10", "11_12"])

    # positive index
    result = ser.str.split("_").str.get(2)
    expected = Series(["3", "8", np.nan])
    tm.assert_series_equal(result, expected)

    # negative index
    result = ser.str.split("_").str.get(-3)
    expected = Series(["3", "8", np.nan])
    tm.assert_series_equal(result, expected)


def test_get_complex():
    # GH 20671, getting value not in dict raising `KeyError`
    ser = Series([(1, 2, 3), [1, 2, 3], {1, 2, 3}, {1: "a", 2: "b", 3: "c"}])

    result = ser.str.get(1)
    expected = Series([2, 2, np.nan, "a"])
    tm.assert_series_equal(result, expected)

    result = ser.str.get(-1)
    expected = Series([3, 3, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("to_type", [tuple, list, np.array])
def test_get_complex_nested(to_type):
    ser = Series([to_type([to_type([1, 2])])])

    result = ser.str.get(0)
    expected = Series([to_type([1, 2])])
    tm.assert_series_equal(result, expected)

    result = ser.str.get(1)
    expected = Series([np.nan])
    tm.assert_series_equal(result, expected)


def test_get_strings(any_string_dtype):
    ser = Series(["a", "ab", np.nan, "abc"], dtype=any_string_dtype)
    result = ser.str.get(2)
    expected = Series([np.nan, np.nan, np.nan, "c"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)
