from datetime import datetime
import re

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Series,
    _testing as tm,
)

# --------------------------------------------------------------------------------------
# str.contains
# --------------------------------------------------------------------------------------


def test_contains(any_string_dtype):
    values = np.array(
        ["foo", np.nan, "fooommm__foo", "mmm_", "foommm[_]+bar"], dtype=np.object_
    )
    values = Series(values, dtype=any_string_dtype)
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series(
        np.array([False, np.nan, True, True, False], dtype=np.object_),
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, regex=False)
    expected = Series(
        np.array([False, np.nan, False, False, True], dtype=np.object_),
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=object),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive using regex
    values = Series(
        np.array(["Foo", "xYz", "fOOomMm__fOo", "MMM_"], dtype=object),
        dtype=any_string_dtype,
    )
    result = values.str.contains("FOO|mmm", case=False)
    expected = Series(np.array([True, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # case insensitive without regex
    result = values.str.contains("foo", regex=False, case=False)
    expected = Series(np.array([True, False, True, False]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    # unicode
    values = Series(
        np.array(["foo", np.nan, "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    pat = "mmm[_]+"

    result = values.str.contains(pat)
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series(
        np.array([False, np.nan, True, True], dtype=np.object_), dtype=expected_dtype
    )
    tm.assert_series_equal(result, expected)

    result = values.str.contains(pat, na=False)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        np.array(["foo", "xyz", "fooommm__foo", "mmm_"], dtype=np.object_),
        dtype=any_string_dtype,
    )
    result = values.str.contains(pat)
    expected = Series(np.array([False, False, True, True]), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_contains_object_mixed():
    mixed = Series(
        np.array(
            ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
            dtype=object,
        )
    )
    result = mixed.str.contains("o")
    expected = Series(
        np.array(
            [False, np.nan, False, np.nan, np.nan, True, np.nan, np.nan, np.nan],
            dtype=np.object_,
        )
    )
    tm.assert_series_equal(result, expected)


def test_contains_na_kwarg_for_object_category():
    # gh 22158

    # na for category
    values = Series(["a", "b", "c", "a", np.nan], dtype="category")
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)

    # na for objects
    values = Series(["a", "b", "c", "a", np.nan])
    result = values.str.contains("a", na=True)
    expected = Series([True, False, False, True, True])
    tm.assert_series_equal(result, expected)

    result = values.str.contains("a", na=False)
    expected = Series([True, False, False, True, False])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "na, expected",
    [
        (None, pd.NA),
        (True, True),
        (False, False),
        (0, False),
        (3, True),
        (np.nan, pd.NA),
    ],
)
@pytest.mark.parametrize("regex", [True, False])
def test_contains_na_kwarg_for_nullable_string_dtype(
    nullable_string_dtype, na, expected, regex
):
    # https://github.com/pandas-dev/pandas/pull/41025#issuecomment-824062416

    values = Series(["a", "b", "c", "a", np.nan], dtype=nullable_string_dtype)
    result = values.str.contains("a", na=na, regex=regex)
    expected = Series([True, False, False, True, expected], dtype="boolean")
    tm.assert_series_equal(result, expected)


def test_contains_moar(any_string_dtype):
    # PR #1179
    s = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = s.str.contains("a")
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series(
        [False, False, False, True, True, False, np.nan, False, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("a", case=False)
    expected = Series(
        [True, False, False, True, True, False, np.nan, True, False, True],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("Aa")
    expected = Series(
        [False, False, False, True, False, False, np.nan, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba")
    expected = Series(
        [False, False, False, True, False, False, np.nan, False, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = s.str.contains("ba", case=False)
    expected = Series(
        [False, False, False, True, True, False, np.nan, True, False, False],
        dtype=expected_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_contains_nan(any_string_dtype):
    # PR #14171
    s = Series([np.nan, np.nan, np.nan], dtype=any_string_dtype)

    result = s.str.contains("foo", na=False)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series([False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo", na=True)
    expected = Series([True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo", na="foo")
    if any_string_dtype == "object":
        expected = Series(["foo", "foo", "foo"], dtype=np.object_)
    else:
        expected = Series([True, True, True], dtype="boolean")
    tm.assert_series_equal(result, expected)

    result = s.str.contains("foo")
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series([np.nan, np.nan, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.startswith
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [None, "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_startswith(dtype, null_value, na):
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.startswith("foo")
    exp = Series([False, np.nan, True, False, False, np.nan, True])
    tm.assert_series_equal(result, exp)

    result = values.str.startswith("foo", na=na)
    exp = Series([False, na, True, False, False, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=np.object_,
    )
    rs = Series(mixed).str.startswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, True, np.nan, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)


@pytest.mark.parametrize("na", [None, True, False])
def test_startswith_nullable_string_dtype(nullable_string_dtype, na):
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=nullable_string_dtype,
    )
    result = values.str.startswith("foo", na=na)
    exp = Series(
        [False, na, True, False, False, na, True, False, False], dtype="boolean"
    )
    tm.assert_series_equal(result, exp)

    result = values.str.startswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype="boolean"
    )
    tm.assert_series_equal(result, exp)


# --------------------------------------------------------------------------------------
# str.endswith
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [None, "category"])
@pytest.mark.parametrize("null_value", [None, np.nan, pd.NA])
@pytest.mark.parametrize("na", [True, False])
def test_endswith(dtype, null_value, na):
    # add category dtype parametrizations for GH-36241
    values = Series(
        ["om", null_value, "foo_nom", "nom", "bar_foo", null_value, "foo"],
        dtype=dtype,
    )

    result = values.str.endswith("foo")
    exp = Series([False, np.nan, False, False, True, np.nan, True])
    tm.assert_series_equal(result, exp)

    result = values.str.endswith("foo", na=na)
    exp = Series([False, na, False, False, True, na, True])
    tm.assert_series_equal(result, exp)

    # mixed
    mixed = np.array(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    rs = Series(mixed).str.endswith("f")
    xp = Series([False, np.nan, False, np.nan, np.nan, False, np.nan, np.nan, np.nan])
    tm.assert_series_equal(rs, xp)


@pytest.mark.parametrize("na", [None, True, False])
def test_endswith_nullable_string_dtype(nullable_string_dtype, na):
    values = Series(
        ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
        dtype=nullable_string_dtype,
    )
    result = values.str.endswith("foo", na=na)
    exp = Series(
        [False, na, False, False, True, na, True, False, False], dtype="boolean"
    )
    tm.assert_series_equal(result, exp)

    result = values.str.endswith("rege.", na=na)
    exp = Series(
        [False, na, False, False, False, na, False, False, True], dtype="boolean"
    )
    tm.assert_series_equal(result, exp)


# --------------------------------------------------------------------------------------
# str.replace
# --------------------------------------------------------------------------------------


def test_replace(any_string_dtype):
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    result = ser.str.replace("BAD[_]*", "", regex=True)
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_max_replacements(any_string_dtype):
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    result = ser.str.replace("BAD[_]*", "", n=1, regex=True)
    tm.assert_series_equal(result, expected)

    expected = Series(["foo__barBAD", np.nan], dtype=any_string_dtype)
    result = ser.str.replace("BAD", "", n=1, regex=False)
    tm.assert_series_equal(result, expected)


def test_replace_mixed_object():
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    result = Series(ser).str.replace("BAD[_]*", "", regex=True)
    expected = Series(["a", np.nan, "b", np.nan, np.nan, "foo", np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_replace_unicode(any_string_dtype):
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    result = ser.str.replace(r"(?<=\w),(?=\w)", ", ", flags=re.UNICODE, regex=True)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("repl", [None, 3, {"a": "b"}])
@pytest.mark.parametrize("data", [["a", "b", None], ["a", "b", "c", "ad"]])
def test_replace_wrong_repl_type_raises(any_string_dtype, index_or_series, repl, data):
    # https://github.com/pandas-dev/pandas/issues/13438
    msg = "repl must be a string or callable"
    obj = index_or_series(data, dtype=any_string_dtype)
    with pytest.raises(TypeError, match=msg):
        obj.str.replace("a", repl)


def test_replace_callable(any_string_dtype):
    # GH 15055
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with callable
    repl = lambda m: m.group(0).swapcase()
    result = ser.str.replace("[a-z][A-Z]{2}", repl, n=2, regex=True)
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "repl", [lambda: None, lambda m, x: None, lambda m, x, y=None: None]
)
def test_replace_callable_raises(any_string_dtype, repl):
    # GH 15055
    values = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with wrong number of arguments, raising an error
    msg = (
        r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
        r"(?(3)required )positional arguments?"
    )
    with pytest.raises(TypeError, match=msg):
        values.str.replace("a", repl)


def test_replace_callable_named_groups(any_string_dtype):
    # test regex named groups
    ser = Series(["Foo Bar Baz", np.nan], dtype=any_string_dtype)
    pat = r"(?P<first>\w+) (?P<middle>\w+) (?P<last>\w+)"
    repl = lambda m: m.group("middle").swapcase()
    result = ser.str.replace(pat, repl, regex=True)
    expected = Series(["bAR", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex(any_string_dtype):
    # GH 15446
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)

    # test with compiled regex
    pat = re.compile(r"BAD_*")
    result = ser.str.replace(pat, "", regex=True)
    expected = Series(["foobar", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace(pat, "", n=1, regex=True)
    expected = Series(["foobarBAD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_mixed_object():
    pat = re.compile(r"BAD_*")
    ser = Series(
        ["aBAD", np.nan, "bBAD", True, datetime.today(), "fooBAD", None, 1, 2.0]
    )
    result = Series(ser).str.replace(pat, "", regex=True)
    expected = Series(["a", np.nan, "b", np.nan, np.nan, "foo", np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_unicode(any_string_dtype):
    ser = Series([b"abcd,\xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    expected = Series([b"abcd, \xc3\xa0".decode("utf-8")], dtype=any_string_dtype)
    pat = re.compile(r"(?<=\w),(?=\w)", flags=re.UNICODE)
    result = ser.str.replace(pat, ", ")
    tm.assert_series_equal(result, expected)


def test_replace_compiled_regex_raises(any_string_dtype):
    # case and flags provided to str.replace will have no effect
    # and will produce warnings
    ser = Series(["fooBAD__barBAD__bad", np.nan], dtype=any_string_dtype)
    pat = re.compile(r"BAD_*")

    msg = "case and flags cannot be set when pat is a compiled regex"

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", flags=re.IGNORECASE)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=False)

    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", case=True)


def test_replace_compiled_regex_callable(any_string_dtype):
    # test with callable
    ser = Series(["fooBAD__barBAD", np.nan], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()
    pat = re.compile("[a-z][A-Z]{2}")
    result = ser.str.replace(pat, repl, n=2)
    expected = Series(["foObaD__baRbaD", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "regex,expected", [(True, ["bao", "bao", np.nan]), (False, ["bao", "foo", np.nan])]
)
def test_replace_literal(regex, expected, any_string_dtype):
    # GH16808 literal replace (regex=False vs regex=True)
    ser = Series(["f.o", "foo", np.nan], dtype=any_string_dtype)
    expected = Series(expected, dtype=any_string_dtype)
    result = ser.str.replace("f.", "ba", regex=regex)
    tm.assert_series_equal(result, expected)


def test_replace_literal_callable_raises(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()

    msg = "Cannot use a callable replacement when regex=False"
    with pytest.raises(ValueError, match=msg):
        ser.str.replace("abc", repl, regex=False)


def test_replace_literal_compiled_raises(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)
    pat = re.compile("[a-z][A-Z]{2}")

    msg = "Cannot use a compiled regex as replacement pattern with regex=False"
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, "", regex=False)


def test_replace_moar(any_string_dtype):
    # PR #1179
    ser = Series(
        ["A", "B", "C", "Aaba", "Baca", "", np.nan, "CABA", "dog", "cat"],
        dtype=any_string_dtype,
    )

    result = ser.str.replace("A", "YYY")
    expected = Series(
        ["YYY", "B", "C", "YYYaba", "Baca", "", np.nan, "CYYYBYYY", "dog", "cat"],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("A", "YYY", case=False)
    expected = Series(
        [
            "YYY",
            "B",
            "C",
            "YYYYYYbYYY",
            "BYYYcYYY",
            "",
            np.nan,
            "CYYYBYYY",
            "dog",
            "cYYYt",
        ],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("^.a|dog", "XX-XX ", case=False, regex=True)
    expected = Series(
        [
            "A",
            "B",
            "C",
            "XX-XX ba",
            "XX-XX ca",
            "",
            np.nan,
            "XX-XX BA",
            "XX-XX ",
            "XX-XX t",
        ],
        dtype=any_string_dtype,
    )
    tm.assert_series_equal(result, expected)


def test_replace_not_case_sensitive_not_regex(any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/41602
    ser = Series(["A.", "a.", "Ab", "ab", np.nan], dtype=any_string_dtype)

    result = ser.str.replace("a", "c", case=False, regex=False)
    expected = Series(["c.", "c.", "cb", "cb", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("a.", "c.", case=False, regex=False)
    expected = Series(["c.", "c.", "Ab", "ab", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_replace_regex_default_warning(any_string_dtype):
    # https://github.com/pandas-dev/pandas/pull/24809
    s = Series(["a", "b", "ac", np.nan, ""], dtype=any_string_dtype)
    msg = (
        "The default value of regex will change from True to False in a "
        "future version\\.$"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.str.replace("^.$", "a")
    expected = Series(["a", "a", "ac", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("regex", [True, False, None])
def test_replace_regex_single_character(regex, any_string_dtype):
    # https://github.com/pandas-dev/pandas/pull/24809

    # The current behavior is to treat single character patterns as literal strings,
    # even when ``regex`` is set to ``True``.

    s = Series(["a.b", ".", "b", np.nan, ""], dtype=any_string_dtype)

    if regex is None:
        msg = re.escape(
            "The default value of regex will change from True to False in a future "
            "version. In addition, single character regular expressions will *not* "
            "be treated as literal strings when regex=True."
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = s.str.replace(".", "a", regex=regex)
    else:
        result = s.str.replace(".", "a", regex=regex)

    expected = Series(["aab", "a", "b", np.nan, ""], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.match
# --------------------------------------------------------------------------------------


def test_match(any_string_dtype):
    # New match behavior introduced in 0.13
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"

    values = Series(["fooBAD__barBAD", np.nan, "foo"], dtype=any_string_dtype)
    result = values.str.match(".*(BAD[_]+).*(BAD)")
    expected = Series([True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match(".*BAD[_]+.*BAD")
    expected = Series([True, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("BAD[_]+.*BAD")
    expected = Series([False, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    values = Series(
        ["fooBAD__barBAD", "^BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = values.str.match("^BAD[_]+.*BAD")
    expected = Series([False, False, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = values.str.match("\\^BAD[_]+.*BAD")
    expected = Series([False, True, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_mixed_object():
    mixed = Series(
        [
            "aBAD_BAD",
            np.nan,
            "BAD_b_BAD",
            True,
            datetime.today(),
            "foo",
            None,
            1,
            2.0,
        ]
    )
    result = Series(mixed).str.match(".*(BAD[_]+).*(BAD)")
    expected = Series(
        [True, np.nan, True, np.nan, np.nan, False, np.nan, np.nan, np.nan]
    )
    assert isinstance(result, Series)
    tm.assert_series_equal(result, expected)


def test_match_na_kwarg(any_string_dtype):
    # GH #6609
    s = Series(["a", "b", np.nan], dtype=any_string_dtype)

    result = s.str.match("a", na=False)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series([True, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = s.str.match("a")
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series([True, False, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_match_case_kwarg(any_string_dtype):
    values = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    result = values.str.match("ab", case=False)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series([True, True, True, True], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.fullmatch
# --------------------------------------------------------------------------------------


def test_fullmatch(any_string_dtype):
    # GH 32806
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = ser.str.fullmatch(".*BAD[_]+.*BAD")
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series([True, False, np.nan, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_fullmatch_na_kwarg(any_string_dtype):
    ser = Series(
        ["fooBAD__barBAD", "BAD_BADleroybrown", np.nan, "foo"], dtype=any_string_dtype
    )
    result = ser.str.fullmatch(".*BAD[_]+.*BAD", na=False)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"
    expected = Series([True, False, False, False], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_fullmatch_case_kwarg(any_string_dtype):
    ser = Series(["ab", "AB", "abc", "ABC"], dtype=any_string_dtype)
    expected_dtype = np.bool_ if any_string_dtype == "object" else "boolean"

    expected = Series([True, False, False, False], dtype=expected_dtype)

    result = ser.str.fullmatch("ab", case=True)
    tm.assert_series_equal(result, expected)

    expected = Series([True, True, False, False], dtype=expected_dtype)

    result = ser.str.fullmatch("ab", case=False)
    tm.assert_series_equal(result, expected)

    result = ser.str.fullmatch("ab", flags=re.IGNORECASE)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.findall
# --------------------------------------------------------------------------------------


def test_findall(any_string_dtype):
    ser = Series(["fooBAD__barBAD", np.nan, "foo", "BAD"], dtype=any_string_dtype)
    result = ser.str.findall("BAD[_]*")
    expected = Series([["BAD__", "BAD"], np.nan, [], ["BAD"]])
    tm.assert_series_equal(result, expected)


def test_findall_mixed_object():
    ser = Series(
        [
            "fooBAD__barBAD",
            np.nan,
            "foo",
            True,
            datetime.today(),
            "BAD",
            None,
            1,
            2.0,
        ]
    )

    result = ser.str.findall("BAD[_]*")
    expected = Series(
        [
            ["BAD__", "BAD"],
            np.nan,
            [],
            np.nan,
            np.nan,
            ["BAD"],
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.find
# --------------------------------------------------------------------------------------


def test_find(any_string_dtype):
    ser = Series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF", "XXXX"], dtype=any_string_dtype
    )
    expected_dtype = np.int64 if any_string_dtype == "object" else "Int64"

    result = ser.str.find("EF")
    expected = Series([4, 3, 1, 0, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF")
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF") for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.find("EF", 3)
    expected = Series([4, 3, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF", 3)
    expected = Series([4, 5, 7, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF", 3) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.find("EF", 3, 6)
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.find("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)

    result = ser.str.rfind("EF", 3, 6)
    expected = Series([4, 3, -1, 4, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)
    expected = np.array([v.rfind("EF", 3, 6) for v in np.array(ser)], dtype=np.int64)
    tm.assert_numpy_array_equal(np.array(result, dtype=np.int64), expected)


def test_find_bad_arg_raises(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)
    with pytest.raises(TypeError, match="expected a string object, not int"):
        ser.str.find(0)

    with pytest.raises(TypeError, match="expected a string object, not int"):
        ser.str.rfind(0)


def test_find_nan(any_string_dtype):
    ser = Series(
        ["ABCDEFG", np.nan, "DEFGHIJEF", np.nan, "XXXX"], dtype=any_string_dtype
    )
    expected_dtype = np.float64 if any_string_dtype == "object" else "Int64"

    result = ser.str.find("EF")
    expected = Series([4, np.nan, 1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF")
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.find("EF", 3)
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF", 3)
    expected = Series([4, np.nan, 7, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.find("EF", 3, 6)
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rfind("EF", 3, 6)
    expected = Series([4, np.nan, -1, np.nan, -1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------
# str.translate
# --------------------------------------------------------------------------------------


def test_translate(index_or_series, any_string_dtype):
    obj = index_or_series(
        ["abcdefg", "abcc", "cdddfg", "cdefggg"], dtype=any_string_dtype
    )
    table = str.maketrans("abc", "cde")
    result = obj.str.translate(table)
    expected = index_or_series(
        ["cdedefg", "cdee", "edddfg", "edefggg"], dtype=any_string_dtype
    )
    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


def test_translate_mixed_object():
    # Series with non-string values
    s = Series(["a", "b", "c", 1.2])
    table = str.maketrans("abc", "cde")
    expected = Series(["c", "d", "e", np.nan])
    result = s.str.translate(table)
    tm.assert_series_equal(result, expected)


# --------------------------------------------------------------------------------------


def test_flags_kwarg(any_string_dtype):
    data = {
        "Dave": "dave@google.com",
        "Steve": "steve@gmail.com",
        "Rob": "rob@gmail.com",
        "Wes": np.nan,
    }
    data = Series(data, dtype=any_string_dtype)

    pat = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"

    result = data.str.extract(pat, flags=re.IGNORECASE, expand=True)
    assert result.iloc[0].tolist() == ["dave", "google", "com"]

    result = data.str.match(pat, flags=re.IGNORECASE)
    assert result[0]

    result = data.str.fullmatch(pat, flags=re.IGNORECASE)
    assert result[0]

    result = data.str.findall(pat, flags=re.IGNORECASE)
    assert result[0][0] == ("dave", "google", "com")

    result = data.str.count(pat, flags=re.IGNORECASE)
    assert result[0] == 1

    msg = "This pattern has match groups"
    with tm.assert_produces_warning(UserWarning, match=msg):
        result = data.str.contains(pat, flags=re.IGNORECASE)
    assert result[0]
