from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
)
import pandas._testing as tm


def assert_series_or_index_equal(left, right):
    if isinstance(left, Series):
        tm.assert_series_equal(left, right)
    else:  # Index
        tm.assert_index_equal(left, right)


def test_iter():
    # GH3638
    strs = "google", "wikimedia", "wikipedia", "wikitravel"
    ser = Series(strs)

    with tm.assert_produces_warning(FutureWarning):
        for s in ser.str:
            # iter must yield a Series
            assert isinstance(s, Series)

            # indices of each yielded Series should be equal to the index of
            # the original Series
            tm.assert_index_equal(s.index, ser.index)

            for el in s:
                # each element of the series is either a basestring/str or nan
                assert isinstance(el, str) or isna(el)

    # desired behavior is to iterate until everything would be nan on the
    # next iter so make sure the last element of the iterator was 'l' in
    # this case since 'wikitravel' is the longest string
    assert s.dropna().values.item() == "l"


def test_iter_empty(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)

    i, s = 100, 1

    with tm.assert_produces_warning(FutureWarning):
        for i, s in enumerate(ser.str):
            pass

    # nothing to iterate over so nothing defined values should remain
    # unchanged
    assert i == 100
    assert s == 1


def test_iter_single_element(any_string_dtype):
    ser = Series(["a"], dtype=any_string_dtype)

    with tm.assert_produces_warning(FutureWarning):
        for i, s in enumerate(ser.str):
            pass

    assert not i
    tm.assert_series_equal(ser, s)


def test_iter_object_try_string():
    ser = Series(
        [
            slice(None, np.random.randint(10), np.random.randint(10, 20))
            for _ in range(4)
        ]
    )

    i, s = 100, "h"

    with tm.assert_produces_warning(FutureWarning):
        for i, s in enumerate(ser.str):
            pass

    assert i == 100
    assert s == "h"


# test integer/float dtypes (inferred by constructor) and mixed


def test_count(any_string_dtype):
    ser = Series(["foo", "foofoo", np.nan, "foooofooofommmfoo"], dtype=any_string_dtype)
    result = ser.str.count("f[o]+")
    expected_dtype = np.float64 if any_string_dtype == "object" else "Int64"
    expected = Series([1, 2, np.nan, 4], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_count_mixed_object():
    ser = Series(
        ["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0],
        dtype=object,
    )
    result = ser.str.count("a")
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_repeat(any_string_dtype):
    ser = Series(["a", "b", np.nan, "c", np.nan, "d"], dtype=any_string_dtype)

    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", "bbb", np.nan, "ccc", np.nan, "ddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.repeat([1, 2, 3, 4, 5, 6])
    expected = Series(
        ["a", "bb", np.nan, "cccc", np.nan, "dddddd"], dtype=any_string_dtype
    )
    tm.assert_series_equal(result, expected)


def test_repeat_mixed_object():
    ser = Series(["a", np.nan, "b", True, datetime.today(), "foo", None, 1, 2.0])
    result = ser.str.repeat(3)
    expected = Series(
        ["aaa", np.nan, "bbb", np.nan, np.nan, "foofoofoo", np.nan, np.nan, np.nan]
    )
    tm.assert_series_equal(result, expected)


def test_repeat_with_null(any_string_dtype):
    # GH: 31632
    ser = Series(["a", None], dtype=any_string_dtype)
    result = ser.str.repeat([3, 4])
    expected = Series(["aaa", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    ser = Series(["a", "b"], dtype=any_string_dtype)
    result = ser.str.repeat([3, None])
    expected = Series(["aaa", np.nan], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_empty_str_methods(any_string_dtype):
    empty_str = empty = Series(dtype=any_string_dtype)
    if any_string_dtype == "object":
        empty_int = Series(dtype="int64")
        empty_bool = Series(dtype=bool)
    else:
        empty_int = Series(dtype="Int64")
        empty_bool = Series(dtype="boolean")
    empty_object = Series(dtype=object)
    empty_bytes = Series(dtype=object)
    empty_df = DataFrame()

    # GH7241
    # (extract) on empty series

    tm.assert_series_equal(empty_str, empty.str.cat(empty))
    assert "" == empty.str.cat()
    tm.assert_series_equal(empty_str, empty.str.title())
    tm.assert_series_equal(empty_int, empty.str.count("a"))
    tm.assert_series_equal(empty_bool, empty.str.contains("a"))
    tm.assert_series_equal(empty_bool, empty.str.startswith("a"))
    tm.assert_series_equal(empty_bool, empty.str.endswith("a"))
    tm.assert_series_equal(empty_str, empty.str.lower())
    tm.assert_series_equal(empty_str, empty.str.upper())
    tm.assert_series_equal(empty_str, empty.str.replace("a", "b"))
    tm.assert_series_equal(empty_str, empty.str.repeat(3))
    tm.assert_series_equal(empty_bool, empty.str.match("^a"))
    tm.assert_frame_equal(
        DataFrame(columns=[0], dtype=any_string_dtype),
        empty.str.extract("()", expand=True),
    )
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=True),
    )
    tm.assert_series_equal(empty_str, empty.str.extract("()", expand=False))
    tm.assert_frame_equal(
        DataFrame(columns=[0, 1], dtype=any_string_dtype),
        empty.str.extract("()()", expand=False),
    )
    tm.assert_frame_equal(empty_df, empty.str.get_dummies())
    tm.assert_series_equal(empty_str, empty_str.str.join(""))
    tm.assert_series_equal(empty_int, empty.str.len())
    tm.assert_series_equal(empty_object, empty_str.str.findall("a"))
    tm.assert_series_equal(empty_int, empty.str.find("a"))
    tm.assert_series_equal(empty_int, empty.str.rfind("a"))
    tm.assert_series_equal(empty_str, empty.str.pad(42))
    tm.assert_series_equal(empty_str, empty.str.center(42))
    tm.assert_series_equal(empty_object, empty.str.split("a"))
    tm.assert_series_equal(empty_object, empty.str.rsplit("a"))
    tm.assert_series_equal(empty_object, empty.str.partition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.partition("a"))
    tm.assert_series_equal(empty_object, empty.str.rpartition("a", expand=False))
    tm.assert_frame_equal(empty_df, empty.str.rpartition("a"))
    tm.assert_series_equal(empty_str, empty.str.slice(stop=1))
    tm.assert_series_equal(empty_str, empty.str.slice(step=1))
    tm.assert_series_equal(empty_str, empty.str.strip())
    tm.assert_series_equal(empty_str, empty.str.lstrip())
    tm.assert_series_equal(empty_str, empty.str.rstrip())
    tm.assert_series_equal(empty_str, empty.str.wrap(42))
    tm.assert_series_equal(empty_str, empty.str.get(0))
    tm.assert_series_equal(empty_object, empty_bytes.str.decode("ascii"))
    tm.assert_series_equal(empty_bytes, empty.str.encode("ascii"))
    # ismethods should always return boolean (GH 29624)
    tm.assert_series_equal(empty_bool, empty.str.isalnum())
    tm.assert_series_equal(empty_bool, empty.str.isalpha())
    tm.assert_series_equal(empty_bool, empty.str.isdigit())
    tm.assert_series_equal(empty_bool, empty.str.isspace())
    tm.assert_series_equal(empty_bool, empty.str.islower())
    tm.assert_series_equal(empty_bool, empty.str.isupper())
    tm.assert_series_equal(empty_bool, empty.str.istitle())
    tm.assert_series_equal(empty_bool, empty.str.isnumeric())
    tm.assert_series_equal(empty_bool, empty.str.isdecimal())
    tm.assert_series_equal(empty_str, empty.str.capitalize())
    tm.assert_series_equal(empty_str, empty.str.swapcase())
    tm.assert_series_equal(empty_str, empty.str.normalize("NFC"))

    table = str.maketrans("a", "b")
    tm.assert_series_equal(empty_str, empty.str.translate(table))


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isalnum", [True, True, True, True, True, False, True, True, False, False]),
        ("isalpha", [True, True, True, False, False, False, True, False, False, False]),
        (
            "isdigit",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isnumeric",
            [False, False, False, True, False, False, False, True, False, False],
        ),
        (
            "isspace",
            [False, False, False, False, False, False, False, False, False, True],
        ),
        (
            "islower",
            [False, True, False, False, False, False, False, False, False, False],
        ),
        (
            "isupper",
            [True, False, False, False, True, False, True, False, False, False],
        ),
        (
            "istitle",
            [True, False, True, False, True, False, False, False, False, False],
        ),
    ],
)
def test_ismethods(method, expected, any_string_dtype):
    ser = Series(
        ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "], dtype=any_string_dtype
    )
    expected_dtype = "bool" if any_string_dtype == "object" else "boolean"
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, True, True, False, True, True, False]),
        ("isdecimal", [False, True, False, False, False, True, False]),
    ],
)
def test_isnumeric_unicode(method, expected, any_string_dtype):
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3
    ser = Series(["A", "3", "¼", "★", "፸", "３", "four"], dtype=any_string_dtype)
    expected_dtype = "bool" if any_string_dtype == "object" else "boolean"
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)() for item in ser]
    assert list(result) == expected


@pytest.mark.parametrize(
    "method, expected",
    [
        ("isnumeric", [False, np.nan, True, False, np.nan, True, False]),
        ("isdecimal", [False, np.nan, False, False, np.nan, True, False]),
    ],
)
def test_isnumeric_unicode_missing(method, expected, any_string_dtype):
    values = ["A", np.nan, "¼", "★", np.nan, "３", "four"]
    ser = Series(values, dtype=any_string_dtype)
    expected_dtype = "object" if any_string_dtype == "object" else "boolean"
    expected = Series(expected, dtype=expected_dtype)
    result = getattr(ser.str, method)()
    tm.assert_series_equal(result, expected)


def test_spilt_join_roundtrip(any_string_dtype):
    ser = Series(["a_b_c", "c_d_e", np.nan, "f_g_h"], dtype=any_string_dtype)
    result = ser.str.split("_").str.join("_")
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)


def test_spilt_join_roundtrip_mixed_object():
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.split("_").str.join("_")
    expected = Series(
        ["a_b", np.nan, "asdf_cas_asdf", np.nan, np.nan, "foo", np.nan, np.nan, np.nan]
    )
    tm.assert_series_equal(result, expected)


def test_len(any_string_dtype):
    ser = Series(
        ["foo", "fooo", "fooooo", np.nan, "fooooooo", "foo\n", "あ"],
        dtype=any_string_dtype,
    )
    result = ser.str.len()
    expected_dtype = "float64" if any_string_dtype == "object" else "Int64"
    expected = Series([3, 4, 6, np.nan, 8, 4, 1], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_len_mixed():
    ser = Series(
        ["a_b", np.nan, "asdf_cas_asdf", True, datetime.today(), "foo", None, 1, 2.0]
    )
    result = ser.str.len()
    expected = Series([3, np.nan, 13, np.nan, np.nan, 3, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method,sub,start,end,expected",
    [
        ("index", "EF", None, None, [4, 3, 1, 0]),
        ("rindex", "EF", None, None, [4, 5, 7, 4]),
        ("index", "EF", 3, None, [4, 3, 7, 4]),
        ("rindex", "EF", 3, None, [4, 5, 7, 4]),
        ("index", "E", 4, 8, [4, 5, 7, 4]),
        ("rindex", "E", 0, 5, [4, 3, 1, 4]),
    ],
)
def test_index(method, sub, start, end, index_or_series, any_string_dtype, expected):
    if index_or_series is Index and not any_string_dtype == "object":
        pytest.skip("Index cannot yet be backed by a StringArray/ArrowStringArray")

    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    expected_dtype = np.int64 if any_string_dtype == "object" else "Int64"
    expected = index_or_series(expected, dtype=expected_dtype)

    result = getattr(obj.str, method)(sub, start, end)

    if index_or_series is Series:
        tm.assert_series_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)

    # compare with standard library
    expected = [getattr(item, method)(sub, start, end) for item in obj]
    assert list(result) == expected


def test_index_not_found_raises(index_or_series, any_string_dtype):
    obj = index_or_series(
        ["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"], dtype=any_string_dtype
    )
    with pytest.raises(ValueError, match="substring not found"):
        obj.str.index("DE")


def test_index_wrong_type_raises(index_or_series, any_string_dtype):
    obj = index_or_series([], dtype=any_string_dtype)
    msg = "expected a string object, not int"

    with pytest.raises(TypeError, match=msg):
        obj.str.index(0)

    with pytest.raises(TypeError, match=msg):
        obj.str.rindex(0)


def test_index_missing(any_string_dtype):
    ser = Series(["abcb", "ab", "bcbe", np.nan], dtype=any_string_dtype)
    expected_dtype = np.float64 if any_string_dtype == "object" else "Int64"

    result = ser.str.index("b")
    expected = Series([1, 1, 0, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rindex("b")
    expected = Series([3, 1, 2, np.nan], dtype=expected_dtype)
    tm.assert_series_equal(result, expected)


def test_pipe_failures(any_string_dtype):
    # #2119
    ser = Series(["A|B|C"], dtype=any_string_dtype)

    result = ser.str.split("|")
    expected = Series([["A", "B", "C"]], dtype=object)
    tm.assert_series_equal(result, expected)

    result = ser.str.replace("|", " ", regex=False)
    expected = Series(["A B C"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", "bar", np.nan, "baz"]),
        (0, 3, -1, ["", "", np.nan, ""]),
        (None, None, -1, ["owtoofaa", "owtrabaa", np.nan, "xuqzabaa"]),
        (3, 10, 2, ["oto", "ato", np.nan, "aqx"]),
        (3, 0, -1, ["ofa", "aba", np.nan, "aba"]),
    ],
)
def test_slice(start, stop, step, expected, any_string_dtype):
    ser = Series(["aafootwo", "aabartwo", np.nan, "aabazqux"], dtype=any_string_dtype)
    result = ser.str.slice(start, stop, step)
    expected = Series(expected, dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (2, 5, None, ["foo", np.nan, "bar", np.nan, np.nan, np.nan, np.nan, np.nan]),
        (4, 1, -1, ["oof", np.nan, "rab", np.nan, np.nan, np.nan, np.nan, np.nan]),
    ],
)
def test_slice_mixed_object(start, stop, step, expected):
    ser = Series(["aafootwo", np.nan, "aabartwo", True, datetime.today(), None, 1, 2.0])
    result = ser.str.slice(start, stop, step)
    expected = Series(expected)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start,stop,repl,expected",
    [
        (2, 3, None, ["shrt", "a it longer", "evnlongerthanthat", "", np.nan]),
        (2, 3, "z", ["shzrt", "a zit longer", "evznlongerthanthat", "z", np.nan]),
        (2, 2, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (2, 1, "z", ["shzort", "a zbit longer", "evzenlongerthanthat", "z", np.nan]),
        (-1, None, "z", ["shorz", "a bit longez", "evenlongerthanthaz", "z", np.nan]),
        (None, -2, "z", ["zrt", "zer", "zat", "z", np.nan]),
        (6, 8, "z", ["shortz", "a bit znger", "evenlozerthanthat", "z", np.nan]),
        (-10, 3, "z", ["zrt", "a zit longer", "evenlongzerthanthat", "z", np.nan]),
    ],
)
def test_slice_replace(start, stop, repl, expected, any_string_dtype):
    ser = Series(
        ["short", "a bit longer", "evenlongerthanthat", "", np.nan],
        dtype=any_string_dtype,
    )
    expected = Series(expected, dtype=any_string_dtype)
    result = ser.str.slice_replace(start, stop, repl)
    tm.assert_series_equal(result, expected)


def test_strip_lstrip_rstrip(any_string_dtype):
    ser = Series(["  aa   ", " bb \n", np.nan, "cc  "], dtype=any_string_dtype)

    result = ser.str.strip()
    expected = Series(["aa", "bb", np.nan, "cc"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.lstrip()
    expected = Series(["aa   ", "bb \n", np.nan, "cc  "], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rstrip()
    expected = Series(["  aa", " bb", np.nan, "cc"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_strip_lstrip_rstrip_mixed_object():
    ser = Series(["  aa  ", np.nan, " bb \t\n", True, datetime.today(), None, 1, 2.0])

    result = ser.str.strip()
    expected = Series(["aa", np.nan, "bb", np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

    result = ser.str.lstrip()
    expected = Series(
        ["aa  ", np.nan, "bb \t\n", np.nan, np.nan, np.nan, np.nan, np.nan]
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.rstrip()
    expected = Series(["  aa", np.nan, " bb", np.nan, np.nan, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_strip_lstrip_rstrip_args(any_string_dtype):
    ser = Series(["xxABCxx", "xx BNSD", "LDFJH xx"], dtype=any_string_dtype)

    result = ser.str.strip("x")
    expected = Series(["ABC", " BNSD", "LDFJH "], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.lstrip("x")
    expected = Series(["ABCxx", " BNSD", "LDFJH xx"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)

    result = ser.str.rstrip("x")
    expected = Series(["xxABC", "xx BNSD", "LDFJH "], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_string_slice_get_syntax(any_string_dtype):
    ser = Series(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", np.nan, "CYYYBYYY", "dog", "cYYYt"],
        dtype=any_string_dtype,
    )

    result = ser.str[0]
    expected = ser.str.get(0)
    tm.assert_series_equal(result, expected)

    result = ser.str[:3]
    expected = ser.str.slice(stop=3)
    tm.assert_series_equal(result, expected)

    result = ser.str[2::-1]
    expected = ser.str.slice(start=2, step=-1)
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds_nested():
    ser = Series([(1, 2), (1,), (3, 4, 5)])
    result = ser.str[1]
    expected = Series([2, np.nan, 4])
    tm.assert_series_equal(result, expected)


def test_string_slice_out_of_bounds(any_string_dtype):
    ser = Series(["foo", "b", "ba"], dtype=any_string_dtype)
    result = ser.str[1]
    expected = Series(["o", np.nan, "a"], dtype=any_string_dtype)
    tm.assert_series_equal(result, expected)


def test_encode_decode(any_string_dtype):
    ser = Series(["a", "b", "a\xe4"], dtype=any_string_dtype).str.encode("utf-8")
    result = ser.str.decode("utf-8")
    expected = ser.map(lambda x: x.decode("utf-8"))
    tm.assert_series_equal(result, expected)


def test_encode_errors_kwarg(any_string_dtype):
    ser = Series(["a", "b", "a\x9d"], dtype=any_string_dtype)

    msg = (
        r"'charmap' codec can't encode character '\\x9d' in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeEncodeError, match=msg):
        ser.str.encode("cp1252")

    result = ser.str.encode("cp1252", "ignore")
    expected = ser.map(lambda x: x.encode("cp1252", "ignore"))
    tm.assert_series_equal(result, expected)


def test_decode_errors_kwarg():
    ser = Series([b"a", b"b", b"a\x9d"])

    msg = (
        "'charmap' codec can't decode byte 0x9d in position 1: "
        "character maps to <undefined>"
    )
    with pytest.raises(UnicodeDecodeError, match=msg):
        ser.str.decode("cp1252")

    result = ser.str.decode("cp1252", "ignore")
    expected = ser.map(lambda x: x.decode("cp1252", "ignore"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "form, expected",
    [
        ("NFKC", ["ABC", "ABC", "123", np.nan, "アイエ"]),
        ("NFC", ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"]),
    ],
)
def test_normalize(form, expected, any_string_dtype):
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    expected = Series(expected, index=["a", "b", "c", "d", "e"], dtype=any_string_dtype)
    result = ser.str.normalize(form)
    tm.assert_series_equal(result, expected)


def test_normalize_bad_arg_raises(any_string_dtype):
    ser = Series(
        ["ABC", "ＡＢＣ", "１２３", np.nan, "ｱｲｴ"],
        index=["a", "b", "c", "d", "e"],
        dtype=any_string_dtype,
    )
    with pytest.raises(ValueError, match="invalid normalization form"):
        ser.str.normalize("xxx")


def test_normalize_index():
    idx = Index(["ＡＢＣ", "１２３", "ｱｲｴ"])
    expected = Index(["ABC", "123", "アイエ"])
    result = idx.str.normalize("NFKC")
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        (["a", "b"], "string"),
        (["a", "b", 1], "mixed-integer"),
        (["a", "b", 1.3], "mixed"),
        (["a", "b", 1.3, 1], "mixed-integer"),
        (["aa", datetime(2011, 1, 1)], "mixed"),
    ],
)
def test_index_str_accessor_visibility(values, inferred_type, index_or_series):
    from pandas.core.strings import StringMethods

    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    assert isinstance(obj.str, StringMethods)


@pytest.mark.parametrize(
    "values,inferred_type",
    [
        ([1, np.nan], "floating"),
        ([datetime(2011, 1, 1)], "datetime64"),
        ([timedelta(1)], "timedelta64"),
    ],
)
def test_index_str_accessor_non_string_values_raises(
    values, inferred_type, index_or_series
):
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type

    msg = "Can only use .str accessor with string values"
    with pytest.raises(AttributeError, match=msg):
        obj.str


def test_index_str_accessor_multiindex_raises():
    # MultiIndex has mixed dtype, but not allow to use accessor
    idx = MultiIndex.from_tuples([("a", "b"), ("a", "b")])
    assert idx.inferred_type == "mixed"

    msg = "Can only use .str accessor with Index, not MultiIndex"
    with pytest.raises(AttributeError, match=msg):
        idx.str


def test_str_accessor_no_new_attributes(any_string_dtype):
    # https://github.com/pandas-dev/pandas/issues/10673
    ser = Series(list("aabbcde"), dtype=any_string_dtype)
    with pytest.raises(AttributeError, match="You cannot add any new attribute"):
        ser.str.xlabel = "a"


def test_cat_on_bytes_raises():
    lhs = Series(np.array(list("abc"), "S1").astype(object))
    rhs = Series(np.array(list("def"), "S1").astype(object))
    msg = "Cannot use .str.cat with values of inferred dtype 'bytes'"
    with pytest.raises(TypeError, match=msg):
        lhs.str.cat(rhs)


def test_str_accessor_in_apply_func():
    # https://github.com/pandas-dev/pandas/issues/38979
    df = DataFrame(zip("abc", "def"))
    expected = Series(["A/D", "B/E", "C/F"])
    result = df.apply(lambda f: "/".join(f.str.upper()), axis=1)
    tm.assert_series_equal(result, expected)
