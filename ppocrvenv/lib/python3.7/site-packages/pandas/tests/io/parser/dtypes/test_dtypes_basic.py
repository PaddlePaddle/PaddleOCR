"""
Tests dtype specification during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserWarning

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm


@pytest.mark.parametrize("dtype", [str, object])
@pytest.mark.parametrize("check_orig", [True, False])
def test_dtype_all_columns(all_parsers, dtype, check_orig):
    # see gh-3795, gh-6607
    parser = all_parsers

    df = DataFrame(
        np.random.rand(5, 2).round(4),
        columns=list("AB"),
        index=["1A", "1B", "1C", "1D", "1E"],
    )

    with tm.ensure_clean("__passing_str_as_dtype__.csv") as path:
        df.to_csv(path)

        result = parser.read_csv(path, dtype=dtype, index_col=0)

        if check_orig:
            expected = df.copy()
            result = result.astype(float)
        else:
            expected = df.astype(str)

        tm.assert_frame_equal(result, expected)


def test_dtype_per_column(all_parsers):
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""
    expected = DataFrame(
        [[1, "2.5"], [2, "3.5"], [3, "4.5"], [4, "5.5"]], columns=["one", "two"]
    )
    expected["one"] = expected["one"].astype(np.float64)
    expected["two"] = expected["two"].astype(object)

    result = parser.read_csv(StringIO(data), dtype={"one": np.float64, 1: str})
    tm.assert_frame_equal(result, expected)


def test_invalid_dtype_per_column(all_parsers):
    parser = all_parsers
    data = """\
one,two
1,2.5
2,3.5
3,4.5
4,5.5"""

    with pytest.raises(TypeError, match="data type [\"']foo[\"'] not understood"):
        parser.read_csv(StringIO(data), dtype={"one": "foo", 1: "int"})


def test_raise_on_passed_int_dtype_with_nas(all_parsers):
    # see gh-2631
    parser = all_parsers
    data = """YEAR, DOY, a
2001,106380451,10
2001,,11
2001,106380451,67"""

    msg = (
        "Integer column has NA values"
        if parser.engine == "c"
        else "Unable to convert column DOY"
    )
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), dtype={"DOY": np.int64}, skipinitialspace=True)


def test_dtype_with_converters(all_parsers):
    parser = all_parsers
    data = """a,b
1.1,2.2
1.2,2.3"""

    # Dtype spec ignored if converted specified.
    with tm.assert_produces_warning(ParserWarning):
        result = parser.read_csv(
            StringIO(data), dtype={"a": "i8"}, converters={"a": lambda x: str(x)}
        )
    expected = DataFrame({"a": ["1.1", "1.2"], "b": [2.2, 2.3]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dtype", list(np.typecodes["AllInteger"] + np.typecodes["Float"])
)
def test_numeric_dtype(all_parsers, dtype):
    data = "0\n1"
    parser = all_parsers
    expected = DataFrame([0, 1], dtype=dtype)

    result = parser.read_csv(StringIO(data), header=None, dtype=dtype)
    tm.assert_frame_equal(expected, result)


def test_boolean_dtype(all_parsers):
    parser = all_parsers
    data = "\n".join(
        [
            "a",
            "True",
            "TRUE",
            "true",
            "1",
            "1.0",
            "False",
            "FALSE",
            "false",
            "0",
            "0.0",
            "NaN",
            "nan",
            "NA",
            "null",
            "NULL",
        ]
    )

    result = parser.read_csv(StringIO(data), dtype="boolean")
    expected = DataFrame(
        {
            "a": pd.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                dtype="boolean",
            )
        }
    )

    tm.assert_frame_equal(result, expected)


def test_delimiter_with_usecols_and_parse_dates(all_parsers):
    # GH#35873
    result = all_parsers.read_csv(
        StringIO('"dump","-9,1","-9,1",20101010'),
        engine="python",
        names=["col", "col1", "col2", "col3"],
        usecols=["col1", "col2", "col3"],
        parse_dates=["col3"],
        decimal=",",
    )
    expected = DataFrame(
        {"col1": [-9.1], "col2": [-9.1], "col3": [Timestamp("2010-10-10")]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("thousands", ["_", None])
def test_decimal_and_exponential(python_parser_only, numeric_decimal, thousands):
    # GH#31920
    decimal_number_check(python_parser_only, numeric_decimal, thousands, None)


@pytest.mark.parametrize("thousands", ["_", None])
@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
def test_1000_sep_decimal_float_precision(
    c_parser_only, numeric_decimal, float_precision, thousands
):
    # test decimal and thousand sep handling in across 'float_precision'
    # parsers
    decimal_number_check(c_parser_only, numeric_decimal, thousands, float_precision)


def decimal_number_check(parser, numeric_decimal, thousands, float_precision):
    # GH#31920
    value = numeric_decimal[0]
    if thousands is None and "_" in value:
        pytest.skip("Skip test if no thousands sep is defined and sep is in value")
    df = parser.read_csv(
        StringIO(value),
        sep="|",
        thousands=thousands,
        decimal=",",
        header=None,
    )
    val = df.iloc[0, 0]
    assert val == numeric_decimal[1]


def test_true_values_cast_to_bool(all_parsers):
    # GH#34655
    text = """a,b
yes,xxx
no,yyy
1,zzz
0,aaa
    """
    parser = all_parsers
    result = parser.read_csv(
        StringIO(text),
        true_values=["yes"],
        false_values=["no"],
        dtype={"a": "boolean"},
    )
    expected = DataFrame(
        {"a": [True, False, True, False], "b": ["xxx", "yyy", "zzz", "aaa"]}
    )
    expected["a"] = expected["a"].astype("boolean")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtypes, exp_value", [({}, "1"), ({"a.1": "int64"}, 1)])
def test_dtype_mangle_dup_cols(all_parsers, dtypes, exp_value):
    # GH#35211
    parser = all_parsers
    data = """a,a\n1,1"""
    dtype_dict = {"a": str, **dtypes}
    # GH#42462
    dtype_dict_copy = dtype_dict.copy()
    result = parser.read_csv(StringIO(data), dtype=dtype_dict)
    expected = DataFrame({"a": ["1"], "a.1": [exp_value]})
    assert dtype_dict == dtype_dict_copy, "dtype dict changed"
    tm.assert_frame_equal(result, expected)


def test_dtype_mangle_dup_cols_single_dtype(all_parsers):
    # GH#42022
    parser = all_parsers
    data = """a,a\n1,1"""
    result = parser.read_csv(StringIO(data), dtype=str)
    expected = DataFrame({"a": ["1"], "a.1": ["1"]})
    tm.assert_frame_equal(result, expected)
