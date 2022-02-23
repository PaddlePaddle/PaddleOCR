"""
Tests that comments are properly handled during parsing
for all of the parsers defined in parsers.py
"""
from io import StringIO

import numpy as np
import pytest

from pandas import DataFrame
import pandas._testing as tm


@pytest.mark.parametrize("na_values", [None, ["NaN"]])
def test_comment(all_parsers, na_values):
    parser = all_parsers
    data = """A,B,C
1,2.,4.#hello world
5.,NaN,10.0
"""
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    result = parser.read_csv(StringIO(data), comment="#", na_values=na_values)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "read_kwargs", [{}, {"lineterminator": "*"}, {"delim_whitespace": True}]
)
def test_line_comment(all_parsers, read_kwargs):
    parser = all_parsers
    data = """# empty
A,B,C
1,2.,4.#hello world
#ignore this line
5.,NaN,10.0
"""
    if read_kwargs.get("delim_whitespace"):
        data = data.replace(",", " ")
    elif read_kwargs.get("lineterminator"):
        if parser.engine != "c":
            pytest.skip("Custom terminator not supported with Python engine")

        data = data.replace("\n", read_kwargs.get("lineterminator"))

    read_kwargs["comment"] = "#"
    result = parser.read_csv(StringIO(data), **read_kwargs)

    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows(all_parsers):
    parser = all_parsers
    data = """# empty
random line
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # This should ignore the first four lines (including comments).
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4)
    tm.assert_frame_equal(result, expected)


def test_comment_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Header should begin at the second non-comment line.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    result = parser.read_csv(StringIO(data), comment="#", header=1)
    tm.assert_frame_equal(result, expected)


def test_comment_skiprows_header(all_parsers):
    parser = all_parsers
    data = """# empty
# second empty line
# third empty line
X,Y,Z
1,2,3
A,B,C
1,2.,4.
5.,NaN,10.0
"""
    # Skiprows should skip the first 4 lines (including comments),
    # while header should start from the second non-commented line,
    # starting with line 5.
    expected = DataFrame(
        [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"]
    )
    result = parser.read_csv(StringIO(data), comment="#", skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("comment_char", ["#", "~", "&", "^", "*", "@"])
def test_custom_comment_char(all_parsers, comment_char):
    parser = all_parsers
    data = "a,b,c\n1,2,3#ignore this!\n4,5,6#ignorethistoo"
    result = parser.read_csv(
        StringIO(data.replace("#", comment_char)), comment=comment_char
    )

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("header", ["infer", None])
def test_comment_first_line(all_parsers, header):
    # see gh-4623
    parser = all_parsers
    data = "# notes\na,b,c\n# more notes\n1,2,3"

    if header is None:
        expected = DataFrame({0: ["a", "1"], 1: ["b", "2"], 2: ["c", "3"]})
    else:
        expected = DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    result = parser.read_csv(StringIO(data), comment="#", header=header)
    tm.assert_frame_equal(result, expected)


def test_comment_char_in_default_value(all_parsers, request):
    # GH#34002
    if all_parsers.engine == "c":
        reason = "see gh-34002: works on the python engine but not the c engine"
        # NA value containing comment char is interpreted as comment
        request.node.add_marker(pytest.mark.xfail(reason=reason, raises=AssertionError))
    parser = all_parsers

    data = (
        "# this is a comment\n"
        "col1,col2,col3,col4\n"
        "1,2,3,4#inline comment\n"
        "4,5#,6,10\n"
        "7,8,#N/A,11\n"
    )
    result = parser.read_csv(StringIO(data), comment="#", na_values="#N/A")
    expected = DataFrame(
        {
            "col1": [1, 4, 7],
            "col2": [2, 5, 8],
            "col3": [3.0, np.nan, np.nan],
            "col4": [4.0, np.nan, 11.0],
        }
    )
    tm.assert_frame_equal(result, expected)
