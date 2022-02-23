"""
Tests that duplicate columns are handled appropriately when parsed by the
CSV engine. In general, the expected result is that they are either thoroughly
de-duplicated (if mangling requested) or ignored otherwise.
"""
from io import StringIO

import pytest

from pandas import DataFrame
import pandas._testing as tm


@pytest.mark.parametrize("kwargs", [{}, {"mangle_dupe_cols": True}])
def test_basic(all_parsers, kwargs):
    # TODO: add test for condition "mangle_dupe_cols=False"
    # once it is actually supported (gh-12935)
    parser = all_parsers

    data = "a,a,b,b,b\n1,2,3,4,5"
    result = parser.read_csv(StringIO(data), sep=",", **kwargs)

    expected = DataFrame([[1, 2, 3, 4, 5]], columns=["a", "a.1", "b", "b.1", "b.2"])
    tm.assert_frame_equal(result, expected)


def test_basic_names(all_parsers):
    # See gh-7160
    parser = all_parsers

    data = "a,b,a\n0,1,2\n3,4,5"
    expected = DataFrame([[0, 1, 2], [3, 4, 5]], columns=["a", "b", "a.1"])

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


def test_basic_names_raise(all_parsers):
    # See gh-7160
    parser = all_parsers

    data = "0,1,2\n3,4,5"
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=["a", "b", "a"])


@pytest.mark.parametrize(
    "data,expected",
    [
        ("a,a,a.1\n1,2,3", DataFrame([[1, 2, 3]], columns=["a", "a.1", "a.1.1"])),
        (
            "a,a,a.1,a.1.1,a.1.1.1,a.1.1.1.1\n1,2,3,4,5,6",
            DataFrame(
                [[1, 2, 3, 4, 5, 6]],
                columns=["a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1", "a.1.1.1.1.1"],
            ),
        ),
        (
            "a,a,a.3,a.1,a.2,a,a\n1,2,3,4,5,6,7",
            DataFrame(
                [[1, 2, 3, 4, 5, 6, 7]],
                columns=["a", "a.1", "a.3", "a.1.1", "a.2", "a.2.1", "a.3.1"],
            ),
        ),
    ],
)
def test_thorough_mangle_columns(all_parsers, data, expected):
    # see gh-17060
    parser = all_parsers

    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,names,expected",
    [
        (
            "a,b,b\n1,2,3",
            ["a.1", "a.1", "a.1.1"],
            DataFrame(
                [["a", "b", "b"], ["1", "2", "3"]], columns=["a.1", "a.1.1", "a.1.1.1"]
            ),
        ),
        (
            "a,b,c,d,e,f\n1,2,3,4,5,6",
            ["a", "a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            DataFrame(
                [["a", "b", "c", "d", "e", "f"], ["1", "2", "3", "4", "5", "6"]],
                columns=["a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1", "a.1.1.1.1.1"],
            ),
        ),
        (
            "a,b,c,d,e,f,g\n1,2,3,4,5,6,7",
            ["a", "a", "a.3", "a.1", "a.2", "a", "a"],
            DataFrame(
                [
                    ["a", "b", "c", "d", "e", "f", "g"],
                    ["1", "2", "3", "4", "5", "6", "7"],
                ],
                columns=["a", "a.1", "a.3", "a.1.1", "a.2", "a.2.1", "a.3.1"],
            ),
        ),
    ],
)
def test_thorough_mangle_names(all_parsers, data, names, expected):
    # see gh-17095
    parser = all_parsers

    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names)


def test_mangled_unnamed_placeholders(all_parsers):
    # xref gh-13017
    orig_key = "0"
    parser = all_parsers

    orig_value = [1, 2, 3]
    df = DataFrame({orig_key: orig_value})

    # This test recursively updates `df`.
    for i in range(3):
        expected = DataFrame()

        for j in range(i + 1):
            expected["Unnamed: 0" + ".1" * j] = [0, 1, 2]

        expected[orig_key] = orig_value
        df = parser.read_csv(StringIO(df.to_csv()))

        tm.assert_frame_equal(df, expected)
