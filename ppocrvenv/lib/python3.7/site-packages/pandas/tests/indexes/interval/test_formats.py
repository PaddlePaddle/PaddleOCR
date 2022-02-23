import numpy as np
import pytest

from pandas import (
    DataFrame,
    Float64Index,
    Interval,
    IntervalIndex,
    Series,
    Timedelta,
    Timestamp,
)
import pandas._testing as tm


class TestIntervalIndexRendering:
    def test_frame_repr(self):
        # https://github.com/pandas-dev/pandas/pull/24134/files
        df = DataFrame(
            {"A": [1, 2, 3, 4]}, index=IntervalIndex.from_breaks([0, 1, 2, 3, 4])
        )
        result = repr(df)
        expected = "        A\n(0, 1]  1\n(1, 2]  2\n(2, 3]  3\n(3, 4]  4"
        assert result == expected

    @pytest.mark.parametrize(
        "constructor,expected",
        [
            (
                Series,
                (
                    "(0.0, 1.0]    a\n"
                    "NaN           b\n"
                    "(2.0, 3.0]    c\n"
                    "dtype: object"
                ),
            ),
            (DataFrame, ("            0\n(0.0, 1.0]  a\nNaN         b\n(2.0, 3.0]  c")),
        ],
    )
    def test_repr_missing(self, constructor, expected):
        # GH 25984
        index = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        obj = constructor(list("abc"), index=index)
        result = repr(obj)
        assert result == expected

    def test_repr_floats(self):
        # GH 32553

        markers = Series(
            ["foo", "bar"],
            index=IntervalIndex(
                [
                    Interval(left, right)
                    for left, right in zip(
                        Float64Index([329.973, 345.137], dtype="float64"),
                        Float64Index([345.137, 360.191], dtype="float64"),
                    )
                ]
            ),
        )
        result = str(markers)
        expected = "(329.973, 345.137]    foo\n(345.137, 360.191]    bar\ndtype: object"
        assert result == expected

    @pytest.mark.parametrize(
        "tuples, closed, expected_data",
        [
            ([(0, 1), (1, 2), (2, 3)], "left", ["[0, 1)", "[1, 2)", "[2, 3)"]),
            (
                [(0.5, 1.0), np.nan, (2.0, 3.0)],
                "right",
                ["(0.5, 1.0]", "NaN", "(2.0, 3.0]"],
            ),
            (
                [
                    (Timestamp("20180101"), Timestamp("20180102")),
                    np.nan,
                    ((Timestamp("20180102"), Timestamp("20180103"))),
                ],
                "both",
                ["[2018-01-01, 2018-01-02]", "NaN", "[2018-01-02, 2018-01-03]"],
            ),
            (
                [
                    (Timedelta("0 days"), Timedelta("1 days")),
                    (Timedelta("1 days"), Timedelta("2 days")),
                    np.nan,
                ],
                "neither",
                [
                    "(0 days 00:00:00, 1 days 00:00:00)",
                    "(1 days 00:00:00, 2 days 00:00:00)",
                    "NaN",
                ],
            ),
        ],
    )
    def test_to_native_types(self, tuples, closed, expected_data):
        # GH 28210
        index = IntervalIndex.from_tuples(tuples, closed=closed)
        result = index._format_native_types()
        expected = np.array(expected_data)
        tm.assert_numpy_array_equal(result, expected)
