from decimal import Decimal

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestDataFrameUnaryOperators:
    # __pos__, __neg__, __inv__

    @pytest.mark.parametrize(
        "df,expected",
        [
            (pd.DataFrame({"a": [-1, 1]}), pd.DataFrame({"a": [1, -1]})),
            (pd.DataFrame({"a": [False, True]}), pd.DataFrame({"a": [True, False]})),
            (
                pd.DataFrame({"a": pd.Series(pd.to_timedelta([-1, 1]))}),
                pd.DataFrame({"a": pd.Series(pd.to_timedelta([1, -1]))}),
            ),
        ],
    )
    def test_neg_numeric(self, df, expected):
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df, expected",
        [
            (np.array([1, 2], dtype=object), np.array([-1, -2], dtype=object)),
            ([Decimal("1.0"), Decimal("2.0")], [Decimal("-1.0"), Decimal("-2.0")]),
        ],
    )
    def test_neg_object(self, df, expected):
        # GH#21380
        df = pd.DataFrame({"a": df})
        expected = pd.DataFrame({"a": expected})
        tm.assert_frame_equal(-df, expected)
        tm.assert_series_equal(-df["a"], expected["a"])

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"a": ["a", "b"]}),
            pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])}),
        ],
    )
    def test_neg_raises(self, df):
        msg = (
            "bad operand type for unary -: 'str'|"
            r"Unary negative expects numeric dtype, not datetime64\[ns\]"
        )
        with pytest.raises(TypeError, match=msg):
            (-df)
        with pytest.raises(TypeError, match=msg):
            (-df["a"])

    def test_invert(self, float_frame):
        df = float_frame

        tm.assert_frame_equal(-(df < 0), ~(df < 0))

    def test_invert_mixed(self):
        shape = (10, 5)
        df = pd.concat(
            [
                pd.DataFrame(np.zeros(shape, dtype="bool")),
                pd.DataFrame(np.zeros(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        result = ~df
        expected = pd.concat(
            [
                pd.DataFrame(np.ones(shape, dtype="bool")),
                pd.DataFrame(-np.ones(shape, dtype=int)),
            ],
            axis=1,
            ignore_index=True,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"a": [-1, 1]}),
            pd.DataFrame({"a": [False, True]}),
            pd.DataFrame({"a": pd.Series(pd.to_timedelta([-1, 1]))}),
        ],
    )
    def test_pos_numeric(self, df):
        # GH#16073
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df",
        [
            # numpy changing behavior in the future
            pytest.param(
                pd.DataFrame({"a": ["a", "b"]}),
                marks=[pytest.mark.filterwarnings("ignore")],
            ),
            pd.DataFrame({"a": np.array([-1, 2], dtype=object)}),
            pd.DataFrame({"a": [Decimal("-1.0"), Decimal("2.0")]}),
        ],
    )
    def test_pos_object(self, df):
        # GH#21380
        tm.assert_frame_equal(+df, df)
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df", [pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])})]
    )
    def test_pos_raises(self, df):
        msg = "Unary plus expects .* dtype, not datetime64\\[ns\\]"
        with pytest.raises(TypeError, match=msg):
            (+df)
        with pytest.raises(TypeError, match=msg):
            (+df["a"])
