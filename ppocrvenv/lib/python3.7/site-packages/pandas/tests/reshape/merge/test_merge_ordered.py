import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    merge_ordered,
)
import pandas._testing as tm


class TestMergeOrdered:
    def setup_method(self, method):
        self.left = DataFrame({"key": ["a", "c", "e"], "lvalue": [1, 2.0, 3]})

        self.right = DataFrame({"key": ["b", "c", "d", "f"], "rvalue": [1, 2, 3.0, 4]})

    def test_basic(self):
        result = merge_ordered(self.left, self.right, on="key")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1, np.nan, 2, np.nan, 3, np.nan],
                "rvalue": [np.nan, 1, 2, 3, np.nan, 4],
            }
        )

        tm.assert_frame_equal(result, expected)

    def test_ffill(self):
        result = merge_ordered(self.left, self.right, on="key", fill_method="ffill")
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1.0, 1, 2, 2, 3, 3.0],
                "rvalue": [np.nan, 1, 2, 3, 3, 4],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_multigroup(self):
        left = pd.concat([self.left, self.left], ignore_index=True)

        left["group"] = ["a"] * 3 + ["b"] * 3

        result = merge_ordered(
            left, self.right, on="key", left_by="group", fill_method="ffill"
        )
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"] * 2,
                "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
            }
        )
        expected["group"] = ["a"] * 6 + ["b"] * 6

        tm.assert_frame_equal(result, expected.loc[:, result.columns])

        result2 = merge_ordered(
            self.right, left, on="key", right_by="group", fill_method="ffill"
        )
        tm.assert_frame_equal(result, result2.loc[:, result.columns])

        result = merge_ordered(left, self.right, on="key", left_by="group")
        assert result["group"].notna().all()

    def test_merge_type(self):
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self):
                return NotADataFrame

        nad = NotADataFrame(self.left)
        result = nad.merge(self.right, on="key")

        assert isinstance(result, NotADataFrame)

    def test_empty_sequence_concat(self):
        # GH 9157
        empty_pat = "[Nn]o objects"
        none_pat = "objects.*None"
        test_cases = [
            ((), empty_pat),
            ([], empty_pat),
            ({}, empty_pat),
            ([None], none_pat),
            ([None, None], none_pat),
        ]
        for df_seq, pattern in test_cases:
            with pytest.raises(ValueError, match=pattern):
                pd.concat(df_seq)

        pd.concat([DataFrame()])
        pd.concat([None, DataFrame()])
        pd.concat([DataFrame(), None])

    def test_doc_example(self):
        left = DataFrame(
            {
                "group": list("aaabbb"),
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3] * 2,
            }
        )

        right = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})

        result = merge_ordered(left, right, fill_method="ffill", left_by="group")

        expected = DataFrame(
            {
                "group": list("aaaaabbbbb"),
                "key": ["a", "b", "c", "d", "e"] * 2,
                "lvalue": [1, 1, 2, 2, 3] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3] * 2,
            }
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left, right, on, left_by, right_by, expected",
        [
            (
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                DataFrame({"T": [2], "E": [1]}),
                ["T"],
                ["G", "H"],
                None,
                DataFrame(
                    {
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                    }
                ),
            ),
            (
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                DataFrame({"T": [2], "E": [1]}),
                "T",
                ["G", "H"],
                None,
                DataFrame(
                    {
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                    }
                ),
            ),
            (
                DataFrame({"T": [2], "E": [1]}),
                DataFrame({"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]}),
                ["T"],
                None,
                ["G", "H"],
                DataFrame(
                    {
                        "T": [1, 2, 3],
                        "E": [np.nan, 1.0, np.nan],
                        "G": ["g"] * 3,
                        "H": ["h"] * 3,
                    }
                ),
            ),
        ],
    )
    def test_list_type_by(self, left, right, on, left_by, right_by, expected):
        # GH 35269
        result = merge_ordered(
            left=left,
            right=right,
            on=on,
            left_by=left_by,
            right_by=right_by,
        )

        tm.assert_frame_equal(result, expected)

    def test_left_by_length_equals_to_right_shape0(self):
        # GH 38166
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        result = merge_ordered(left, right, on="E", left_by=["G", "H"])
        expected = DataFrame(
            {"G": ["g"] * 3, "H": ["h"] * 3, "E": [1, 2, 3], "T": [np.nan, 1.0, np.nan]}
        )

        tm.assert_frame_equal(result, expected)

    def test_elements_not_in_by_but_in_df(self):
        # GH 38167
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        right = DataFrame([[2, 1]], columns=list("ET"))
        msg = r"\{'h'\} not found in left columns"
        with pytest.raises(KeyError, match=msg):
            merge_ordered(left, right, on="E", left_by=["G", "h"])
