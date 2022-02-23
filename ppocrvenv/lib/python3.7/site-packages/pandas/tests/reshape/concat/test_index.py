import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    concat,
)
import pandas._testing as tm


class TestIndexConcat:
    def test_concat_ignore_index(self, sort):
        frame1 = DataFrame(
            {"test1": ["a", "b", "c"], "test2": [1, 2, 3], "test3": [4.5, 3.2, 1.2]}
        )
        frame2 = DataFrame({"test3": [5.2, 2.2, 4.3]})
        frame1.index = Index(["x", "y", "z"])
        frame2.index = Index(["x", "y", "q"])

        v1 = concat([frame1, frame2], axis=1, ignore_index=True, sort=sort)

        nan = np.nan
        expected = DataFrame(
            [
                [nan, nan, nan, 4.3],
                ["a", 1, 4.5, 5.2],
                ["b", 2, 3.2, 2.2],
                ["c", 3, 1.2, nan],
            ],
            index=Index(["q", "x", "y", "z"]),
        )
        if not sort:
            expected = expected.loc[["x", "y", "z", "q"]]

        tm.assert_frame_equal(v1, expected)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_concat_same_index_names(self, name_in1, name_in2, name_in3, name_out):
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        frames = [
            DataFrame({c: [0, 1, 2]}, index=i) for i, c in zip(indices, ["x", "y", "z"])
        ]
        result = concat(frames, axis=1)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_concat_rename_index(self):
        a = DataFrame(
            np.random.rand(3, 3),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_a"),
        )
        b = DataFrame(
            np.random.rand(3, 3),
            columns=list("ABC"),
            index=Index(list("abc"), name="index_b"),
        )

        result = concat([a, b], keys=["key0", "key1"], names=["lvl0", "lvl1"])

        exp = concat([a, b], keys=["key0", "key1"], names=["lvl0"])
        names = list(exp.index.names)
        names[1] = "lvl1"
        exp.index.set_names(names, inplace=True)

        tm.assert_frame_equal(result, exp)
        assert result.index.names == exp.index.names

    @pytest.mark.parametrize("test_series", [True, False])
    def test_concat_copy_index(self, test_series, axis):
        # GH 29879
        if test_series:
            ser = Series([1, 2])
            comb = concat([ser, ser], axis=axis, copy=True)
            assert comb.index is not ser.index
        else:
            df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
            comb = concat([df, df], axis=axis, copy=True)
            assert comb.index is not df.index
            assert comb.columns is not df.columns

    def test_default_index(self):
        # is_series and ignore_index
        s1 = Series([1, 2, 3], name="x")
        s2 = Series([4, 5, 6], name="y")
        res = concat([s1, s2], axis=1, ignore_index=True)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        # use check_index_type=True to check the result have
        # RangeIndex (default index)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_series and all inputs have no names
        s1 = Series([1, 2, 3])
        s2 = Series([4, 5, 6])
        res = concat([s1, s2], axis=1, ignore_index=False)
        assert isinstance(res.columns, pd.RangeIndex)
        exp = DataFrame([[1, 4], [2, 5], [3, 6]])
        exp.columns = pd.RangeIndex(2)
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        # is_dataframe and ignore_index
        df1 = DataFrame({"A": [1, 2], "B": [5, 6]})
        df2 = DataFrame({"A": [3, 4], "B": [7, 8]})

        res = concat([df1, df2], axis=0, ignore_index=True)
        exp = DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]], columns=["A", "B"])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

        res = concat([df1, df2], axis=1, ignore_index=True)
        exp = DataFrame([[1, 5, 3, 7], [2, 6, 4, 8]])
        tm.assert_frame_equal(res, exp, check_index_type=True, check_column_type=True)

    def test_dups_index(self):
        # GH 4771

        # single dtypes
        df = DataFrame(
            np.random.randint(0, 10, size=40).reshape(10, 4),
            columns=["A", "A", "C", "C"],
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :4], df)
        tm.assert_frame_equal(result.iloc[:, 4:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)

        # multi dtypes
        df = concat(
            [
                DataFrame(np.random.randn(10, 4), columns=["A", "A", "B", "B"]),
                DataFrame(
                    np.random.randint(0, 10, size=20).reshape(10, 2), columns=["A", "C"]
                ),
            ],
            axis=1,
        )

        result = concat([df, df], axis=1)
        tm.assert_frame_equal(result.iloc[:, :6], df)
        tm.assert_frame_equal(result.iloc[:, 6:], df)

        result = concat([df, df], axis=0)
        tm.assert_frame_equal(result.iloc[:10], df)
        tm.assert_frame_equal(result.iloc[10:], df)

        # append
        result = df.iloc[0:8, :].append(df.iloc[8:])
        tm.assert_frame_equal(result, df)

        result = df.iloc[0:8, :].append(df.iloc[8:9]).append(df.iloc[9:10])
        tm.assert_frame_equal(result, df)

        expected = concat([df, df], axis=0)
        result = df.append(df)
        tm.assert_frame_equal(result, expected)


class TestMultiIndexConcat:
    def test_concat_multiindex_with_keys(self):
        index = MultiIndex(
            levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
            codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
            names=["first", "second"],
        )
        frame = DataFrame(
            np.random.randn(10, 3),
            index=index,
            columns=Index(["A", "B", "C"], name="exp"),
        )
        result = concat([frame, frame], keys=[0, 1], names=["iteration"])

        assert result.index.names == ("iteration",) + index.names
        tm.assert_frame_equal(result.loc[0], frame)
        tm.assert_frame_equal(result.loc[1], frame)
        assert result.index.nlevels == 3

    def test_concat_multiindex_with_none_in_index_names(self):
        # GH 15787
        index = MultiIndex.from_product([[1], range(5)], names=["level1", None])
        df = DataFrame({"col": range(5)}, index=index, dtype=np.int32)

        result = concat([df, df], keys=[1, 2], names=["level2"])
        index = MultiIndex.from_product(
            [[1, 2], [1], range(5)], names=["level2", "level1", None]
        )
        expected = DataFrame({"col": list(range(5)) * 2}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

        result = concat([df, df[:2]], keys=[1, 2], names=["level2"])
        level2 = [1] * 5 + [2] * 2
        level1 = [1] * 7
        no_name = list(range(5)) + list(range(2))
        tuples = list(zip(level2, level1, no_name))
        index = MultiIndex.from_tuples(tuples, names=["level2", "level1", None])
        expected = DataFrame({"col": no_name}, index=index, dtype=np.int32)
        tm.assert_frame_equal(result, expected)

    def test_concat_multiindex_rangeindex(self):
        # GH13542
        # when multi-index levels are RangeIndex objects
        # there is a bug in concat with objects of len 1

        df = DataFrame(np.random.randn(9, 2))
        df.index = MultiIndex(
            levels=[pd.RangeIndex(3), pd.RangeIndex(3)],
            codes=[np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)],
        )

        res = concat([df.iloc[[2, 3, 4], :], df.iloc[[5], :]])
        exp = df.iloc[[2, 3, 4, 5], :]
        tm.assert_frame_equal(res, exp)

    def test_concat_multiindex_dfs_with_deepcopy(self):
        # GH 9967
        from copy import deepcopy

        example_multiindex1 = MultiIndex.from_product([["a"], ["b"]])
        example_dataframe1 = DataFrame([0], index=example_multiindex1)

        example_multiindex2 = MultiIndex.from_product([["a"], ["c"]])
        example_dataframe2 = DataFrame([1], index=example_multiindex2)

        example_dict = {"s1": example_dataframe1, "s2": example_dataframe2}
        expected_index = MultiIndex(
            levels=[["s1", "s2"], ["a"], ["b", "c"]],
            codes=[[0, 1], [0, 0], [0, 1]],
            names=["testname", None, None],
        )
        expected = DataFrame([[0], [1]], index=expected_index)
        result_copy = concat(deepcopy(example_dict), names=["testname"])
        tm.assert_frame_equal(result_copy, expected)
        result_no_copy = concat(example_dict, names=["testname"])
        tm.assert_frame_equal(result_no_copy, expected)
