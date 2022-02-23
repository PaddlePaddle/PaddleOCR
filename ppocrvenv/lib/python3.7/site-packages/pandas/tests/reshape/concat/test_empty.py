import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    concat,
    date_range,
)
import pandas._testing as tm


class TestEmptyConcat:
    def test_handle_empty_objects(self, sort):
        df = DataFrame(np.random.randn(10, 4), columns=list("abcd"))

        baz = df[:5].copy()
        baz["foo"] = "bar"
        empty = df[5:5]

        frames = [baz, empty, empty, df[5:]]
        concatted = concat(frames, axis=0, sort=sort)

        expected = df.reindex(columns=["a", "b", "c", "d", "foo"])
        expected["foo"] = expected["foo"].astype("O")
        expected.loc[0:4, "foo"] = "bar"

        tm.assert_frame_equal(concatted, expected)

        # empty as first element with time series
        # GH3259
        df = DataFrame(
            {"A": range(10000)}, index=date_range("20130101", periods=10000, freq="s")
        )
        empty = DataFrame()
        result = concat([df, empty], axis=1)
        tm.assert_frame_equal(result, df)
        result = concat([empty, df], axis=1)
        tm.assert_frame_equal(result, df)

        result = concat([df, empty])
        tm.assert_frame_equal(result, df)
        result = concat([empty, df])
        tm.assert_frame_equal(result, df)

    def test_concat_empty_series(self):
        # GH 11082
        s1 = Series([1, 2, 3], name="x")
        s2 = Series(name="y", dtype="float64")
        res = concat([s1, s2], axis=1)
        exp = DataFrame(
            {"x": [1, 2, 3], "y": [np.nan, np.nan, np.nan]},
            index=Index([0, 1, 2], dtype="O"),
        )
        tm.assert_frame_equal(res, exp)

        s1 = Series([1, 2, 3], name="x")
        s2 = Series(name="y", dtype="float64")
        res = concat([s1, s2], axis=0)
        # name will be reset
        exp = Series([1, 2, 3])
        tm.assert_series_equal(res, exp)

        # empty Series with no name
        s1 = Series([1, 2, 3], name="x")
        s2 = Series(name=None, dtype="float64")
        res = concat([s1, s2], axis=1)
        exp = DataFrame(
            {"x": [1, 2, 3], 0: [np.nan, np.nan, np.nan]},
            columns=["x", 0],
            index=Index([0, 1, 2], dtype="O"),
        )
        tm.assert_frame_equal(res, exp)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    @pytest.mark.parametrize("values", [[], [1, 2, 3]])
    def test_concat_empty_series_timelike(self, tz, values):
        # GH 18447

        first = Series([], dtype="M8[ns]").dt.tz_localize(tz)
        dtype = None if values else np.float64
        second = Series(values, dtype=dtype)

        expected = DataFrame(
            {
                0: Series([pd.NaT] * len(values), dtype="M8[ns]").dt.tz_localize(tz),
                1: values,
            }
        )
        result = concat([first, second], axis=1)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left,right,expected",
        [
            # booleans
            (np.bool_, np.int32, np.int32),
            (np.bool_, np.float32, np.object_),
            # datetime-like
            ("m8[ns]", np.bool_, np.object_),
            ("m8[ns]", np.int64, np.object_),
            ("M8[ns]", np.bool_, np.object_),
            ("M8[ns]", np.int64, np.object_),
            # categorical
            ("category", "category", "category"),
            ("category", "object", "object"),
        ],
    )
    def test_concat_empty_series_dtypes(self, left, right, expected):
        result = concat([Series(dtype=left), Series(dtype=right)])
        assert result.dtype == expected

    @pytest.mark.parametrize(
        "dtype", ["float64", "int8", "uint8", "bool", "m8[ns]", "M8[ns]"]
    )
    def test_concat_empty_series_dtypes_match_roundtrips(self, dtype):
        dtype = np.dtype(dtype)

        result = concat([Series(dtype=dtype)])
        assert result.dtype == dtype

        result = concat([Series(dtype=dtype), Series(dtype=dtype)])
        assert result.dtype == dtype

    def test_concat_empty_series_dtypes_roundtrips(self):

        # round-tripping with self & like self
        dtypes = map(np.dtype, ["float64", "int8", "uint8", "bool", "m8[ns]", "M8[ns]"])

        def int_result_type(dtype, dtype2):
            typs = {dtype.kind, dtype2.kind}
            if not len(typs - {"i", "u", "b"}) and (
                dtype.kind == "i" or dtype2.kind == "i"
            ):
                return "i"
            elif not len(typs - {"u", "b"}) and (
                dtype.kind == "u" or dtype2.kind == "u"
            ):
                return "u"
            return None

        def float_result_type(dtype, dtype2):
            typs = {dtype.kind, dtype2.kind}
            if not len(typs - {"f", "i", "u"}) and (
                dtype.kind == "f" or dtype2.kind == "f"
            ):
                return "f"
            return None

        def get_result_type(dtype, dtype2):
            result = float_result_type(dtype, dtype2)
            if result is not None:
                return result
            result = int_result_type(dtype, dtype2)
            if result is not None:
                return result
            return "O"

        for dtype in dtypes:
            for dtype2 in dtypes:
                if dtype == dtype2:
                    continue

                expected = get_result_type(dtype, dtype2)
                result = concat([Series(dtype=dtype), Series(dtype=dtype2)]).dtype
                assert result.kind == expected

    def test_concat_empty_series_dtypes_triple(self):

        assert (
            concat(
                [Series(dtype="M8[ns]"), Series(dtype=np.bool_), Series(dtype=np.int64)]
            ).dtype
            == np.object_
        )

    def test_concat_empty_series_dtype_category_with_array(self):
        # GH#18515
        assert (
            concat(
                [Series(np.array([]), dtype="category"), Series(dtype="float64")]
            ).dtype
            == "float64"
        )

    def test_concat_empty_series_dtypes_sparse(self):
        result = concat(
            [
                Series(dtype="float64").astype("Sparse"),
                Series(dtype="float64").astype("Sparse"),
            ]
        )
        assert result.dtype == "Sparse[float64]"

        result = concat(
            [Series(dtype="float64").astype("Sparse"), Series(dtype="float64")]
        )
        # TODO: release-note: concat sparse dtype
        expected = pd.SparseDtype(np.float64)
        assert result.dtype == expected

        result = concat(
            [Series(dtype="float64").astype("Sparse"), Series(dtype="object")]
        )
        # TODO: release-note: concat sparse dtype
        expected = pd.SparseDtype("object")
        assert result.dtype == expected

    def test_concat_empty_df_object_dtype(self):
        # GH 9149
        df_1 = DataFrame({"Row": [0, 1, 1], "EmptyCol": np.nan, "NumberCol": [1, 2, 3]})
        df_2 = DataFrame(columns=df_1.columns)
        result = concat([df_1, df_2], axis=0)
        expected = df_1.astype(object)
        tm.assert_frame_equal(result, expected)

    def test_concat_empty_dataframe_dtypes(self):
        df = DataFrame(columns=list("abc"))
        df["a"] = df["a"].astype(np.bool_)
        df["b"] = df["b"].astype(np.int32)
        df["c"] = df["c"].astype(np.float64)

        result = concat([df, df])
        assert result["a"].dtype == np.bool_
        assert result["b"].dtype == np.int32
        assert result["c"].dtype == np.float64

        result = concat([df, df.astype(np.float64)])
        assert result["a"].dtype == np.object_
        assert result["b"].dtype == np.float64
        assert result["c"].dtype == np.float64

    def test_concat_inner_join_empty(self):
        # GH 15328
        df_empty = DataFrame()
        df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
        df_expected = DataFrame({"a": []}, index=[], dtype="int64")

        for how, expected in [("inner", df_expected), ("outer", df_a)]:
            result = concat([df_a, df_empty], axis=1, join=how)
            tm.assert_frame_equal(result, expected)

    def test_empty_dtype_coerce(self):

        # xref to #12411
        # xref to #12045
        # xref to #11594
        # see below

        # 10571
        df1 = DataFrame(data=[[1, None], [2, None]], columns=["a", "b"])
        df2 = DataFrame(data=[[3, None], [4, None]], columns=["a", "b"])
        result = concat([df1, df2])
        expected = df1.dtypes
        tm.assert_series_equal(result.dtypes, expected)

    def test_concat_empty_dataframe(self):
        # 39037
        df1 = DataFrame(columns=["a", "b"])
        df2 = DataFrame(columns=["b", "c"])
        result = concat([df1, df2, df1])
        expected = DataFrame(columns=["a", "b", "c"])
        tm.assert_frame_equal(result, expected)

        df3 = DataFrame(columns=["a", "b"])
        df4 = DataFrame(columns=["b"])
        result = concat([df3, df4])
        expected = DataFrame(columns=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_concat_empty_dataframe_different_dtypes(self):
        # 39037
        df1 = DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        df2 = DataFrame({"a": [1, 2, 3]})

        result = concat([df1[:0], df2[:0]])
        assert result["a"].dtype == np.int64
        assert result["b"].dtype == np.object_
