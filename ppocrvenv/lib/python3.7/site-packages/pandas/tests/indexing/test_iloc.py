""" test positional based indexing with iloc """

from datetime import datetime
import re
from warnings import (
    catch_warnings,
    simplefilter,
)

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    NA,
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    Interval,
    NaT,
    Series,
    array,
    concat,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import IndexingError
from pandas.tests.indexing.common import Base

# We pass through the error message from numpy
_slice_iloc_msg = re.escape(
    "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) "
    "and integer or boolean arrays are valid indices"
)


class TestiLoc(Base):
    @pytest.mark.parametrize("key", [2, -1, [0, 1, 2]])
    def test_iloc_getitem_int_and_list_int(self, key):
        self.check_result(
            "iloc",
            key,
            typs=["labels", "mixed", "ts", "floats", "empty"],
            fails=IndexError,
        )

        # array of ints (GH5006), make sure that a single indexer is returning
        # the correct type


class TestiLocBaseIndependent:
    """Tests Independent Of Base Class"""

    @pytest.mark.parametrize(
        "key",
        [
            slice(None),
            slice(3),
            range(3),
            [0, 1, 2],
            Index(range(3)),
            np.asarray([0, 1, 2]),
        ],
    )
    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_iloc_setitem_fullcol_categorical(self, indexer, key, using_array_manager):
        frame = DataFrame({0: range(3)}, dtype=object)

        cat = Categorical(["alpha", "beta", "gamma"])

        if not using_array_manager:
            assert frame._mgr.blocks[0]._can_hold_element(cat)

        df = frame.copy()
        orig_vals = df.values
        indexer(df)[key, 0] = cat

        overwrite = isinstance(key, slice) and key == slice(None)

        if overwrite or using_array_manager:
            # TODO(ArrayManager) we always overwrite because ArrayManager takes
            #  the "split" path, which still overwrites
            # TODO: GH#39986 this probably shouldn't behave differently
            expected = DataFrame({0: cat})
            assert not np.shares_memory(df.values, orig_vals)
        else:
            expected = DataFrame({0: cat}).astype(object)
            if not using_array_manager:
                assert np.shares_memory(df[0].values, orig_vals)

        tm.assert_frame_equal(df, expected)

        # check we dont have a view on cat (may be undesired GH#39986)
        df.iloc[0, 0] = "gamma"
        if overwrite:
            assert cat[0] != "gamma"
        else:
            assert cat[0] != "gamma"

        # TODO with mixed dataframe ("split" path), we always overwrite the column
        frame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: range(3)})
        df = frame.copy()
        orig_vals = df.values
        indexer(df)[key, 0] = cat
        expected = DataFrame({0: cat, 1: range(3)})
        tm.assert_frame_equal(df, expected)

    # TODO(ArrayManager) does not yet update parent
    @td.skip_array_manager_not_yet_implemented
    @pytest.mark.parametrize("box", [array, Series])
    def test_iloc_setitem_ea_inplace(self, frame_or_series, box, using_array_manager):
        # GH#38952 Case with not setting a full column
        #  IntegerArray without NAs
        arr = array([1, 2, 3, 4])
        obj = frame_or_series(arr.to_numpy("i8"))

        if frame_or_series is Series or not using_array_manager:
            values = obj.values
        else:
            values = obj[0].values

        if frame_or_series is Series:
            obj.iloc[:2] = box(arr[2:])
        else:
            obj.iloc[:2, 0] = box(arr[2:])

        expected = frame_or_series(np.array([3, 4, 3, 4], dtype="i8"))
        tm.assert_equal(obj, expected)

        # Check that we are actually in-place
        if frame_or_series is Series:
            assert obj.values is values
        else:
            if using_array_manager:
                assert obj[0].values is values
            else:
                assert obj.values.base is values.base and values.base is not None

    def test_is_scalar_access(self):
        # GH#32085 index with duplicates doesn't matter for _is_scalar_access
        index = Index([1, 2, 1])
        ser = Series(range(3), index=index)

        assert ser.iloc._is_scalar_access((1,))

        df = ser.to_frame()
        assert df.iloc._is_scalar_access((1, 0))

    def test_iloc_exceeds_bounds(self):

        # GH6296
        # iloc should allow indexers that exceed the bounds
        df = DataFrame(np.random.random_sample((20, 5)), columns=list("ABCDE"))

        # lists of positions should raise IndexError!
        msg = "positional indexers are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[:, [0, 1, 2, 3, 4, 5]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, 30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[1, -30]]
        with pytest.raises(IndexError, match=msg):
            df.iloc[[100]]

        s = df["A"]
        with pytest.raises(IndexError, match=msg):
            s.iloc[[100]]
        with pytest.raises(IndexError, match=msg):
            s.iloc[[-100]]

        # still raise on a single indexer
        msg = "single positional indexer is out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[30]
        with pytest.raises(IndexError, match=msg):
            df.iloc[-30]

        # GH10779
        # single positive/negative indexer exceeding Series bounds should raise
        # an IndexError
        with pytest.raises(IndexError, match=msg):
            s.iloc[30]
        with pytest.raises(IndexError, match=msg):
            s.iloc[-30]

        # slices are ok
        result = df.iloc[:, 4:10]  # 0 < start < len < stop
        expected = df.iloc[:, 4:]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -4:-10]  # stop < 0 < start < len
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:4:-1]  # 0 < stop < len < start (down)
        expected = df.iloc[:, :4:-1]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 4:-10:-1]  # stop < 0 < start < len (down)
        expected = df.iloc[:, 4::-1]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -10:4]  # start < 0 < stop < len
        expected = df.iloc[:, :4]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:4]  # 0 < stop < len < start
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, -10:-11:-1]  # stop < start < 0 < len (down)
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 10:11]  # 0 < len < start < stop
        expected = df.iloc[:, :0]
        tm.assert_frame_equal(result, expected)

        # slice bounds exceeding is ok
        result = s.iloc[18:30]
        expected = s.iloc[18:]
        tm.assert_series_equal(result, expected)

        result = s.iloc[30:]
        expected = s.iloc[:0]
        tm.assert_series_equal(result, expected)

        result = s.iloc[30::-1]
        expected = s.iloc[::-1]
        tm.assert_series_equal(result, expected)

        # doc example
        def check(result, expected):
            str(result)
            result.dtypes
            tm.assert_frame_equal(result, expected)

        dfl = DataFrame(np.random.randn(5, 2), columns=list("AB"))
        check(dfl.iloc[:, 2:3], DataFrame(index=dfl.index))
        check(dfl.iloc[:, 1:3], dfl.iloc[:, [1]])
        check(dfl.iloc[4:6], dfl.iloc[[4]])

        msg = "positional indexers are out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[[4, 5, 6]]
        msg = "single positional indexer is out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            dfl.iloc[:, 4]

    @pytest.mark.parametrize("index,columns", [(np.arange(20), list("ABCDE"))])
    @pytest.mark.parametrize(
        "index_vals,column_vals",
        [
            ([slice(None), ["A", "D"]]),
            (["1", "2"], slice(None)),
            ([datetime(2019, 1, 1)], slice(None)),
        ],
    )
    def test_iloc_non_integer_raises(self, index, columns, index_vals, column_vals):
        # GH 25753
        df = DataFrame(
            np.random.randn(len(index), len(columns)), index=index, columns=columns
        )
        msg = ".iloc requires numeric indexers, got"
        with pytest.raises(IndexError, match=msg):
            df.iloc[index_vals, column_vals]

    @pytest.mark.parametrize("dims", [1, 2])
    def test_iloc_getitem_invalid_scalar(self, dims):
        # GH 21982

        if dims == 1:
            s = Series(np.arange(10))
        else:
            s = DataFrame(np.arange(100).reshape(10, 10))

        with pytest.raises(TypeError, match="Cannot index by location index"):
            s.iloc["a"]

    def test_iloc_array_not_mutating_negative_indices(self):

        # GH 21867
        array_with_neg_numbers = np.array([1, 2, -1])
        array_copy = array_with_neg_numbers.copy()
        df = DataFrame(
            {"A": [100, 101, 102], "B": [103, 104, 105], "C": [106, 107, 108]},
            index=[1, 2, 3],
        )
        df.iloc[array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)
        df.iloc[:, array_with_neg_numbers]
        tm.assert_numpy_array_equal(array_with_neg_numbers, array_copy)

    def test_iloc_getitem_neg_int_can_reach_first_index(self):
        # GH10547 and GH10779
        # negative integers should be able to reach index 0
        df = DataFrame({"A": [2, 3, 5], "B": [7, 11, 13]})
        s = df["A"]

        expected = df.iloc[0]
        result = df.iloc[-3]
        tm.assert_series_equal(result, expected)

        expected = df.iloc[[0]]
        result = df.iloc[[-3]]
        tm.assert_frame_equal(result, expected)

        expected = s.iloc[0]
        result = s.iloc[-3]
        assert result == expected

        expected = s.iloc[[0]]
        result = s.iloc[[-3]]
        tm.assert_series_equal(result, expected)

        # check the length 1 Series case highlighted in GH10547
        expected = Series(["a"], index=["A"])
        result = expected.iloc[[-1]]
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_dups(self):
        # GH 6766
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)

        # cross-sectional indexing
        result = df.iloc[0, 0]
        assert isna(result)

        result = df.iloc[0, :]
        expected = Series([np.nan, 1, 3, 3], index=["A", "B", "A", "B"], name=0)
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_array(self):
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}])
        tm.assert_frame_equal(df.iloc[[0]], expected)

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        tm.assert_frame_equal(df.iloc[[0, 1]], expected)

        expected = DataFrame([{"B": 2, "C": 3}, {"B": 2000, "C": 3000}], index=[0, 2])
        result = df.iloc[[0, 2], [1, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_bool(self):
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        result = df.iloc[[True, True, False]]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            [{"A": 1, "B": 2, "C": 3}, {"A": 1000, "B": 2000, "C": 3000}], index=[0, 2]
        )
        result = df.iloc[lambda x: x.index % 2 == 0]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_iloc_getitem_bool_diff_len(self, index):
        # GH26658
        s = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.iloc[index]

    def test_iloc_getitem_slice(self):
        df = DataFrame(
            [
                {"A": 1, "B": 2, "C": 3},
                {"A": 100, "B": 200, "C": 300},
                {"A": 1000, "B": 2000, "C": 3000},
            ]
        )

        expected = DataFrame([{"A": 1, "B": 2, "C": 3}, {"A": 100, "B": 200, "C": 300}])
        result = df.iloc[:2]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([{"A": 100, "B": 200}], index=[1])
        result = df.iloc[1:2, 0:2]
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            [{"A": 1, "C": 3}, {"A": 100, "C": 300}, {"A": 1000, "C": 3000}]
        )
        result = df.iloc[:, lambda df: [0, 2]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_slice_dups(self):

        df1 = DataFrame(np.random.randn(10, 4), columns=["A", "A", "B", "B"])
        df2 = DataFrame(
            np.random.randint(0, 10, size=20).reshape(10, 2), columns=["A", "C"]
        )

        # axis=1
        df = concat([df1, df2], axis=1)
        tm.assert_frame_equal(df.iloc[:, :4], df1)
        tm.assert_frame_equal(df.iloc[:, 4:], df2)

        df = concat([df2, df1], axis=1)
        tm.assert_frame_equal(df.iloc[:, :2], df2)
        tm.assert_frame_equal(df.iloc[:, 2:], df1)

        exp = concat([df2, df1.iloc[:, [0]]], axis=1)
        tm.assert_frame_equal(df.iloc[:, 0:3], exp)

        # axis=0
        df = concat([df, df], axis=0)
        tm.assert_frame_equal(df.iloc[0:10, :2], df2)
        tm.assert_frame_equal(df.iloc[0:10, 2:], df1)
        tm.assert_frame_equal(df.iloc[10:, :2], df2)
        tm.assert_frame_equal(df.iloc[10:, 2:], df1)

    def test_iloc_setitem(self):
        df = DataFrame(
            np.random.randn(4, 4), index=np.arange(0, 8, 2), columns=np.arange(0, 12, 3)
        )

        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1

        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)

        # GH5771
        s = Series(0, index=[4, 5, 6])
        s.iloc[1:2] += 1
        expected = Series([0, 1, 0], index=[4, 5, 6])
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_list(self):

        # setitem with an iloc list
        df = DataFrame(
            np.arange(9).reshape((3, 3)), index=["A", "B", "C"], columns=["A", "B", "C"]
        )
        df.iloc[[0, 1], [1, 2]]
        df.iloc[[0, 1], [1, 2]] += 100

        expected = DataFrame(
            np.array([0, 101, 102, 3, 104, 105, 6, 7, 8]).reshape((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_pandas_object(self):
        # GH 17193
        s_orig = Series([0, 1, 2, 3])
        expected = Series([0, -1, -2, 3])

        s = s_orig.copy()
        s.iloc[Series([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

        s = s_orig.copy()
        s.iloc[Index([1, 2])] = [-1, -2]
        tm.assert_series_equal(s, expected)

    def test_iloc_setitem_dups(self):

        # GH 6766
        # iloc with a mask aligning from another iloc
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)

        expected = df.fillna(3)
        inds = np.isnan(df.iloc[:, 0])
        mask = inds[inds].index
        df.iloc[mask, 0] = df.iloc[mask, 2]
        tm.assert_frame_equal(df, expected)

        # del a dup column across blocks
        expected = DataFrame({0: [1, 2], 1: [3, 4]})
        expected.columns = ["B", "B"]
        del df["A"]
        tm.assert_frame_equal(df, expected)

        # assign back to self
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]
        tm.assert_frame_equal(df, expected)

        # reversed x 2
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        df.iloc[[1, 0], [0, 1]] = df.iloc[[1, 0], [0, 1]].reset_index(drop=True)
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_frame_duplicate_columns_multiple_blocks(
        self, using_array_manager
    ):
        # Same as the "assign back to self" check in test_iloc_setitem_dups
        #  but on a DataFrame with multiple blocks
        df = DataFrame([[0, 1], [2, 3]], columns=["B", "B"])

        df.iloc[:, 0] = df.iloc[:, 0].astype("f8")
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2
        expected = df.copy()

        # assign back to self
        df.iloc[[0, 1], [0, 1]] = df.iloc[[0, 1], [0, 1]]

        tm.assert_frame_equal(df, expected)

    # TODO: GH#27620 this test used to compare iloc against ix; check if this
    #  is redundant with another test comparing iloc against loc
    def test_iloc_getitem_frame(self):
        df = DataFrame(
            np.random.randn(10, 4), index=range(0, 20, 2), columns=range(0, 8, 2)
        )

        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        result = df.iloc[2, 2]
        exp = df.loc[4, 4]
        assert result == exp

        # slice
        result = df.iloc[4:8]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[:, 2:3]
        expected = df.loc[:, 4:5]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[[0, 1, 3]]
        expected = df.loc[[0, 2, 6]]
        tm.assert_frame_equal(result, expected)

        result = df.iloc[[0, 1, 3], [0, 1]]
        expected = df.loc[[0, 2, 6], [0, 2]]
        tm.assert_frame_equal(result, expected)

        # neg indices
        result = df.iloc[[-1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # dups indices
        result = df.iloc[[-1, -1, 1, 3], [-1, 1]]
        expected = df.loc[[18, 18, 2, 6], [6, 2]]
        tm.assert_frame_equal(result, expected)

        # with index-like
        s = Series(index=range(1, 5), dtype=object)
        result = df.iloc[s.index]
        expected = df.loc[[2, 4, 6, 8]]
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_labelled_frame(self):
        # try with labelled frame
        df = DataFrame(
            np.random.randn(10, 4), index=list("abcdefghij"), columns=list("ABCD")
        )

        result = df.iloc[1, 1]
        exp = df.loc["b", "B"]
        assert result == exp

        result = df.iloc[:, 2:3]
        expected = df.loc[:, ["C"]]
        tm.assert_frame_equal(result, expected)

        # negative indexing
        result = df.iloc[-1, -1]
        exp = df.loc["j", "D"]
        assert result == exp

        # out-of-bounds exception
        msg = "single positional indexer is out-of-bounds"
        with pytest.raises(IndexError, match=msg):
            df.iloc[10, 5]

        # trying to use a label
        msg = (
            r"Location based indexing can only have \[integer, integer "
            r"slice \(START point is INCLUDED, END point is EXCLUDED\), "
            r"listlike of integers, boolean array\] types"
        )
        with pytest.raises(ValueError, match=msg):
            df.iloc["j", "D"]

    def test_iloc_getitem_doc_issue(self, using_array_manager):

        # multi axis slicing issue with single block
        # surfaced in GH 6059

        arr = np.random.randn(6, 4)
        index = date_range("20130101", periods=6)
        columns = list("ABCD")
        df = DataFrame(arr, index=index, columns=columns)

        # defines ref_locs
        df.describe()

        result = df.iloc[3:5, 0:2]
        str(result)
        result.dtypes

        expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=columns[0:2])
        tm.assert_frame_equal(result, expected)

        # for dups
        df.columns = list("aaaa")
        result = df.iloc[3:5, 0:2]
        str(result)
        result.dtypes

        expected = DataFrame(arr[3:5, 0:2], index=index[3:5], columns=list("aa"))
        tm.assert_frame_equal(result, expected)

        # related
        arr = np.random.randn(6, 4)
        index = list(range(0, 12, 2))
        columns = list(range(0, 8, 2))
        df = DataFrame(arr, index=index, columns=columns)

        if not using_array_manager:
            df._mgr.blocks[0].mgr_locs
        result = df.iloc[1:5, 2:4]
        str(result)
        result.dtypes
        expected = DataFrame(arr[1:5, 2:4], index=index[1:5], columns=columns[2:4])
        tm.assert_frame_equal(result, expected)

    def test_iloc_setitem_series(self):
        df = DataFrame(
            np.random.randn(10, 4), index=list("abcdefghij"), columns=list("ABCD")
        )

        df.iloc[1, 1] = 1
        result = df.iloc[1, 1]
        assert result == 1

        df.iloc[:, 2:3] = 0
        expected = df.iloc[:, 2:3]
        result = df.iloc[:, 2:3]
        tm.assert_frame_equal(result, expected)

        s = Series(np.random.randn(10), index=range(0, 20, 2))

        s.iloc[1] = 1
        result = s.iloc[1]
        assert result == 1

        s.iloc[:4] = 0
        expected = s.iloc[:4]
        result = s.iloc[:4]
        tm.assert_series_equal(result, expected)

        s = Series([-1] * 6)
        s.iloc[0::2] = [0, 2, 4]
        s.iloc[1::2] = [1, 3, 5]
        result = s
        expected = Series([0, 1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)

    def test_iloc_setitem_list_of_lists(self):

        # GH 7551
        # list-of-list is set incorrectly in mixed vs. single dtyped frames
        df = DataFrame(
            {"A": np.arange(5, dtype="int64"), "B": np.arange(5, 10, dtype="int64")}
        )
        df.iloc[2:4] = [[10, 11], [12, 13]]
        expected = DataFrame({"A": [0, 1, 10, 12, 4], "B": [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)

        df = DataFrame(
            {"A": ["a", "b", "c", "d", "e"], "B": np.arange(5, 10, dtype="int64")}
        )
        df.iloc[2:4] = [["x", 11], ["y", 13]]
        expected = DataFrame({"A": ["a", "b", "x", "y", "e"], "B": [5, 6, 11, 13, 9]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [[0], slice(None, 1, None), np.array([0])])
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_iloc_setitem_with_scalar_index(self, indexer, value):
        # GH #19474
        # assigning like "df.iloc[0, [0]] = ['Z']" should be evaluated
        # elementwisely, not using "setter('A', ['Z'])".

        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
        df.iloc[0, indexer] = value
        result = df.iloc[0, 0]

        assert is_scalar(result) and result == "Z"

    def test_iloc_mask(self):

        # GH 3631, iloc with a mask (of a series) should raise
        df = DataFrame(list(range(5)), index=list("ABCDE"), columns=["a"])
        mask = df.a % 2 == 0
        msg = "iLocation based boolean indexing cannot use an indexable as a mask"
        with pytest.raises(ValueError, match=msg):
            df.iloc[mask]
        mask.index = range(len(mask))
        msg = "iLocation based boolean indexing on an integer type is not available"
        with pytest.raises(NotImplementedError, match=msg):
            df.iloc[mask]

        # ndarray ok
        result = df.iloc[np.array([True] * len(mask), dtype=bool)]
        tm.assert_frame_equal(result, df)

        # the possibilities
        locs = np.arange(4)
        nums = 2 ** locs
        reps = [bin(num) for num in nums]
        df = DataFrame({"locs": locs, "nums": nums}, reps)

        expected = {
            (None, ""): "0b1100",
            (None, ".loc"): "0b1100",
            (None, ".iloc"): "0b1100",
            ("index", ""): "0b11",
            ("index", ".loc"): "0b11",
            ("index", ".iloc"): (
                "iLocation based boolean indexing cannot use an indexable as a mask"
            ),
            ("locs", ""): "Unalignable boolean Series provided as indexer "
            "(index of the boolean Series and of the indexed "
            "object do not match).",
            ("locs", ".loc"): "Unalignable boolean Series provided as indexer "
            "(index of the boolean Series and of the "
            "indexed object do not match).",
            ("locs", ".iloc"): (
                "iLocation based boolean indexing on an "
                "integer type is not available"
            ),
        }

        # UserWarnings from reindex of a boolean mask
        with catch_warnings(record=True):
            simplefilter("ignore", UserWarning)
            for idx in [None, "index", "locs"]:
                mask = (df.nums > 2).values
                if idx:
                    mask = Series(mask, list(reversed(getattr(df, idx))))
                for method in ["", ".loc", ".iloc"]:
                    try:
                        if method:
                            accessor = getattr(df, method[1:])
                        else:
                            accessor = df
                        answer = str(bin(accessor[mask]["nums"].sum()))
                    except (ValueError, IndexingError, NotImplementedError) as e:
                        answer = str(e)

                    key = (
                        idx,
                        method,
                    )
                    r = expected.get(key)
                    if r != answer:
                        raise AssertionError(
                            f"[{key}] does not match [{answer}], received [{r}]"
                        )

    def test_iloc_non_unique_indexing(self):

        # GH 4017, non-unique indexing (on the axis)
        df = DataFrame({"A": [0.1] * 3000, "B": [1] * 3000})
        idx = np.arange(30) * 99
        expected = df.iloc[idx]

        df3 = concat([df, 2 * df, 3 * df])
        result = df3.iloc[idx]

        tm.assert_frame_equal(result, expected)

        df2 = DataFrame({"A": [0.1] * 1000, "B": [1] * 1000})
        df2 = concat([df2, 2 * df2, 3 * df2])

        with pytest.raises(KeyError, match="not in index"):
            df2.loc[idx]

    def test_iloc_empty_list_indexer_is_ok(self):

        df = tm.makeCustomDataframe(5, 2)
        # vertical empty
        tm.assert_frame_equal(
            df.iloc[:, []],
            df.iloc[:, :0],
            check_index_type=True,
            check_column_type=True,
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.iloc[[], :],
            df.iloc[:0, :],
            check_index_type=True,
            check_column_type=True,
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.iloc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )

    def test_identity_slice_returns_new_object(self, using_array_manager):
        # GH13873
        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.iloc[:]
        assert sliced_df is not original_df

        # should be a shallow copy
        original_df["a"] = [4, 4, 4]
        if using_array_manager:
            # TODO(ArrayManager) verify it is expected that the original didn't change
            # setitem is replacing full column, so doesn't update "viewing" dataframe
            assert not (sliced_df["a"] == 4).all()
        else:
            assert (sliced_df["a"] == 4).all()

        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.iloc[:]
        assert sliced_series is not original_series

        # should also be a shallow copy
        original_series[:3] = [7, 8, 9]
        assert all(sliced_series[:3] == [7, 8, 9])

    def test_indexing_zerodim_np_array(self):
        # GH24919
        df = DataFrame([[1, 2], [3, 4]])
        result = df.iloc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # GH24919
        s = Series([1, 2])
        result = s.iloc[np.array(0)]
        assert result == 1

    @pytest.mark.xfail(reason="https://github.com/pandas-dev/pandas/issues/33457")
    def test_iloc_setitem_categorical_updates_inplace(self):
        # Mixed dtype ensures we go through take_split_path in setitem_with_indexer
        cat = Categorical(["A", "B", "C"])
        df = DataFrame({1: cat, 2: [1, 2, 3]})

        # This should modify our original values in-place
        df.iloc[:, 0] = cat[::-1]

        expected = Categorical(["C", "B", "A"])
        tm.assert_categorical_equal(cat, expected)

    def test_iloc_with_boolean_operation(self):
        # GH 20627
        result = DataFrame([[0, 1], [2, 3], [4, 5], [6, np.nan]])
        result.iloc[result.index <= 2] *= 2
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [6, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[result.index > 2] *= 2
        expected = DataFrame([[0, 2], [4, 6], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[[True, True, False, False]] *= 2
        expected = DataFrame([[0, 4], [8, 12], [8, 10], [12, np.nan]])
        tm.assert_frame_equal(result, expected)

        result.iloc[[False, False, True, True]] /= 2
        expected = DataFrame([[0.0, 4.0], [8.0, 12.0], [4.0, 5.0], [6.0, np.nan]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_getitem_singlerow_slice_categoricaldtype_gives_series(self):
        # GH#29521
        df = DataFrame({"x": Categorical("a b c d e".split())})
        result = df.iloc[0]
        raw_cat = Categorical(["a"], categories=["a", "b", "c", "d", "e"])
        expected = Series(raw_cat, index=["x"], name=0, dtype="category")

        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_categorical_values(self):
        # GH#14580
        # test iloc() on Series with Categorical data

        ser = Series([1, 2, 3]).astype("category")

        # get slice
        result = ser.iloc[0:2]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # get list of indexes
        result = ser.iloc[[0, 1]]
        expected = Series([1, 2]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

        # get boolean array
        result = ser.iloc[[True, False, False]]
        expected = Series([1]).astype(CategoricalDtype([1, 2, 3]))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("value", [None, NaT, np.nan])
    def test_iloc_setitem_td64_values_cast_na(self, value):
        # GH#18586
        series = Series([0, 1, 2], dtype="timedelta64[ns]")
        series.iloc[0] = value
        expected = Series([NaT, 1, 2], dtype="timedelta64[ns]")
        tm.assert_series_equal(series, expected)

    def test_iloc_setitem_empty_frame_raises_with_3d_ndarray(self):
        idx = Index([])
        obj = DataFrame(np.random.randn(len(idx), len(idx)), index=idx, columns=idx)
        nd3 = np.random.randint(5, size=(2, 2, 2))

        msg = f"Cannot set values with ndim > {obj.ndim}"
        with pytest.raises(ValueError, match=msg):
            obj.iloc[nd3] = 0

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_iloc_getitem_read_only_values(self, indexer):
        # GH#10043 this is fundamentally a test for iloc, but test loc while
        #  we're here
        rw_array = np.eye(10)
        rw_df = DataFrame(rw_array)

        ro_array = np.eye(10)
        ro_array.setflags(write=False)
        ro_df = DataFrame(ro_array)

        tm.assert_frame_equal(indexer(rw_df)[[1, 2, 3]], indexer(ro_df)[[1, 2, 3]])
        tm.assert_frame_equal(indexer(rw_df)[[1]], indexer(ro_df)[[1]])
        tm.assert_series_equal(indexer(rw_df)[1], indexer(ro_df)[1])
        tm.assert_frame_equal(indexer(rw_df)[1:3], indexer(ro_df)[1:3])

    def test_iloc_getitem_readonly_key(self):
        # GH#17192 iloc with read-only array raising TypeError
        df = DataFrame({"data": np.ones(100, dtype="float64")})
        indices = np.array([1, 3, 6])
        indices.flags.writeable = False

        result = df.iloc[indices]
        expected = df.loc[[1, 3, 6]]
        tm.assert_frame_equal(result, expected)

        result = df["data"].iloc[indices]
        expected = df["data"].loc[[1, 3, 6]]
        tm.assert_series_equal(result, expected)

    # TODO(ArrayManager) setting single item with an iterable doesn't work yet
    # in the "split" path
    @td.skip_array_manager_not_yet_implemented
    def test_iloc_assign_series_to_df_cell(self):
        # GH 37593
        df = DataFrame(columns=["a"], index=[0])
        df.iloc[0, 0] = Series([1, 2, 3])
        expected = DataFrame({"a": [Series([1, 2, 3])]}, columns=["a"], index=[0])
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("klass", [list, np.array])
    def test_iloc_setitem_bool_indexer(self, klass):
        # GH#36741
        df = DataFrame({"flag": ["x", "y", "z"], "value": [1, 3, 4]})
        indexer = klass([True, False, False])
        df.iloc[indexer, 1] = df.iloc[indexer, 1] * 2
        expected = DataFrame({"flag": ["x", "y", "z"], "value": [2, 3, 4]})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("indexer", [[1], slice(1, 2)])
    def test_iloc_setitem_pure_position_based(self, indexer):
        # GH#22046
        df1 = DataFrame({"a2": [11, 12, 13], "b2": [14, 15, 16]})
        df2 = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        df2.iloc[:, indexer] = df1.iloc[:, [0]]
        expected = DataFrame({"a": [1, 2, 3], "b": [11, 12, 13], "c": [7, 8, 9]})
        tm.assert_frame_equal(df2, expected)

    def test_iloc_setitem_dictionary_value(self):
        # GH#37728
        df = DataFrame({"x": [1, 2], "y": [2, 2]})
        rhs = {"x": 9, "y": 99}
        df.iloc[1] = rhs
        expected = DataFrame({"x": [1, 9], "y": [2, 99]})
        tm.assert_frame_equal(df, expected)

        # GH#38335 same thing, mixed dtypes
        df = DataFrame({"x": [1, 2], "y": [2.0, 2.0]})
        df.iloc[1] = rhs
        expected = DataFrame({"x": [1, 9], "y": [2.0, 99.0]})
        tm.assert_frame_equal(df, expected)

    def test_iloc_getitem_float_duplicates(self):
        df = DataFrame(
            np.random.randn(3, 3), index=[0.1, 0.2, 0.2], columns=list("abc")
        )
        expect = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df.index = [1, 0.2, 0.2]
        expect = df.iloc[1:]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df = DataFrame(
            np.random.randn(4, 3), index=[1, 0.2, 0.2, 1], columns=list("abc")
        )
        expect = df.iloc[1:-1]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[1:-1, 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

        df.index = [0.1, 0.2, 2, 0.2]
        expect = df.iloc[[1, -1]]
        tm.assert_frame_equal(df.loc[0.2], expect)

        expect = df.iloc[[1, -1], 0]
        tm.assert_series_equal(df.loc[0.2, "a"], expect)

    def test_iloc_setitem_custom_object(self):
        # iloc with an object
        class TO:
            def __init__(self, value):
                self.value = value

            def __str__(self) -> str:
                return f"[{self.value}]"

            __repr__ = __str__

            def __eq__(self, other) -> bool:
                return self.value == other.value

            def view(self):
                return self

        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = TO(2)

        result = DataFrame(index=[0, 1], columns=[0])
        result.iloc[1, 0] = TO(2)

        tm.assert_frame_equal(result, df)

        # remains object dtype even after setting it back
        df = DataFrame(index=[0, 1], columns=[0])
        df.iloc[1, 0] = TO(1)
        df.iloc[1, 0] = np.nan
        result = DataFrame(index=[0, 1], columns=[0])

        tm.assert_frame_equal(result, df)

    def test_iloc_getitem_with_duplicates(self):

        df = DataFrame(np.random.rand(3, 3), columns=list("ABC"), index=list("aab"))

        result = df.iloc[0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])

        result = df.T.iloc[:, 0]
        assert isinstance(result, Series)
        tm.assert_almost_equal(result.values, df.values[0])

    def test_iloc_getitem_with_duplicates2(self):
        # GH#2259
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1, 1, 2])
        result = df.iloc[:, [0]]
        expected = df.take([0], axis=1)
        tm.assert_frame_equal(result, expected)

    def test_iloc_interval(self):
        # GH#17130
        df = DataFrame({Interval(1, 2): [1, 2]})

        result = df.iloc[0]
        expected = Series({Interval(1, 2): 1}, name=0)
        tm.assert_series_equal(result, expected)

        result = df.iloc[:, 0]
        expected = Series([1, 2], name=Interval(1, 2))
        tm.assert_series_equal(result, expected)

        result = df.copy()
        result.iloc[:, 0] += 1
        expected = DataFrame({Interval(1, 2): [2, 3]})
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("indexing_func", [list, np.array])
    @pytest.mark.parametrize("rhs_func", [list, np.array])
    def test_loc_setitem_boolean_list(self, rhs_func, indexing_func):
        # GH#20438 testing specifically list key, not arraylike
        ser = Series([0, 1, 2])
        ser.iloc[indexing_func([True, False, True])] = rhs_func([5, 10])
        expected = Series([5, 1, 10])
        tm.assert_series_equal(ser, expected)

        df = DataFrame({"a": [0, 1, 2]})
        df.iloc[indexing_func([True, False, True])] = rhs_func([[5], [10]])
        expected = DataFrame({"a": [5, 1, 10]})
        tm.assert_frame_equal(df, expected)


class TestILocErrors:
    # NB: this test should work for _any_ Series we can pass as
    #  series_with_simple_index
    def test_iloc_float_raises(self, series_with_simple_index, frame_or_series):
        # GH#4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors
        # this duplicates the code below
        # but is specifically testing for the error
        # message

        obj = series_with_simple_index
        if frame_or_series is DataFrame:
            obj = obj.to_frame()

        msg = "Cannot index by location index with a non-integer key"
        with pytest.raises(TypeError, match=msg):
            obj.iloc[3.0]

        with pytest.raises(IndexError, match=_slice_iloc_msg):
            obj.iloc[3.0] = 0

    def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame):
        with pytest.raises(IndexingError, match="Too many indexers"):
            float_frame.iloc[:, :, :]

        with pytest.raises(IndexError, match="too many indices for array"):
            # GH#32257 we let numpy do validation, get their exception
            float_frame.iloc[:, :, :] = 1

    # TODO(ArrayManager) "split" path doesn't properly implement DataFrame indexer
    @td.skip_array_manager_not_yet_implemented
    def test_iloc_frame_indexer(self):
        # GH#39004
        df = DataFrame({"a": [1, 2, 3]})
        indexer = DataFrame({"a": [True, False, True]})
        with tm.assert_produces_warning(FutureWarning):
            df.iloc[indexer] = 1

        msg = (
            "DataFrame indexer is not allowed for .iloc\n"
            "Consider using .loc for automatic alignment."
        )
        with pytest.raises(IndexError, match=msg):
            df.iloc[indexer]


class TestILocSetItemDuplicateColumns:
    def test_iloc_setitem_scalar_duplicate_columns(self):
        # GH#15686, duplicate columns and mixed dtype
        df1 = DataFrame([{"A": None, "B": 1}, {"A": 2, "B": 2}])
        df2 = DataFrame([{"A": 3, "B": 3}, {"A": 4, "B": 4}])
        df = concat([df1, df2], axis=1)
        df.iloc[0, 0] = -1

        assert df.iloc[0, 0] == -1
        assert df.iloc[0, 2] == 3
        assert df.dtypes.iloc[2] == np.int64

    def test_iloc_setitem_list_duplicate_columns(self):
        # GH#22036 setting with same-sized list
        df = DataFrame([[0, "str", "str2"]], columns=["a", "b", "b"])

        df.iloc[:, 2] = ["str3"]

        expected = DataFrame([[0, "str", "str3"]], columns=["a", "b", "b"])
        tm.assert_frame_equal(df, expected)

    def test_iloc_setitem_series_duplicate_columns(self):
        df = DataFrame(
            np.arange(8, dtype=np.int64).reshape(2, 4), columns=["A", "B", "A", "B"]
        )
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.float64)
        assert df.dtypes.iloc[2] == np.int64

    @pytest.mark.parametrize(
        ["dtypes", "init_value", "expected_value"],
        [("int64", "0", 0), ("float", "1.2", 1.2)],
    )
    def test_iloc_setitem_dtypes_duplicate_columns(
        self, dtypes, init_value, expected_value
    ):
        # GH#22035
        df = DataFrame([[init_value, "str", "str2"]], columns=["a", "b", "b"])
        df.iloc[:, 0] = df.iloc[:, 0].astype(dtypes)
        expected_df = DataFrame(
            [[expected_value, "str", "str2"]], columns=["a", "b", "b"]
        )
        tm.assert_frame_equal(df, expected_df)


class TestILocCallable:
    def test_frame_iloc_getitem_callable(self):
        # GH#11485
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # return location
        res = df.iloc[lambda x: [1, 3]]
        tm.assert_frame_equal(res, df.iloc[[1, 3]])

        res = df.iloc[lambda x: [1, 3], :]
        tm.assert_frame_equal(res, df.iloc[[1, 3], :])

        res = df.iloc[lambda x: [1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[lambda x: [1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        # mixture
        res = df.iloc[[1, 3], lambda x: 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[[1, 3], lambda x: [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

        res = df.iloc[lambda x: [1, 3], 0]
        tm.assert_series_equal(res, df.iloc[[1, 3], 0])

        res = df.iloc[lambda x: [1, 3], [0]]
        tm.assert_frame_equal(res, df.iloc[[1, 3], [0]])

    def test_frame_iloc_setitem_callable(self):
        # GH#11485
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # return location
        res = df.copy()
        res.iloc[lambda x: [1, 3]] = 0
        exp = df.copy()
        exp.iloc[[1, 3]] = 0
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], :] = -1
        exp = df.copy()
        exp.iloc[[1, 3], :] = -1
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: 0] = 5
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 5
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], lambda x: [0]] = 25
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = 25
        tm.assert_frame_equal(res, exp)

        # mixture
        res = df.copy()
        res.iloc[[1, 3], lambda x: 0] = -3
        exp = df.copy()
        exp.iloc[[1, 3], 0] = -3
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[[1, 3], lambda x: [0]] = -5
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = -5
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], 0] = 10
        exp = df.copy()
        exp.iloc[[1, 3], 0] = 10
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.iloc[lambda x: [1, 3], [0]] = [-5, -5]
        exp = df.copy()
        exp.iloc[[1, 3], [0]] = [-5, -5]
        tm.assert_frame_equal(res, exp)


class TestILocSeries:
    def test_iloc(self):
        ser = Series(np.random.randn(10), index=list(range(0, 20, 2)))

        for i in range(len(ser)):
            result = ser.iloc[i]
            exp = ser[ser.index[i]]
            tm.assert_almost_equal(result, exp)

        # pass a slice
        result = ser.iloc[slice(1, 3)]
        expected = ser.loc[2:4]
        tm.assert_series_equal(result, expected)

        # test slice is a view
        result[:] = 0
        assert (ser[1:3] == 0).all()

        # list of integers
        result = ser.iloc[[0, 2, 3, 4, 5]]
        expected = ser.reindex(ser.index[[0, 2, 3, 4, 5]])
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_nonunique(self):
        ser = Series([0, 1, 2], index=[0, 1, 0])
        assert ser.iloc[2] == 2

    def test_iloc_setitem_pure_position_based(self):
        # GH#22046
        ser1 = Series([1, 2, 3])
        ser2 = Series([4, 5, 6], index=[1, 0, 2])
        ser1.iloc[1:3] = ser2.iloc[1:3]
        expected = Series([1, 5, 6])
        tm.assert_series_equal(ser1, expected)

    def test_iloc_nullable_int64_size_1_nan(self):
        # GH 31861
        result = DataFrame({"a": ["test"], "b": [np.nan]})
        result.loc[:, "b"] = result.loc[:, "b"].astype("Int64")
        expected = DataFrame({"a": ["test"], "b": array([NA], dtype="Int64")})
        tm.assert_frame_equal(result, expected)
