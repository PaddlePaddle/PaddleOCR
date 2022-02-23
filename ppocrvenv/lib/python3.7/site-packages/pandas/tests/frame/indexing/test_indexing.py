from datetime import (
    datetime,
    timedelta,
)
import re

import numpy as np
import pytest

from pandas._libs import iNaT
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    isna,
    notna,
)
import pandas._testing as tm
import pandas.core.common as com

# We pass through a TypeError raised by numpy
_slice_msg = "slice indices must be integers or None or have an __index__ method"


class TestDataFrameIndexing:
    def test_getitem(self, float_frame):
        # Slicing
        sl = float_frame[:20]
        assert len(sl.index) == 20

        # Column access
        for _, series in sl.items():
            assert len(series.index) == 20
            assert tm.equalContents(series.index, sl.index)

        for key, _ in float_frame._series.items():
            assert float_frame[key] is not None

        assert "random" not in float_frame
        with pytest.raises(KeyError, match="random"):
            float_frame["random"]

    def test_getitem2(self, float_frame):

        df = float_frame.copy()
        df["$10"] = np.random.randn(len(df))

        ad = np.random.randn(len(df))
        df["@awesome_domain"] = ad

        with pytest.raises(KeyError, match=re.escape("'df[\"$10\"]'")):
            df.__getitem__('df["$10"]')

        res = df["@awesome_domain"]
        tm.assert_numpy_array_equal(ad, res.values)

    def test_setitem_list(self, float_frame):

        float_frame["E"] = "foo"
        data = float_frame[["A", "B"]]
        float_frame[["B", "A"]] = data

        tm.assert_series_equal(float_frame["B"], data["A"], check_names=False)
        tm.assert_series_equal(float_frame["A"], data["B"], check_names=False)

        msg = "Columns must be same length as key"
        with pytest.raises(ValueError, match=msg):
            data[["A"]] = float_frame[["A", "B"]]
        newcolumndata = range(len(data.index) - 1)
        msg = (
            rf"Length of values \({len(newcolumndata)}\) "
            rf"does not match length of index \({len(data)}\)"
        )
        with pytest.raises(ValueError, match=msg):
            data["A"] = newcolumndata

    def test_setitem_list2(self):

        df = DataFrame(0, index=range(3), columns=["tt1", "tt2"], dtype=np.int_)
        df.loc[1, ["tt1", "tt2"]] = [1, 2]

        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series([1, 2], df.columns, dtype=np.int_, name=1)
        tm.assert_series_equal(result, expected)

        df["tt1"] = df["tt2"] = "0"
        df.loc[df.index[1], ["tt1", "tt2"]] = ["1", "2"]
        result = df.loc[df.index[1], ["tt1", "tt2"]]
        expected = Series(["1", "2"], df.columns, name=1)
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean(self, mixed_float_frame, mixed_int_frame, datetime_frame):
        # boolean indexing
        d = datetime_frame.index[10]
        indexer = datetime_frame.index > d
        indexer_obj = indexer.astype(object)

        subindex = datetime_frame.index[indexer]
        subframe = datetime_frame[indexer]

        tm.assert_index_equal(subindex, subframe.index)
        with pytest.raises(ValueError, match="Item wrong length"):
            datetime_frame[indexer[:-1]]

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        with pytest.raises(ValueError, match="Boolean array expected"):
            datetime_frame[datetime_frame]

        # test that Series work
        indexer_obj = Series(indexer_obj, datetime_frame.index)

        subframe_obj = datetime_frame[indexer_obj]
        tm.assert_frame_equal(subframe_obj, subframe)

        # test that Series indexers reindex
        # we are producing a warning that since the passed boolean
        # key is not the same as the given index, we will reindex
        # not sure this is really necessary
        with tm.assert_produces_warning(UserWarning):
            indexer_obj = indexer_obj.reindex(datetime_frame.index[::-1])
            subframe_obj = datetime_frame[indexer_obj]
            tm.assert_frame_equal(subframe_obj, subframe)

        # test df[df > 0]
        for df in [
            datetime_frame,
            mixed_float_frame,
            mixed_int_frame,
        ]:

            data = df._get_numeric_data()
            bif = df[df > 0]
            bifw = DataFrame(
                {c: np.where(data[c] > 0, data[c], np.nan) for c in data.columns},
                index=data.index,
                columns=data.columns,
            )

            # add back other columns to compare
            for c in df.columns:
                if c not in bifw:
                    bifw[c] = df[c]
            bifw = bifw.reindex(columns=df.columns)

            tm.assert_frame_equal(bif, bifw, check_dtype=False)
            for c in df.columns:
                if bif[c].dtype != bifw[c].dtype:
                    assert bif[c].dtype == df[c].dtype

    def test_getitem_boolean_casting(self, datetime_frame):

        # don't upcast if we don't need to
        df = datetime_frame.copy()
        df["E"] = 1
        df["E"] = df["E"].astype("int32")
        df["E1"] = df["E"].copy()
        df["F"] = 1
        df["F"] = df["F"].astype("int64")
        df["F1"] = df["F"].copy()

        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")] * 2
            + [np.dtype("int64")] * 2,
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

        # int block splitting
        df.loc[df.index[1:3], ["E1", "F1"]] = 0
        casted = df[df > 0]
        result = casted.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [np.dtype("int32")]
            + [np.dtype("float64")]
            + [np.dtype("int64")]
            + [np.dtype("float64")],
            index=["A", "B", "C", "D", "E", "E1", "F", "F1"],
        )
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_list(self):
        df = DataFrame(np.arange(12).reshape(3, 4))

        def _checkit(lst):
            result = df[lst]
            expected = df.loc[df.index[lst]]
            tm.assert_frame_equal(result, expected)

        _checkit([True, False, True])
        _checkit([True, True, True])
        _checkit([False, False, False])

    def test_getitem_boolean_iadd(self):
        arr = np.random.randn(5, 5)

        df = DataFrame(arr.copy(), columns=["A", "B", "C", "D", "E"])

        df[df < 0] += 1
        arr[arr < 0] += 1

        tm.assert_almost_equal(df.values, arr)

    def test_boolean_index_empty_corner(self):
        # #2096
        blah = DataFrame(np.empty([0, 1]), columns=["A"], index=DatetimeIndex([]))

        # both of these should succeed trivially
        k = np.array([], bool)

        blah[k]
        blah[k] = 0

    def test_getitem_ix_mixed_integer(self):
        df = DataFrame(
            np.random.randn(4, 3), index=[1, 10, "C", "E"], columns=[1, 2, 3]
        )

        result = df.iloc[:-1]
        expected = df.loc[df.index[:-1]]
        tm.assert_frame_equal(result, expected)

        result = df.loc[[1, 10]]
        expected = df.loc[Index([1, 10])]
        tm.assert_frame_equal(result, expected)

    def test_getitem_ix_mixed_integer2(self):
        # 11320
        df = DataFrame(
            {
                "rna": (1.5, 2.2, 3.2, 4.5),
                -1000: [11, 21, 36, 40],
                0: [10, 22, 43, 34],
                1000: [0, 10, 20, 30],
            },
            columns=["rna", -1000, 0, 1000],
        )
        result = df[[1000]]
        expected = df.iloc[:, [3]]
        tm.assert_frame_equal(result, expected)
        result = df[[-1000]]
        expected = df.iloc[:, [1]]
        tm.assert_frame_equal(result, expected)

    def test_getattr(self, float_frame):
        tm.assert_series_equal(float_frame.A, float_frame["A"])
        msg = "'DataFrame' object has no attribute 'NONEXISTENT_NAME'"
        with pytest.raises(AttributeError, match=msg):
            float_frame.NONEXISTENT_NAME

    def test_setattr_column(self):
        df = DataFrame({"foobar": 1}, index=range(10))

        df.foobar = 5
        assert (df.foobar == 5).all()

    def test_setitem(self, float_frame):
        # not sure what else to do here
        series = float_frame["A"][::2]
        float_frame["col5"] = series
        assert "col5" in float_frame

        assert len(series) == 15
        assert len(float_frame) == 30

        exp = np.ravel(np.column_stack((series.values, [np.nan] * 15)))
        exp = Series(exp, index=float_frame.index, name="col5")
        tm.assert_series_equal(float_frame["col5"], exp)

        series = float_frame["A"]
        float_frame["col6"] = series
        tm.assert_series_equal(series, float_frame["col6"], check_names=False)

        # set ndarray
        arr = np.random.randn(len(float_frame))
        float_frame["col9"] = arr
        assert (float_frame["col9"] == arr).all()

        float_frame["col7"] = 5
        assert (float_frame["col7"] == 5).all()

        float_frame["col0"] = 3.14
        assert (float_frame["col0"] == 3.14).all()

        float_frame["col8"] = "foo"
        assert (float_frame["col8"] == "foo").all()

        # this is partially a view (e.g. some blocks are view)
        # so raise/warn
        smaller = float_frame[:2]

        msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(com.SettingWithCopyError, match=msg):
            smaller["col10"] = ["1", "2"]

        assert smaller["col10"].dtype == np.object_
        assert (smaller["col10"] == ["1", "2"]).all()

    def test_setitem2(self):
        # dtype changing GH4204
        df = DataFrame([[0, 0]])
        df.iloc[0] = np.nan
        expected = DataFrame([[np.nan, np.nan]])
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[0, 0]])
        df.loc[0] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_setitem_boolean(self, float_frame):
        df = float_frame.copy()
        values = float_frame.values

        df[df["A"] > 0] = 4
        values[values[:, 0] > 0] = 4
        tm.assert_almost_equal(df.values, values)

        # test that column reindexing works
        series = df["A"] == 4
        series = series.reindex(df.index[::-1])
        df[series] = 1
        values[values[:, 0] == 4] = 1
        tm.assert_almost_equal(df.values, values)

        df[df > 0] = 5
        values[values > 0] = 5
        tm.assert_almost_equal(df.values, values)

        df[df == 5] = 0
        values[values == 5] = 0
        tm.assert_almost_equal(df.values, values)

        # a df that needs alignment first
        df[df[:-1] < 0] = 2
        np.putmask(values[:-1], values[:-1] < 0, 2)
        tm.assert_almost_equal(df.values, values)

        # indexed with same shape but rows-reversed df
        df[df[::-1] == 2] = 3
        values[values == 2] = 3
        tm.assert_almost_equal(df.values, values)

        msg = "Must pass DataFrame or 2-d ndarray with boolean values only"
        with pytest.raises(TypeError, match=msg):
            df[df * 0] = 2

        # index with DataFrame
        mask = df > np.abs(df)
        expected = df.copy()
        df[df > np.abs(df)] = np.nan
        expected.values[mask.values] = np.nan
        tm.assert_frame_equal(df, expected)

        # set from DataFrame
        expected = df.copy()
        df[df > np.abs(df)] = df * 2
        np.putmask(expected.values, mask.values, df.values * 2)
        tm.assert_frame_equal(df, expected)

    def test_setitem_cast(self, float_frame):
        float_frame["D"] = float_frame["D"].astype("i8")
        assert float_frame["D"].dtype == np.int64

        # #669, should not cast?
        # this is now set to int64, which means a replacement of the column to
        # the value dtype (and nothing to do with the existing dtype)
        float_frame["B"] = 0
        assert float_frame["B"].dtype == np.int64

        # cast if pass array of course
        float_frame["B"] = np.arange(len(float_frame))
        assert issubclass(float_frame["B"].dtype.type, np.integer)

        float_frame["foo"] = "bar"
        float_frame["foo"] = 0
        assert float_frame["foo"].dtype == np.int64

        float_frame["foo"] = "bar"
        float_frame["foo"] = 2.5
        assert float_frame["foo"].dtype == np.float64

        float_frame["something"] = 0
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2
        assert float_frame["something"].dtype == np.int64
        float_frame["something"] = 2.5
        assert float_frame["something"].dtype == np.float64

    def test_setitem_corner(self, float_frame):
        # corner case
        df = DataFrame({"B": [1.0, 2.0, 3.0], "C": ["a", "b", "c"]}, index=np.arange(3))
        del df["B"]
        df["B"] = [1.0, 2.0, 3.0]
        assert "B" in df
        assert len(df.columns) == 2

        df["A"] = "beginning"
        df["E"] = "foo"
        df["D"] = "bar"
        df[datetime.now()] = "date"
        df[datetime.now()] = 5.0

        # what to do when empty frame with index
        dm = DataFrame(index=float_frame.index)
        dm["A"] = "foo"
        dm["B"] = "bar"
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.object_

        # upcast
        dm["C"] = 1
        assert dm["C"].dtype == np.int64

        dm["E"] = 1.0
        assert dm["E"].dtype == np.float64

        # set existing column
        dm["A"] = "bar"
        assert "bar" == dm["A"][0]

        dm = DataFrame(index=np.arange(3))
        dm["A"] = 1
        dm["foo"] = "bar"
        del dm["foo"]
        dm["foo"] = "bar"
        assert dm["foo"].dtype == np.object_

        dm["coercible"] = ["1", "2", "3"]
        assert dm["coercible"].dtype == np.object_

    def test_setitem_corner2(self):
        data = {
            "title": ["foobar", "bar", "foobar"] + ["foobar"] * 17,
            "cruft": np.random.random(20),
        }

        df = DataFrame(data)
        ix = df[df["title"] == "bar"].index

        df.loc[ix, ["title"]] = "foobar"
        df.loc[ix, ["cruft"]] = 0

        assert df.loc[1, "title"] == "foobar"
        assert df.loc[1, "cruft"] == 0

    def test_setitem_ambig(self):
        # Difficulties with mixed-type data
        from decimal import Decimal

        # Created as float type
        dm = DataFrame(index=range(3), columns=range(3))

        coercable_series = Series([Decimal(1) for _ in range(3)], index=range(3))
        uncoercable_series = Series(["foo", "bzr", "baz"], index=range(3))

        dm[0] = np.ones(3)
        assert len(dm.columns) == 3

        dm[1] = coercable_series
        assert len(dm.columns) == 3

        dm[2] = uncoercable_series
        assert len(dm.columns) == 3
        assert dm[2].dtype == np.object_

    def test_setitem_None(self, float_frame):
        # GH #766
        float_frame[None] = float_frame["A"]
        tm.assert_series_equal(
            float_frame.iloc[:, -1], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(
            float_frame.loc[:, None], float_frame["A"], check_names=False
        )
        tm.assert_series_equal(float_frame[None], float_frame["A"], check_names=False)
        repr(float_frame)

    def test_loc_setitem_boolean_mask_allfalse(self):
        # GH 9596
        df = DataFrame(
            {"a": ["1", "2", "3"], "b": ["11", "22", "33"], "c": ["111", "222", "333"]}
        )

        result = df.copy()
        result.loc[result.b.isna(), "a"] = result.a
        tm.assert_frame_equal(result, df)

    def test_getitem_fancy_slice_integers_step(self):
        df = DataFrame(np.random.randn(10, 5))

        # this is OK
        result = df.iloc[:8:2]  # noqa
        df.iloc[:8:2] = np.nan
        assert isna(df.iloc[:8:2]).values.all()

    def test_getitem_setitem_integer_slice_keyerrors(self):
        df = DataFrame(np.random.randn(10, 5), index=range(0, 20, 2))

        # this is OK
        cp = df.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).values.all()

        # so is this
        cp = df.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        result = df.iloc[2:6]
        result2 = df.loc[3:11]
        expected = df.reindex([4, 6, 8, 10])

        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        # non-monotonic, raise KeyError
        df2 = df.iloc[list(range(5)) + list(range(5, 10))[::-1]]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11]
        with pytest.raises(KeyError, match=r"^3$"):
            df2.loc[3:11] = 0

    @td.skip_array_manager_invalid_test  # already covered in test_iloc_col_slice_view
    def test_fancy_getitem_slice_mixed(self, float_frame, float_string_frame):
        sliced = float_string_frame.iloc[:, -3:]
        assert sliced["D"].dtype == np.float64

        # get view with single block
        # setting it triggers setting with copy
        sliced = float_frame.iloc[:, -3:]

        msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(com.SettingWithCopyError, match=msg):
            sliced["C"] = 4.0

        assert (float_frame["C"] == 4).all()

    def test_getitem_setitem_non_ix_labels(self):
        df = tm.makeTimeDataFrame()

        start, end = df.index[[5, 10]]

        result = df.loc[start:end]
        result2 = df[start:end]
        expected = df[5:11]
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

        result = df.copy()
        result.loc[start:end] = 0
        result2 = df.copy()
        result2[start:end] = 0
        expected = df.copy()
        expected[5:11] = 0
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(result2, expected)

    def test_ix_multi_take(self):
        df = DataFrame(np.random.randn(3, 2))
        rs = df.loc[df.index == 0, :]
        xp = df.reindex([0])
        tm.assert_frame_equal(rs, xp)

        # FIXME: dont leave commented-out
        """ #1321
        df = DataFrame(np.random.randn(3, 2))
        rs = df.loc[df.index==0, df.columns==1]
        xp = df.reindex([0], [1])
        tm.assert_frame_equal(rs, xp)
        """

    def test_getitem_fancy_scalar(self, float_frame):
        f = float_frame
        ix = f.loc

        # individual value
        for col in f.columns:
            ts = f[col]
            for idx in f.index[::5]:
                assert ix[idx, col] == ts[idx]

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    def test_setitem_fancy_scalar(self, float_frame):
        f = float_frame
        expected = float_frame.copy()
        ix = f.loc

        # individual value
        for j, col in enumerate(f.columns):
            ts = f[col]  # noqa
            for idx in f.index[::5]:
                i = f.index.get_loc(idx)
                val = np.random.randn()
                expected.values[i, j] = val

                ix[idx, col] = val
                tm.assert_frame_equal(f, expected)

    def test_getitem_fancy_boolean(self, float_frame):
        f = float_frame
        ix = f.loc

        expected = f.reindex(columns=["B", "D"])
        result = ix[:, [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        expected = f.reindex(index=f.index[5:10], columns=["B", "D"])
        result = ix[f.index[5:10], [False, True, False, True]]
        tm.assert_frame_equal(result, expected)

        boolvec = f.index > f.index[7]
        expected = f.reindex(index=f.index[boolvec])
        result = ix[boolvec]
        tm.assert_frame_equal(result, expected)
        result = ix[boolvec, :]
        tm.assert_frame_equal(result, expected)

        result = ix[boolvec, f.columns[2:]]
        expected = f.reindex(index=f.index[boolvec], columns=["C", "D"])
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    def test_setitem_fancy_boolean(self, float_frame):
        # from 2d, set with booleans
        frame = float_frame.copy()
        expected = float_frame.copy()

        mask = frame["A"] > 0
        frame.loc[mask] = 0.0
        expected.values[mask.values] = 0.0
        tm.assert_frame_equal(frame, expected)

        frame = float_frame.copy()
        expected = float_frame.copy()
        frame.loc[mask, ["A", "B"]] = 0.0
        expected.values[mask.values, :2] = 0.0
        tm.assert_frame_equal(frame, expected)

    def test_getitem_fancy_ints(self, float_frame):
        result = float_frame.iloc[[1, 4, 7]]
        expected = float_frame.loc[float_frame.index[[1, 4, 7]]]
        tm.assert_frame_equal(result, expected)

        result = float_frame.iloc[:, [2, 0, 1]]
        expected = float_frame.loc[:, float_frame.columns[[2, 0, 1]]]
        tm.assert_frame_equal(result, expected)

    def test_getitem_setitem_boolean_misaligned(self, float_frame):
        # boolean index misaligned labels
        mask = float_frame["A"][::-1] > 1

        result = float_frame.loc[mask]
        expected = float_frame.loc[mask[::-1]]
        tm.assert_frame_equal(result, expected)

        cp = float_frame.copy()
        expected = float_frame.copy()
        cp.loc[mask] = 0
        expected.loc[mask] = 0
        tm.assert_frame_equal(cp, expected)

    def test_getitem_setitem_boolean_multi(self):
        df = DataFrame(np.random.randn(3, 2))

        # get
        k1 = np.array([True, False, True])
        k2 = np.array([False, True])
        result = df.loc[k1, k2]
        expected = df.loc[[0, 2], [1]]
        tm.assert_frame_equal(result, expected)

        expected = df.copy()
        df.loc[np.array([True, False, True]), np.array([False, True])] = 5
        expected.loc[[0, 2], [1]] = 5
        tm.assert_frame_equal(df, expected)

    def test_getitem_setitem_float_labels(self):
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.random.randn(5, 5), index=index)

        result = df.loc[1.5:4]
        expected = df.reindex([1.5, 2, 3, 4])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4:5]
        expected = df.reindex([4, 5])  # reindex with int
        tm.assert_frame_equal(result, expected, check_index_type=False)
        assert len(result) == 2

        result = df.loc[4:5]
        expected = df.reindex([4.0, 5.0])  # reindex with float
        tm.assert_frame_equal(result, expected)
        assert len(result) == 2

        # loc_float changes this to work properly
        result = df.loc[1:2]
        expected = df.iloc[0:2]
        tm.assert_frame_equal(result, expected)

        df.loc[1:2] = 0
        result = df[1:2]
        assert (result == 0).all().all()

        # #2727
        index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
        df = DataFrame(np.random.randn(5, 5), index=index)

        # positional slicing only via iloc!
        msg = (
            "cannot do positional indexing on Float64Index with "
            r"these indexers \[1.0\] of type float"
        )
        with pytest.raises(TypeError, match=msg):
            df.iloc[1.0:5]

        result = df.iloc[4:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        cp = df.copy()

        with pytest.raises(TypeError, match=_slice_msg):
            cp.iloc[1.0:5] = 0

        with pytest.raises(TypeError, match=msg):
            result = cp.iloc[1.0:5] == 0

        assert result.values.all()
        assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()

        cp = df.copy()
        cp.iloc[4:5] = 0
        assert (cp.iloc[4:5] == 0).values.all()
        assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()

        # float slicing
        result = df.loc[1.0:5]
        expected = df
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        result = df.loc[1.1:5]
        expected = df.reindex([2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 4

        result = df.loc[4.51:5]
        expected = df.reindex([5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 1

        result = df.loc[1.0:5.0]
        expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
        tm.assert_frame_equal(result, expected)
        assert len(result) == 5

        cp = df.copy()
        cp.loc[1.0:5.0] = 0
        result = cp.loc[1.0:5.0]
        assert (result == 0).values.all()

    def test_setitem_single_column_mixed_datetime(self):
        df = DataFrame(
            np.random.randn(5, 3),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )

        df["timestamp"] = Timestamp("20010102")

        # check our dtypes
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("datetime64[ns]")],
            index=["foo", "bar", "baz", "timestamp"],
        )
        tm.assert_series_equal(result, expected)

        # GH#16674 iNaT is treated as an integer when given by the user
        df.loc["b", "timestamp"] = iNaT
        assert not isna(df.loc["b", "timestamp"])
        assert df["timestamp"].dtype == np.object_
        assert df.loc["b", "timestamp"] == iNaT

        # allow this syntax
        df.loc["c", "timestamp"] = np.nan
        assert isna(df.loc["c", "timestamp"])

        # allow this syntax
        df.loc["d", :] = np.nan
        assert not isna(df.loc["c", :]).all()

        # FIXME: don't leave commented-out
        # as of GH 3216 this will now work!
        # try to set with a list like item
        # pytest.raises(
        #    Exception, df.loc.__setitem__, ('d', 'timestamp'), [np.nan])

    def test_setitem_mixed_datetime(self):
        # GH 9336
        expected = DataFrame(
            {
                "a": [0, 0, 0, 0, 13, 14],
                "b": [
                    datetime(2012, 1, 1),
                    1,
                    "x",
                    "y",
                    datetime(2013, 1, 1),
                    datetime(2014, 1, 1),
                ],
            }
        )
        df = DataFrame(0, columns=list("ab"), index=range(6))
        df["b"] = pd.NaT
        df.loc[0, "b"] = datetime(2012, 1, 1)
        df.loc[1, "b"] = 1
        df.loc[[2, 3], "b"] = "x", "y"
        A = np.array(
            [
                [13, np.datetime64("2013-01-01T00:00:00")],
                [14, np.datetime64("2014-01-01T00:00:00")],
            ]
        )
        df.loc[[4, 5], ["a", "b"]] = A
        tm.assert_frame_equal(df, expected)

    def test_setitem_frame_float(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        float_frame.loc[float_frame.index[-2] :, ["A", "B"]] = piece.values
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_setitem_frame_mixed(self, float_string_frame):
        # GH 3216

        # already aligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=f.index[0:2], columns=["A", "B"]
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(f.loc[f.index[0:2], ["A", "B"]].values, piece.values)

    def test_setitem_frame_mixed_rows_unaligned(self, float_string_frame):
        # GH#3216 rows unaligned
        f = float_string_frame.copy()
        piece = DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            index=list(f.index[0:2]) + ["foo", "bar"],
            columns=["A", "B"],
        )
        key = (f.index[slice(None, 2)], ["A", "B"])
        f.loc[key] = piece
        tm.assert_almost_equal(
            f.loc[f.index[0:2:], ["A", "B"]].values, piece.values[0:2]
        )

    def test_setitem_frame_mixed_key_unaligned(self, float_string_frame):
        # GH#3216 key is unaligned with values
        f = float_string_frame.copy()
        piece = f.loc[f.index[:2], ["A"]]
        piece.index = f.index[-2:]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece
        piece["B"] = np.nan
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_mixed_ndarray(self, float_string_frame):
        # GH#3216 ndarray
        f = float_string_frame.copy()
        piece = float_string_frame.loc[f.index[:2], ["A", "B"]]
        key = (f.index[slice(-2, None)], ["A", "B"])
        f.loc[key] = piece.values
        tm.assert_almost_equal(f.loc[f.index[-2:], ["A", "B"]].values, piece.values)

    def test_setitem_frame_upcast(self):
        # needs upcasting
        df = DataFrame([[1, 2, "foo"], [3, 4, "bar"]], columns=["A", "B", "C"])
        df2 = df.copy()
        df2.loc[:, ["A", "B"]] = df.loc[:, ["A", "B"]] + 0.5
        expected = df.reindex(columns=["A", "B"])
        expected += 0.5
        expected["C"] = df["C"]
        tm.assert_frame_equal(df2, expected)

    def test_setitem_frame_align(self, float_frame):
        piece = float_frame.loc[float_frame.index[:2], ["A", "B"]]
        piece.index = float_frame.index[-2:]
        piece.columns = ["A", "B"]
        float_frame.loc[float_frame.index[-2:], ["A", "B"]] = piece
        result = float_frame.loc[float_frame.index[-2:], ["A", "B"]].values
        expected = piece.values
        tm.assert_almost_equal(result, expected)

    def test_getitem_setitem_ix_duplicates(self):
        # #1201
        df = DataFrame(np.random.randn(5, 3), index=["foo", "foo", "bar", "baz", "bar"])

        result = df.loc["foo"]
        expected = df[:2]
        tm.assert_frame_equal(result, expected)

        result = df.loc["bar"]
        expected = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, expected)

        result = df.loc["baz"]
        expected = df.iloc[3]
        tm.assert_series_equal(result, expected)

    def test_getitem_ix_boolean_duplicates_multiple(self):
        # #1201
        df = DataFrame(np.random.randn(5, 3), index=["foo", "foo", "bar", "baz", "bar"])

        result = df.loc[["bar"]]
        exp = df.iloc[[2, 4]]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[1] > 0]
        exp = df[df[1] > 0]
        tm.assert_frame_equal(result, exp)

        result = df.loc[df[0] > 0]
        exp = df[df[0] > 0]
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("bool_value", [True, False])
    def test_getitem_setitem_ix_bool_keyerror(self, bool_value):
        # #2199
        df = DataFrame({"a": [1, 2, 3]})
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        with pytest.raises(KeyError, match=message):
            df.loc[bool_value]

        msg = "cannot use a single bool to index into setitem"
        with pytest.raises(KeyError, match=msg):
            df.loc[bool_value] = 0

    # TODO: rename?  remove?
    def test_single_element_ix_dont_upcast(self, float_frame):
        float_frame["E"] = 1
        assert issubclass(float_frame["E"].dtype.type, (int, np.integer))

        result = float_frame.loc[float_frame.index[5], "E"]
        assert is_integer(result)

        # GH 11617
        df = DataFrame({"a": [1.23]})
        df["b"] = 666

        result = df.loc[0, "b"]
        assert is_integer(result)

        expected = Series([666], [0], name="b")
        result = df.loc[[0], "b"]
        tm.assert_series_equal(result, expected)

    def test_iloc_row(self):
        df = DataFrame(np.random.randn(10, 4), index=range(0, 20, 2))

        result = df.iloc[1]
        exp = df.loc[2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[2]
        exp = df.loc[4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[slice(4, 8)]
        expected = df.loc[8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[[1, 2, 4, 6]]
        expected = df.reindex(df.index[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_row_slice_view(self, using_array_manager):
        df = DataFrame(np.random.randn(10, 4), index=range(0, 20, 2))
        original = df.copy()

        # verify slice is view
        # setting it makes it raise/warn
        subset = df.iloc[slice(4, 8)]

        msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
        with pytest.raises(com.SettingWithCopyError, match=msg):
            subset[2] = 0.0

        exp_col = original[2].copy()
        # TODO(ArrayManager) verify it is expected that the original didn't change
        if not using_array_manager:
            exp_col[4:8] = 0.0
        tm.assert_series_equal(df[2], exp_col)

    def test_iloc_col(self):

        df = DataFrame(np.random.randn(4, 10), columns=range(0, 20, 2))

        result = df.iloc[:, 1]
        exp = df.loc[:, 2]
        tm.assert_series_equal(result, exp)

        result = df.iloc[:, 2]
        exp = df.loc[:, 4]
        tm.assert_series_equal(result, exp)

        # slice
        result = df.iloc[:, slice(4, 8)]
        expected = df.loc[:, 8:14]
        tm.assert_frame_equal(result, expected)

        # list of integers
        result = df.iloc[:, [1, 2, 4, 6]]
        expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
        tm.assert_frame_equal(result, expected)

    def test_iloc_col_slice_view(self, using_array_manager):
        df = DataFrame(np.random.randn(4, 10), columns=range(0, 20, 2))
        original = df.copy()
        subset = df.iloc[:, slice(4, 8)]

        if not using_array_manager:
            # verify slice is view
            # and that we are setting a copy
            msg = r"\nA value is trying to be set on a copy of a slice from a DataFrame"
            with pytest.raises(com.SettingWithCopyError, match=msg):
                subset[8] = 0.0

            assert (df[8] == 0).all()
        else:
            # TODO(ArrayManager) verify this is the desired behaviour
            subset[8] = 0.0
            # subset changed
            assert (subset[8] == 0).all()
            # but df itself did not change (setitem replaces full column)
            tm.assert_frame_equal(df, original)

    def test_loc_duplicates(self):
        # gh-17105

        # insert a duplicate element to the index
        trange = date_range(
            start=Timestamp(year=2017, month=1, day=1),
            end=Timestamp(year=2017, month=1, day=5),
        )

        trange = trange.insert(loc=5, item=Timestamp(year=2017, month=1, day=5))

        df = DataFrame(0, index=trange, columns=["A", "B"])
        bool_idx = np.array([False, False, False, False, False, True])

        # assignment
        df.loc[trange[bool_idx], "A"] = 6

        expected = DataFrame(
            {"A": [0, 0, 0, 0, 6, 6], "B": [0, 0, 0, 0, 0, 0]}, index=trange
        )
        tm.assert_frame_equal(df, expected)

        # in-place
        df = DataFrame(0, index=trange, columns=["A", "B"])
        df.loc[trange[bool_idx], "A"] += 6
        tm.assert_frame_equal(df, expected)

    def test_setitem_with_unaligned_tz_aware_datetime_column(self):
        # GH 12981
        # Assignment of unaligned offset-aware datetime series.
        # Make sure timezone isn't lost
        column = Series(date_range("2015-01-01", periods=3, tz="utc"), name="dates")
        df = DataFrame({"dates": column})
        df["dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

        df = DataFrame({"dates": column})
        df.loc[[0, 1, 2], "dates"] = column[[1, 0, 2]]
        tm.assert_series_equal(df["dates"], column)

    def test_loc_setitem_datetimelike_with_inference(self):
        # GH 7592
        # assignment of timedeltas with NaT

        one_hour = timedelta(hours=1)
        df = DataFrame(index=date_range("20130101", periods=4))
        df["A"] = np.array([1 * one_hour] * 4, dtype="m8[ns]")
        df.loc[:, "B"] = np.array([2 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "C"] = np.array([3 * one_hour] * 3, dtype="m8[ns]")
        df.loc[:, "D"] = np.array([4 * one_hour] * 4, dtype="m8[ns]")
        df.loc[df.index[:3], "E"] = np.array([5 * one_hour] * 3, dtype="m8[ns]")
        df["F"] = np.timedelta64("NaT")
        df.loc[df.index[:-1], "F"] = np.array([6 * one_hour] * 3, dtype="m8[ns]")
        df.loc[df.index[-3] :, "G"] = date_range("20130101", periods=3)
        df["H"] = np.datetime64("NaT")
        result = df.dtypes
        expected = Series(
            [np.dtype("timedelta64[ns]")] * 6 + [np.dtype("datetime64[ns]")] * 2,
            index=list("ABCDEFGH"),
        )
        tm.assert_series_equal(result, expected)

    def test_getitem_boolean_indexing_mixed(self):
        df = DataFrame(
            {
                0: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                1: {
                    35: np.nan,
                    40: 0.32632316859446198,
                    43: np.nan,
                    49: 0.32632316859446198,
                    50: 0.39114724480578139,
                },
                2: {
                    35: np.nan,
                    40: np.nan,
                    43: 0.29012581014105987,
                    49: np.nan,
                    50: np.nan,
                },
                3: {35: np.nan, 40: np.nan, 43: np.nan, 49: np.nan, 50: np.nan},
                4: {
                    35: 0.34215328467153283,
                    40: np.nan,
                    43: np.nan,
                    49: np.nan,
                    50: np.nan,
                },
                "y": {35: 0, 40: 0, 43: 0, 49: 0, 50: 1},
            }
        )

        # mixed int/float ok
        df2 = df.copy()
        df2[df2 > 0.3] = 1
        expected = df.copy()
        expected.loc[40, 1] = 1
        expected.loc[49, 1] = 1
        expected.loc[50, 1] = 1
        expected.loc[35, 4] = 1
        tm.assert_frame_equal(df2, expected)

        df["foo"] = "test"
        msg = "not supported between instances|unorderable types"

        with pytest.raises(TypeError, match=msg):
            df[df > 0.3] = 1

    def test_type_error_multiindex(self):
        # See gh-12218
        df = DataFrame(
            columns=["i", "c", "x", "y"],
            data=[[0, 0, 1, 2], [1, 0, 3, 4], [0, 1, 1, 2], [1, 1, 3, 4]],
        )
        dg = df.pivot_table(index="i", columns="c", values=["x", "y"])
        # TODO: Is this test for pivot_table?
        with pytest.raises(TypeError, match="unhashable type"):
            dg[:, 0]

        index = Index(range(2), name="i")
        columns = MultiIndex(
            levels=[["x", "y"], [0, 1]], codes=[[0, 1], [0, 0]], names=[None, "c"]
        )
        expected = DataFrame([[1, 2], [3, 4]], columns=columns, index=index)

        result = dg.loc[:, (slice(None), 0)]
        tm.assert_frame_equal(result, expected)

        name = ("x", 0)
        index = Index(range(2), name="i")
        expected = Series([1, 3], index=index, name=name)

        result = dg["x", 0]
        tm.assert_series_equal(result, expected)

    def test_getitem_interval_index_partial_indexing(self):
        # GH#36490
        df = DataFrame(
            np.ones((3, 4)), columns=pd.IntervalIndex.from_breaks(np.arange(5))
        )

        expected = df.iloc[:, 0]

        res = df[0.5]
        tm.assert_series_equal(res, expected)

        res = df.loc[:, 0.5]
        tm.assert_series_equal(res, expected)

    def test_setitem_array_as_cell_value(self):
        # GH#43422
        df = DataFrame(columns=["a", "b"], dtype=object)
        df.loc[0] = {"a": np.zeros((2,)), "b": np.zeros((2, 2))}
        expected = DataFrame({"a": [np.zeros((2,))], "b": [np.zeros((2, 2))]})
        tm.assert_frame_equal(df, expected)


class TestDataFrameIndexingUInt64:
    def test_setitem(self, uint64_frame):

        df = uint64_frame
        idx = df["A"].rename("foo")

        # setitem
        assert "C" not in df.columns
        df["C"] = idx
        tm.assert_series_equal(df["C"], Series(idx, name="C"))

        assert "D" not in df.columns
        df["D"] = "foo"
        df["D"] = idx
        tm.assert_series_equal(df["D"], Series(idx, name="D"))
        del df["D"]

        # With NaN: because uint64 has no NaN element,
        # the column should be cast to object.
        df2 = df.copy()
        df2.iloc[1, 1] = pd.NaT
        df2.iloc[1, 2] = pd.NaT
        result = df2["B"]
        tm.assert_series_equal(notna(result), Series([True, False, True], name="B"))
        tm.assert_series_equal(
            df2.dtypes,
            Series(
                [np.dtype("uint64"), np.dtype("O"), np.dtype("O")],
                index=["A", "B", "C"],
            ),
        )


def test_object_casting_indexing_wraps_datetimelike(using_array_manager):
    # GH#31649, check the indexing methods all the way down the stack
    df = DataFrame(
        {
            "A": [1, 2],
            "B": date_range("2000", periods=2),
            "C": pd.timedelta_range("1 Day", periods=2),
        }
    )

    ser = df.loc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.iloc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    ser = df.xs(0, axis=0)
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)

    if using_array_manager:
        # remainder of the test checking BlockManager internals
        return

    mgr = df._mgr
    mgr._rebuild_blknos_and_blklocs()
    arr = mgr.fast_xs(0)
    assert isinstance(arr[1], Timestamp)
    assert isinstance(arr[2], pd.Timedelta)

    blk = mgr.blocks[mgr.blknos[1]]
    assert blk.dtype == "M8[ns]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, Timestamp)

    blk = mgr.blocks[mgr.blknos[2]]
    assert blk.dtype == "m8[ns]"  # we got the right block
    val = blk.iget((0, 0))
    assert isinstance(val, pd.Timedelta)


msg1 = "Cannot setitem on a Categorical with a new category, set the categories first"
msg2 = "Cannot set a Categorical with another, without identical categories"


class TestLocILocDataFrameCategorical:
    @pytest.fixture
    def orig(self):
        cats = Categorical(["a", "a", "a", "a", "a", "a", "a"], categories=["a", "b"])
        idx = Index(["h", "i", "j", "k", "l", "m", "n"])
        values = [1, 1, 1, 1, 1, 1, 1]
        orig = DataFrame({"cats": cats, "values": values}, index=idx)
        return orig

    @pytest.fixture
    def exp_single_row(self):
        # The expected values if we change a single row
        cats1 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx1 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values1 = [1, 1, 2, 1, 1, 1, 1]
        exp_single_row = DataFrame({"cats": cats1, "values": values1}, index=idx1)
        return exp_single_row

    @pytest.fixture
    def exp_multi_row(self):
        # assign multiple rows (mixed values) (-> array) -> exp_multi_row
        # changed multiple rows
        cats2 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx2 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values2 = [1, 1, 2, 2, 1, 1, 1]
        exp_multi_row = DataFrame({"cats": cats2, "values": values2}, index=idx2)
        return exp_multi_row

    @pytest.fixture
    def exp_parts_cats_col(self):
        # changed part of the cats column
        cats3 = Categorical(["a", "a", "b", "b", "a", "a", "a"], categories=["a", "b"])
        idx3 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values3 = [1, 1, 1, 1, 1, 1, 1]
        exp_parts_cats_col = DataFrame({"cats": cats3, "values": values3}, index=idx3)
        return exp_parts_cats_col

    @pytest.fixture
    def exp_single_cats_value(self):
        # changed single value in cats col
        cats4 = Categorical(["a", "a", "b", "a", "a", "a", "a"], categories=["a", "b"])
        idx4 = Index(["h", "i", "j", "k", "l", "m", "n"])
        values4 = [1, 1, 1, 1, 1, 1, 1]
        exp_single_cats_value = DataFrame(
            {"cats": cats4, "values": values4}, index=idx4
        )
        return exp_single_cats_value

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_list_of_lists(self, orig, exp_multi_row, indexer):
        #   - assign multiple rows (mixed values) -> exp_multi_row
        df = orig.copy()

        key = slice(2, 4)
        if indexer is tm.loc:
            key = slice("j", "k")

        indexer(df)[key, :] = [["b", 2], ["b", 2]]
        tm.assert_frame_equal(df, exp_multi_row)

        df = orig.copy()
        with pytest.raises(ValueError, match=msg1):
            indexer(df)[key, :] = [["c", 2], ["c", 2]]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc, tm.at, tm.iat])
    def test_loc_iloc_at_iat_setitem_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        #   - assign a single value -> exp_single_cats_value
        df = orig.copy()

        key = (2, 0)
        if indexer in [tm.loc, tm.at]:
            key = (df.index[2], df.columns[0])

        # "b" is among the categories for df["cat"}]
        indexer(df)[key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(ValueError, match=msg1):
            indexer(df)[key] = "c"

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_mask_single_value_in_categories(
        self, orig, exp_single_cats_value, indexer
    ):
        # mask with single True
        df = orig.copy()

        mask = df.index == "j"
        key = 0
        if indexer is tm.loc:
            key = df.columns[key]

        indexer(df)[mask, key] = "b"
        tm.assert_frame_equal(df, exp_single_cats_value)

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_full_row_non_categorical_rhs(
        self, orig, exp_single_row, indexer
    ):
        #   - assign a complete row (mixed values) -> exp_single_row
        df = orig.copy()

        key = 2
        if indexer is tm.loc:
            key = df.index[2]

        # not categorical dtype, but "b" _is_ among the categories for df["cat"]
        indexer(df)[key, :] = ["b", 2]
        tm.assert_frame_equal(df, exp_single_row)

        # "c" is not among the categories for df["cat"]
        with pytest.raises(ValueError, match=msg1):
            indexer(df)[key, :] = ["c", 2]

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_partial_col_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer
    ):
        # assign a part of a column with dtype == categorical ->
        # exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # same categories as we currently have in df["cats"]
        compat = Categorical(["b", "b"], categories=["a", "b"])
        indexer(df)[key] = compat
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # categories do not match df["cat"]'s, but "b" is among them
        semi_compat = Categorical(list("bb"), categories=list("abc"))
        with pytest.raises(ValueError, match=msg2):
            # different categories but holdable values
            #  -> not sure if this should fail or pass
            indexer(df)[key] = semi_compat

        # categories do not match df["cat"]'s, and "c" is not among them
        incompat = Categorical(list("cc"), categories=list("abc"))
        with pytest.raises(ValueError, match=msg2):
            # different values
            indexer(df)[key] = incompat

    @pytest.mark.parametrize("indexer", [tm.loc, tm.iloc])
    def test_loc_iloc_setitem_non_categorical_rhs(
        self, orig, exp_parts_cats_col, indexer
    ):
        # assign a part of a column with dtype != categorical -> exp_parts_cats_col
        df = orig.copy()

        key = (slice(2, 4), 0)
        if indexer is tm.loc:
            key = (slice("j", "k"), df.columns[0])

        # "b" is among the categories for df["cat"]
        indexer(df)[key] = ["b", "b"]
        tm.assert_frame_equal(df, exp_parts_cats_col)

        # "c" not part of the categories
        with pytest.raises(ValueError, match=msg1):
            indexer(df)[key] = ["c", "c"]
