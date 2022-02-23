from datetime import (
    date,
    datetime,
)

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.core.indexing import IndexingError

from pandas.tseries.offsets import BDay


class TestSetitemDT64Values:
    def test_setitem_none_nan(self):
        series = Series(date_range("1/1/2000", periods=10))
        series[3] = None
        assert series[3] is NaT

        series[3:5] = None
        assert series[4] is NaT

        series[5] = np.nan
        assert series[5] is NaT

        series[5:7] = np.nan
        assert series[6] is NaT

    def test_setitem_multiindex_empty_slice(self):
        # https://github.com/pandas-dev/pandas/issues/35878
        idx = MultiIndex.from_tuples([("a", 1), ("b", 2)])
        result = Series([1, 2], index=idx)
        expected = result.copy()
        result.loc[[]] = 0
        tm.assert_series_equal(result, expected)

    def test_setitem_with_string_index(self):
        # GH#23451
        ser = Series([1, 2, 3], index=["Date", "b", "other"])
        ser["Date"] = date.today()
        assert ser.Date == date.today()
        assert ser["Date"] == date.today()

    def test_setitem_tuple_with_datetimetz_values(self):
        # GH#20441
        arr = date_range("2017", periods=4, tz="US/Eastern")
        index = [(0, 1), (0, 2), (0, 3), (0, 4)]
        result = Series(arr, index=index)
        expected = result.copy()
        result[(0, 1)] = np.nan
        expected.iloc[0] = np.nan
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Tokyo"])
    def test_setitem_with_tz(self, tz, indexer_sli):
        orig = Series(date_range("2016-01-01", freq="H", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2016-01-01 02:00", tz=tz),
            ]
        )

        # scalar
        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        # vector
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-01-01 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ]
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)

    def test_setitem_with_tz_dst(self, indexer_sli):
        # GH XXX TODO: fill in GH ref
        tz = "US/Eastern"
        orig = Series(date_range("2016-11-06", freq="H", periods=3, tz=tz))
        assert orig.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00-04:00", tz=tz),
                Timestamp("2011-01-01 00:00-05:00", tz=tz),
                Timestamp("2016-11-06 01:00-05:00", tz=tz),
            ]
        )

        # scalar
        ser = orig.copy()
        indexer_sli(ser)[1] = Timestamp("2011-01-01", tz=tz)
        tm.assert_series_equal(ser, exp)

        # vector
        vals = Series(
            [Timestamp("2011-01-01", tz=tz), Timestamp("2012-01-01", tz=tz)],
            index=[1, 2],
        )
        assert vals.dtype == f"datetime64[ns, {tz}]"

        exp = Series(
            [
                Timestamp("2016-11-06 00:00", tz=tz),
                Timestamp("2011-01-01 00:00", tz=tz),
                Timestamp("2012-01-01 00:00", tz=tz),
            ]
        )

        ser = orig.copy()
        indexer_sli(ser)[[1, 2]] = vals
        tm.assert_series_equal(ser, exp)


class TestSetitemScalarIndexer:
    def test_setitem_negative_out_of_bounds(self):
        ser = Series(tm.rands_array(5, 10), index=tm.rands_array(10, 10))

        msg = "index -11 is out of bounds for axis 0 with size 10"
        with pytest.raises(IndexError, match=msg):
            ser[-11] = "foo"

    @pytest.mark.parametrize("indexer", [tm.loc, tm.at])
    @pytest.mark.parametrize("ser_index", [0, 1])
    def test_setitem_series_object_dtype(self, indexer, ser_index):
        # GH#38303
        ser = Series([0, 0], dtype="object")
        idxr = indexer(ser)
        idxr[0] = Series([42], index=[ser_index])
        expected = Series([Series([42], index=[ser_index]), 0], dtype="object")
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("index, exp_value", [(0, 42), (1, np.nan)])
    def test_setitem_series(self, index, exp_value):
        # GH#38303
        ser = Series([0, 0])
        ser.loc[0] = Series([42], index=[index])
        expected = Series([exp_value, 0])
        tm.assert_series_equal(ser, expected)


class TestSetitemSlices:
    def test_setitem_slice_float_raises(self, datetime_series):
        msg = (
            "cannot do slice indexing on DatetimeIndex with these indexers "
            r"\[{key}\] of type float"
        )
        with pytest.raises(TypeError, match=msg.format(key=r"4\.0")):
            datetime_series[4.0:10.0] = 0

        with pytest.raises(TypeError, match=msg.format(key=r"4\.5")):
            datetime_series[4.5:10.0] = 0

    def test_setitem_slice(self):
        ser = Series(range(10), index=list(range(10)))
        ser[-12:] = 0
        assert (ser == 0).all()

        ser[:-12] = 5
        assert (ser == 0).all()

    def test_setitem_slice_integers(self):
        ser = Series(np.random.randn(8), index=[2, 4, 6, 8, 10, 12, 14, 16])

        ser[:4] = 0
        assert (ser[:4] == 0).all()
        assert not (ser[4:] == 0).any()

    def test_setitem_slicestep(self):
        # caught this bug when writing tests
        series = Series(tm.makeIntIndex(20).astype(float), index=tm.makeIntIndex(20))

        series[::2] = 0
        assert (series[::2] == 0).all()

    def test_setitem_multiindex_slice(self, indexer_sli):
        # GH 8856
        mi = MultiIndex.from_product(([0, 1], list("abcde")))
        result = Series(np.arange(10, dtype=np.int64), mi)
        indexer_sli(result)[::4] = 100
        expected = Series([100, 1, 2, 3, 100, 5, 6, 7, 100, 9], mi)
        tm.assert_series_equal(result, expected)


class TestSetitemBooleanMask:
    def test_setitem_boolean(self, string_series):
        mask = string_series > string_series.median()

        # similar indexed series
        result = string_series.copy()
        result[mask] = string_series * 2
        expected = string_series * 2
        tm.assert_series_equal(result[mask], expected[mask])

        # needs alignment
        result = string_series.copy()
        result[mask] = (string_series * 2)[0:5]
        expected = (string_series * 2)[0:5].reindex_like(string_series)
        expected[-mask] = string_series[mask]
        tm.assert_series_equal(result[mask], expected[mask])

    def test_setitem_boolean_corner(self, datetime_series):
        ts = datetime_series
        mask_shifted = ts.shift(1, freq=BDay()) > ts.median()

        msg = (
            r"Unalignable boolean Series provided as indexer \(index of "
            r"the boolean Series and of the indexed object do not match"
        )
        with pytest.raises(IndexingError, match=msg):
            ts[mask_shifted] = 1

        with pytest.raises(IndexingError, match=msg):
            ts.loc[mask_shifted] = 1

    def test_setitem_boolean_different_order(self, string_series):
        ordered = string_series.sort_values()

        copy = string_series.copy()
        copy[ordered > 0] = 0

        expected = string_series.copy()
        expected[expected > 0] = 0

        tm.assert_series_equal(copy, expected)

    @pytest.mark.parametrize("func", [list, np.array, Series])
    def test_setitem_boolean_python_list(self, func):
        # GH19406
        ser = Series([None, "b", None])
        mask = func([True, False, True])
        ser[mask] = ["a", "c"]
        expected = Series(["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_boolean_nullable_int_types(self, any_nullable_numeric_dtype):
        # GH: 26468
        ser = Series([5, 6, 7, 8], dtype=any_nullable_numeric_dtype)
        ser[ser > 6] = Series(range(4), dtype=any_nullable_numeric_dtype)
        expected = Series([5, 6, 2, 3], dtype=any_nullable_numeric_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_nullable_numeric_dtype)
        ser.loc[ser > 6] = Series(range(4), dtype=any_nullable_numeric_dtype)
        tm.assert_series_equal(ser, expected)

        ser = Series([5, 6, 7, 8], dtype=any_nullable_numeric_dtype)
        loc_ser = Series(range(4), dtype=any_nullable_numeric_dtype)
        ser.loc[ser > 6] = loc_ser.loc[loc_ser > 1]
        tm.assert_series_equal(ser, expected)

    def test_setitem_with_bool_mask_and_values_matching_n_trues_in_length(self):
        # GH#30567
        ser = Series([None] * 10)
        mask = [False] * 3 + [True] * 5 + [False] * 2
        ser[mask] = range(5)
        result = ser
        expected = Series([None] * 3 + list(range(5)) + [None] * 2).astype("object")
        tm.assert_series_equal(result, expected)

    def test_setitem_nan_with_bool(self):
        # GH 13034
        result = Series([True, False, True])
        result[0] = np.nan
        expected = Series([np.nan, False, True], dtype=object)
        tm.assert_series_equal(result, expected)


class TestSetitemViewCopySemantics:
    def test_setitem_invalidates_datetime_index_freq(self):
        # GH#24096 altering a datetime64tz Series inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._data.base is not dti._data._data.base
        assert dti.freq == "D"
        ser.iloc[1] = NaT
        assert ser._values.freq is None

        # check that the DatetimeIndex was not altered in place
        assert ser._values is not dti
        assert ser._values._data.base is not dti._data._data.base
        assert dti[1] == ts
        assert dti.freq == "D"

    def test_dt64tz_setitem_does_not_mutate_dti(self):
        # GH#21907, GH#24096
        dti = date_range("2016-01-01", periods=10, tz="US/Pacific")
        ts = dti[0]
        ser = Series(dti)
        assert ser._values is not dti
        assert ser._values._data.base is not dti._data._data.base
        assert ser._mgr.arrays[0] is not dti
        assert ser._mgr.arrays[0]._data.base is not dti._data._data.base

        ser[::3] = NaT
        assert ser[0] is NaT
        assert dti[0] == ts


class TestSetitemCallable:
    def test_setitem_callable_key(self):
        # GH#12533
        ser = Series([1, 2, 3, 4], index=list("ABCD"))
        ser[lambda x: "A"] = -1

        expected = Series([-1, 2, 3, 4], index=list("ABCD"))
        tm.assert_series_equal(ser, expected)

    def test_setitem_callable_other(self):
        # GH#13299
        inc = lambda x: x + 1

        ser = Series([1, 2, -1, 4])
        ser[ser < 0] = inc

        expected = Series([1, 2, inc, 4])
        tm.assert_series_equal(ser, expected)


class TestSetitemWithExpansion:
    def test_setitem_empty_series(self):
        # GH#10193
        key = Timestamp("2012-01-01")
        series = Series(dtype=object)
        series[key] = 47
        expected = Series(47, [key])
        tm.assert_series_equal(series, expected)

    def test_setitem_empty_series_datetimeindex_preserves_freq(self):
        # GH#33573 our index should retain its freq
        series = Series([], DatetimeIndex([], freq="D"), dtype=object)
        key = Timestamp("2012-01-01")
        series[key] = 47
        expected = Series(47, DatetimeIndex([key], freq="D"))
        tm.assert_series_equal(series, expected)
        assert series.index.freq == expected.index.freq

    def test_setitem_empty_series_timestamp_preserves_dtype(self):
        # GH 21881
        timestamp = Timestamp(1412526600000000000)
        series = Series([timestamp], index=["timestamp"], dtype=object)
        expected = series["timestamp"]

        series = Series([], dtype=object)
        series["anything"] = 300.0
        series["timestamp"] = timestamp
        result = series["timestamp"]
        assert result == expected

    @pytest.mark.parametrize(
        "td",
        [
            Timedelta("9 days"),
            Timedelta("9 days").to_timedelta64(),
            Timedelta("9 days").to_pytimedelta(),
        ],
    )
    def test_append_timedelta_does_not_cast(self, td):
        # GH#22717 inserting a Timedelta should _not_ cast to int64
        expected = Series(["x", td], index=[0, "td"], dtype=object)

        ser = Series(["x"])
        ser["td"] = td
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

        ser = Series(["x"])
        ser.loc["td"] = Timedelta("9 days")
        tm.assert_series_equal(ser, expected)
        assert isinstance(ser["td"], Timedelta)

    def test_setitem_with_expansion_type_promotion(self):
        # GH#12599
        ser = Series(dtype=object)
        ser["a"] = Timestamp("2016-01-01")
        ser["b"] = 3.0
        ser["c"] = "foo"
        expected = Series([Timestamp("2016-01-01"), 3.0, "foo"], index=["a", "b", "c"])
        tm.assert_series_equal(ser, expected)

    def test_setitem_not_contained(self, string_series):
        # set item that's not contained
        ser = string_series.copy()
        assert "foobar" not in ser.index
        ser["foobar"] = 1

        app = Series([1], index=["foobar"], name="series")
        expected = string_series.append(app)
        tm.assert_series_equal(ser, expected)


def test_setitem_scalar_into_readonly_backing_data():
    # GH#14359: test that you cannot mutate a read only buffer

    array = np.zeros(5)
    array.flags.writeable = False  # make the array immutable
    series = Series(array)

    for n in range(len(series)):
        msg = "assignment destination is read-only"
        with pytest.raises(ValueError, match=msg):
            series[n] = 1

        assert array[n] == 0


def test_setitem_slice_into_readonly_backing_data():
    # GH#14359: test that you cannot mutate a read only buffer

    array = np.zeros(5)
    array.flags.writeable = False  # make the array immutable
    series = Series(array)

    msg = "assignment destination is read-only"
    with pytest.raises(ValueError, match=msg):
        series[1:3] = 1

    assert not array.any()


def test_setitem_categorical_assigning_ops():
    orig = Series(Categorical(["b", "b"], categories=["a", "b"]))
    ser = orig.copy()
    ser[:] = "a"
    exp = Series(Categorical(["a", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[1] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[ser.index > 0] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser[[False, True]] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]))
    tm.assert_series_equal(ser, exp)

    ser = orig.copy()
    ser.index = ["x", "y"]
    ser["y"] = "a"
    exp = Series(Categorical(["b", "a"], categories=["a", "b"]), index=["x", "y"])
    tm.assert_series_equal(ser, exp)


def test_setitem_nan_into_categorical():
    # ensure that one can set something to np.nan
    ser = Series(Categorical([1, 2, 3]))
    exp = Series(Categorical([1, np.nan, 3], categories=[1, 2, 3]))
    ser[1] = np.nan
    tm.assert_series_equal(ser, exp)


class TestSetitemCasting:
    @pytest.mark.parametrize("unique", [True, False])
    @pytest.mark.parametrize("val", [3, 3.0, "3"], ids=type)
    def test_setitem_non_bool_into_bool(self, val, indexer_sli, unique):
        # dont cast these 3-like values to bool
        ser = Series([True, False])
        if not unique:
            ser.index = [1, 1]

        indexer_sli(ser)[1] = val
        assert type(ser.iloc[1]) == type(val)

        expected = Series([True, val], dtype=object, index=ser.index)
        if not unique and indexer_sli is not tm.iloc:
            expected = Series([val, val], dtype=object, index=[1, 1])
        tm.assert_series_equal(ser, expected)


class SetitemCastingEquivalents:
    """
    Check each of several methods that _should_ be equivalent to `obj[key] = val`

    We assume that
        - obj.index is the default Index(range(len(obj)))
        - the setitem does not expand the obj
    """

    @pytest.fixture
    def is_inplace(self):
        """
        Indicate that we are not (yet) checking whether or not setting is inplace.
        """
        return None

    def check_indexer(self, obj, key, expected, val, indexer, is_inplace):
        orig = obj
        obj = obj.copy()
        arr = obj._values

        indexer(obj)[key] = val
        tm.assert_series_equal(obj, expected)

        self._check_inplace(is_inplace, orig, arr, obj)

    def _check_inplace(self, is_inplace, orig, arr, obj):
        if is_inplace is None:
            # We are not (yet) checking whether setting is inplace or not
            pass
        elif is_inplace:
            if arr.dtype.kind in ["m", "M"]:
                # We may not have the same DTA/TDA, but will have the same
                #  underlying data
                assert arr._data is obj._values._data
            else:
                assert obj._values is arr
        else:
            # otherwise original array should be unchanged
            tm.assert_equal(arr, orig._values)

    def test_int_key(self, obj, key, expected, val, indexer_sli, is_inplace):
        if not isinstance(key, int):
            return

        self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)

        if indexer_sli is tm.loc:
            self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
        elif indexer_sli is tm.iloc:
            self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)

        rng = range(key, key + 1)
        self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)

        if indexer_sli is not tm.loc:
            # Note: no .loc because that handles slice edges differently
            slc = slice(key, key + 1)
            self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)

        ilkey = [key]
        self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)

        indkey = np.array(ilkey)
        self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)

        genkey = (x for x in [key])
        self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_slice_key(self, obj, key, expected, val, indexer_sli, is_inplace):
        if not isinstance(key, slice):
            return

        if indexer_sli is not tm.loc:
            # Note: no .loc because that handles slice edges differently
            self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)

        ilkey = list(range(len(obj)))[key]
        self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)

        indkey = np.array(ilkey)
        self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)

        genkey = (x for x in indkey)
        self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)

    def test_mask_key(self, obj, key, expected, val, indexer_sli):
        # setitem with boolean mask
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        obj = obj.copy()
        indexer_sli(obj)[mask] = val
        tm.assert_series_equal(obj, expected)

    def test_series_where(self, obj, key, expected, val, is_inplace):
        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        orig = obj
        obj = obj.copy()
        arr = obj._values

        res = obj.where(~mask, val)
        tm.assert_series_equal(res, expected)

        self._check_inplace(is_inplace, orig, arr, obj)

    def test_index_where(self, obj, key, expected, val, request):
        if Index(obj).dtype != obj.dtype:
            pytest.skip("test not applicable for this dtype")

        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        if obj.dtype == bool:
            msg = "Index/Series casting behavior inconsistent GH#38692"
            mark = pytest.mark.xfail(reason=msg)
            request.node.add_marker(mark)

        res = Index(obj).where(~mask, val)
        tm.assert_index_equal(res, Index(expected))

    def test_index_putmask(self, obj, key, expected, val):
        if Index(obj).dtype != obj.dtype:
            pytest.skip("test not applicable for this dtype")

        mask = np.zeros(obj.shape, dtype=bool)
        mask[key] = True

        res = Index(obj).putmask(mask, val)
        tm.assert_index_equal(res, Index(expected))


@pytest.mark.parametrize(
    "obj,expected,key",
    [
        pytest.param(
            # these induce dtype changes
            Series([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            Series([np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]),
            slice(None, None, 2),
            id="int_series_slice_key_step",
        ),
        pytest.param(
            Series([True, True, False, False]),
            Series([np.nan, True, np.nan, False], dtype=object),
            slice(None, None, 2),
            id="bool_series_slice_key_step",
        ),
        pytest.param(
            # these induce dtype changes
            Series(np.arange(10)),
            Series([np.nan, np.nan, np.nan, np.nan, np.nan, 5, 6, 7, 8, 9]),
            slice(None, 5),
            id="int_series_slice_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([1, 2, 3]),
            Series([np.nan, 2, 3]),
            0,
            id="int_series_int_key",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False]),
            Series([np.nan], dtype=object),
            # TODO: maybe go to float64 since we are changing the _whole_ Series?
            0,
            id="bool_series_int_key_change_all",
        ),
        pytest.param(
            # changes dtype GH#4463
            Series([False, True]),
            Series([np.nan, True], dtype=object),
            0,
            id="bool_series_int_key",
        ),
    ],
)
class TestSetitemCastingEquivalents(SetitemCastingEquivalents):
    @pytest.fixture(params=[np.nan, np.float64("NaN")])
    def val(self, request):
        """
        One python float NaN, one np.float64.  Only np.float64 has a `dtype`
        attribute.
        """
        return request.param


class TestSetitemTimedelta64IntoNumeric(SetitemCastingEquivalents):
    # timedelta64 should not be treated as integers when setting into
    #  numeric Series

    @pytest.fixture
    def val(self):
        td = np.timedelta64(4, "ns")
        return td
        # TODO: could also try np.full((1,), td)

    @pytest.fixture(params=[complex, int, float])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        return ser

    @pytest.fixture
    def expected(self, dtype):
        arr = np.arange(5).astype(dtype)
        ser = Series(arr)
        ser = ser.astype(object)
        ser.values[0] = np.timedelta64(4, "ns")
        return ser

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def is_inplace(self):
        """
        Indicate we do _not_ expect the setting to be done inplace.
        """
        return False


class TestSetitemDT64IntoInt(SetitemCastingEquivalents):
    # GH#39619 dont cast dt64 to int when doing this setitem

    @pytest.fixture(params=["M8[ns]", "m8[ns]"])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def scalar(self, dtype):
        val = np.datetime64("2021-01-18 13:25:00", "ns")
        if dtype == "m8[ns]":
            val = val - val
        return val

    @pytest.fixture
    def expected(self, scalar):
        expected = Series([scalar, scalar, 3], dtype=object)
        assert isinstance(expected[0], type(scalar))
        return expected

    @pytest.fixture
    def obj(self):
        return Series([1, 2, 3])

    @pytest.fixture
    def key(self):
        return slice(None, -1)

    @pytest.fixture(params=[None, list, np.array])
    def val(self, scalar, request):
        box = request.param
        if box is None:
            return scalar
        return box([scalar, scalar])

    @pytest.fixture
    def is_inplace(self):
        return False


class TestSetitemNAPeriodDtype(SetitemCastingEquivalents):
    # Setting compatible NA values into Series with PeriodDtype

    @pytest.fixture
    def expected(self, key):
        exp = Series(period_range("2000-01-01", periods=10, freq="D"))
        exp._values.view("i8")[key] = NaT.value
        assert exp[key] is NaT or all(x is NaT for x in exp[key])
        return exp

    @pytest.fixture
    def obj(self):
        return Series(period_range("2000-01-01", periods=10, freq="D"))

    @pytest.fixture(params=[3, slice(3, 5)])
    def key(self, request):
        return request.param

    @pytest.fixture(params=[None, np.nan])
    def val(self, request):
        return request.param

    @pytest.fixture
    def is_inplace(self):
        return True


class TestSetitemNADatetimeLikeDtype(SetitemCastingEquivalents):
    # some nat-like values should be cast to datetime64/timedelta64 when
    #  inserting into a datetime64/timedelta64 series.  Others should coerce
    #  to object and retain their dtypes.
    # GH#18586 for td64 and boolean mask case

    @pytest.fixture(
        params=["m8[ns]", "M8[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Central]"]
    )
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def obj(self, dtype):
        i8vals = date_range("2016-01-01", periods=3).asi8
        idx = Index(i8vals, dtype=dtype)
        assert idx.dtype == dtype
        return Series(idx)

    @pytest.fixture(
        params=[
            None,
            np.nan,
            NaT,
            np.timedelta64("NaT", "ns"),
            np.datetime64("NaT", "ns"),
        ]
    )
    def val(self, request):
        return request.param

    @pytest.fixture
    def is_inplace(self, val, obj):
        # td64   -> cast to object iff val is datetime64("NaT")
        # dt64   -> cast to object iff val is timedelta64("NaT")
        # dt64tz -> cast to object with anything _but_ NaT
        return val is NaT or val is None or val is np.nan or obj.dtype == val.dtype

    @pytest.fixture
    def expected(self, obj, val, is_inplace):
        dtype = obj.dtype if is_inplace else object
        expected = Series([val] + list(obj[1:]), dtype=dtype)
        return expected

    @pytest.fixture
    def key(self):
        return 0


class TestSetitemMismatchedTZCastsToObject(SetitemCastingEquivalents):
    # GH#24024
    @pytest.fixture
    def obj(self):
        return Series(date_range("2000", periods=2, tz="US/Central"))

    @pytest.fixture
    def val(self):
        return Timestamp("2000", tz="US/Eastern")

    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def expected(self):
        expected = Series(
            [
                Timestamp("2000-01-01 00:00:00-05:00", tz="US/Eastern"),
                Timestamp("2000-01-02 00:00:00-06:00", tz="US/Central"),
            ],
            dtype=object,
        )
        return expected


@pytest.mark.parametrize(
    "obj,expected",
    [
        # For numeric series, we should coerce to NaN.
        (Series([1, 2, 3]), Series([np.nan, 2, 3])),
        (Series([1.0, 2.0, 3.0]), Series([np.nan, 2.0, 3.0])),
        # For datetime series, we should coerce to NaT.
        (
            Series([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)]),
            Series([NaT, datetime(2000, 1, 2), datetime(2000, 1, 3)]),
        ),
        # For objects, we should preserve the None value.
        (Series(["foo", "bar", "baz"]), Series([None, "bar", "baz"])),
    ],
)
class TestSeriesNoneCoercion(SetitemCastingEquivalents):
    @pytest.fixture
    def key(self):
        return 0

    @pytest.fixture
    def val(self):
        return None

    @pytest.fixture
    def is_inplace(self, obj):
        # This is specific to the 4 cases currently implemented for this class.
        return obj.dtype.kind != "i"


def test_setitem_with_bool_indexer():
    # GH#42530

    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.pop("b")
    result[[True, False, False]] = 9
    expected = Series(data=[9, 5, 6], name="b")
    tm.assert_series_equal(result, expected)

    df.loc[[True, False, False], "a"] = 10
    expected = DataFrame({"a": [10, 2, 3]})
    tm.assert_frame_equal(df, expected)
