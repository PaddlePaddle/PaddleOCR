import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


class TestSeriesReplace:
    def test_replace(self, datetime_series):
        N = 100
        ser = pd.Series(np.random.randn(N))
        ser[0:4] = np.nan
        ser[6:10] = 0

        # replace list with a single value
        return_value = ser.replace([np.nan], -1, inplace=True)
        assert return_value is None

        exp = ser.fillna(-1)
        tm.assert_series_equal(ser, exp)

        rs = ser.replace(0.0, np.nan)
        ser[ser == 0.0] = np.nan
        tm.assert_series_equal(rs, ser)

        ser = pd.Series(np.fabs(np.random.randn(N)), tm.makeDateIndex(N), dtype=object)
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None

        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

        ser = pd.Series([np.nan, 0, np.inf])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        ser = pd.Series([np.nan, 0, "foo", "bar", np.inf, None, pd.NaT])
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
        filled = ser.copy()
        filled[4] = 0
        tm.assert_series_equal(ser.replace(np.inf, 0), filled)

        ser = pd.Series(datetime_series.index)
        tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))

        # malformed
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            ser.replace([1, 2, 3], [np.nan, 0])

        # ser is dt64 so can't hold 1 or 2, so this replace is a no-op
        result = ser.replace([1, 2], [np.nan, 0])
        tm.assert_series_equal(result, ser)

        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))

    def test_replace_gh5319(self):
        # API change from 0.12?
        # GH 5319
        ser = pd.Series([0, np.nan, 2, 3, 4])
        expected = ser.ffill()
        result = ser.replace([np.nan])
        tm.assert_series_equal(result, expected)

        ser = pd.Series([0, np.nan, 2, 3, 4])
        expected = ser.ffill()
        result = ser.replace(np.nan)
        tm.assert_series_equal(result, expected)

    def test_replace_datetime64(self):
        # GH 5797
        ser = pd.Series(pd.date_range("20130101", periods=5))
        expected = ser.copy()
        expected.loc[2] = pd.Timestamp("20120101")
        result = ser.replace({pd.Timestamp("20130103"): pd.Timestamp("20120101")})
        tm.assert_series_equal(result, expected)
        result = ser.replace(pd.Timestamp("20130103"), pd.Timestamp("20120101"))
        tm.assert_series_equal(result, expected)

    def test_replace_nat_with_tz(self):
        # GH 11792: Test with replacing NaT in a list with tz data
        ts = pd.Timestamp("2015/01/01", tz="UTC")
        s = pd.Series([pd.NaT, pd.Timestamp("2015/01/01", tz="UTC")])
        result = s.replace([np.nan, pd.NaT], pd.Timestamp.min)
        expected = pd.Series([pd.Timestamp.min, ts], dtype=object)
        tm.assert_series_equal(expected, result)

    def test_replace_timedelta_td64(self):
        tdi = pd.timedelta_range(0, periods=5)
        ser = pd.Series(tdi)

        # Using a single dict argument means we go through replace_list
        result = ser.replace({ser[1]: ser[3]})

        expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
        tm.assert_series_equal(result, expected)

    def test_replace_with_single_list(self):
        ser = pd.Series([0, 1, 2, 3, 4])
        result = ser.replace([1, 2, 3])
        tm.assert_series_equal(result, pd.Series([0, 0, 0, 0, 4]))

        s = ser.copy()
        return_value = s.replace([1, 2, 3], inplace=True)
        assert return_value is None
        tm.assert_series_equal(s, pd.Series([0, 0, 0, 0, 4]))

        # make sure things don't get corrupted when fillna call fails
        s = ser.copy()
        msg = (
            r"Invalid fill method\. Expecting pad \(ffill\) or backfill "
            r"\(bfill\)\. Got crash_cymbal"
        )
        with pytest.raises(ValueError, match=msg):
            return_value = s.replace([1, 2, 3], inplace=True, method="crash_cymbal")
            assert return_value is None
        tm.assert_series_equal(s, ser)

    def test_replace_mixed_types(self):
        s = pd.Series(np.arange(5), dtype="int64")

        def check_replace(to_rep, val, expected):
            sc = s.copy()
            r = s.replace(to_rep, val)
            return_value = sc.replace(to_rep, val, inplace=True)
            assert return_value is None
            tm.assert_series_equal(expected, r)
            tm.assert_series_equal(expected, sc)

        # MUST upcast to float
        e = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
        tr, v = [3], [3.0]
        check_replace(tr, v, e)

        # MUST upcast to float
        e = pd.Series([0, 1, 2, 3.5, 4])
        tr, v = [3], [3.5]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, "a"])
        tr, v = [3, 4], [3.5, "a"]
        check_replace(tr, v, e)

        # again casts to object
        e = pd.Series([0, 1, 2, 3.5, pd.Timestamp("20130101")])
        tr, v = [3, 4], [3.5, pd.Timestamp("20130101")]
        check_replace(tr, v, e)

        # casts to object
        e = pd.Series([0, 1, 2, 3.5, True], dtype="object")
        tr, v = [3, 4], [3.5, True]
        check_replace(tr, v, e)

        # test an object with dates + floats + integers + strings
        dr = pd.Series(pd.date_range("1/1/2001", "1/10/2001", freq="D"))
        result = dr.astype(object).replace([dr[0], dr[1], dr[2]], [1.0, 2, "a"])
        expected = pd.Series([1.0, 2, "a"] + dr[3:].tolist(), dtype=object)
        tm.assert_series_equal(result, expected)

    def test_replace_bool_with_string_no_op(self):
        s = pd.Series([True, False, True])
        result = s.replace("fun", "in-the-sun")
        tm.assert_series_equal(s, result)

    def test_replace_bool_with_string(self):
        # nonexistent elements
        s = pd.Series([True, False, True])
        result = s.replace(True, "2u")
        expected = pd.Series(["2u", False, "2u"])
        tm.assert_series_equal(expected, result)

    def test_replace_bool_with_bool(self):
        s = pd.Series([True, False, True])
        result = s.replace(True, False)
        expected = pd.Series([False] * len(s))
        tm.assert_series_equal(expected, result)

    def test_replace_with_dict_with_bool_keys(self):
        s = pd.Series([True, False, True])
        result = s.replace({"asdf": "asdb", True: "yes"})
        expected = pd.Series(["yes", False, "yes"])
        tm.assert_series_equal(result, expected)

    def test_replace_Int_with_na(self, any_nullable_int_dtype):
        # GH 38267
        result = pd.Series([0, None], dtype=any_nullable_int_dtype).replace(0, pd.NA)
        expected = pd.Series([pd.NA, pd.NA], dtype=any_nullable_int_dtype)
        tm.assert_series_equal(result, expected)
        result = pd.Series([0, 1], dtype=any_nullable_int_dtype).replace(0, pd.NA)
        result.replace(1, pd.NA, inplace=True)
        tm.assert_series_equal(result, expected)

    def test_replace2(self):
        N = 100
        ser = pd.Series(np.fabs(np.random.randn(N)), tm.makeDateIndex(N), dtype=object)
        ser[:5] = np.nan
        ser[6:10] = "foo"
        ser[20:30] = "bar"

        # replace list with a single value
        rs = ser.replace([np.nan, "foo", "bar"], -1)

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -1).all()
        assert (rs[20:30] == -1).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values
        rs = ser.replace({np.nan: -1, "foo": -2, "bar": -3})

        assert (rs[:5] == -1).all()
        assert (rs[6:10] == -2).all()
        assert (rs[20:30] == -3).all()
        assert (pd.isna(ser[:5])).all()

        # replace with different values with 2 lists
        rs2 = ser.replace([np.nan, "foo", "bar"], [-1, -2, -3])
        tm.assert_series_equal(rs, rs2)

        # replace inplace
        return_value = ser.replace([np.nan, "foo", "bar"], -1, inplace=True)
        assert return_value is None
        assert (ser[:5] == -1).all()
        assert (ser[6:10] == -1).all()
        assert (ser[20:30] == -1).all()

    def test_replace_with_dictlike_and_string_dtype(self, nullable_string_dtype):
        # GH 32621
        s = pd.Series(["one", "two", np.nan], dtype=nullable_string_dtype)
        expected = pd.Series(["1", "2", np.nan])
        result = s.replace({"one": "1", "two": "2"})
        tm.assert_series_equal(expected, result)

    def test_replace_with_empty_dictlike(self):
        # GH 15289
        s = pd.Series(list("abcd"))
        tm.assert_series_equal(s, s.replace({}))

        with tm.assert_produces_warning(DeprecationWarning):
            empty_series = pd.Series([])
        tm.assert_series_equal(s, s.replace(empty_series))

    def test_replace_string_with_number(self):
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_replacer_equals_replacement(self):
        # GH 20656
        # make sure all replacers are matching against original values
        s = pd.Series(["a", "b"])
        expected = pd.Series(["b", "a"])
        result = s.replace({"a": "b", "b": "a"})
        tm.assert_series_equal(expected, result)

    def test_replace_unicode_with_number(self):
        # GH 15743
        s = pd.Series([1, 2, 3])
        result = s.replace("2", np.nan)
        expected = pd.Series([1, 2, 3])
        tm.assert_series_equal(expected, result)

    def test_replace_mixed_types_with_string(self):
        # Testing mixed
        s = pd.Series([1, 2, 3, "4", 4, 5])
        result = s.replace([2, "4"], np.nan)
        expected = pd.Series([1, np.nan, 3, np.nan, 4, 5])
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "categorical, numeric",
        [
            (pd.Categorical(["A"], categories=["A", "B"]), [1]),
            (pd.Categorical(("A",), categories=["A", "B"]), [1]),
            (pd.Categorical(("A", "B"), categories=["A", "B"]), [1, 2]),
        ],
    )
    def test_replace_categorical(self, categorical, numeric):
        # GH 24971
        # Do not check if dtypes are equal due to a known issue that
        # Categorical.replace sometimes coerces to object (GH 23305)
        s = pd.Series(categorical)
        result = s.replace({"A": 1, "B": 2})
        expected = pd.Series(numeric)
        tm.assert_series_equal(expected, result)

    def test_replace_categorical_single(self):
        # GH 26988
        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        s = pd.Series(dti)
        c = s.astype("category")

        expected = c.copy()
        expected = expected.cat.add_categories("foo")
        expected[2] = "foo"
        expected = expected.cat.remove_unused_categories()
        assert c[2] != "foo"

        result = c.replace(c[2], "foo")
        tm.assert_series_equal(expected, result)
        assert c[2] != "foo"  # ensure non-inplace call does not alter original

        return_value = c.replace(c[2], "foo", inplace=True)
        assert return_value is None
        tm.assert_series_equal(expected, c)

        first_value = c[0]
        return_value = c.replace(c[1], c[0], inplace=True)
        assert return_value is None
        assert c[0] == c[1] == first_value  # test replacing with existing value

    def test_replace_with_no_overflowerror(self):
        # GH 25616
        # casts to object without Exception from OverflowError
        s = pd.Series([0, 1, 2, 3, 4])
        result = s.replace([3], ["100000000000000000000"])
        expected = pd.Series([0, 1, 2, "100000000000000000000", 4])
        tm.assert_series_equal(result, expected)

        s = pd.Series([0, "100000000000000000000", "100000000000000000001"])
        result = s.replace(["100000000000000000000"], [1])
        expected = pd.Series([0, 1, "100000000000000000001"])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, to_replace, exp",
        [
            ([1, 2, 3], {1: 2, 2: 3, 3: 4}, [2, 3, 4]),
            (["1", "2", "3"], {"1": "2", "2": "3", "3": "4"}, ["2", "3", "4"]),
        ],
    )
    def test_replace_commutative(self, ser, to_replace, exp):
        # GH 16051
        # DataFrame.replace() overwrites when values are non-numeric

        series = pd.Series(ser)

        expected = pd.Series(exp)
        result = series.replace(to_replace)

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "ser, exp", [([1, 2, 3], [1, True, 3]), (["x", 2, 3], ["x", True, 3])]
    )
    def test_replace_no_cast(self, ser, exp):
        # GH 9113
        # BUG: replace int64 dtype with bool coerces to int64

        series = pd.Series(ser)
        result = series.replace(2, True)
        expected = pd.Series(exp)

        tm.assert_series_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # GH 18634
        # API: replace() should raise an exception if invalid argument is given
        series = pd.Series(["a", "b", "c "])
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        with pytest.raises(TypeError, match=msg):
            series.replace(lambda x: x.strip())

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_nonbool_regex(self, frame):
        obj = pd.Series(["a", "b", "c "])
        if frame:
            obj = obj.to_frame()

        msg = "'to_replace' must be 'None' if 'regex' is not a bool"
        with pytest.raises(ValueError, match=msg):
            obj.replace(to_replace=["a"], regex="foo")

    @pytest.mark.parametrize("frame", [False, True])
    def test_replace_empty_copy(self, frame):
        obj = pd.Series([], dtype=np.float64)
        if frame:
            obj = obj.to_frame()

        res = obj.replace(4, 5, inplace=True)
        assert res is None

        res = obj.replace(4, 5, inplace=False)
        tm.assert_equal(res, obj)
        assert res is not obj

    def test_replace_only_one_dictlike_arg(self):
        # GH#33340

        ser = pd.Series([1, 2, "A", pd.Timestamp.now(), True])
        to_replace = {0: 1, 2: "A"}
        value = "foo"
        msg = "Series.replace cannot use dict-like to_replace and non-None value"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

        to_replace = 1
        value = {0: "foo", 2: "bar"}
        msg = "Series.replace cannot use dict-value and non-None to_replace"
        with pytest.raises(ValueError, match=msg):
            ser.replace(to_replace, value)

    def test_replace_extension_other(self, frame_or_series):
        # https://github.com/pandas-dev/pandas/issues/34530
        obj = frame_or_series(pd.array([1, 2, 3], dtype="Int64"))
        result = obj.replace("", "")  # no exception
        # should not have changed dtype
        tm.assert_equal(obj, result)

    def test_replace_with_compiled_regex(self):
        # https://github.com/pandas-dev/pandas/issues/35680
        s = pd.Series(["a", "b", "c"])
        regex = re.compile("^a$")
        result = s.replace({regex: "z"}, regex=True)
        expected = pd.Series(["z", "b", "c"])
        tm.assert_series_equal(result, expected)
