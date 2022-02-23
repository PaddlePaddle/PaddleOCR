from __future__ import annotations

from datetime import timedelta
import itertools

import numpy as np
import pytest

from pandas.compat import (
    IS64,
    is_platform_windows,
)

import pandas as pd
import pandas._testing as tm

###############################################################
# Index / Series common tests which may trigger dtype coercions
###############################################################


@pytest.fixture(autouse=True, scope="class")
def check_comprehensiveness(request):
    # Iterate over combination of dtype, method and klass
    # and ensure that each are contained within a collected test
    cls = request.cls
    combos = itertools.product(cls.klasses, cls.dtypes, [cls.method])

    def has_test(combo):
        klass, dtype, method = combo
        cls_funcs = request.node.session.items
        return any(
            klass in x.name and dtype in x.name and method in x.name for x in cls_funcs
        )

    opts = request.config.option
    if opts.lf or opts.keyword:
        # If we are running with "last-failed" or -k foo, we expect to only
        #  run a subset of tests.
        yield

    else:

        for combo in combos:
            if not has_test(combo):
                raise AssertionError(
                    f"test method is not defined: {cls.__name__}, {combo}"
                )

        yield


class CoercionBase:

    klasses = ["index", "series"]
    dtypes = [
        "object",
        "int64",
        "float64",
        "complex128",
        "bool",
        "datetime64",
        "datetime64tz",
        "timedelta64",
        "period",
    ]

    @property
    def method(self):
        raise NotImplementedError(self)


class TestSetitemCoercion(CoercionBase):

    method = "setitem"

    def _assert_setitem_series_conversion(
        self, original_series, loc_value, expected_series, expected_dtype
    ):
        """test series value's coercion triggered by assignment"""
        temp = original_series.copy()
        temp[1] = loc_value
        tm.assert_series_equal(temp, expected_series)
        # check dtype explicitly for sure
        assert temp.dtype == expected_dtype

        # FIXME: dont leave commented-out
        # .loc works different rule, temporary disable
        # temp = original_series.copy()
        # temp.loc[1] = loc_value
        # tm.assert_series_equal(temp, expected_series)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(1, object), (1.1, object), (1 + 1j, object), (True, object)]
    )
    def test_setitem_series_object(self, val, exp_dtype):
        obj = pd.Series(list("abcd"))
        assert obj.dtype == object

        exp = pd.Series(["a", val, "c", "d"])
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_setitem_series_int64(self, val, exp_dtype, request):
        obj = pd.Series([1, 2, 3, 4])
        assert obj.dtype == np.int64

        if exp_dtype is np.float64:
            exp = pd.Series([1, 1, 3, 4])
            self._assert_setitem_series_conversion(obj, 1.1, exp, np.int64)
            mark = pytest.mark.xfail(reason="GH12747 The result must be float")
            request.node.add_marker(mark)

        exp = pd.Series([1, val, 3, 4])
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(np.int32(1), np.int8), (np.int16(2 ** 9), np.int16)]
    )
    def test_setitem_series_int8(self, val, exp_dtype, request):
        obj = pd.Series([1, 2, 3, 4], dtype=np.int8)
        assert obj.dtype == np.int8

        if exp_dtype is np.int16:
            exp = pd.Series([1, 0, 3, 4], dtype=np.int8)
            self._assert_setitem_series_conversion(obj, val, exp, np.int8)
            mark = pytest.mark.xfail(
                reason="BUG: it must be pd.Series([1, 1, 3, 4], dtype=np.int16"
            )
            request.node.add_marker(mark)

        warn = None if exp_dtype is np.int8 else FutureWarning
        msg = "Values are too large to be losslessly cast to int8"
        with tm.assert_produces_warning(warn, match=msg):
            exp = pd.Series([1, val, 3, 4], dtype=np.int8)
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_setitem_series_float64(self, val, exp_dtype):
        obj = pd.Series([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64

        exp = pd.Series([1.1, val, 3.3, 4.4])
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_setitem_series_complex128(self, val, exp_dtype):
        obj = pd.Series([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
        assert obj.dtype == np.complex128

        exp = pd.Series([1 + 1j, val, 3 + 3j, 4 + 4j])
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [
            (1, object),
            ("3", object),
            (3, object),
            (1.1, object),
            (1 + 1j, object),
            (True, np.bool_),
        ],
    )
    def test_setitem_series_bool(self, val, exp_dtype):
        obj = pd.Series([True, False, True, False])
        assert obj.dtype == np.bool_

        exp = pd.Series([True, val, True, False], dtype=exp_dtype)
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [(pd.Timestamp("2012-01-01"), "datetime64[ns]"), (1, object), ("x", object)],
    )
    def test_setitem_series_datetime64(self, val, exp_dtype):
        obj = pd.Series(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"

        exp = pd.Series(
            [
                pd.Timestamp("2011-01-01"),
                val,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
            (pd.Timestamp("2012-01-01", tz="US/Pacific"), object),
            (pd.Timestamp("2012-01-01"), object),
            (1, object),
        ],
    )
    def test_setitem_series_datetime64tz(self, val, exp_dtype):
        tz = "US/Eastern"
        obj = pd.Series(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.Timestamp("2011-01-02", tz=tz),
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        assert obj.dtype == "datetime64[ns, US/Eastern]"

        exp = pd.Series(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                val,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype",
        [(pd.Timedelta("12 day"), "timedelta64[ns]"), (1, object), ("x", object)],
    )
    def test_setitem_series_timedelta64(self, val, exp_dtype):
        obj = pd.Series(
            [
                pd.Timedelta("1 day"),
                pd.Timedelta("2 day"),
                pd.Timedelta("3 day"),
                pd.Timedelta("4 day"),
            ]
        )
        assert obj.dtype == "timedelta64[ns]"

        exp = pd.Series(
            [pd.Timedelta("1 day"), val, pd.Timedelta("3 day"), pd.Timedelta("4 day")]
        )
        self._assert_setitem_series_conversion(obj, val, exp, exp_dtype)

    def test_setitem_series_no_coercion_from_values_list(self):
        # GH35865 - int casted to str when internally calling np.array(ser.values)
        ser = pd.Series(["a", 1])
        ser[:] = list(ser.values)

        expected = pd.Series(["a", 1])

        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(
        self, original_series, loc_key, expected_index, expected_dtype
    ):
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        # check dtype explicitly for sure
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize(
        "val,exp_dtype", [("x", object), (5, IndexError), (1.1, object)]
    )
    def test_setitem_index_object(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4], index=list("abcd"))
        assert obj.index.dtype == object

        if exp_dtype is IndexError:
            temp = obj.copy()
            msg = "index 5 is out of bounds for axis 0 with size 4"
            with pytest.raises(exp_dtype, match=msg):
                temp[5] = 5
        else:
            exp_index = pd.Index(list("abcd") + [val])
            self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, np.int64), (1.1, np.float64), ("x", object)]
    )
    def test_setitem_index_int64(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64

        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize(
        "val,exp_dtype", [(5, IndexError), (5.1, np.float64), ("x", object)]
    )
    def test_setitem_index_float64(self, val, exp_dtype, request):
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64

        if exp_dtype is IndexError:
            # float + int -> int
            temp = obj.copy()
            msg = "index 5 is out of bounds for axis 0 with size 4"
            with pytest.raises(exp_dtype, match=msg):
                temp[5] = 5
            mark = pytest.mark.xfail(reason="TODO_GH12747 The result must be float")
            request.node.add_marker(mark)
        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_series_period(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_datetime64tz(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_setitem_index_period(self):
        raise NotImplementedError


class TestInsertIndexCoercion(CoercionBase):

    klasses = ["index"]
    method = "insert"

    def _assert_insert_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by insert"""
        target = original.copy()
        res = target.insert(1, value)
        tm.assert_index_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, object),
            (1.1, 1.1, object),
            (False, False, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_object(self, insert, coerced_val, coerced_dtype):
        obj = pd.Index(list("abcd"))
        assert obj.dtype == object

        exp = pd.Index(["a", coerced_val, "b", "c", "d"])
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1, np.int64),
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_index_int64(self, insert, coerced_val, coerced_dtype):
        obj = pd.Int64Index([1, 2, 3, 4])
        assert obj.dtype == np.int64

        exp = pd.Index([1, coerced_val, 2, 3, 4])
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (1, 1.0, np.float64),
            (1.1, 1.1, np.float64),
            (False, False, object),  # GH#36319
            ("x", "x", object),
        ],
    )
    def test_insert_index_float64(self, insert, coerced_val, coerced_dtype):
        obj = pd.Float64Index([1.0, 2.0, 3.0, 4.0])
        assert obj.dtype == np.float64

        exp = pd.Index([1.0, coerced_val, 2.0, 3.0, 4.0])
        self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    @pytest.mark.parametrize(
        "insert_value",
        [pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), 1],
    )
    def test_insert_index_datetimes(self, request, fill_val, exp_dtype, insert_value):

        obj = pd.DatetimeIndex(
            ["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], tz=fill_val.tz
        )
        assert obj.dtype == exp_dtype

        exp = pd.DatetimeIndex(
            ["2011-01-01", fill_val.date(), "2011-01-02", "2011-01-03", "2011-01-04"],
            tz=fill_val.tz,
        )
        self._assert_insert_conversion(obj, fill_val, exp, exp_dtype)

        if fill_val.tz:

            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

            # mismatched tz --> cast to object (could reasonably cast to common tz)
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

        else:
            # mismatched tzawareness
            ts = pd.Timestamp("2012-01-01", tz="Asia/Tokyo")
            result = obj.insert(1, ts)
            expected = obj.astype(object).insert(1, ts)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

        item = 1
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected[1] == item
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)

    def test_insert_index_timedelta64(self):
        obj = pd.TimedeltaIndex(["1 day", "2 day", "3 day", "4 day"])
        assert obj.dtype == "timedelta64[ns]"

        # timedelta64 + timedelta64 => timedelta64
        exp = pd.TimedeltaIndex(["1 day", "10 day", "2 day", "3 day", "4 day"])
        self._assert_insert_conversion(
            obj, pd.Timedelta("10 day"), exp, "timedelta64[ns]"
        )

        for item in [pd.Timestamp("2012-01-01"), 1]:
            result = obj.insert(1, item)
            expected = obj.astype(object).insert(1, item)
            assert expected.dtype == object
            tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "insert, coerced_val, coerced_dtype",
        [
            (pd.Period("2012-01", freq="M"), "2012-01", "period[M]"),
            (pd.Timestamp("2012-01-01"), pd.Timestamp("2012-01-01"), object),
            (1, 1, object),
            ("x", "x", object),
        ],
    )
    def test_insert_index_period(self, insert, coerced_val, coerced_dtype):
        obj = pd.PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq="M")
        assert obj.dtype == "period[M]"

        data = [
            pd.Period("2011-01", freq="M"),
            coerced_val,
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-04", freq="M"),
        ]
        if isinstance(insert, pd.Period):
            exp = pd.PeriodIndex(data, freq="M")
            self._assert_insert_conversion(obj, insert, exp, coerced_dtype)

            # string that can be parsed to appropriate PeriodDtype
            self._assert_insert_conversion(obj, str(insert), exp, coerced_dtype)

        else:
            result = obj.insert(0, insert)
            expected = obj.astype(object).insert(0, insert)
            tm.assert_index_equal(result, expected)

            # TODO: ATM inserting '2012-01-01 00:00:00' when we have obj.freq=="M"
            #  casts that string to Period[M], not clear that is desirable
            if not isinstance(insert, pd.Timestamp):
                # non-castable string
                result = obj.insert(0, str(insert))
                expected = obj.astype(object).insert(0, str(insert))
                tm.assert_index_equal(result, expected)

            msg = r"Unexpected keyword arguments {'freq'}"
            with pytest.raises(TypeError, match=msg):
                with tm.assert_produces_warning(FutureWarning):
                    # passing keywords to pd.Index
                    pd.Index(data, freq="M")

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_insert_index_bool(self):
        raise NotImplementedError


class TestWhereCoercion(CoercionBase):

    method = "where"

    def _assert_where_conversion(
        self, original, cond, values, expected, expected_dtype
    ):
        """test coercion triggered by where"""
        target = original.copy()
        res = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_where_object(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass(list("abcd"))
        assert obj.dtype == object
        cond = klass([True, False, True, False])

        if fill_val is True and klass is pd.Series:
            ret_val = 1
        else:
            ret_val = fill_val

        exp = klass(["a", ret_val, "c", ret_val])
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        if fill_val is True:
            values = klass([True, False, True, True])
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])

        exp = klass(["a", values[1], "c", values[3]])
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_int64(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        if klass is pd.Index and exp_dtype is np.complex128:
            pytest.skip("Complex Index not supported")
        obj = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        cond = klass([True, False, True, False])

        exp = klass([1, fill_val, 3, fill_val])
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        if fill_val is True:
            values = klass([True, False, True, True])
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])
        exp = klass([1, values[1], 3, values[3]])
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val, exp_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_where_float64(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        if klass is pd.Index and exp_dtype is np.complex128:
            pytest.skip("Complex Index not supported")
        obj = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        cond = klass([True, False, True, False])

        exp = klass([1.1, fill_val, 3.3, fill_val])
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        if fill_val is True:
            values = klass([True, False, True, True])
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])
        exp = klass([1.1, values[1], 3.3, values[3]])
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_where_series_complex128(self, fill_val, exp_dtype):
        klass = pd.Series
        obj = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
        assert obj.dtype == np.complex128
        cond = klass([True, False, True, False])

        exp = klass([1 + 1j, fill_val, 3 + 3j, fill_val])
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        if fill_val is True:
            values = klass([True, False, True, True])
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])
        exp = klass([1 + 1j, values[1], 3 + 3j, values[3]], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)],
    )
    def test_where_series_bool(self, fill_val, exp_dtype):
        klass = pd.Series

        obj = klass([True, False, True, False])
        assert obj.dtype == np.bool_
        cond = klass([True, False, True, False])

        exp = klass([True, fill_val, True, fill_val])
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        if fill_val is True:
            values = klass([True, False, True, True])
        else:
            values = klass(x * fill_val for x in [5, 6, 7, 8])
        exp = klass([True, values[1], True, values[3]])
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val,exp_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
        ],
        ids=["datetime64", "datetime64tz"],
    )
    def test_where_series_datetime64(self, fill_val, exp_dtype):
        obj = pd.Series(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"
        cond = pd.Series([True, False, True, False])

        exp = pd.Series(
            [pd.Timestamp("2011-01-01"), fill_val, pd.Timestamp("2011-01-03"), fill_val]
        )
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)

        values = pd.Series(pd.date_range(fill_val, periods=4))
        if fill_val.tz:
            exp = pd.Series(
                [
                    pd.Timestamp("2011-01-01"),
                    pd.Timestamp("2012-01-02 00:00", tz="US/Eastern"),
                    pd.Timestamp("2011-01-03"),
                    pd.Timestamp("2012-01-04 00:00", tz="US/Eastern"),
                ]
            )
            self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

        exp = pd.Series(
            [
                pd.Timestamp("2011-01-01"),
                values[1],
                pd.Timestamp("2011-01-03"),
                values[3],
            ]
        )
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize(
        "fill_val",
        [
            pd.Timestamp("2012-01-01"),
            pd.Timestamp("2012-01-01").to_datetime64(),
            pd.Timestamp("2012-01-01").to_pydatetime(),
        ],
    )
    def test_where_index_datetime(self, fill_val):
        exp_dtype = "datetime64[ns]"
        obj = pd.Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"
        cond = pd.Index([True, False, True, False])

        result = obj.where(cond, fill_val)
        expected = pd.DatetimeIndex([obj[0], fill_val, obj[2], fill_val])
        tm.assert_index_equal(result, expected)

        values = pd.Index(pd.date_range(fill_val, periods=4))
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2012-01-02"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2012-01-04"),
            ]
        )

        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.xfail(reason="GH 22839: do not ignore timezone, must be object")
    def test_where_index_datetime64tz(self):
        fill_val = pd.Timestamp("2012-01-01", tz="US/Eastern")
        exp_dtype = object
        obj = pd.Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2011-01-02"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"
        cond = pd.Index([True, False, True, False])

        msg = "Index\\(\\.\\.\\.\\) must be called with a collection of some kind"
        with pytest.raises(TypeError, match=msg):
            obj.where(cond, fill_val)

        values = pd.Index(pd.date_range(fill_val, periods=4))
        exp = pd.Index(
            [
                pd.Timestamp("2011-01-01"),
                pd.Timestamp("2012-01-02", tz="US/Eastern"),
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2012-01-04", tz="US/Eastern"),
            ],
            dtype=exp_dtype,
        )

        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_where_series_period(self):
        raise NotImplementedError

    @pytest.mark.parametrize(
        "value", [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, "D")]
    )
    def test_where_index_timedelta64(self, value):
        tdi = pd.timedelta_range("1 Day", periods=4)
        cond = np.array([True, False, False, True])

        expected = pd.TimedeltaIndex(["1 Day", value, value, "4 Days"])
        result = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # wrong-dtyped NaT
        dtnat = np.datetime64("NaT", "ns")
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        assert expected[1] is dtnat

        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)

    def test_where_index_period(self):
        dti = pd.date_range("2016-01-01", periods=3, freq="QS")
        pi = dti.to_period("Q")

        cond = np.array([False, True, False])

        # Passinga  valid scalar
        value = pi[-1] + pi.freq * 10
        expected = pd.PeriodIndex([value, pi[1], value])
        result = pi.where(cond, value)
        tm.assert_index_equal(result, expected)

        # Case passing ndarray[object] of Periods
        other = np.asarray(pi + pi.freq * 10, dtype=object)
        result = pi.where(cond, other)
        expected = pd.PeriodIndex([other[0], pi[1], other[2]])
        tm.assert_index_equal(result, expected)

        # Passing a mismatched scalar -> casts to object
        td = pd.Timedelta(days=4)
        expected = pd.Index([td, pi[1], td], dtype=object)
        result = pi.where(cond, td)
        tm.assert_index_equal(result, expected)

        per = pd.Period("2020-04-21", "D")
        expected = pd.Index([per, pi[1], per], dtype=object)
        result = pi.where(cond, per)
        tm.assert_index_equal(result, expected)


class TestFillnaSeriesCoercion(CoercionBase):

    # not indexing, but place here for consistency

    method = "fillna"

    @pytest.mark.xfail(reason="Test not implemented")
    def test_has_comprehensive_tests(self):
        raise NotImplementedError

    def _assert_fillna_conversion(self, original, value, expected, expected_dtype):
        """test coercion triggered by fillna"""
        target = original.copy()
        res = target.fillna(value)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    @pytest.mark.parametrize(
        "fill_val, fill_dtype",
        [(1, object), (1.1, object), (1 + 1j, object), (True, object)],
    )
    def test_fillna_object(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass(["a", np.nan, "c", "d"])
        assert obj.dtype == object

        exp = klass(["a", fill_val, "c", "d"])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)],
    )
    def test_fillna_float64(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass([1.1, np.nan, 3.3, 4.4])
        assert obj.dtype == np.float64

        exp = klass([1.1, fill_val, 3.3, 4.4])
        # float + complex -> we don't support a complex Index
        # complex for Series,
        # object for Index
        if fill_dtype == np.complex128 and klass == pd.Index:
            fill_dtype = object
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (1, np.complex128),
            (1.1, np.complex128),
            (1 + 1j, np.complex128),
            (True, object),
        ],
    )
    def test_fillna_series_complex128(self, fill_val, fill_dtype):
        obj = pd.Series([1 + 1j, np.nan, 3 + 3j, 4 + 4j])
        assert obj.dtype == np.complex128

        exp = pd.Series([1 + 1j, fill_val, 3 + 3j, 4 + 4j])
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01"), "datetime64[ns]"),
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), object),
            (1, object),
            ("x", object),
        ],
        ids=["datetime64", "datetime64tz", "object", "object"],
    )
    def test_fillna_datetime(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        obj = klass(
            [
                pd.Timestamp("2011-01-01"),
                pd.NaT,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        assert obj.dtype == "datetime64[ns]"

        exp = klass(
            [
                pd.Timestamp("2011-01-01"),
                fill_val,
                pd.Timestamp("2011-01-03"),
                pd.Timestamp("2011-01-04"),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.parametrize(
        "fill_val,fill_dtype",
        [
            (pd.Timestamp("2012-01-01", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
            (pd.Timestamp("2012-01-01"), object),
            (pd.Timestamp("2012-01-01", tz="Asia/Tokyo"), object),
            (1, object),
            ("x", object),
        ],
    )
    def test_fillna_datetime64tz(self, index_or_series, fill_val, fill_dtype):
        klass = index_or_series
        tz = "US/Eastern"

        obj = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                pd.NaT,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        assert obj.dtype == "datetime64[ns, US/Eastern]"

        exp = klass(
            [
                pd.Timestamp("2011-01-01", tz=tz),
                fill_val,
                pd.Timestamp("2011-01-03", tz=tz),
                pd.Timestamp("2011-01-04", tz=tz),
            ]
        )
        self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_int64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_series_period(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="Test not implemented")
    def test_fillna_index_period(self):
        raise NotImplementedError


class TestReplaceSeriesCoercion(CoercionBase):

    klasses = ["series"]
    method = "replace"

    rep: dict[str, list] = {}
    rep["object"] = ["a", "b"]
    rep["int64"] = [4, 5]
    rep["float64"] = [1.1, 2.2]
    rep["complex128"] = [1 + 1j, 2 + 2j]
    rep["bool"] = [True, False]
    rep["datetime64[ns]"] = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-03")]

    for tz in ["UTC", "US/Eastern"]:
        # to test tz => different tz replacement
        key = f"datetime64[ns, {tz}]"
        rep[key] = [
            pd.Timestamp("2011-01-01", tz=tz),
            pd.Timestamp("2011-01-03", tz=tz),
        ]

    rep["timedelta64[ns]"] = [pd.Timedelta("1 day"), pd.Timedelta("2 day")]

    @pytest.fixture(params=["dict", "series"])
    def how(self, request):
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ]
    )
    def from_key(self, request):
        return request.param

    @pytest.fixture(
        params=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[ns, US/Eastern]",
            "timedelta64[ns]",
        ],
        ids=[
            "object",
            "int64",
            "float64",
            "complex128",
            "bool",
            "datetime64",
            "datetime64tz",
            "datetime64tz",
            "timedelta64",
        ],
    )
    def to_key(self, request):
        return request.param

    @pytest.fixture
    def replacer(self, how, from_key, to_key):
        """
        Object we will pass to `Series.replace`
        """
        if how == "dict":
            replacer = dict(zip(self.rep[from_key], self.rep[to_key]))
        elif how == "series":
            replacer = pd.Series(self.rep[to_key], index=self.rep[from_key])
        else:
            raise ValueError
        return replacer

    def test_replace_series(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name="xxx")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        assert obj.dtype == from_key

        if from_key.startswith("datetime") and to_key.startswith("datetime"):
            # tested below
            return
        elif from_key in ["datetime64[ns, US/Eastern]", "datetime64[ns, UTC]"]:
            # tested below
            return

        result = obj.replace(replacer)

        if (from_key == "float64" and to_key in ("int64")) or (
            from_key == "complex128" and to_key in ("int64", "float64")
        ):

            if not IS64 or is_platform_windows():
                pytest.skip(f"32-bit platform buggy: {from_key} -> {to_key}")

            # Expected: do not downcast by replacement
            exp = pd.Series(self.rep[to_key], index=index, name="yyy", dtype=from_key)

        else:
            exp = pd.Series(self.rep[to_key], index=index, name="yyy")
            assert exp.dtype == to_key

        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "to_key",
        ["timedelta64[ns]", "bool", "object", "complex128", "float64", "int64"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key", ["datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"], indirect=True
    )
    def test_replace_series_datetime_tz(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        assert obj.dtype == from_key

        result = obj.replace(replacer)
        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        assert exp.dtype == to_key

        tm.assert_series_equal(result, exp)

    @pytest.mark.parametrize(
        "to_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "from_key",
        ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        indirect=True,
    )
    def test_replace_series_datetime_datetime(self, how, to_key, from_key, replacer):
        index = pd.Index([3, 4], name="xyz")
        obj = pd.Series(self.rep[from_key], index=index, name="yyy")
        assert obj.dtype == from_key

        result = obj.replace(replacer)
        exp = pd.Series(self.rep[to_key], index=index, name="yyy")
        assert exp.dtype == to_key

        tm.assert_series_equal(result, exp)

    @pytest.mark.xfail(reason="Test not implemented")
    def test_replace_series_period(self):
        raise NotImplementedError
