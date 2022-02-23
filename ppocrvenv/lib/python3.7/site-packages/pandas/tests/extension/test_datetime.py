"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.

The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.

"""
import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas.core.arrays import DatetimeArray
from pandas.tests.extension import base


@pytest.fixture(params=["US/Central"])
def dtype(request):
    return DatetimeTZDtype(unit="ns", tz=request.param)


@pytest.fixture
def data(dtype):
    data = DatetimeArray(pd.date_range("2000", periods=100, tz=dtype.tz), dtype=dtype)
    return data


@pytest.fixture
def data_missing(dtype):
    return DatetimeArray(
        np.array(["NaT", "2000-01-01"], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def data_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    return DatetimeArray(np.array([b, c, a], dtype="datetime64[ns]"), dtype=dtype)


@pytest.fixture
def data_missing_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    return DatetimeArray(np.array([b, "NaT", a], dtype="datetime64[ns]"), dtype=dtype)


@pytest.fixture
def data_for_grouping(dtype):
    """
    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    na = "NaT"
    return DatetimeArray(
        np.array([b, b, na, na, a, a, b, c], dtype="datetime64[ns]"), dtype=dtype
    )


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return a is pd.NaT and a is b

    return cmp


@pytest.fixture
def na_value():
    return pd.NaT


# ----------------------------------------------------------------------------
class BaseDatetimeTests:
    pass


# ----------------------------------------------------------------------------
# Tests
class TestDatetimeDtype(BaseDatetimeTests, base.BaseDtypeTests):
    pass


class TestConstructors(BaseDatetimeTests, base.BaseConstructorsTests):
    def test_series_constructor(self, data):
        # Series construction drops any .freq attr
        data = data._with_freq(None)
        super().test_series_constructor(data)


class TestGetitem(BaseDatetimeTests, base.BaseGetitemTests):
    pass


class TestMethods(BaseDatetimeTests, base.BaseMethodsTests):
    @pytest.mark.skip(reason="Incorrect expected")
    def test_value_counts(self, all_data, dropna):
        pass

    def test_combine_add(self, data_repeated):
        # Timestamp.__add__(Timestamp) not defined
        pass


class TestInterface(BaseDatetimeTests, base.BaseInterfaceTests):
    def test_array_interface(self, data):
        if data.tz:
            # np.asarray(DTA) is currently always tz-naive.
            pytest.skip("GH-23569")
        else:
            super().test_array_interface(data)


class TestArithmeticOps(BaseDatetimeTests, base.BaseArithmeticOpsTests):
    implements = {"__sub__", "__rsub__"}

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        if all_arithmetic_operators in self.implements:
            df = pd.DataFrame({"A": data})
            self.check_opname(df, all_arithmetic_operators, data[0], exc=None)
        else:
            # ... but not the rest.
            super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        if all_arithmetic_operators in self.implements:
            s = pd.Series(data)
            self.check_opname(s, all_arithmetic_operators, s.iloc[0], exc=None)
        else:
            # ... but not the rest.
            super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_add_series_with_extension_array(self, data):
        # Datetime + Datetime not implemented
        s = pd.Series(data)
        msg = "cannot add DatetimeArray and DatetimeArray"
        with pytest.raises(TypeError, match=msg):
            s + data

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        if all_arithmetic_operators in self.implements:
            s = pd.Series(data)
            self.check_opname(s, all_arithmetic_operators, s.iloc[0], exc=None)
        else:
            # ... but not the rest.
            super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_divmod_series_array(self):
        # GH 23287
        # skipping because it is not implemented
        pass


class TestCasting(BaseDatetimeTests, base.BaseCastingTests):
    pass


class TestComparisonOps(BaseDatetimeTests, base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        # the base test is not appropriate for us. We raise on comparison
        # with (some) integers, depending on the value.
        pass


class TestMissing(BaseDatetimeTests, base.BaseMissingTests):
    pass


class TestReshaping(BaseDatetimeTests, base.BaseReshapingTests):
    @pytest.mark.skip(reason="We have DatetimeTZBlock")
    def test_concat(self, data, in_frame):
        pass

    def test_concat_mixed_dtypes(self, data):
        # concat(Series[datetimetz], Series[category]) uses a
        # plain np.array(values) on the DatetimeArray, which
        # drops the tz.
        super().test_concat_mixed_dtypes(data)


class TestSetitem(BaseDatetimeTests, base.BaseSetitemTests):
    pass


class TestGroupby(BaseDatetimeTests, base.BaseGroupbyTests):
    pass


class TestPrinting(BaseDatetimeTests, base.BasePrintingTests):
    pass


class Test2DCompat(BaseDatetimeTests, base.Dim2CompatTests):
    pass
