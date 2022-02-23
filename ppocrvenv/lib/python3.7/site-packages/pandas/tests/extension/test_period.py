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

from pandas._libs import iNaT

from pandas.core.dtypes.dtypes import PeriodDtype

import pandas as pd
from pandas.core.arrays import PeriodArray
from pandas.tests.extension import base


@pytest.fixture
def dtype():
    return PeriodDtype(freq="D")


@pytest.fixture
def data(dtype):
    return PeriodArray(np.arange(1970, 2070), freq=dtype.freq)


@pytest.fixture
def data_for_twos(dtype):
    return PeriodArray(np.ones(100) * 2, freq=dtype.freq)


@pytest.fixture
def data_for_sorting(dtype):
    return PeriodArray([2018, 2019, 2017], freq=dtype.freq)


@pytest.fixture
def data_missing(dtype):
    return PeriodArray([iNaT, 2017], freq=dtype.freq)


@pytest.fixture
def data_missing_for_sorting(dtype):
    return PeriodArray([2018, iNaT, 2017], freq=dtype.freq)


@pytest.fixture
def data_for_grouping(dtype):
    B = 2018
    NA = iNaT
    A = 2017
    C = 2019
    return PeriodArray([B, B, NA, NA, A, A, B, C], freq=dtype.freq)


@pytest.fixture
def na_value():
    return pd.NaT


class BasePeriodTests:
    pass


class TestPeriodDtype(BasePeriodTests, base.BaseDtypeTests):
    pass


class TestConstructors(BasePeriodTests, base.BaseConstructorsTests):
    pass


class TestGetitem(BasePeriodTests, base.BaseGetitemTests):
    pass


class TestMethods(BasePeriodTests, base.BaseMethodsTests):
    def test_combine_add(self, data_repeated):
        # Period + Period is not defined.
        pass


class TestInterface(BasePeriodTests, base.BaseInterfaceTests):

    pass


class TestArithmeticOps(BasePeriodTests, base.BaseArithmeticOpsTests):
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
        # we implement substitution...
        if all_arithmetic_operators in self.implements:
            s = pd.Series(data)
            self.check_opname(s, all_arithmetic_operators, s.iloc[0], exc=None)
        else:
            # ... but not the rest.
            super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        if all_arithmetic_operators in self.implements:
            s = pd.Series(data)
            self.check_opname(s, all_arithmetic_operators, s.iloc[0], exc=None)
        else:
            # ... but not the rest.
            super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def _check_divmod_op(self, s, op, other, exc=NotImplementedError):
        super()._check_divmod_op(s, op, other, exc=TypeError)

    def test_add_series_with_extension_array(self, data):
        # we don't implement + for Period
        s = pd.Series(data)
        msg = (
            r"unsupported operand type\(s\) for \+: "
            r"\'PeriodArray\' and \'PeriodArray\'"
        )
        with pytest.raises(TypeError, match=msg):
            s + data

    @pytest.mark.parametrize("box", [pd.Series, pd.DataFrame])
    def test_direct_arith_with_ndframe_returns_not_implemented(self, data, box):
        # Override to use __sub__ instead of __add__
        other = pd.Series(data)
        if box is pd.DataFrame:
            other = other.to_frame()

        result = data.__sub__(other)
        assert result is NotImplemented


class TestCasting(BasePeriodTests, base.BaseCastingTests):
    pass


class TestComparisonOps(BasePeriodTests, base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        # the base test is not appropriate for us. We raise on comparison
        # with (some) integers, depending on the value.
        pass


class TestMissing(BasePeriodTests, base.BaseMissingTests):
    pass


class TestReshaping(BasePeriodTests, base.BaseReshapingTests):
    pass


class TestSetitem(BasePeriodTests, base.BaseSetitemTests):
    pass


class TestGroupby(BasePeriodTests, base.BaseGroupbyTests):
    pass


class TestPrinting(BasePeriodTests, base.BasePrintingTests):
    pass


class TestParsing(BasePeriodTests, base.BaseParsingTests):
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data):
        super().test_EA_types(engine, data)


class Test2DCompat(BasePeriodTests, base.Dim2CompatTests):
    pass
