import decimal
import math
import operator

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.api.types import infer_dtype
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
    make_data,
    to_decimal,
)


@pytest.fixture
def dtype():
    return DecimalDtype()


@pytest.fixture
def data():
    return DecimalArray(make_data())


@pytest.fixture
def data_for_twos():
    return DecimalArray([decimal.Decimal(2) for _ in range(100)])


@pytest.fixture
def data_missing():
    return DecimalArray([decimal.Decimal("NaN"), decimal.Decimal(1)])


@pytest.fixture
def data_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("2"), decimal.Decimal("0")]
    )


@pytest.fixture
def data_missing_for_sorting():
    return DecimalArray(
        [decimal.Decimal("1"), decimal.Decimal("NaN"), decimal.Decimal("0")]
    )


@pytest.fixture
def na_cmp():
    return lambda x, y: x.is_nan() and y.is_nan()


@pytest.fixture
def na_value():
    return decimal.Decimal("NaN")


@pytest.fixture
def data_for_grouping():
    b = decimal.Decimal("1.0")
    a = decimal.Decimal("0.0")
    c = decimal.Decimal("2.0")
    na = decimal.Decimal("NaN")
    return DecimalArray([b, b, na, na, a, a, b, c])


class BaseDecimal:
    @classmethod
    def assert_series_equal(cls, left, right, *args, **kwargs):
        def convert(x):
            # need to convert array([Decimal(NaN)], dtype='object') to np.NaN
            # because Series[object].isnan doesn't recognize decimal(NaN) as
            # NA.
            try:
                return math.isnan(x)
            except TypeError:
                return False

        if left.dtype == "object":
            left_na = left.apply(convert)
        else:
            left_na = left.isna()
        if right.dtype == "object":
            right_na = right.apply(convert)
        else:
            right_na = right.isna()

        tm.assert_series_equal(left_na, right_na)
        return tm.assert_series_equal(left[~left_na], right[~right_na], *args, **kwargs)

    @classmethod
    def assert_frame_equal(cls, left, right, *args, **kwargs):
        # TODO(EA): select_dtypes
        tm.assert_index_equal(
            left.columns,
            right.columns,
            exact=kwargs.get("check_column_type", "equiv"),
            check_names=kwargs.get("check_names", True),
            check_exact=kwargs.get("check_exact", False),
            check_categorical=kwargs.get("check_categorical", True),
            obj=f"{kwargs.get('obj', 'DataFrame')}.columns",
        )

        decimals = (left.dtypes == "decimal").index

        for col in decimals:
            cls.assert_series_equal(left[col], right[col], *args, **kwargs)

        left = left.drop(columns=decimals)
        right = right.drop(columns=decimals)
        tm.assert_frame_equal(left, right, *args, **kwargs)


class TestDtype(BaseDecimal, base.BaseDtypeTests):
    def test_hashable(self, dtype):
        pass

    @pytest.mark.parametrize("skipna", [True, False])
    def test_infer_dtype(self, data, data_missing, skipna):
        # here overriding base test to ensure we fall back to return
        # "unknown-array" for an EA pandas doesn't know
        assert infer_dtype(data, skipna=skipna) == "unknown-array"
        assert infer_dtype(data_missing, skipna=skipna) == "unknown-array"


class TestInterface(BaseDecimal, base.BaseInterfaceTests):
    pass


class TestConstructors(BaseDecimal, base.BaseConstructorsTests):
    pass


class TestReshaping(BaseDecimal, base.BaseReshapingTests):
    pass


class TestGetitem(BaseDecimal, base.BaseGetitemTests):
    def test_take_na_value_other_decimal(self):
        arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
        result = arr.take([0, -1], allow_fill=True, fill_value=decimal.Decimal("-1.0"))
        expected = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("-1.0")])
        self.assert_extension_array_equal(result, expected)


class TestMissing(BaseDecimal, base.BaseMissingTests):
    pass


class Reduce:
    def check_reduce(self, s, op_name, skipna):

        if op_name in ["median", "skew", "kurt"]:
            msg = r"decimal does not support the .* operation"
            with pytest.raises(NotImplementedError, match=msg):
                getattr(s, op_name)(skipna=skipna)

        else:
            result = getattr(s, op_name)(skipna=skipna)
            expected = getattr(np.asarray(s), op_name)()
            tm.assert_almost_equal(result, expected)


class TestNumericReduce(Reduce, base.BaseNumericReduceTests):
    pass


class TestBooleanReduce(Reduce, base.BaseBooleanReduceTests):
    pass


class TestMethods(BaseDecimal, base.BaseMethodsTests):
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, request):
        all_data = all_data[:10]
        if dropna:
            other = np.array(all_data[~all_data.isna()])
        else:
            other = all_data

        vcs = pd.Series(all_data).value_counts(dropna=dropna)
        vcs_ex = pd.Series(other).value_counts(dropna=dropna)

        with decimal.localcontext() as ctx:
            # avoid raising when comparing Decimal("NAN") < Decimal(2)
            ctx.traps[decimal.InvalidOperation] = False

            result = vcs.sort_index()
            expected = vcs_ex.sort_index()

        tm.assert_series_equal(result, expected)

    def test_value_counts_with_normalize(self, data):
        return super().test_value_counts_with_normalize(data)


class TestCasting(BaseDecimal, base.BaseCastingTests):
    pass


class TestGroupby(BaseDecimal, base.BaseGroupbyTests):
    def test_groupby_agg_extension(self, data_for_grouping):
        super().test_groupby_agg_extension(data_for_grouping)


class TestSetitem(BaseDecimal, base.BaseSetitemTests):
    pass


class TestPrinting(BaseDecimal, base.BasePrintingTests):
    def test_series_repr(self, data):
        # Overriding this base test to explicitly test that
        # the custom _formatter is used
        ser = pd.Series(data)
        assert data.dtype.name in repr(ser)
        assert "Decimal: " in repr(ser)


# TODO(extension)
@pytest.mark.xfail(
    reason=(
        "raising AssertionError as this is not implemented, though easy enough to do"
    )
)
def test_series_constructor_coerce_data_to_extension_dtype_raises():
    xpr = (
        "Cannot cast data to extension dtype 'decimal'. Pass the "
        "extension array directly."
    )
    with pytest.raises(ValueError, match=xpr):
        pd.Series([0, 1, 2], dtype=DecimalDtype())


def test_series_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.Series(arr, dtype=DecimalDtype())
    expected = pd.Series(arr)
    tm.assert_series_equal(result, expected)

    result = pd.Series(arr, dtype="int64")
    expected = pd.Series([10])
    tm.assert_series_equal(result, expected)


def test_dataframe_constructor_with_dtype():
    arr = DecimalArray([decimal.Decimal("10.0")])

    result = pd.DataFrame({"A": arr}, dtype=DecimalDtype())
    expected = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, expected)

    arr = DecimalArray([decimal.Decimal("10.0")])
    result = pd.DataFrame({"A": arr}, dtype="int64")
    expected = pd.DataFrame({"A": [10]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("frame", [True, False])
def test_astype_dispatches(frame):
    # This is a dtype-specific test that ensures Series[decimal].astype
    # gets all the way through to ExtensionArray.astype
    # Designing a reliable smoke test that works for arbitrary data types
    # is difficult.
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name="a")
    ctx = decimal.Context()
    ctx.prec = 5

    if frame:
        data = data.to_frame()

    result = data.astype(DecimalDtype(ctx))

    if frame:
        result = result["a"]

    assert result.dtype.context.prec == ctx.prec


class TestArithmeticOps(BaseDecimal, base.BaseArithmeticOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        super().check_opname(s, op_name, other, exc=None)

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        op_name = all_arithmetic_operators
        s = pd.Series(data)

        context = decimal.getcontext()
        divbyzerotrap = context.traps[decimal.DivisionByZero]
        invalidoptrap = context.traps[decimal.InvalidOperation]
        context.traps[decimal.DivisionByZero] = 0
        context.traps[decimal.InvalidOperation] = 0

        # Decimal supports ops with int, but not float
        other = pd.Series([int(d * 100) for d in data])
        self.check_opname(s, op_name, other)

        if "mod" not in op_name:
            self.check_opname(s, op_name, s * 2)

        self.check_opname(s, op_name, 0)
        self.check_opname(s, op_name, 5)
        context.traps[decimal.DivisionByZero] = divbyzerotrap
        context.traps[decimal.InvalidOperation] = invalidoptrap

    def _check_divmod_op(self, s, op, other, exc=NotImplementedError):
        # We implement divmod
        super()._check_divmod_op(s, op, other, exc=None)


class TestComparisonOps(BaseDecimal, base.BaseComparisonOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        super().check_opname(s, op_name, other, exc=None)

    def _compare_other(self, s, data, op_name, other):
        self.check_opname(s, op_name, other)

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        self._compare_other(s, data, op_name, 0.5)

    def test_compare_array(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)

        alter = np.random.choice([-1, 0, 1], len(data))
        # Randomly double, halve or keep same value
        other = pd.Series(data) * [decimal.Decimal(pow(2.0, i)) for i in alter]
        self._compare_other(s, data, op_name, other)


class DecimalArrayWithoutFromSequence(DecimalArray):
    """Helper class for testing error handling in _from_sequence."""

    def _from_sequence(cls, scalars, dtype=None, copy=False):
        raise KeyError("For the test")


class DecimalArrayWithoutCoercion(DecimalArrayWithoutFromSequence):
    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)


DecimalArrayWithoutCoercion._add_arithmetic_ops()


def test_combine_from_sequence_raises(monkeypatch):
    # https://github.com/pandas-dev/pandas/issues/22850
    cls = DecimalArrayWithoutFromSequence

    @classmethod
    def construct_array_type(cls):
        return DecimalArrayWithoutFromSequence

    monkeypatch.setattr(DecimalDtype, "construct_array_type", construct_array_type)

    arr = cls([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    ser = pd.Series(arr)
    result = ser.combine(ser, operator.add)

    # note: object dtype
    expected = pd.Series(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "class_", [DecimalArrayWithoutFromSequence, DecimalArrayWithoutCoercion]
)
def test_scalar_ops_from_sequence_raises(class_):
    # op(EA, EA) should return an EA, or an ndarray if it's not possible
    # to return an EA with the return values.
    arr = class_([decimal.Decimal("1.0"), decimal.Decimal("2.0")])
    result = arr + arr
    expected = np.array(
        [decimal.Decimal("2.0"), decimal.Decimal("4.0")], dtype="object"
    )
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "reverse, expected_div, expected_mod",
    [(False, [0, 1, 1, 2], [1, 0, 1, 0]), (True, [2, 1, 0, 0], [0, 0, 2, 2])],
)
def test_divmod_array(reverse, expected_div, expected_mod):
    # https://github.com/pandas-dev/pandas/issues/22930
    arr = to_decimal([1, 2, 3, 4])
    if reverse:
        div, mod = divmod(2, arr)
    else:
        div, mod = divmod(arr, 2)
    expected_div = to_decimal(expected_div)
    expected_mod = to_decimal(expected_mod)

    tm.assert_extension_array_equal(div, expected_div)
    tm.assert_extension_array_equal(mod, expected_mod)


def test_ufunc_fallback(data):
    a = data[:5]
    s = pd.Series(a, index=range(3, 8))
    result = np.abs(s)
    expected = pd.Series(np.abs(a), index=range(3, 8))
    tm.assert_series_equal(result, expected)


def test_array_ufunc():
    a = to_decimal([1, 2, 3])
    result = np.exp(a)
    expected = to_decimal(np.exp(a._data))
    tm.assert_extension_array_equal(result, expected)


def test_array_ufunc_series():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.exp(s)
    expected = pd.Series(to_decimal(np.exp(a._data)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_scalar_other():
    # check _HANDLED_TYPES
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)
    result = np.add(s, decimal.Decimal(1))
    expected = pd.Series(np.add(a, decimal.Decimal(1)))
    tm.assert_series_equal(result, expected)


def test_array_ufunc_series_defer():
    a = to_decimal([1, 2, 3])
    s = pd.Series(a)

    expected = pd.Series(to_decimal([2, 4, 6]))
    r1 = np.add(s, a)
    r2 = np.add(a, s)

    tm.assert_series_equal(r1, expected)
    tm.assert_series_equal(r2, expected)


def test_groupby_agg():
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    data = make_data()[:5]
    df = pd.DataFrame(
        {"id1": [0, 0, 0, 1, 1], "id2": [0, 1, 0, 1, 1], "decimals": DecimalArray(data)}
    )

    # single key, selected column
    expected = pd.Series(to_decimal([data[0], data[3]]))
    result = df.groupby("id1")["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby(df["id1"]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple keys, selected column
    expected = pd.Series(
        to_decimal([data[0], data[1], data[3]]),
        index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 1)]),
    )
    result = df.groupby(["id1", "id2"])["decimals"].agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)
    result = df["decimals"].groupby([df["id1"], df["id2"]]).agg(lambda x: x.iloc[0])
    tm.assert_series_equal(result, expected, check_names=False)

    # multiple columns
    expected = pd.DataFrame({"id2": [0, 1], "decimals": to_decimal([data[0], data[3]])})
    result = df.groupby("id1").agg(lambda x: x.iloc[0])
    tm.assert_frame_equal(result, expected, check_names=False)


def test_groupby_agg_ea_method(monkeypatch):
    # Ensure that the result of agg is inferred to be decimal dtype
    # https://github.com/pandas-dev/pandas/issues/29141

    def DecimalArray__my_sum(self):
        return np.sum(np.array(self))

    monkeypatch.setattr(DecimalArray, "my_sum", DecimalArray__my_sum, raising=False)

    data = make_data()[:5]
    df = pd.DataFrame({"id": [0, 0, 0, 1, 1], "decimals": DecimalArray(data)})
    expected = pd.Series(to_decimal([data[0] + data[1] + data[2], data[3] + data[4]]))

    result = df.groupby("id")["decimals"].agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)
    s = pd.Series(DecimalArray(data))
    result = s.groupby(np.array([0, 0, 0, 1, 1])).agg(lambda x: x.values.my_sum())
    tm.assert_series_equal(result, expected, check_names=False)


def test_indexing_no_materialize(monkeypatch):
    # See https://github.com/pandas-dev/pandas/issues/29708
    # Ensure that indexing operations do not materialize (convert to a numpy
    # array) the ExtensionArray unnecessary

    def DecimalArray__array__(self, dtype=None):
        raise Exception("tried to convert a DecimalArray to a numpy array")

    monkeypatch.setattr(DecimalArray, "__array__", DecimalArray__array__, raising=False)

    data = make_data()
    s = pd.Series(DecimalArray(data))
    df = pd.DataFrame({"a": s, "b": range(len(s))})

    # ensure the following operations do not raise an error
    s[s > 0.5]
    df[s > 0.5]
    s.at[0]
    df.at[0, "a"]


def test_to_numpy_keyword():
    # test the extra keyword
    values = [decimal.Decimal("1.1111"), decimal.Decimal("2.2222")]
    expected = np.array(
        [decimal.Decimal("1.11"), decimal.Decimal("2.22")], dtype="object"
    )
    a = pd.array(values, dtype="decimal")
    result = a.to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)

    result = pd.Series(a).to_numpy(decimals=2)
    tm.assert_numpy_array_equal(result, expected)
