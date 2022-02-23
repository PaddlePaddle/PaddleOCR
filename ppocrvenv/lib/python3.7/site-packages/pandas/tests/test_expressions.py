import operator
import re

import numpy as np
import pytest

import pandas._testing as tm
from pandas.core.api import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.computation import expressions as expr

_frame = DataFrame(np.random.randn(10001, 4), columns=list("ABCD"), dtype="float64")
_frame2 = DataFrame(np.random.randn(100, 4), columns=list("ABCD"), dtype="float64")
_mixed = DataFrame(
    {
        "A": _frame["A"].copy(),
        "B": _frame["B"].astype("float32"),
        "C": _frame["C"].astype("int64"),
        "D": _frame["D"].astype("int32"),
    }
)
_mixed2 = DataFrame(
    {
        "A": _frame2["A"].copy(),
        "B": _frame2["B"].astype("float32"),
        "C": _frame2["C"].astype("int64"),
        "D": _frame2["D"].astype("int32"),
    }
)
_integer = DataFrame(
    np.random.randint(1, 100, size=(10001, 4)), columns=list("ABCD"), dtype="int64"
)
_integer2 = DataFrame(
    np.random.randint(1, 100, size=(101, 4)), columns=list("ABCD"), dtype="int64"
)
_array = _frame["A"].values.copy()
_array2 = _frame2["A"].values.copy()

_array_mixed = _mixed["D"].values.copy()
_array_mixed2 = _mixed2["D"].values.copy()


@pytest.mark.skipif(not expr.USE_NUMEXPR, reason="not using numexpr")
class TestExpressions:
    def setup_method(self, method):

        self.frame = _frame.copy()
        self.frame2 = _frame2.copy()
        self.mixed = _mixed.copy()
        self.mixed2 = _mixed2.copy()
        self._MIN_ELEMENTS = expr._MIN_ELEMENTS

    def teardown_method(self, method):
        expr._MIN_ELEMENTS = self._MIN_ELEMENTS

    @staticmethod
    def call_op(df, other, flex: bool, opname: str):
        if flex:
            op = lambda x, y: getattr(x, opname)(y)
            op.__name__ = opname
        else:
            op = getattr(operator, opname)

        expr.set_use_numexpr(False)
        expected = op(df, other)
        expr.set_use_numexpr(True)

        expr.get_test_result()

        result = op(df, other)
        return result, expected

    def run_arithmetic(self, df, other, flex: bool):
        expr._MIN_ELEMENTS = 0
        operations = ["add", "sub", "mul", "mod", "truediv", "floordiv"]
        for arith in operations:
            result, expected = self.call_op(df, other, flex, arith)

            if arith == "truediv":
                if expected.ndim == 1:
                    assert expected.dtype.kind == "f"
                else:
                    assert all(x.kind == "f" for x in expected.dtypes.values)
            tm.assert_equal(expected, result)

    def run_binary(self, df, other, flex: bool):
        """
        tests solely that the result is the same whether or not numexpr is
        enabled.  Need to test whether the function does the correct thing
        elsewhere.
        """
        expr._MIN_ELEMENTS = 0
        expr.set_test_mode(True)
        operations = ["gt", "lt", "ge", "le", "eq", "ne"]

        for arith in operations:
            result, expected = self.call_op(df, other, flex, arith)

            used_numexpr = expr.get_test_result()
            assert used_numexpr, "Did not use numexpr as expected."
            tm.assert_equal(expected, result)

    def run_frame(self, df, other, flex: bool):
        self.run_arithmetic(df, other, flex)

        expr.set_use_numexpr(False)
        binary_comp = other + 1
        expr.set_use_numexpr(True)
        self.run_binary(df, binary_comp, flex)

        for i in range(len(df.columns)):
            self.run_arithmetic(df.iloc[:, i], other.iloc[:, i], flex)
            # FIXME: dont leave commented-out
            # series doesn't uses vec_compare instead of numexpr...
            # binary_comp = other.iloc[:, i] + 1
            # self.run_binary(df.iloc[:, i], binary_comp, flex)

    @pytest.mark.parametrize(
        "df",
        [
            _integer,
            _integer2,
            # randint to get a case with zeros
            _integer * np.random.randint(0, 2, size=np.shape(_integer)),
            _frame,
            _frame2,
            _mixed,
            _mixed2,
        ],
    )
    @pytest.mark.parametrize("flex", [True, False])
    def test_arithmetic(self, df, flex):
        self.run_frame(df, df, flex)

    def test_invalid(self):
        array = np.random.randn(1_000_001)
        array2 = np.random.randn(100)

        # no op
        result = expr._can_use_numexpr(operator.add, None, array, array, "evaluate")
        assert not result

        # min elements
        result = expr._can_use_numexpr(operator.add, "+", array2, array2, "evaluate")
        assert not result

        # ok, we only check on first part of expression
        result = expr._can_use_numexpr(operator.add, "+", array, array2, "evaluate")
        assert result

    @pytest.mark.parametrize(
        "opname,op_str",
        [("add", "+"), ("sub", "-"), ("mul", "*"), ("truediv", "/"), ("pow", "**")],
    )
    @pytest.mark.parametrize(
        "left,right", [(_array, _array2), (_array_mixed, _array_mixed2)]
    )
    def test_binary_ops(self, opname, op_str, left, right):
        def testit():

            if opname == "pow":
                # TODO: get this working
                return

            op = getattr(operator, opname)

            result = expr.evaluate(op, left, left, use_numexpr=True)
            expected = expr.evaluate(op, left, left, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op_str, right, right, "evaluate")
            assert not result

        expr.set_use_numexpr(False)
        testit()
        expr.set_use_numexpr(True)
        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        "opname,op_str",
        [
            ("gt", ">"),
            ("lt", "<"),
            ("ge", ">="),
            ("le", "<="),
            ("eq", "=="),
            ("ne", "!="),
        ],
    )
    @pytest.mark.parametrize(
        "left,right", [(_array, _array2), (_array_mixed, _array_mixed2)]
    )
    def test_comparison_ops(self, opname, op_str, left, right):
        def testit():
            f12 = left + 1
            f22 = right + 1

            op = getattr(operator, opname)

            result = expr.evaluate(op, left, f12, use_numexpr=True)
            expected = expr.evaluate(op, left, f12, use_numexpr=False)
            tm.assert_numpy_array_equal(result, expected)

            result = expr._can_use_numexpr(op, op_str, right, f22, "evaluate")
            assert not result

        expr.set_use_numexpr(False)
        testit()
        expr.set_use_numexpr(True)
        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize("cond", [True, False])
    @pytest.mark.parametrize("df", [_frame, _frame2, _mixed, _mixed2])
    def test_where(self, cond, df):
        def testit():
            c = np.empty(df.shape, dtype=np.bool_)
            c.fill(cond)
            result = expr.where(c, df.values, df.values + 1)
            expected = np.where(c, df.values, df.values + 1)
            tm.assert_numpy_array_equal(result, expected)

        expr.set_use_numexpr(False)
        testit()
        expr.set_use_numexpr(True)
        expr.set_numexpr_threads(1)
        testit()
        expr.set_numexpr_threads()
        testit()

    @pytest.mark.parametrize(
        "op_str,opname", [("/", "truediv"), ("//", "floordiv"), ("**", "pow")]
    )
    def test_bool_ops_raise_on_arithmetic(self, op_str, opname):
        df = DataFrame({"a": np.random.rand(10) > 0.5, "b": np.random.rand(10) > 0.5})

        msg = f"operator '{opname}' not implemented for bool dtypes"
        f = getattr(operator, opname)
        err_msg = re.escape(msg)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, df.b)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df.a, True)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df.a)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(False, df)

        with pytest.raises(NotImplementedError, match=err_msg):
            f(df, True)

    @pytest.mark.parametrize(
        "op_str,opname", [("+", "add"), ("*", "mul"), ("-", "sub")]
    )
    def test_bool_ops_warn_on_arithmetic(self, op_str, opname):
        n = 10
        df = DataFrame({"a": np.random.rand(n) > 0.5, "b": np.random.rand(n) > 0.5})

        subs = {"+": "|", "*": "&", "-": "^"}
        sub_funcs = {"|": "or_", "&": "and_", "^": "xor"}

        f = getattr(operator, opname)
        fe = getattr(operator, sub_funcs[subs[op_str]])

        if op_str == "-":
            # raises TypeError
            return

        with tm.use_numexpr(True, min_elements=5):
            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(df, df)
                e = fe(df, df)
                tm.assert_frame_equal(r, e)

            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(df.a, df.b)
                e = fe(df.a, df.b)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(df.a, True)
                e = fe(df.a, True)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(False, df.a)
                e = fe(False, df.a)
                tm.assert_series_equal(r, e)

            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(False, df)
                e = fe(False, df)
                tm.assert_frame_equal(r, e)

            with tm.assert_produces_warning(check_stacklevel=False):
                r = f(df, True)
                e = fe(df, True)
                tm.assert_frame_equal(r, e)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                DataFrame(
                    [[0, 1, 2, "aa"], [0, 1, 2, "aa"]], columns=["a", "b", "c", "dtype"]
                ),
                DataFrame([[False, False], [False, False]], columns=["a", "dtype"]),
            ),
            (
                DataFrame(
                    [[0, 3, 2, "aa"], [0, 4, 2, "aa"], [0, 1, 1, "bb"]],
                    columns=["a", "b", "c", "dtype"],
                ),
                DataFrame(
                    [[False, False], [False, False], [False, False]],
                    columns=["a", "dtype"],
                ),
            ),
        ],
    )
    def test_bool_ops_column_name_dtype(self, test_input, expected):
        # GH 22383 - .ne fails if columns containing column name 'dtype'
        result = test_input.loc[:, ["a", "dtype"]].ne(test_input.loc[:, ["a", "dtype"]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "arith", ("add", "sub", "mul", "mod", "truediv", "floordiv")
    )
    @pytest.mark.parametrize("axis", (0, 1))
    def test_frame_series_axis(self, axis, arith):
        # GH#26736 Dataframe.floordiv(Series, axis=1) fails

        df = self.frame
        if axis == 1:
            other = self.frame.iloc[0, :]
        else:
            other = self.frame.iloc[:, 0]

        expr._MIN_ELEMENTS = 0

        op_func = getattr(df, arith)

        expr.set_use_numexpr(False)
        expected = op_func(other, axis=axis)
        expr.set_use_numexpr(True)

        result = op_func(other, axis=axis)
        tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize(
        "op",
        [
            "__mod__",
            pytest.param("__rmod__", marks=pytest.mark.xfail(reason="GH-36552")),
            "__floordiv__",
            "__rfloordiv__",
        ],
    )
    @pytest.mark.parametrize("box", [DataFrame, Series, Index])
    @pytest.mark.parametrize("scalar", [-5, 5])
    def test_python_semantics_with_numexpr_installed(self, op, box, scalar):
        # https://github.com/pandas-dev/pandas/issues/36047
        expr._MIN_ELEMENTS = 0
        data = np.arange(-50, 50)
        obj = box(data)
        method = getattr(obj, op)
        result = method(scalar)

        # compare result with numpy
        expr.set_use_numexpr(False)
        expected = method(scalar)
        expr.set_use_numexpr(True)
        tm.assert_equal(result, expected)

        # compare result element-wise with Python
        for i, elem in enumerate(data):
            if box == DataFrame:
                scalar_result = result.iloc[i, 0]
            else:
                scalar_result = result[i]
            try:
                expected = getattr(int(elem), op)(scalar)
            except ZeroDivisionError:
                pass
            else:
                assert scalar_result == expected
