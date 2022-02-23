from __future__ import annotations

from functools import reduce
from itertools import product
import operator
import warnings

import numpy as np
import pytest

from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import (
    is_bool,
    is_list_like,
    is_scalar,
)

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    compat,
    date_range,
)
import pandas._testing as tm
from pandas.core.computation import pytables
from pandas.core.computation.engines import (
    ENGINES,
    NumExprClobberingError,
)
import pandas.core.computation.expr as expr
from pandas.core.computation.expr import (
    BaseExprVisitor,
    PandasExprVisitor,
    PythonExprVisitor,
)
from pandas.core.computation.expressions import (
    NUMEXPR_INSTALLED,
    USE_NUMEXPR,
)
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    SPECIAL_CASE_ARITH_OPS_SYMS,
    _binary_math_ops,
    _binary_ops_dict,
    _unary_math_ops,
)


@pytest.fixture(
    params=(
        pytest.param(
            engine,
            marks=pytest.mark.skipif(
                engine == "numexpr" and not USE_NUMEXPR,
                reason=f"numexpr enabled->{USE_NUMEXPR}, "
                f"installed->{NUMEXPR_INSTALLED}",
            ),
        )
        for engine in ENGINES
    )
)
def engine(request):
    return request.param


@pytest.fixture(params=expr.PARSERS)
def parser(request):
    return request.param


def _get_unary_fns_for_ne():
    return list(_unary_math_ops) if NUMEXPR_INSTALLED else []


@pytest.fixture(params=_get_unary_fns_for_ne())
def unary_fns_for_ne(request):
    return request.param


def engine_has_neg_frac(engine):
    return ENGINES[engine].has_neg_frac


def _eval_single_bin(lhs, cmp1, rhs, engine):
    c = _binary_ops_dict[cmp1]
    if engine_has_neg_frac(engine):
        try:
            return c(lhs, rhs)
        except ValueError as e:
            if str(e).startswith(
                "negative number cannot be raised to a fractional power"
            ):
                return np.nan
            raise
    return c(lhs, rhs)


def _series_and_2d_ndarray(lhs, rhs):
    return (
        isinstance(lhs, Series) and isinstance(rhs, np.ndarray) and rhs.ndim > 1
    ) or (isinstance(rhs, Series) and isinstance(lhs, np.ndarray) and lhs.ndim > 1)


def _series_and_frame(lhs, rhs):
    return (isinstance(lhs, Series) and isinstance(rhs, DataFrame)) or (
        isinstance(rhs, Series) and isinstance(lhs, DataFrame)
    )


def _bool_and_frame(lhs, rhs):
    return isinstance(lhs, bool) and isinstance(rhs, pd.core.generic.NDFrame)


def _is_py3_complex_incompat(result, expected):
    return isinstance(expected, (complex, np.complexfloating)) and np.isnan(result)


_good_arith_ops = sorted(set(ARITH_OPS_SYMS).difference(SPECIAL_CASE_ARITH_OPS_SYMS))


# TODO: using range(5) here is a kludge
@pytest.fixture(params=list(range(5)))
def lhs(request):

    nan_df1 = DataFrame(np.random.rand(10, 5))
    nan_df1[nan_df1 > 0.5] = np.nan

    opts = (
        DataFrame(np.random.randn(10, 5)),
        Series(np.random.randn(5)),
        Series([1, 2, np.nan, np.nan, 5]),
        nan_df1,
        np.random.randn(),
    )
    return opts[request.param]


rhs = lhs
midhs = lhs


@td.skip_if_no_ne
class TestEvalNumexprPandas:
    exclude_cmp: list[str] = []
    exclude_bool: list[str] = []

    engine = "numexpr"
    parser = "pandas"

    @classmethod
    def setup_class(cls):
        import numexpr as ne

        cls.ne = ne

    @property
    def current_engines(self):
        return (engine for engine in ENGINES if engine != self.engine)

    @pytest.mark.parametrize(
        "cmp1",
        ["!=", "==", "<=", ">=", "<", ">"],
        ids=["ne", "eq", "le", "ge", "lt", "gt"],
    )
    @pytest.mark.parametrize("cmp2", [">", "<"], ids=["gt", "lt"])
    @pytest.mark.parametrize("binop", expr.BOOL_OPS_SYMS)
    def test_complex_cmp_ops(self, cmp1, cmp2, binop, lhs, rhs):
        if binop in self.exclude_bool:
            pytest.skip()

        lhs_new = _eval_single_bin(lhs, cmp1, rhs, self.engine)
        rhs_new = _eval_single_bin(lhs, cmp2, rhs, self.engine)
        expected = _eval_single_bin(lhs_new, binop, rhs_new, self.engine)

        ex = f"(lhs {cmp1} rhs) {binop} (lhs {cmp2} rhs)"
        result = pd.eval(ex, engine=self.engine, parser=self.parser)
        self.check_equal(result, expected)

    @pytest.mark.parametrize("cmp_op", expr.CMP_OPS_SYMS)
    def test_simple_cmp_ops(self, cmp_op):
        if cmp_op in self.exclude_cmp:
            pytest.skip()

        bool_lhses = (
            DataFrame(tm.randbool(size=(10, 5))),
            Series(tm.randbool((5,))),
            tm.randbool(),
        )
        bool_rhses = (
            DataFrame(tm.randbool(size=(10, 5))),
            Series(tm.randbool((5,))),
            tm.randbool(),
        )
        for lhs, rhs in product(bool_lhses, bool_rhses):
            self.check_simple_cmp_op(lhs, cmp_op, rhs)

    @pytest.mark.parametrize("op", _good_arith_ops)
    def test_binary_arith_ops(self, op, lhs, rhs, request):
        self.check_binary_arith_op(lhs, op, rhs)

    def test_modulus(self, lhs, rhs):
        self.check_modulus(lhs, "%", rhs)

    def test_floor_division(self, lhs, rhs):
        self.check_floor_division(lhs, "//", rhs)

    @td.skip_if_windows
    def test_pow(self, lhs, rhs):
        # odd failure on win32 platform, so skip
        self.check_pow(lhs, "**", rhs)

    @pytest.mark.parametrize("op", expr.CMP_OPS_SYMS)
    def test_single_invert_op(self, op, lhs):
        if op in self.exclude_cmp:
            pytest.skip()

        self.check_single_invert_op(lhs, op)

    @pytest.mark.parametrize("op", expr.CMP_OPS_SYMS)
    def test_compound_invert_op(self, op, lhs, rhs):
        if op in self.exclude_cmp:
            pytest.skip()

        self.check_compound_invert_op(lhs, op, rhs)

    @pytest.mark.parametrize("cmp1", ["<", ">"])
    @pytest.mark.parametrize("cmp2", ["<", ">"])
    def test_chained_cmp_op(self, cmp1, cmp2, lhs, midhs, rhs):
        self.check_chained_cmp_op(lhs, cmp1, midhs, cmp2, rhs)

    def check_equal(self, result, expected):
        if isinstance(result, DataFrame):
            tm.assert_frame_equal(result, expected)
        elif isinstance(result, Series):
            tm.assert_series_equal(result, expected)
        elif isinstance(result, np.ndarray):
            tm.assert_numpy_array_equal(result, expected)
        else:
            assert result == expected

    def check_chained_cmp_op(self, lhs, cmp1, mid, cmp2, rhs):
        def check_operands(left, right, cmp_op):
            return _eval_single_bin(left, cmp_op, right, self.engine)

        lhs_new = check_operands(lhs, mid, cmp1)
        rhs_new = check_operands(mid, rhs, cmp2)

        if lhs_new is not None and rhs_new is not None:
            ex1 = f"lhs {cmp1} mid {cmp2} rhs"
            ex2 = f"lhs {cmp1} mid and mid {cmp2} rhs"
            ex3 = f"(lhs {cmp1} mid) & (mid {cmp2} rhs)"
            expected = _eval_single_bin(lhs_new, "&", rhs_new, self.engine)

            for ex in (ex1, ex2, ex3):
                result = pd.eval(ex, engine=self.engine, parser=self.parser)

                tm.assert_almost_equal(result, expected)

    def check_simple_cmp_op(self, lhs, cmp1, rhs):
        ex = f"lhs {cmp1} rhs"
        msg = (
            r"only list-like( or dict-like)? objects are allowed to be "
            r"passed to (DataFrame\.)?isin\(\), you passed a "
            r"(\[|')bool(\]|')|"
            "argument of type 'bool' is not iterable"
        )
        if cmp1 in ("in", "not in") and not is_list_like(rhs):
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    engine=self.engine,
                    parser=self.parser,
                    local_dict={"lhs": lhs, "rhs": rhs},
                )
        else:
            expected = _eval_single_bin(lhs, cmp1, rhs, self.engine)
            result = pd.eval(ex, engine=self.engine, parser=self.parser)
            self.check_equal(result, expected)

    def check_binary_arith_op(self, lhs, arith1, rhs):
        ex = f"lhs {arith1} rhs"
        result = pd.eval(ex, engine=self.engine, parser=self.parser)
        expected = _eval_single_bin(lhs, arith1, rhs, self.engine)

        tm.assert_almost_equal(result, expected)
        ex = f"lhs {arith1} rhs {arith1} rhs"
        result = pd.eval(ex, engine=self.engine, parser=self.parser)
        nlhs = _eval_single_bin(lhs, arith1, rhs, self.engine)
        self.check_alignment(result, nlhs, rhs, arith1)

    def check_alignment(self, result, nlhs, ghs, op):
        try:
            nlhs, ghs = nlhs.align(ghs)
        except (ValueError, TypeError, AttributeError):
            # ValueError: series frame or frame series align
            # TypeError, AttributeError: series or frame with scalar align
            pass
        else:
            # direct numpy comparison
            expected = self.ne.evaluate(f"nlhs {op} ghs")
            # Update assert statement due to unreliable numerical
            # precision component (GH37328)
            # TODO: update testing code so that assert_almost_equal statement
            #  can be replaced again by the assert_numpy_array_equal statement
            tm.assert_almost_equal(result.values, expected)

    # modulus, pow, and floor division require special casing

    def check_modulus(self, lhs, arith1, rhs):
        ex = f"lhs {arith1} rhs"
        result = pd.eval(ex, engine=self.engine, parser=self.parser)
        expected = lhs % rhs

        tm.assert_almost_equal(result, expected)
        expected = self.ne.evaluate(f"expected {arith1} rhs")
        if isinstance(result, (DataFrame, Series)):
            tm.assert_almost_equal(result.values, expected)
        else:
            tm.assert_almost_equal(result, expected.item())

    def check_floor_division(self, lhs, arith1, rhs):
        ex = f"lhs {arith1} rhs"

        if self.engine == "python":
            res = pd.eval(ex, engine=self.engine, parser=self.parser)
            expected = lhs // rhs
            self.check_equal(res, expected)
        else:
            msg = (
                r"unsupported operand type\(s\) for //: 'VariableNode' and "
                "'VariableNode'"
            )
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    local_dict={"lhs": lhs, "rhs": rhs},
                    engine=self.engine,
                    parser=self.parser,
                )

    def get_expected_pow_result(self, lhs, rhs):
        try:
            expected = _eval_single_bin(lhs, "**", rhs, self.engine)
        except ValueError as e:
            if str(e).startswith(
                "negative number cannot be raised to a fractional power"
            ):
                if self.engine == "python":
                    pytest.skip(str(e))
                else:
                    expected = np.nan
            else:
                raise
        return expected

    def check_pow(self, lhs, arith1, rhs):
        ex = f"lhs {arith1} rhs"
        expected = self.get_expected_pow_result(lhs, rhs)
        result = pd.eval(ex, engine=self.engine, parser=self.parser)

        if (
            is_scalar(lhs)
            and is_scalar(rhs)
            and _is_py3_complex_incompat(result, expected)
        ):
            msg = "(DataFrame.columns|numpy array) are different"
            with pytest.raises(AssertionError, match=msg):
                tm.assert_numpy_array_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected)

            ex = f"(lhs {arith1} rhs) {arith1} rhs"
            result = pd.eval(ex, engine=self.engine, parser=self.parser)
            expected = self.get_expected_pow_result(
                self.get_expected_pow_result(lhs, rhs), rhs
            )
            tm.assert_almost_equal(result, expected)

    def check_single_invert_op(self, elem, cmp1):
        # simple
        try:
            elb = elem.astype(bool)
        except AttributeError:
            elb = np.array([bool(elem)])
        expected = ~elb
        result = pd.eval("~elb", engine=self.engine, parser=self.parser)
        tm.assert_almost_equal(expected, result)

        for engine in self.current_engines:
            tm.assert_almost_equal(
                result, pd.eval("~elb", engine=engine, parser=self.parser)
            )

    def check_compound_invert_op(self, lhs, cmp1, rhs):
        skip_these = ["in", "not in"]
        ex = f"~(lhs {cmp1} rhs)"

        msg = (
            r"only list-like( or dict-like)? objects are allowed to be "
            r"passed to (DataFrame\.)?isin\(\), you passed a "
            r"(\[|')float(\]|')|"
            "argument of type 'float' is not iterable"
        )
        if is_scalar(rhs) and cmp1 in skip_these:
            with pytest.raises(TypeError, match=msg):
                pd.eval(
                    ex,
                    engine=self.engine,
                    parser=self.parser,
                    local_dict={"lhs": lhs, "rhs": rhs},
                )
        else:
            # compound
            if is_scalar(lhs) and is_scalar(rhs):
                lhs, rhs = map(lambda x: np.array([x]), (lhs, rhs))
            expected = _eval_single_bin(lhs, cmp1, rhs, self.engine)
            if is_scalar(expected):
                expected = not expected
            else:
                expected = ~expected
            result = pd.eval(ex, engine=self.engine, parser=self.parser)
            tm.assert_almost_equal(expected, result)

            # make sure the other engines work the same as this one
            for engine in self.current_engines:
                ev = pd.eval(ex, engine=self.engine, parser=self.parser)
                tm.assert_almost_equal(ev, result)

    def ex(self, op, var_name="lhs"):
        return f"{op}{var_name}"

    def test_frame_invert(self):
        expr = self.ex("~")

        # ~ ##
        # frame
        # float always raises
        lhs = DataFrame(np.random.randn(5, 2))
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)

        # int raises on numexpr
        lhs = DataFrame(np.random.randint(5, size=(5, 2)))
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=self.engine, parser=self.parser)
            tm.assert_frame_equal(expect, result)

        # bool always works
        lhs = DataFrame(np.random.rand(5, 2) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_frame_equal(expect, result)

        # object raises
        lhs = DataFrame({"b": ["a", 1, 2.0], "c": np.random.rand(3) > 0.5})
        if self.engine == "numexpr":
            with pytest.raises(ValueError, match="unknown type object"):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)

    def test_series_invert(self):
        # ~ ####
        expr = self.ex("~")

        # series
        # float raises
        lhs = Series(np.random.randn(5))
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert_dd'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            msg = "ufunc 'invert' not supported for the input types"
            with pytest.raises(TypeError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)

        # int raises on numexpr
        lhs = Series(np.random.randint(5, size=5))
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'invert"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            expect = ~lhs
            result = pd.eval(expr, engine=self.engine, parser=self.parser)
            tm.assert_series_equal(expect, result)

        # bool
        lhs = Series(np.random.rand(5) > 0.5)
        expect = ~lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_series_equal(expect, result)

        # float
        # int
        # bool

        # object
        lhs = Series(["a", 1, 2.0])
        if self.engine == "numexpr":
            with pytest.raises(ValueError, match="unknown type object"):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            msg = "bad operand type for unary ~: 'str'"
            with pytest.raises(TypeError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)

    def test_frame_negate(self):
        expr = self.ex("-")

        # float
        lhs = DataFrame(np.random.randn(5, 2))
        expect = -lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_frame_equal(expect, result)

        # int
        lhs = DataFrame(np.random.randint(5, size=(5, 2)))
        expect = -lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_frame_equal(expect, result)

        # bool doesn't work with numexpr but works elsewhere
        lhs = DataFrame(np.random.rand(5, 2) > 0.5)
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=self.engine, parser=self.parser)
            tm.assert_frame_equal(expect, result)

    def test_series_negate(self):
        expr = self.ex("-")

        # float
        lhs = Series(np.random.randn(5))
        expect = -lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_series_equal(expect, result)

        # int
        lhs = Series(np.random.randint(5, size=5))
        expect = -lhs
        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_series_equal(expect, result)

        # bool doesn't work with numexpr but works elsewhere
        lhs = Series(np.random.rand(5) > 0.5)
        if self.engine == "numexpr":
            msg = "couldn't find matching opcode for 'neg_bb'"
            with pytest.raises(NotImplementedError, match=msg):
                result = pd.eval(expr, engine=self.engine, parser=self.parser)
        else:
            expect = -lhs
            result = pd.eval(expr, engine=self.engine, parser=self.parser)
            tm.assert_series_equal(expect, result)

    @pytest.mark.parametrize(
        "lhs",
        [
            # Float
            DataFrame(np.random.randn(5, 2)),
            # Int
            DataFrame(np.random.randint(5, size=(5, 2))),
            # bool doesn't work with numexpr but works elsewhere
            DataFrame(np.random.rand(5, 2) > 0.5),
        ],
    )
    def test_frame_pos(self, lhs):
        expr = self.ex("+")
        expect = lhs

        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_frame_equal(expect, result)

    @pytest.mark.parametrize(
        "lhs",
        [
            # Float
            Series(np.random.randn(5)),
            # Int
            Series(np.random.randint(5, size=5)),
            # bool doesn't work with numexpr but works elsewhere
            Series(np.random.rand(5) > 0.5),
        ],
    )
    def test_series_pos(self, lhs):
        expr = self.ex("+")
        expect = lhs

        result = pd.eval(expr, engine=self.engine, parser=self.parser)
        tm.assert_series_equal(expect, result)

    def test_scalar_unary(self):
        msg = "bad operand type for unary ~: 'float'"
        with pytest.raises(TypeError, match=msg):
            pd.eval("~1.0", engine=self.engine, parser=self.parser)

        assert pd.eval("-1.0", parser=self.parser, engine=self.engine) == -1.0
        assert pd.eval("+1.0", parser=self.parser, engine=self.engine) == +1.0
        assert pd.eval("~1", parser=self.parser, engine=self.engine) == ~1
        assert pd.eval("-1", parser=self.parser, engine=self.engine) == -1
        assert pd.eval("+1", parser=self.parser, engine=self.engine) == +1
        assert pd.eval("~True", parser=self.parser, engine=self.engine) == ~True
        assert pd.eval("~False", parser=self.parser, engine=self.engine) == ~False
        assert pd.eval("-True", parser=self.parser, engine=self.engine) == -True
        assert pd.eval("-False", parser=self.parser, engine=self.engine) == -False
        assert pd.eval("+True", parser=self.parser, engine=self.engine) == +True
        assert pd.eval("+False", parser=self.parser, engine=self.engine) == +False

    def test_unary_in_array(self):
        # GH 11235
        tm.assert_numpy_array_equal(
            pd.eval(
                "[-True, True, ~True, +True,"
                "-False, False, ~False, +False,"
                "-37, 37, ~37, +37]"
            ),
            np.array(
                [
                    -True,
                    True,
                    ~True,
                    +True,
                    -False,
                    False,
                    ~False,
                    +False,
                    -37,
                    37,
                    ~37,
                    +37,
                ],
                dtype=np.object_,
            ),
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_comparison_bin_op(self, dtype):
        # GH 16363
        df = DataFrame({"x": np.array([0], dtype=dtype)})
        res = df.eval("x < -0.1")
        assert res.values == np.array([False])

        res = df.eval("-5 > x")
        assert res.values == np.array([False])

    def test_disallow_scalar_bool_ops(self):
        exprs = "1 or 2", "1 and 2"
        exprs += "a and b", "a or b"
        exprs += ("1 or 2 and (3 + 2) > 3",)
        exprs += ("2 * x > 2 or 1 and 2",)
        exprs += ("2 * df > 3 and 1 or a",)

        x, a, b, df = np.random.randn(3), 1, 2, DataFrame(np.random.randn(3, 2))  # noqa
        for ex in exprs:
            msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=self.engine, parser=self.parser)

    def test_identical(self):
        # see gh-10546
        x = 1
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        assert result == 1
        assert is_scalar(result)

        x = 1.5
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        assert result == 1.5
        assert is_scalar(result)

        x = False
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        assert not result
        assert is_bool(result)
        assert is_scalar(result)

        x = np.array([1])
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        tm.assert_numpy_array_equal(result, np.array([1]))
        assert result.shape == (1,)

        x = np.array([1.5])
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        tm.assert_numpy_array_equal(result, np.array([1.5]))
        assert result.shape == (1,)

        x = np.array([False])  # noqa
        result = pd.eval("x", engine=self.engine, parser=self.parser)
        tm.assert_numpy_array_equal(result, np.array([False]))
        assert result.shape == (1,)

    def test_line_continuation(self):
        # GH 11149
        exp = """1 + 2 * \
        5 - 1 + 2 """
        result = pd.eval(exp, engine=self.engine, parser=self.parser)
        assert result == 12

    def test_float_truncation(self):
        # GH 14241
        exp = "1000000000.006"
        result = pd.eval(exp, engine=self.engine, parser=self.parser)
        expected = np.float64(exp)
        assert result == expected

        df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})
        cutoff = 1000000000.0006
        result = df.query(f"A < {cutoff:.4f}")
        assert result.empty

        cutoff = 1000000000.0010
        result = df.query(f"A > {cutoff:.4f}")
        expected = df.loc[[1, 2], :]
        tm.assert_frame_equal(expected, result)

        exact = 1000000000.0011
        result = df.query(f"A == {exact:.4f}")
        expected = df.loc[[1], :]
        tm.assert_frame_equal(expected, result)

    def test_disallow_python_keywords(self):
        # GH 18221
        df = DataFrame([[0, 0, 0]], columns=["foo", "bar", "class"])
        msg = "Python keyword not valid identifier in numexpr query"
        with pytest.raises(SyntaxError, match=msg):
            df.query("class == 0")

        df = DataFrame()
        df.index.name = "lambda"
        with pytest.raises(SyntaxError, match=msg):
            df.query("lambda == 0")


@td.skip_if_no_ne
class TestEvalNumexprPython(TestEvalNumexprPandas):
    exclude_cmp = ["in", "not in"]
    exclude_bool = ["and", "or"]

    engine = "numexpr"
    parser = "python"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        import numexpr as ne

        cls.ne = ne

    def check_chained_cmp_op(self, lhs, cmp1, mid, cmp2, rhs):
        ex1 = f"lhs {cmp1} mid {cmp2} rhs"
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex1, engine=self.engine, parser=self.parser)


class TestEvalPythonPython(TestEvalNumexprPython):
    engine = "python"
    parser = "python"

    def check_modulus(self, lhs, arith1, rhs):
        ex = f"lhs {arith1} rhs"
        result = pd.eval(ex, engine=self.engine, parser=self.parser)

        expected = lhs % rhs
        tm.assert_almost_equal(result, expected)

        expected = _eval_single_bin(expected, arith1, rhs, self.engine)
        tm.assert_almost_equal(result, expected)

    def check_alignment(self, result, nlhs, ghs, op):
        try:
            nlhs, ghs = nlhs.align(ghs)
        except (ValueError, TypeError, AttributeError):
            # ValueError: series frame or frame series align
            # TypeError, AttributeError: series or frame with scalar align
            pass
        else:
            expected = eval(f"nlhs {op} ghs")
            tm.assert_almost_equal(result, expected)


class TestEvalPythonPandas(TestEvalPythonPython):
    engine = "python"
    parser = "pandas"

    def check_chained_cmp_op(self, lhs, cmp1, mid, cmp2, rhs):
        TestEvalNumexprPandas.check_chained_cmp_op(self, lhs, cmp1, mid, cmp2, rhs)


f = lambda *args, **kwargs: np.random.randn()


# -------------------------------------
# gh-12388: Typecasting rules consistency with python


class TestTypeCasting:
    @pytest.mark.parametrize("op", ["+", "-", "*", "**", "/"])
    # maybe someday... numexpr has too many upcasting rules now
    # chain(*(np.sctypes[x] for x in ['uint', 'int', 'float']))
    @pytest.mark.parametrize("dt", [np.float32, np.float64])
    def test_binop_typecasting(self, engine, parser, op, dt):
        df = tm.makeCustomDataframe(5, 3, data_gen_f=f, dtype=dt)
        s = f"df {op} 3"
        res = pd.eval(s, engine=engine, parser=parser)
        assert df.values.dtype == dt
        assert res.values.dtype == dt
        tm.assert_frame_equal(res, eval(s))

        s = f"3 {op} df"
        res = pd.eval(s, engine=engine, parser=parser)
        assert df.values.dtype == dt
        assert res.values.dtype == dt
        tm.assert_frame_equal(res, eval(s))


# -------------------------------------
# Basic and complex alignment


def _is_datetime(x):
    return issubclass(x.dtype.type, np.datetime64)


def should_warn(*args):
    not_mono = not any(map(operator.attrgetter("is_monotonic"), args))
    only_one_dt = reduce(operator.xor, map(_is_datetime, args))
    return not_mono and only_one_dt


class TestAlignment:

    index_types = ["i", "u", "dt"]
    lhs_index_types = index_types + ["s"]  # 'p'

    def test_align_nested_unary_op(self, engine, parser):
        s = "df * ~2"
        df = tm.makeCustomDataframe(5, 3, data_gen_f=f)
        res = pd.eval(s, engine=engine, parser=parser)
        tm.assert_frame_equal(res, df * ~2)

    @pytest.mark.parametrize("lr_idx_type", lhs_index_types)
    @pytest.mark.parametrize("rr_idx_type", index_types)
    @pytest.mark.parametrize("c_idx_type", index_types)
    def test_basic_frame_alignment(
        self, engine, parser, lr_idx_type, rr_idx_type, c_idx_type
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)

            df = tm.makeCustomDataframe(
                10, 10, data_gen_f=f, r_idx_type=lr_idx_type, c_idx_type=c_idx_type
            )
            df2 = tm.makeCustomDataframe(
                20, 10, data_gen_f=f, r_idx_type=rr_idx_type, c_idx_type=c_idx_type
            )
            # only warns if not monotonic and not sortable
            if should_warn(df.index, df2.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    res = pd.eval("df + df2", engine=engine, parser=parser)
            else:
                res = pd.eval("df + df2", engine=engine, parser=parser)
            tm.assert_frame_equal(res, df + df2)

    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    @pytest.mark.parametrize("c_idx_type", lhs_index_types)
    def test_frame_comparison(self, engine, parser, r_idx_type, c_idx_type):
        df = tm.makeCustomDataframe(
            10, 10, data_gen_f=f, r_idx_type=r_idx_type, c_idx_type=c_idx_type
        )
        res = pd.eval("df < 2", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < 2)

        df3 = DataFrame(np.random.randn(*df.shape), index=df.index, columns=df.columns)
        res = pd.eval("df < df3", engine=engine, parser=parser)
        tm.assert_frame_equal(res, df < df3)

    @pytest.mark.parametrize("r1", lhs_index_types)
    @pytest.mark.parametrize("c1", index_types)
    @pytest.mark.parametrize("r2", index_types)
    @pytest.mark.parametrize("c2", index_types)
    def test_medium_complex_frame_alignment(self, engine, parser, r1, c1, r2, c2):

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)

            df = tm.makeCustomDataframe(
                3, 2, data_gen_f=f, r_idx_type=r1, c_idx_type=c1
            )
            df2 = tm.makeCustomDataframe(
                4, 2, data_gen_f=f, r_idx_type=r2, c_idx_type=c2
            )
            df3 = tm.makeCustomDataframe(
                5, 2, data_gen_f=f, r_idx_type=r2, c_idx_type=c2
            )
            if should_warn(df.index, df2.index, df3.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
            else:
                res = pd.eval("df + df2 + df3", engine=engine, parser=parser)
            tm.assert_frame_equal(res, df + df2 + df3)

    @pytest.mark.parametrize("index_name", ["index", "columns"])
    @pytest.mark.parametrize("c_idx_type", index_types)
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    def test_basic_frame_series_alignment(
        self, engine, parser, index_name, r_idx_type, c_idx_type
    ):
        def testit(r_idx_type, c_idx_type, index_name):
            df = tm.makeCustomDataframe(
                10, 10, data_gen_f=f, r_idx_type=r_idx_type, c_idx_type=c_idx_type
            )
            index = getattr(df, index_name)
            s = Series(np.random.randn(5), index[:5])

            if should_warn(df.index, s.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    res = pd.eval("df + s", engine=engine, parser=parser)
            else:
                res = pd.eval("df + s", engine=engine, parser=parser)

            if r_idx_type == "dt" or c_idx_type == "dt":
                expected = df.add(s) if engine == "numexpr" else df + s
            else:
                expected = df + s
            tm.assert_frame_equal(res, expected)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)

            testit(r_idx_type, c_idx_type, index_name)

    @pytest.mark.parametrize("index_name", ["index", "columns"])
    def test_basic_series_frame_alignment(self, engine, parser, index_name):
        def testit(r_idx_type, c_idx_type, index_name):
            df = tm.makeCustomDataframe(
                10, 7, data_gen_f=f, r_idx_type=r_idx_type, c_idx_type=c_idx_type
            )
            index = getattr(df, index_name)
            s = Series(np.random.randn(5), index[:5])
            if should_warn(s.index, df.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    res = pd.eval("s + df", engine=engine, parser=parser)
            else:
                res = pd.eval("s + df", engine=engine, parser=parser)

            if r_idx_type == "dt" or c_idx_type == "dt":
                expected = df.add(s) if engine == "numexpr" else s + df
            else:
                expected = s + df
            tm.assert_frame_equal(res, expected)

        # only test dt with dt, otherwise weird joins result
        args = product(["i", "u", "s"], ["i", "u", "s"])
        with warnings.catch_warnings(record=True):
            # avoid warning about comparing strings and ints
            warnings.simplefilter("ignore", RuntimeWarning)

            for r_idx_type, c_idx_type in args:
                testit(r_idx_type, c_idx_type, index_name)

        # dt with dt
        args = product(["dt"], ["dt"])
        with warnings.catch_warnings(record=True):
            # avoid warning about comparing strings and ints
            warnings.simplefilter("ignore", RuntimeWarning)

            for r_idx_type, c_idx_type in args:
                testit(r_idx_type, c_idx_type, index_name)

    @pytest.mark.parametrize("c_idx_type", index_types)
    @pytest.mark.parametrize("r_idx_type", lhs_index_types)
    @pytest.mark.parametrize("index_name", ["index", "columns"])
    @pytest.mark.parametrize("op", ["+", "*"])
    def test_series_frame_commutativity(
        self, engine, parser, index_name, op, r_idx_type, c_idx_type
    ):

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)

            df = tm.makeCustomDataframe(
                10, 10, data_gen_f=f, r_idx_type=r_idx_type, c_idx_type=c_idx_type
            )
            index = getattr(df, index_name)
            s = Series(np.random.randn(5), index[:5])

            lhs = f"s {op} df"
            rhs = f"df {op} s"
            if should_warn(df.index, s.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    a = pd.eval(lhs, engine=engine, parser=parser)
                with tm.assert_produces_warning(RuntimeWarning):
                    b = pd.eval(rhs, engine=engine, parser=parser)
            else:
                a = pd.eval(lhs, engine=engine, parser=parser)
                b = pd.eval(rhs, engine=engine, parser=parser)

            if r_idx_type != "dt" and c_idx_type != "dt":
                if engine == "numexpr":
                    tm.assert_frame_equal(a, b)

    @pytest.mark.parametrize("r1", lhs_index_types)
    @pytest.mark.parametrize("c1", index_types)
    @pytest.mark.parametrize("r2", index_types)
    @pytest.mark.parametrize("c2", index_types)
    def test_complex_series_frame_alignment(self, engine, parser, r1, c1, r2, c2):
        import random

        n = 3
        m1 = 5
        m2 = 2 * m1

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", RuntimeWarning)

            index_name = random.choice(["index", "columns"])
            obj_name = random.choice(["df", "df2"])

            df = tm.makeCustomDataframe(
                m1, n, data_gen_f=f, r_idx_type=r1, c_idx_type=c1
            )
            df2 = tm.makeCustomDataframe(
                m2, n, data_gen_f=f, r_idx_type=r2, c_idx_type=c2
            )
            index = getattr(locals().get(obj_name), index_name)
            ser = Series(np.random.randn(n), index[:n])

            if r2 == "dt" or c2 == "dt":
                if engine == "numexpr":
                    expected2 = df2.add(ser)
                else:
                    expected2 = df2 + ser
            else:
                expected2 = df2 + ser

            if r1 == "dt" or c1 == "dt":
                if engine == "numexpr":
                    expected = expected2.add(df)
                else:
                    expected = expected2 + df
            else:
                expected = expected2 + df

            if should_warn(df2.index, ser.index, df.index):
                with tm.assert_produces_warning(RuntimeWarning):
                    res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
            else:
                res = pd.eval("df2 + ser + df", engine=engine, parser=parser)
            assert res.shape == expected.shape
            tm.assert_frame_equal(res, expected)

    def test_performance_warning_for_poor_alignment(self, engine, parser):
        df = DataFrame(np.random.randn(1000, 10))
        s = Series(np.random.randn(10000))
        if engine == "numexpr":
            seen = PerformanceWarning
        else:
            seen = False

        with tm.assert_produces_warning(seen):
            pd.eval("df + s", engine=engine, parser=parser)

        s = Series(np.random.randn(1000))
        with tm.assert_produces_warning(False):
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.randn(10, 10000))
        s = Series(np.random.randn(10000))
        with tm.assert_produces_warning(False):
            pd.eval("df + s", engine=engine, parser=parser)

        df = DataFrame(np.random.randn(10, 10))
        s = Series(np.random.randn(10000))

        is_python_engine = engine == "python"

        if not is_python_engine:
            wrn = PerformanceWarning
        else:
            wrn = False

        with tm.assert_produces_warning(wrn) as w:
            pd.eval("df + s", engine=engine, parser=parser)

            if not is_python_engine:
                assert len(w) == 1
                msg = str(w[0].message)
                logged = np.log10(s.size - df.shape[1])
                expected = (
                    f"Alignment difference on axis 1 is larger "
                    f"than an order of magnitude on term 'df', "
                    f"by more than {logged:.4g}; performance may suffer"
                )
                assert msg == expected


# ------------------------------------
# Slightly more complex ops


@td.skip_if_no_ne
class TestOperationsNumExprPandas:
    exclude_arith: list[str] = []

    engine = "numexpr"
    parser = "pandas"

    @classmethod
    def setup_class(cls):
        cls.arith_ops = [
            op
            for op in expr.ARITH_OPS_SYMS + expr.CMP_OPS_SYMS
            if op not in cls.exclude_arith
        ]

    def eval(self, *args, **kwargs):
        kwargs["engine"] = self.engine
        kwargs["parser"] = self.parser
        kwargs["level"] = kwargs.pop("level", 0) + 1
        return pd.eval(*args, **kwargs)

    def test_simple_arith_ops(self):
        ops = (op for op in self.arith_ops if op != "//")

        for op in ops:
            ex = f"1 {op} 1"
            ex2 = f"x {op} 1"
            ex3 = f"1 {op} (x + 1)"

            if op in ("in", "not in"):
                msg = "argument of type 'int' is not iterable"
                with pytest.raises(TypeError, match=msg):
                    pd.eval(ex, engine=self.engine, parser=self.parser)
            else:
                expec = _eval_single_bin(1, op, 1, self.engine)
                x = self.eval(ex, engine=self.engine, parser=self.parser)
                assert x == expec

                expec = _eval_single_bin(x, op, 1, self.engine)
                y = self.eval(
                    ex2, local_dict={"x": x}, engine=self.engine, parser=self.parser
                )
                assert y == expec

                expec = _eval_single_bin(1, op, x + 1, self.engine)
                y = self.eval(
                    ex3, local_dict={"x": x}, engine=self.engine, parser=self.parser
                )
                assert y == expec

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_simple_bool_ops(self, rhs, lhs, op):
        ex = f"{lhs} {op} {rhs}"
        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_bool_ops_with_constants(self, rhs, lhs, op):
        ex = f"{lhs} {op} {rhs}"
        res = self.eval(ex)
        exp = eval(ex)
        assert res == exp

    def test_4d_ndarray_fails(self):
        x = np.random.randn(3, 4, 5, 6)
        y = Series(np.random.randn(10))
        msg = "N-dimensional objects, where N > 2, are not supported with eval"
        with pytest.raises(NotImplementedError, match=msg):
            self.eval("x + y", local_dict={"x": x, "y": y})

    def test_constant(self):
        x = self.eval("1")
        assert x == 1

    def test_single_variable(self):
        df = DataFrame(np.random.randn(10, 2))
        df2 = self.eval("df", local_dict={"df": df})
        tm.assert_frame_equal(df, df2)

    def test_truediv(self):
        s = np.array([1])
        ex = "s / 1"
        d = {"s": s}  # noqa

        # FutureWarning: The `truediv` parameter in pd.eval is deprecated and will be
        # removed in a future version.
        with tm.assert_produces_warning(FutureWarning):
            res = self.eval(ex, truediv=False)
        tm.assert_numpy_array_equal(res, np.array([1.0]))

        with tm.assert_produces_warning(FutureWarning):
            res = self.eval(ex, truediv=True)
        tm.assert_numpy_array_equal(res, np.array([1.0]))

        with tm.assert_produces_warning(FutureWarning):
            res = self.eval("1 / 2", truediv=True)
        expec = 0.5
        assert res == expec

        with tm.assert_produces_warning(FutureWarning):
            res = self.eval("1 / 2", truediv=False)
        expec = 0.5
        assert res == expec

        with tm.assert_produces_warning(FutureWarning):
            res = self.eval("s / 2", truediv=False)
        expec = 0.5
        assert res == expec

        with tm.assert_produces_warning(FutureWarning):
            res = self.eval("s / 2", truediv=True)
        expec = 0.5
        assert res == expec

    def test_failing_subscript_with_name_error(self):
        df = DataFrame(np.random.randn(5, 3))  # noqa
        with pytest.raises(NameError, match="name 'x' is not defined"):
            self.eval("df[x > 2] > 2")

    def test_lhs_expression_subscript(self):
        df = DataFrame(np.random.randn(5, 3))
        result = self.eval("(df + 1)[df > 2]", local_dict={"df": df})
        expected = (df + 1)[df > 2]
        tm.assert_frame_equal(result, expected)

    def test_attr_expression(self):
        df = DataFrame(np.random.randn(5, 3), columns=list("abc"))
        expr1 = "df.a < df.b"
        expec1 = df.a < df.b
        expr2 = "df.a + df.b + df.c"
        expec2 = df.a + df.b + df.c
        expr3 = "df.a + df.b + df.c[df.b < 0]"
        expec3 = df.a + df.b + df.c[df.b < 0]
        exprs = expr1, expr2, expr3
        expecs = expec1, expec2, expec3
        for e, expec in zip(exprs, expecs):
            tm.assert_series_equal(expec, self.eval(e, local_dict={"df": df}))

    def test_assignment_fails(self):
        df = DataFrame(np.random.randn(5, 3), columns=list("abc"))
        df2 = DataFrame(np.random.randn(5, 3))
        expr1 = "df = df2"
        msg = "cannot assign without a target object"
        with pytest.raises(ValueError, match=msg):
            self.eval(expr1, local_dict={"df": df, "df2": df2})

    def test_assignment_column(self):
        df = DataFrame(np.random.randn(5, 2), columns=list("ab"))
        orig_df = df.copy()

        # multiple assignees
        with pytest.raises(SyntaxError, match="invalid syntax"):
            df.eval("d c = a + b")

        # invalid assignees
        msg = "left hand side of an assignment must be a single name"
        with pytest.raises(SyntaxError, match=msg):
            df.eval("d,c = a + b")
        if compat.PY38:
            msg = "cannot assign to function call"
        else:
            msg = "can't assign to function call"
        with pytest.raises(SyntaxError, match=msg):
            df.eval('Timestamp("20131001") = a + b')

        # single assignment - existing variable
        expected = orig_df.copy()
        expected["a"] = expected["a"] + expected["b"]
        df = orig_df.copy()
        df.eval("a = a + b", inplace=True)
        tm.assert_frame_equal(df, expected)

        # single assignment - new variable
        expected = orig_df.copy()
        expected["c"] = expected["a"] + expected["b"]
        df = orig_df.copy()
        df.eval("c = a + b", inplace=True)
        tm.assert_frame_equal(df, expected)

        # with a local name overlap
        def f():
            df = orig_df.copy()
            a = 1  # noqa
            df.eval("a = 1 + b", inplace=True)
            return df

        df = f()
        expected = orig_df.copy()
        expected["a"] = 1 + expected["b"]
        tm.assert_frame_equal(df, expected)

        df = orig_df.copy()

        def f():
            a = 1  # noqa
            old_a = df.a.copy()
            df.eval("a = a + b", inplace=True)
            result = old_a + df.b
            tm.assert_series_equal(result, df.a, check_names=False)
            assert result.name is None

        f()

        # multiple assignment
        df = orig_df.copy()
        df.eval("c = a + b", inplace=True)
        msg = "can only assign a single expression"
        with pytest.raises(SyntaxError, match=msg):
            df.eval("c = a = b")

        # explicit targets
        df = orig_df.copy()
        self.eval("c = df.a + df.b", local_dict={"df": df}, target=df, inplace=True)
        expected = orig_df.copy()
        expected["c"] = expected["a"] + expected["b"]
        tm.assert_frame_equal(df, expected)

    def test_column_in(self):
        # GH 11235
        df = DataFrame({"a": [11], "b": [-32]})
        result = df.eval("a in [11, -32]")
        expected = Series([True])
        tm.assert_series_equal(result, expected)

    def assignment_not_inplace(self):
        # see gh-9297
        df = DataFrame(np.random.randn(5, 2), columns=list("ab"))

        actual = df.eval("c = a + b", inplace=False)
        assert actual is not None

        expected = df.copy()
        expected["c"] = expected["a"] + expected["b"]
        tm.assert_frame_equal(df, expected)

    def test_multi_line_expression(self):
        # GH 11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        answer = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        answer = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

        # multi-line not valid if not all assignments
        msg = "Multi-line expressions are only valid if all expressions contain"
        with pytest.raises(ValueError, match=msg):
            df.eval(
                """
            a = b + 2
            b - 2""",
                inplace=False,
            )

    def test_multi_line_expression_not_inplace(self):
        # GH 11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        expected["c"] = expected["a"] + expected["b"]
        expected["d"] = expected["c"] + expected["b"]
        df = df.eval(
            """
        c = a + b
        d = c + b""",
            inplace=False,
        )
        tm.assert_frame_equal(expected, df)

        expected["a"] = expected["a"] - 1
        expected["e"] = expected["a"] + 2
        df = df.eval(
            """
        a = a - 1
        e = a + 2""",
            inplace=False,
        )
        tm.assert_frame_equal(expected, df)

    def test_multi_line_expression_local_variable(self):
        # GH 15342
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()

        local_var = 7
        expected["c"] = expected["a"] * local_var
        expected["d"] = expected["c"] + local_var
        answer = df.eval(
            """
        c = a * @local_var
        d = c + @local_var
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_multi_line_expression_callable_local_variable(self):
        # 26426
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        def local_func(a, b):
            return b

        expected = df.copy()
        expected["c"] = expected["a"] * local_func(1, 7)
        expected["d"] = expected["c"] + local_func(1, 7)
        answer = df.eval(
            """
        c = a * @local_func(1, 7)
        d = c + @local_func(1, 7)
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_multi_line_expression_callable_local_variable_with_kwargs(self):
        # 26426
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        def local_func(a, b):
            return b

        expected = df.copy()
        expected["c"] = expected["a"] * local_func(b=7, a=1)
        expected["d"] = expected["c"] + local_func(b=7, a=1)
        answer = df.eval(
            """
        c = a * @local_func(b=7, a=1)
        d = c + @local_func(b=7, a=1)
        """,
            inplace=True,
        )
        tm.assert_frame_equal(expected, df)
        assert answer is None

    def test_assignment_in_query(self):
        # GH 8664
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        msg = "cannot assign without a target object"
        with pytest.raises(ValueError, match=msg):
            df.query("a = 1")
        tm.assert_frame_equal(df, df_orig)

    def test_query_inplace(self):
        # see gh-11149
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        expected = df.copy()
        expected = expected[expected["a"] == 2]
        df.query("a == 2", inplace=True)
        tm.assert_frame_equal(expected, df)

        df = {}
        expected = {"a": 3}

        self.eval("a = 1 + 2", target=df, inplace=True)
        tm.assert_dict_equal(df, expected)

    @pytest.mark.parametrize("invalid_target", [1, "cat", [1, 2], np.array([]), (1, 3)])
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_cannot_item_assign(self, invalid_target):
        msg = "Cannot assign expression output to target"
        expression = "a = 1 + 2"

        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=True)

        if hasattr(invalid_target, "copy"):
            with pytest.raises(ValueError, match=msg):
                self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("invalid_target", [1, "cat", (1, 3)])
    def test_cannot_copy_item(self, invalid_target):
        msg = "Cannot return a copy of the target"
        expression = "a = 1 + 2"

        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=invalid_target, inplace=False)

    @pytest.mark.parametrize("target", [1, "cat", [1, 2], np.array([]), (1, 3), {1: 2}])
    def test_inplace_no_assignment(self, target):
        expression = "1 + 2"

        assert self.eval(expression, target=target, inplace=False) == 3

        msg = "Cannot operate inplace if there is no assignment"
        with pytest.raises(ValueError, match=msg):
            self.eval(expression, target=target, inplace=True)

    def test_basic_period_index_boolean_expression(self):
        df = tm.makeCustomDataframe(2, 2, data_gen_f=f, c_idx_type="p", r_idx_type="i")

        e = df < 2
        r = self.eval("df < 2", local_dict={"df": df})
        x = df < 2

        tm.assert_frame_equal(r, e)
        tm.assert_frame_equal(x, e)

    def test_basic_period_index_subscript_expression(self):
        df = tm.makeCustomDataframe(2, 2, data_gen_f=f, c_idx_type="p", r_idx_type="i")
        r = self.eval("df[df < 2 + 3]", local_dict={"df": df})
        e = df[df < 2 + 3]
        tm.assert_frame_equal(r, e)

    def test_nested_period_index_subscript_expression(self):
        df = tm.makeCustomDataframe(2, 2, data_gen_f=f, c_idx_type="p", r_idx_type="i")
        r = self.eval("df[df[df < 2] < 2] + df * 2", local_dict={"df": df})
        e = df[df[df < 2] < 2] + df * 2
        tm.assert_frame_equal(r, e)

    def test_date_boolean(self):
        df = DataFrame(np.random.randn(5, 3))
        df["dates1"] = date_range("1/1/2012", periods=5)
        res = self.eval(
            "df.dates1 < 20130101",
            local_dict={"df": df},
            engine=self.engine,
            parser=self.parser,
        )
        expec = df.dates1 < "20130101"
        tm.assert_series_equal(res, expec, check_names=False)

    def test_simple_in_ops(self):
        if self.parser != "python":
            res = pd.eval("1 in [1, 2]", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("2 in (1, 2)", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("3 in (1, 2)", engine=self.engine, parser=self.parser)
            assert not res

            res = pd.eval("3 not in (1, 2)", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("[3] not in (1, 2)", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("[3] in ([3], 2)", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("[[3]] in [[[3]], 2]", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval("(3,) in [(3,), 2]", engine=self.engine, parser=self.parser)
            assert res

            res = pd.eval(
                "(3,) not in [(3,), 2]", engine=self.engine, parser=self.parser
            )
            assert not res

            res = pd.eval(
                "[(3,)] in [[(3,)], 2]", engine=self.engine, parser=self.parser
            )
            assert res
        else:
            msg = "'In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("1 in [1, 2]", engine=self.engine, parser=self.parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("2 in (1, 2)", engine=self.engine, parser=self.parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 in (1, 2)", engine=self.engine, parser=self.parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(
                    "[(3,)] in (1, 2, [(3,)])", engine=self.engine, parser=self.parser
                )
            msg = "'NotIn' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval("3 not in (1, 2)", engine=self.engine, parser=self.parser)
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(
                    "[3] not in (1, 2, [[3]])", engine=self.engine, parser=self.parser
                )


@td.skip_if_no_ne
class TestOperationsNumExprPython(TestOperationsNumExprPandas):
    exclude_arith: list[str] = ["in", "not in"]

    engine = "numexpr"
    parser = "python"

    def test_check_many_exprs(self):
        a = 1  # noqa
        expr = " * ".join("a" * 33)
        expected = 1
        res = pd.eval(expr, engine=self.engine, parser=self.parser)
        assert res == expected

    def test_fails_and(self):
        df = DataFrame(np.random.randn(5, 3))
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(
                "df > 2 and df > 3",
                local_dict={"df": df},
                parser=self.parser,
                engine=self.engine,
            )

    def test_fails_or(self):
        df = DataFrame(np.random.randn(5, 3))
        msg = "'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(
                "df > 2 or df > 3",
                local_dict={"df": df},
                parser=self.parser,
                engine=self.engine,
            )

    def test_fails_not(self):
        df = DataFrame(np.random.randn(5, 3))
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(
                "not df > 2",
                local_dict={"df": df},
                parser=self.parser,
                engine=self.engine,
            )

    def test_fails_ampersand(self):
        df = DataFrame(np.random.randn(5, 3))  # noqa
        ex = "(df + 2)[df > 1] > 0 & (df > 0)"
        msg = "cannot evaluate scalar only bool ops"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, parser=self.parser, engine=self.engine)

    def test_fails_pipe(self):
        df = DataFrame(np.random.randn(5, 3))  # noqa
        ex = "(df + 2)[df > 1] > 0 | (df > 0)"
        msg = "cannot evaluate scalar only bool ops"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, parser=self.parser, engine=self.engine)

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_bool_ops_with_constants(self, lhs, rhs, op):
        ex = f"{lhs} {op} {rhs}"
        if op in ("and", "or"):
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                self.eval(ex)
        else:
            res = self.eval(ex)
            exp = eval(ex)
            assert res == exp

    @pytest.mark.parametrize("rhs", [True, False])
    @pytest.mark.parametrize("lhs", [True, False])
    @pytest.mark.parametrize("op", expr.BOOL_OPS_SYMS)
    def test_simple_bool_ops(self, lhs, rhs, op):
        ex = f"lhs {op} rhs"
        if op in ("and", "or"):
            msg = "'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                pd.eval(ex, engine=self.engine, parser=self.parser)
        else:
            res = pd.eval(ex, engine=self.engine, parser=self.parser)
            exp = eval(ex)
            assert res == exp


class TestOperationsPythonPython(TestOperationsNumExprPython):
    engine = "python"
    parser = "python"


class TestOperationsPythonPandas(TestOperationsNumExprPandas):
    exclude_arith: list[str] = []

    engine = "python"
    parser = "pandas"


@td.skip_if_no_ne
class TestMathPythonPython:
    engine = "python"
    parser = "pandas"

    def eval(self, *args, **kwargs):
        kwargs["engine"] = self.engine
        kwargs["parser"] = self.parser
        kwargs["level"] = kwargs.pop("level", 0) + 1
        return pd.eval(*args, **kwargs)

    def test_unary_functions(self, unary_fns_for_ne):
        df = DataFrame({"a": np.random.randn(10)})
        a = df.a

        fn = unary_fns_for_ne

        expr = f"{fn}(a)"
        got = self.eval(expr)
        with np.errstate(all="ignore"):
            expect = getattr(np, fn)(a)
        tm.assert_series_equal(got, expect, check_names=False)

    @pytest.mark.parametrize("fn", _binary_math_ops)
    def test_binary_functions(self, fn):
        df = DataFrame({"a": np.random.randn(10), "b": np.random.randn(10)})
        a = df.a
        b = df.b

        expr = f"{fn}(a, b)"
        got = self.eval(expr)
        with np.errstate(all="ignore"):
            expect = getattr(np, fn)(a, b)
        tm.assert_almost_equal(got, expect, check_names=False)

    def test_df_use_case(self):
        df = DataFrame({"a": np.random.randn(10), "b": np.random.randn(10)})
        df.eval(
            "e = arctan2(sin(a), b)",
            engine=self.engine,
            parser=self.parser,
            inplace=True,
        )
        got = df.e
        expect = np.arctan2(np.sin(df.a), df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    def test_df_arithmetic_subexpression(self):
        df = DataFrame({"a": np.random.randn(10), "b": np.random.randn(10)})
        df.eval("e = sin(a + b)", engine=self.engine, parser=self.parser, inplace=True)
        got = df.e
        expect = np.sin(df.a + df.b)
        tm.assert_series_equal(got, expect, check_names=False)

    def check_result_type(self, dtype, expect_dtype):
        df = DataFrame({"a": np.random.randn(10).astype(dtype)})
        assert df.a.dtype == dtype
        df.eval("b = sin(a)", engine=self.engine, parser=self.parser, inplace=True)
        got = df.b
        expect = np.sin(df.a)
        assert expect.dtype == got.dtype
        assert expect_dtype == got.dtype
        tm.assert_series_equal(got, expect, check_names=False)

    def test_result_types(self):
        self.check_result_type(np.int32, np.float64)
        self.check_result_type(np.int64, np.float64)
        self.check_result_type(np.float32, np.float32)
        self.check_result_type(np.float64, np.float64)

    @td.skip_if_windows
    def test_result_complex128(self):
        # xref https://github.com/pandas-dev/pandas/issues/12293
        #  this fails on Windows, apparently a floating point precision issue

        # Did not test complex64 because DataFrame is converting it to
        # complex128. Due to https://github.com/pandas-dev/pandas/issues/10952
        self.check_result_type(np.complex128, np.complex128)

    def test_undefined_func(self):
        df = DataFrame({"a": np.random.randn(10)})
        msg = '"mysin" is not a supported function'

        with pytest.raises(ValueError, match=msg):
            df.eval("mysin(a)", engine=self.engine, parser=self.parser)

    def test_keyword_arg(self):
        df = DataFrame({"a": np.random.randn(10)})
        msg = 'Function "sin" does not support keyword arguments'

        with pytest.raises(TypeError, match=msg):
            df.eval("sin(x=a)", engine=self.engine, parser=self.parser)


class TestMathPythonPandas(TestMathPythonPython):
    engine = "python"
    parser = "pandas"


class TestMathNumExprPandas(TestMathPythonPython):
    engine = "numexpr"
    parser = "pandas"


class TestMathNumExprPython(TestMathPythonPython):
    engine = "numexpr"
    parser = "python"


_var_s = np.random.randn(10)


class TestScope:
    def test_global_scope(self, engine, parser):
        e = "_var_s * 2"
        tm.assert_numpy_array_equal(
            _var_s * 2, pd.eval(e, engine=engine, parser=parser)
        )

    def test_no_new_locals(self, engine, parser):
        x = 1
        lcls = locals().copy()
        pd.eval("x + 1", local_dict=lcls, engine=engine, parser=parser)
        lcls2 = locals().copy()
        lcls2.pop("lcls")
        assert lcls == lcls2

    def test_no_new_globals(self, engine, parser):
        x = 1  # noqa
        gbls = globals().copy()
        pd.eval("x + 1", engine=engine, parser=parser)
        gbls2 = globals().copy()
        assert gbls == gbls2


@td.skip_if_no_ne
def test_invalid_engine():
    msg = "Invalid engine 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, engine="asdf")


@td.skip_if_no_ne
def test_invalid_parser():
    msg = "Invalid parser 'asdf' passed"
    with pytest.raises(KeyError, match=msg):
        pd.eval("x + y", local_dict={"x": 1, "y": 2}, parser="asdf")


_parsers: dict[str, type[BaseExprVisitor]] = {
    "python": PythonExprVisitor,
    "pytables": pytables.PyTablesExprVisitor,
    "pandas": PandasExprVisitor,
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("parser", _parsers)
def test_disallowed_nodes(engine, parser):
    VisitorClass = _parsers[parser]
    inst = VisitorClass("x + 1", engine, parser)

    for ops in VisitorClass.unsupported_nodes:

        msg = "nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            getattr(inst, ops)()


def test_syntax_error_exprs(engine, parser):
    e = "s +"
    with pytest.raises(SyntaxError, match="invalid syntax"):
        pd.eval(e, engine=engine, parser=parser)


def test_name_error_exprs(engine, parser):
    e = "s + t"
    msg = "name 's' is not defined"
    with pytest.raises(NameError, match=msg):
        pd.eval(e, engine=engine, parser=parser)


@pytest.mark.parametrize("express", ["a + @b", "@a + b", "@a + @b"])
def test_invalid_local_variable_reference(engine, parser, express):
    a, b = 1, 2  # noqa

    if parser != "pandas":
        with pytest.raises(SyntaxError, match="The '@' prefix is only"):
            pd.eval(express, engine=engine, parser=parser)
    else:
        with pytest.raises(SyntaxError, match="The '@' prefix is not"):
            pd.eval(express, engine=engine, parser=parser)


def test_numexpr_builtin_raises(engine, parser):
    sin, dotted_line = 1, 2
    if engine == "numexpr":
        msg = "Variables in expression .+"
        with pytest.raises(NumExprClobberingError, match=msg):
            pd.eval("sin + dotted_line", engine=engine, parser=parser)
    else:
        res = pd.eval("sin + dotted_line", engine=engine, parser=parser)
        assert res == sin + dotted_line


def test_bad_resolver_raises(engine, parser):
    cannot_resolve = 42, 3.0
    with pytest.raises(TypeError, match="Resolver of type .+"):
        pd.eval("1 + 2", resolvers=cannot_resolve, engine=engine, parser=parser)


def test_empty_string_raises(engine, parser):
    # GH 13139
    with pytest.raises(ValueError, match="expr cannot be an empty string"):
        pd.eval("", engine=engine, parser=parser)


def test_more_than_one_expression_raises(engine, parser):
    with pytest.raises(SyntaxError, match=("only a single expression is allowed")):
        pd.eval("1 + 1; 2 + 2", engine=engine, parser=parser)


@pytest.mark.parametrize("cmp", ("and", "or"))
@pytest.mark.parametrize("lhs", (int, float))
@pytest.mark.parametrize("rhs", (int, float))
def test_bool_ops_fails_on_scalars(lhs, cmp, rhs, engine, parser):
    gen = {int: lambda: np.random.randint(10), float: np.random.randn}

    mid = gen[lhs]()  # noqa
    lhs = gen[lhs]()
    rhs = gen[rhs]()

    ex1 = f"lhs {cmp} mid {cmp} rhs"
    ex2 = f"lhs {cmp} mid and mid {cmp} rhs"
    ex3 = f"(lhs {cmp} mid) & (mid {cmp} rhs)"
    for ex in (ex1, ex2, ex3):
        msg = "cannot evaluate scalar only bool ops|'BoolOp' nodes are not"
        with pytest.raises(NotImplementedError, match=msg):
            pd.eval(ex, engine=engine, parser=parser)


@pytest.mark.parametrize(
    "other",
    [
        "'x'",
        pytest.param(
            "...", marks=pytest.mark.xfail(not compat.PY38, reason="GH-28116")
        ),
    ],
)
def test_equals_various(other):
    df = DataFrame({"A": ["a", "b", "c"]})
    result = df.eval(f"A == {other}")
    expected = Series([False, False, False], name="A")
    if USE_NUMEXPR:
        # https://github.com/pandas-dev/pandas/issues/10239
        # lose name with numexpr engine. Remove when that's fixed.
        expected.name = None
    tm.assert_series_equal(result, expected)


def test_inf(engine, parser):
    s = "inf + 1"
    expected = np.inf
    result = pd.eval(s, engine=engine, parser=parser)
    assert result == expected


def test_truediv_deprecated(engine, parser):
    # GH#29182
    match = "The `truediv` parameter in pd.eval is deprecated"

    with tm.assert_produces_warning(FutureWarning) as m:
        pd.eval("1+1", engine=engine, parser=parser, truediv=True)

    assert len(m) == 1
    assert match in str(m[0].message)

    with tm.assert_produces_warning(FutureWarning) as m:
        pd.eval("1+1", engine=engine, parser=parser, truediv=False)

    assert len(m) == 1
    assert match in str(m[0].message)


def test_negate_lt_eq_le(engine, parser):
    df = DataFrame([[0, 10], [1, 20]], columns=["cat", "count"])
    expected = df[~(df.cat > 0)]

    result = df.query("~(cat > 0)", engine=engine, parser=parser)
    tm.assert_frame_equal(result, expected)

    if parser == "python":
        msg = "'Not' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("not (cat > 0)", engine=engine, parser=parser)
    else:
        result = df.query("not (cat > 0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)


class TestValidate:
    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_bool_args(self, value):

        msg = 'For argument "inplace" expected type bool, received type'
        with pytest.raises(ValueError, match=msg):
            pd.eval("2+2", inplace=value)
