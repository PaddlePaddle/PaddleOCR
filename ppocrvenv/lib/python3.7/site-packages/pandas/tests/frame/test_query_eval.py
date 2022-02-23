from io import StringIO
import operator

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED

PARSERS = "python", "pandas"
ENGINES = "python", pytest.param("numexpr", marks=td.skip_if_no_ne)


@pytest.fixture(params=PARSERS, ids=lambda x: x)
def parser(request):
    return request.param


@pytest.fixture(params=ENGINES, ids=lambda x: x)
def engine(request):
    return request.param


def skip_if_no_pandas_parser(parser):
    if parser != "pandas":
        pytest.skip(f"cannot evaluate with parser {repr(parser)}")


class TestCompat:
    def setup_method(self, method):
        self.df = DataFrame({"A": [1, 2, 3]})
        self.expected1 = self.df[self.df.A > 0]
        self.expected2 = self.df.A + 1

    def test_query_default(self):

        # GH 12749
        # this should always work, whether NUMEXPR_INSTALLED or not
        df = self.df
        result = df.query("A>0")
        tm.assert_frame_equal(result, self.expected1)
        result = df.eval("A+1")
        tm.assert_series_equal(result, self.expected2, check_names=False)

    def test_query_None(self):

        df = self.df
        result = df.query("A>0", engine=None)
        tm.assert_frame_equal(result, self.expected1)
        result = df.eval("A+1", engine=None)
        tm.assert_series_equal(result, self.expected2, check_names=False)

    def test_query_python(self):

        df = self.df
        result = df.query("A>0", engine="python")
        tm.assert_frame_equal(result, self.expected1)
        result = df.eval("A+1", engine="python")
        tm.assert_series_equal(result, self.expected2, check_names=False)

    def test_query_numexpr(self):

        df = self.df
        if NUMEXPR_INSTALLED:
            result = df.query("A>0", engine="numexpr")
            tm.assert_frame_equal(result, self.expected1)
            result = df.eval("A+1", engine="numexpr")
            tm.assert_series_equal(result, self.expected2, check_names=False)
        else:
            msg = (
                r"'numexpr' is not installed or an unsupported version. "
                r"Cannot use engine='numexpr' for query/eval if 'numexpr' is "
                r"not installed"
            )
            with pytest.raises(ImportError, match=msg):
                df.query("A>0", engine="numexpr")
            with pytest.raises(ImportError, match=msg):
                df.eval("A+1", engine="numexpr")


class TestDataFrameEval:

    # smaller hits python, larger hits numexpr
    @pytest.mark.parametrize("n", [4, 4000])
    @pytest.mark.parametrize(
        "op_str,op,rop",
        [
            ("+", "__add__", "__radd__"),
            ("-", "__sub__", "__rsub__"),
            ("*", "__mul__", "__rmul__"),
            ("/", "__truediv__", "__rtruediv__"),
        ],
    )
    def test_ops(self, op_str, op, rop, n):

        # tst ops and reversed ops in evaluation
        # GH7198

        df = DataFrame(1, index=range(n), columns=list("abcd"))
        df.iloc[0] = 2
        m = df.mean()

        base = DataFrame(  # noqa
            np.tile(m.values, n).reshape(n, -1), columns=list("abcd")
        )

        expected = eval(f"base {op_str} df")

        # ops as strings
        result = eval(f"m {op_str} df")
        tm.assert_frame_equal(result, expected)

        # these are commutative
        if op in ["+", "*"]:
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)

        # these are not
        elif op in ["-", "/"]:
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self):
        # GH7192: Note we need a large number of rows to ensure this
        #  goes through the numexpr path
        df = DataFrame({"A": np.random.randn(25000)})
        df.iloc[0:5] = np.nan
        expected = 1 - np.isnan(df.iloc[0:25])
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self):
        # GH 11485
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "b"]})

        msg = "expr must be a string to be evaluated"
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == "b")

        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self):
        # GH 13139
        df = DataFrame({"A": [1, 2, 3]})

        msg = "expr cannot be an empty string"
        with pytest.raises(ValueError, match=msg):
            df.query("")

    def test_eval_resolvers_as_list(self):
        # GH 14095
        df = DataFrame(np.random.randn(10, 2), columns=list("ab"))
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        assert df.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]
        assert pd.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]

    def test_eval_object_dtype_binop(self):
        # GH#24883
        df = DataFrame({"a1": ["Y", "N"]})
        res = df.eval("c = ((a1 == 'Y') & True)")
        expected = DataFrame({"a1": ["Y", "N"], "c": [True, False]})
        tm.assert_frame_equal(res, expected)


class TestDataFrameQueryWithMultiIndex:
    def test_query_with_named_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.choice(["red", "green"], size=10)
        b = np.random.choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b], names=["color", "food"])
        df = DataFrame(np.random.randn(10, 2), index=index)
        ind = Series(
            df.index.get_level_values("color").values, index=index, name="color"
        )

        # equality
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_unnamed_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.choice(["red", "green"], size=10)
        b = np.random.choice(["eggs", "ham"], size=10)
        index = MultiIndex.from_arrays([a, b])
        df = DataFrame(np.random.randn(10, 2), index=index)
        ind = Series(df.index.get_level_values(0).values, index=index)

        res1 = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == ilevel_0', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != ilevel_0', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_0 == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_0 != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["red"] in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" in ilevel_0', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in ilevel_0', parser=parser, engine=engine)
        res2 = df.query('"red" not in ilevel_0', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # ## LEVEL 1
        ind = Series(df.index.get_level_values(1).values, index=index)
        res1 = df.query('ilevel_1 == "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" == ilevel_1', parser=parser, engine=engine)
        exp = df[ind == "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # inequality
        res1 = df.query('ilevel_1 != "eggs"', parser=parser, engine=engine)
        res2 = df.query('"eggs" != ilevel_1', parser=parser, engine=engine)
        exp = df[ind != "eggs"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # list equality (really just set membership)
        res1 = df.query('ilevel_1 == ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] == ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('ilevel_1 != ["eggs"]', parser=parser, engine=engine)
        res2 = df.query('["eggs"] != ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # in/not in ops
        res1 = df.query('["eggs"] in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" in ilevel_1', parser=parser, engine=engine)
        exp = df[ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["eggs"] not in ilevel_1', parser=parser, engine=engine)
        res2 = df.query('"eggs" not in ilevel_1', parser=parser, engine=engine)
        exp = df[~ind.isin(["eggs"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

    def test_query_with_partially_named_multiindex(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        a = np.random.choice(["red", "green"], size=10)
        b = np.arange(10)
        index = MultiIndex.from_arrays([a, b])
        index.names = [None, "rating"]
        df = DataFrame(np.random.randn(10, 2), index=index)
        res = df.query("rating == 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind == 1]
        tm.assert_frame_equal(res, exp)

        res = df.query("rating != 1", parser=parser, engine=engine)
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        exp = df[ind != 1]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res, exp)

        res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        ind = Series(df.index.get_level_values(0).values, index=index)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res, exp)

    def test_query_multiindex_get_index_resolvers(self):
        df = tm.makeCustomDataframe(
            10, 3, r_idx_nlevels=2, r_idx_names=["spam", "eggs"]
        )
        resolvers = df._get_index_resolvers()

        def to_series(mi, level):
            level_values = mi.get_level_values(level)
            s = level_values.to_series()
            s.index = mi
            return s

        col_series = df.columns.to_series()
        expected = {
            "index": df.index,
            "columns": col_series,
            "spam": to_series(df.index, "spam"),
            "eggs": to_series(df.index, "eggs"),
            "C0": col_series,
        }
        for k, v in resolvers.items():
            if isinstance(v, Index):
                assert v.is_(expected[k])
            elif isinstance(v, Series):
                tm.assert_series_equal(v, expected[k])
            else:
                raise AssertionError("object must be a Series or Index")


@td.skip_if_no_ne
class TestDataFrameQueryNumExprPandas:
    @classmethod
    def setup_class(cls):
        cls.engine = "numexpr"
        cls.parser = "pandas"

    @classmethod
    def teardown_class(cls):
        del cls.engine, cls.parser

    def test_date_query_with_attribute_access(self):
        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)
        df = DataFrame(np.random.randn(5, 3))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "@df.dates1 < 20130101 < @df.dates3", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_no_attribute_access(self):
        engine, parser = self.engine, self.parser
        df = DataFrame(np.random.randn(5, 3))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.rand(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.rand(n) > 0.5, "dates3"] = pd.NaT
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self):
        engine, parser = self.engine, self.parser
        n = 10
        d = {}
        d["dates1"] = date_range("1/1/2012", periods=n)
        d["dates3"] = date_range("1/1/2014", periods=n)
        df = DataFrame(d)
        df.loc[np.random.rand(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        expec = df[(df.index.to_series() < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_non_date(self):
        engine, parser = self.engine, self.parser

        n = 10
        df = DataFrame(
            {"dates": date_range("1/1/2012", periods=n), "nondate": np.arange(n)}
        )

        result = df.query("dates == nondate", parser=parser, engine=engine)
        assert len(result) == 0

        result = df.query("dates != nondate", parser=parser, engine=engine)
        tm.assert_frame_equal(result, df)

        msg = r"Invalid comparison between dtype=datetime64\[ns\] and ndarray"
        for op in ["<", ">", "<=", ">="]:
            with pytest.raises(TypeError, match=msg):
                df.query(f"dates {op} nondate", parser=parser, engine=engine)

    def test_query_syntax_error(self):
        engine, parser = self.engine, self.parser
        df = DataFrame({"i": range(10), "+": range(3, 13), "r": range(4, 14)})
        msg = "invalid syntax"
        with pytest.raises(SyntaxError, match=msg):
            df.query("i - +", engine=engine, parser=parser)

    def test_query_scope(self):
        from pandas.core.computation.ops import UndefinedVariableError

        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.randn(20, 2), columns=list("ab"))

        a, b = 1, 2  # noqa
        res = df.query("a > b", engine=engine, parser=parser)
        expected = df[df.a > df.b]
        tm.assert_frame_equal(res, expected)

        res = df.query("@a > b", engine=engine, parser=parser)
        expected = df[a > df.b]
        tm.assert_frame_equal(res, expected)

        # no local variable c
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("@a > b > @c", engine=engine, parser=parser)

        # no column named 'c'
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query("@a > b > c", engine=engine, parser=parser)

    def test_query_doesnt_pickup_local(self):
        from pandas.core.computation.ops import UndefinedVariableError

        engine, parser = self.engine, self.parser
        n = m = 10
        df = DataFrame(np.random.randint(m, size=(n, 3)), columns=list("abc"))

        # we don't pick up the local 'sin'
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query_builtin(self):
        from pandas.core.computation.engines import NumExprClobberingError

        engine, parser = self.engine, self.parser

        n = m = 10
        df = DataFrame(np.random.randint(m, size=(n, 3)), columns=list("abc"))

        df.index.name = "sin"
        msg = "Variables in expression.+"
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query("sin > 5", engine=engine, parser=parser)

    def test_query(self):
        engine, parser = self.engine, self.parser
        df = DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])

        tm.assert_frame_equal(
            df.query("a < b", engine=engine, parser=parser), df[df.a < df.b]
        )
        tm.assert_frame_equal(
            df.query("a + b > b * c", engine=engine, parser=parser),
            df[df.a + df.b > df.b * df.c],
        )

    def test_query_index_with_name(self):
        engine, parser = self.engine, self.parser
        df = DataFrame(
            np.random.randint(10, size=(10, 3)),
            index=Index(range(10), name="blob"),
            columns=["a", "b", "c"],
        )
        res = df.query("(blob < 5) & (a < b)", engine=engine, parser=parser)
        expec = df[(df.index < 5) & (df.a < df.b)]
        tm.assert_frame_equal(res, expec)

        res = df.query("blob < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]

        tm.assert_frame_equal(res, expec)

    def test_query_index_without_name(self):
        engine, parser = self.engine, self.parser
        df = DataFrame(
            np.random.randint(10, size=(10, 3)),
            index=range(10),
            columns=["a", "b", "c"],
        )

        # "index" should refer to the index
        res = df.query("index < b", engine=engine, parser=parser)
        expec = df[df.index < df.b]
        tm.assert_frame_equal(res, expec)

        # test against a scalar
        res = df.query("index < 5", engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    def test_nested_scope(self):
        engine = self.engine
        parser = self.parser

        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.randn(5, 3))
        df2 = DataFrame(np.random.randn(5, 3))
        expected = df[(df > 0) & (df2 > 0)]

        result = df.query("(@df > 0) & (@df2 > 0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[df > 0 and df2 > 0]", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

        result = pd.eval(
            "df[df > 0 and df2 > 0 and df[df > 0] > 0]", engine=engine, parser=parser
        )
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        tm.assert_frame_equal(result, expected)

        result = pd.eval("df[(df>0) & (df2>0)]", engine=engine, parser=parser)
        expected = df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expected)

    def test_nested_raises_on_local_self_reference(self):
        from pandas.core.computation.ops import UndefinedVariableError

        df = DataFrame(np.random.randn(5, 3))

        # can't reference ourself b/c we're a local so @ is necessary
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("df > 0", engine=self.engine, parser=self.parser)

    def test_local_syntax(self):
        skip_if_no_pandas_parser(self.parser)

        engine, parser = self.engine, self.parser
        df = DataFrame(np.random.randn(100, 10), columns=list("abcdefghij"))
        b = 1
        expect = df[df.a < b]
        result = df.query("a < @b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

        expect = df[df.a < df.b]
        result = df.query("a < b", engine=engine, parser=parser)
        tm.assert_frame_equal(result, expect)

    def test_chained_cmp_and_in(self):
        skip_if_no_pandas_parser(self.parser)
        engine, parser = self.engine, self.parser
        cols = list("abc")
        df = DataFrame(np.random.randn(100, len(cols)), columns=cols)
        res = df.query(
            "a < b < c and a not in b not in c", engine=engine, parser=parser
        )
        ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec = df[ind]
        tm.assert_frame_equal(res, expec)

    def test_local_variable_with_in(self):
        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)
        a = Series(np.random.randint(3, size=15), name="a")
        b = Series(np.random.randint(10, size=15), name="b")
        df = DataFrame({"a": a, "b": b})

        expected = df.loc[(df.b - 1).isin(a)]
        result = df.query("b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        b = Series(np.random.randint(10, size=15), name="b")
        expected = df.loc[(b - 1).isin(a)]
        result = df.query("@b - 1 in a", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

    def test_at_inside_string(self):
        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)
        c = 1  # noqa
        df = DataFrame({"a": ["a", "a", "b", "b", "@c", "@c"]})
        result = df.query('a == "@c"', engine=engine, parser=parser)
        expected = df[df.a == "@c"]
        tm.assert_frame_equal(result, expected)

    def test_query_undefined_local(self):
        from pandas.core.computation.ops import UndefinedVariableError

        engine, parser = self.engine, self.parser
        skip_if_no_pandas_parser(parser)

        df = DataFrame(np.random.rand(10, 2), columns=list("ab"))
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("a == @c", engine=engine, parser=parser)

    def test_index_resolvers_come_after_columns_with_the_same_name(self):
        n = 1  # noqa
        a = np.r_[20:101:20]

        df = DataFrame({"index": a, "b": np.random.randn(a.size)})
        df.index.name = "index"
        result = df.query("index > 5", engine=self.engine, parser=self.parser)
        expected = df[df["index"] > 5]
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"index": a, "b": np.random.randn(a.size)})
        result = df.query("ilevel_0 > 5", engine=self.engine, parser=self.parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

        df = DataFrame({"a": a, "b": np.random.randn(a.size)})
        df.index.name = "a"
        result = df.query("a > 5", engine=self.engine, parser=self.parser)
        expected = df[df.a > 5]
        tm.assert_frame_equal(result, expected)

        result = df.query("index > 5", engine=self.engine, parser=self.parser)
        expected = df.loc[df.index[df.index > 5]]
        tm.assert_frame_equal(result, expected)

    def test_inf(self):
        n = 10
        df = DataFrame({"a": np.random.rand(n), "b": np.random.rand(n)})
        df.loc[::2, 0] = np.inf
        d = {"==": operator.eq, "!=": operator.ne}
        for op, f in d.items():
            q = f"a {op} inf"
            expected = df[f(df.a, np.inf)]
            result = df.query(q, engine=self.engine, parser=self.parser)
            tm.assert_frame_equal(result, expected)

    def test_check_tz_aware_index_query(self, tz_aware_fixture):
        # https://github.com/pandas-dev/pandas/issues/29463
        tz = tz_aware_fixture
        df_index = date_range(
            start="2019-01-01", freq="1d", periods=10, tz=tz, name="time"
        )
        expected = DataFrame(index=df_index)
        df = DataFrame(index=df_index)
        result = df.query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(df_index)
        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        tm.assert_frame_equal(result, expected)


@td.skip_if_no_ne
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.engine = "numexpr"
        cls.parser = "python"

    def test_date_query_no_attribute_access(self):
        engine, parser = self.engine, self.parser
        df = DataFrame(np.random.randn(5, 3))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_query_with_NaT(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.rand(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.rand(n) > 0.5, "dates3"] = pd.NaT
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        tm.assert_frame_equal(res, expec)

    def test_date_index_query_with_NaT_duplicates(self):
        engine, parser = self.engine, self.parser
        n = 10
        df = DataFrame(np.random.randn(n, 3))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.rand(n) > 0.5, "dates1"] = pd.NaT
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        msg = r"'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            df.query("index < 20130101 < dates3", engine=engine, parser=parser)

    def test_nested_scope(self):
        from pandas.core.computation.ops import UndefinedVariableError

        engine = self.engine
        parser = self.parser
        # smoke test
        x = 1  # noqa
        result = pd.eval("x + 1", engine=engine, parser=parser)
        assert result == 2

        df = DataFrame(np.random.randn(5, 3))
        df2 = DataFrame(np.random.randn(5, 3))

        # don't have the pandas parser
        msg = r"The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)

        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("(df>0) & (df2>0)", engine=engine, parser=parser)

        expected = df[(df > 0) & (df2 > 0)]
        result = pd.eval("df[(df > 0) & (df2 > 0)]", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)

        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        result = pd.eval(
            "df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]", engine=engine, parser=parser
        )
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryPythonPandas(TestDataFrameQueryNumExprPandas):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.engine = "python"
        cls.parser = "pandas"

    def test_query_builtin(self):
        engine, parser = self.engine, self.parser

        n = m = 10
        df = DataFrame(np.random.randint(m, size=(n, 3)), columns=list("abc"))

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryPythonPython(TestDataFrameQueryNumExprPython):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.engine = cls.parser = "python"

    def test_query_builtin(self):
        engine, parser = self.engine, self.parser

        n = m = 10
        df = DataFrame(np.random.randint(m, size=(n, 3)), columns=list("abc"))

        df.index.name = "sin"
        expected = df[df.index > 5]
        result = df.query("sin > 5", engine=engine, parser=parser)
        tm.assert_frame_equal(expected, result)


class TestDataFrameQueryStrings:
    def test_str_query_method(self, parser, engine):
        df = DataFrame(np.random.randn(10, 1), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings == "a"]

        if parser != "pandas":
            col = "strings"
            lst = '"a"'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lhs, op, rhs in zip(lhs, ops, rhs):
                ex = f"{lhs} {op} {rhs}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(
                        ex,
                        engine=engine,
                        parser=parser,
                        local_dict={"strings": df.strings},
                    )
        else:
            res = df.query('"a" == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('strings == "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[df.strings.isin(["a"])])

            expect = df[df.strings != "a"]
            res = df.query('strings != "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('"a" != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)
            tm.assert_frame_equal(res, df[~df.strings.isin(["a"])])

    def test_str_list_query_method(self, parser, engine):
        df = DataFrame(np.random.randn(10, 1), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))
        expect = df[df.strings.isin(["a", "b"])]

        if parser != "pandas":
            col = "strings"
            lst = '["a", "b"]'

            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            for lhs, op, rhs in zip(lhs, ops, rhs):
                ex = f"{lhs} {op} {rhs}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            res = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            expect = df[~df.strings.isin(["a", "b"])]

            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    def test_query_with_string_columns(self, parser, engine):
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.randint(5, size=12),
                "d": np.random.randint(9, size=12),
            }
        )
        if parser == "pandas":
            res = df.query("a in b", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)

            res = df.query("a in b and c < d", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            msg = r"'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b", parser=parser, engine=engine)

            msg = r"'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b and c < d", parser=parser, engine=engine)

    def test_object_array_eq_ne(self, parser, engine):
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.randint(5, size=12),
                "d": np.random.randint(9, size=12),
            }
        )
        res = df.query("a == b", parser=parser, engine=engine)
        exp = df[df.a == df.b]
        tm.assert_frame_equal(res, exp)

        res = df.query("a != b", parser=parser, engine=engine)
        exp = df[df.a != df.b]
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        raw = """id          event          timestamp
        1   "page 1 load"   1/1/2014 0:00:01
        1   "page 1 exit"   1/1/2014 0:00:31
        2   "page 2 load"   1/1/2014 0:01:01
        2   "page 2 exit"   1/1/2014 0:01:31
        3   "page 3 load"   1/1/2014 0:02:01
        3   "page 3 exit"   1/1/2014 0:02:31
        4   "page 1 load"   2/1/2014 1:00:01
        4   "page 1 exit"   2/1/2014 1:00:31
        5   "page 2 load"   2/1/2014 1:01:01
        5   "page 2 exit"   2/1/2014 1:01:31
        6   "page 3 load"   2/1/2014 1:02:01
        6   "page 3 exit"   2/1/2014 1:02:31
        """
        df = pd.read_csv(
            StringIO(raw), sep=r"\s{2,}", engine="python", parse_dates=["timestamp"]
        )
        expected = df[df.event == '"page 1 load"']
        res = df.query("""'"page 1 load"' in event""", parser=parser, engine=engine)
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame({"a": ["a", "b", "test & test"], "b": [1, 2, 3]})
        res = df.query('a == "test & test"', parser=parser, engine=engine)
        expec = df[df.a == "test & test"]
        tm.assert_frame_equal(res, expec)

    def test_query_lex_compare_strings(self, parser, engine):

        a = Series(np.random.choice(list("abcde"), 20))
        b = Series(np.arange(a.size))
        df = DataFrame({"X": a, "Y": b})

        ops = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}

        for op, func in ops.items():
            res = df.query(f'X {op} "d"', engine=engine, parser=parser)
            expected = df[func(df.X, "d")]
            tm.assert_frame_equal(res, expected)

    def test_query_single_element_booleans(self, parser, engine):
        columns = "bid", "bidsize", "ask", "asksize"
        data = np.random.randint(2, size=(1, len(columns))).astype(bool)
        df = DataFrame(data, columns=columns)
        res = df.query("bid & ask", engine=engine, parser=parser)
        expected = df[df.bid & df.ask]
        tm.assert_frame_equal(res, expected)

    def test_query_string_scalar_variable(self, parser, engine):
        skip_if_no_pandas_parser(parser)
        df = DataFrame(
            {
                "Symbol": ["BUD US", "BUD US", "IBM US", "IBM US"],
                "Price": [109.70, 109.72, 183.30, 183.35],
            }
        )
        e = df[df.Symbol == "BUD US"]
        symb = "BUD US"  # noqa
        r = df.query("Symbol == @symb", parser=parser, engine=engine)
        tm.assert_frame_equal(e, r)


class TestDataFrameEvalWithFrame:
    def setup_method(self, method):
        self.frame = DataFrame(np.random.randn(10, 3), columns=list("abc"))

    def teardown_method(self, method):
        del self.frame

    def test_simple_expr(self, parser, engine):
        res = self.frame.eval("a + b", engine=engine, parser=parser)
        expect = self.frame.a + self.frame.b
        tm.assert_series_equal(res, expect)

    def test_bool_arith_expr(self, parser, engine):
        res = self.frame.eval("a[a < 1] + b", engine=engine, parser=parser)
        expect = self.frame.a[self.frame.a < 1] + self.frame.b
        tm.assert_series_equal(res, expect)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_invalid_type_for_operator_raises(self, parser, engine, op):
        df = DataFrame({"a": [1, 2], "b": ["c", "d"]})
        msg = r"unsupported operand type\(s\) for .+: '.+' and '.+'"

        with pytest.raises(TypeError, match=msg):
            df.eval(f"a {op} b", engine=engine, parser=parser)


class TestDataFrameQueryBacktickQuoting:
    @pytest.fixture(scope="class")
    def df(self):
        """
        Yields a dataframe with strings that may or may not need escaping
        by backticks. The last two columns cannot be escaped by backticks
        and should raise a ValueError.
        """
        yield DataFrame(
            {
                "A": [1, 2, 3],
                "B B": [3, 2, 1],
                "C C": [4, 5, 6],
                "C  C": [7, 4, 3],
                "C_C": [8, 9, 10],
                "D_D D": [11, 1, 101],
                "E.E": [6, 3, 5],
                "F-F": [8, 1, 10],
                "1e1": [2, 4, 8],
                "def": [10, 11, 2],
                "A (x)": [4, 1, 3],
                "B(x)": [1, 1, 5],
                "B (x)": [2, 7, 4],
                "  &^ :!€$?(} >    <++*''  ": [2, 5, 6],
                "": [10, 11, 1],
                " A": [4, 7, 9],
                "  ": [1, 2, 1],
                "it's": [6, 3, 1],
                "that's": [9, 1, 8],
                "☺": [8, 7, 6],
                "foo#bar": [2, 4, 5],
                1: [5, 7, 9],
            }
        )

    def test_single_backtick_variable_query(self, df):
        res = df.query("1 < `B B`")
        expect = df[1 < df["B B"]]
        tm.assert_frame_equal(res, expect)

    def test_two_backtick_variables_query(self, df):
        res = df.query("1 < `B B` and 4 < `C C`")
        expect = df[(1 < df["B B"]) & (4 < df["C C"])]
        tm.assert_frame_equal(res, expect)

    def test_single_backtick_variable_expr(self, df):
        res = df.eval("A + `B B`")
        expect = df["A"] + df["B B"]
        tm.assert_series_equal(res, expect)

    def test_two_backtick_variables_expr(self, df):
        res = df.eval("`B B` + `C C`")
        expect = df["B B"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_already_underscore_variable(self, df):
        res = df.eval("`C_C` + A")
        expect = df["C_C"] + df["A"]
        tm.assert_series_equal(res, expect)

    def test_same_name_but_underscores(self, df):
        res = df.eval("C_C + `C C`")
        expect = df["C_C"] + df["C C"]
        tm.assert_series_equal(res, expect)

    def test_mixed_underscores_and_spaces(self, df):
        res = df.eval("A + `D_D D`")
        expect = df["A"] + df["D_D D"]
        tm.assert_series_equal(res, expect)

    def test_backtick_quote_name_with_no_spaces(self, df):
        res = df.eval("A + `C_C`")
        expect = df["A"] + df["C_C"]
        tm.assert_series_equal(res, expect)

    def test_special_characters(self, df):
        res = df.eval("`E.E` + `F-F` - A")
        expect = df["E.E"] + df["F-F"] - df["A"]
        tm.assert_series_equal(res, expect)

    def test_start_with_digit(self, df):
        res = df.eval("A + `1e1`")
        expect = df["A"] + df["1e1"]
        tm.assert_series_equal(res, expect)

    def test_keyword(self, df):
        res = df.eval("A + `def`")
        expect = df["A"] + df["def"]
        tm.assert_series_equal(res, expect)

    def test_unneeded_quoting(self, df):
        res = df.query("`A` > 2")
        expect = df[df["A"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_parenthesis(self, df):
        res = df.query("`A (x)` > 2")
        expect = df[df["A (x)"] > 2]
        tm.assert_frame_equal(res, expect)

    def test_empty_string(self, df):
        res = df.query("`` > 5")
        expect = df[df[""] > 5]
        tm.assert_frame_equal(res, expect)

    def test_multiple_spaces(self, df):
        res = df.query("`C  C` > 5")
        expect = df[df["C  C"] > 5]
        tm.assert_frame_equal(res, expect)

    def test_start_with_spaces(self, df):
        res = df.eval("` A` + `  `")
        expect = df[" A"] + df["  "]
        tm.assert_series_equal(res, expect)

    def test_lots_of_operators_string(self, df):
        res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
        expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        tm.assert_frame_equal(res, expect)

    def test_missing_attribute(self, df):
        message = "module 'pandas' has no attribute 'thing'"
        with pytest.raises(AttributeError, match=message):
            df.eval("@pd.thing")

    def test_failing_quote(self, df):
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`it's` > `that's`")

    def test_failing_character_outside_range(self, df):
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`☺` > 4")

    def test_failing_hashtag(self, df):
        msg = "Failed to parse backticks"
        with pytest.raises(SyntaxError, match=msg):
            df.query("`foo#bar` > 4")

    def test_call_non_named_expression(self, df):
        """
        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        """

        def func(*_):
            return 1

        funcs = [func]  # noqa

        df.eval("@func()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0]()")

        with pytest.raises(TypeError, match="Only named functions are supported"):
            df.eval("@funcs[0].__call__()")
