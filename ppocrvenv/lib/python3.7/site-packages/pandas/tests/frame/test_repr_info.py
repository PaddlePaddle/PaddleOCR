from datetime import (
    datetime,
    timedelta,
)
from io import StringIO
import warnings

import numpy as np
import pytest

from pandas import (
    Categorical,
    DataFrame,
    MultiIndex,
    NaT,
    PeriodIndex,
    Series,
    Timestamp,
    date_range,
    option_context,
    period_range,
)
import pandas._testing as tm

import pandas.io.formats.format as fmt


class TestDataFrameReprInfoEtc:
    def test_repr_bytes_61_lines(self, using_array_manager):
        # GH#12857
        lets = list("ACDEFGHIJKLMNOP")
        slen = 50
        nseqs = 1000
        words = [[np.random.choice(lets) for x in range(slen)] for _ in range(nseqs)]
        df = DataFrame(words).astype("U1")
        # TODO(Arraymanager) astype("U1") actually gives this dtype instead of object
        if not using_array_manager:
            assert (df.dtypes == object).all()

        # smoke tests; at one point this raised with 61 but not 60
        repr(df)
        repr(df.iloc[:60, :])
        repr(df.iloc[:61, :])

    def test_repr_unicode_level_names(self, frame_or_series):
        index = MultiIndex.from_tuples([(0, 0), (1, 1)], names=["\u0394", "i1"])

        obj = DataFrame(np.random.randn(2, 4), index=index)
        if frame_or_series is Series:
            obj = obj[0]
        repr(obj)

    def test_assign_index_sequences(self):
        # GH#2200
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).set_index(
            ["a", "b"]
        )
        index = list(df.index)
        index[0] = ("faz", "boo")
        df.index = index
        repr(df)

        # this travels an improper code path
        index[0] = ["faz", "boo"]
        df.index = index
        repr(df)

    def test_repr_with_mi_nat(self, float_string_frame):
        df = DataFrame({"X": [1, 2]}, index=[[NaT, Timestamp("20130101")], ["a", "b"]])
        result = repr(df)
        expected = "              X\nNaT        a  1\n2013-01-01 b  2"
        assert result == expected

    def test_multiindex_na_repr(self):
        # only an issue with long columns
        df3 = DataFrame(
            {
                "A" * 30: {("A", "A0006000", "nuit"): "A0006000"},
                "B" * 30: {("A", "A0006000", "nuit"): np.nan},
                "C" * 30: {("A", "A0006000", "nuit"): np.nan},
                "D" * 30: {("A", "A0006000", "nuit"): np.nan},
                "E" * 30: {("A", "A0006000", "nuit"): "A"},
                "F" * 30: {("A", "A0006000", "nuit"): np.nan},
            }
        )

        idf = df3.set_index(["A" * 30, "C" * 30])
        repr(idf)

    def test_repr_name_coincide(self):
        index = MultiIndex.from_tuples(
            [("a", 0, "foo"), ("b", 1, "bar")], names=["a", "b", "c"]
        )

        df = DataFrame({"value": [0, 1]}, index=index)

        lines = repr(df).split("\n")
        assert lines[2].startswith("a 0 foo")

    def test_repr_to_string(
        self,
        multiindex_year_month_day_dataframe_random_data,
        multiindex_dataframe_random_data,
    ):
        ymd = multiindex_year_month_day_dataframe_random_data
        frame = multiindex_dataframe_random_data

        repr(frame)
        repr(ymd)
        repr(frame.T)
        repr(ymd.T)

        buf = StringIO()
        frame.to_string(buf=buf)
        ymd.to_string(buf=buf)
        frame.T.to_string(buf=buf)
        ymd.T.to_string(buf=buf)

    def test_repr_empty(self):
        # empty
        repr(DataFrame())

        # empty with index
        frame = DataFrame(index=np.arange(1000))
        repr(frame)

    def test_repr_mixed(self, float_string_frame):
        buf = StringIO()

        # mixed
        repr(float_string_frame)
        float_string_frame.info(verbose=False, buf=buf)

    @pytest.mark.slow
    def test_repr_mixed_big(self):
        # big mixed
        biggie = DataFrame(
            {"A": np.random.randn(200), "B": tm.makeStringIndex(200)}, index=range(200)
        )
        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan

        repr(biggie)

    def test_repr(self, float_frame):
        buf = StringIO()

        # small one
        repr(float_frame)
        float_frame.info(verbose=False, buf=buf)

        # even smaller
        float_frame.reindex(columns=["A"]).info(verbose=False, buf=buf)
        float_frame.reindex(columns=["A", "B"]).info(verbose=False, buf=buf)

        # exhausting cases in DataFrame.info

        # columns but no index
        no_index = DataFrame(columns=[0, 1, 3])
        repr(no_index)

        # no columns or index
        DataFrame().info(buf=buf)

        df = DataFrame(["a\n\r\tb"], columns=["a\n\r\td"], index=["a\n\r\tf"])
        assert "\t" not in repr(df)
        assert "\r" not in repr(df)
        assert "a\n" not in repr(df)

    def test_repr_dimensions(self):
        df = DataFrame([[1, 2], [3, 4]])
        with option_context("display.show_dimensions", True):
            assert "2 rows x 2 columns" in repr(df)

        with option_context("display.show_dimensions", False):
            assert "2 rows x 2 columns" not in repr(df)

        with option_context("display.show_dimensions", "truncate"):
            assert "2 rows x 2 columns" not in repr(df)

    @pytest.mark.slow
    def test_repr_big(self):
        # big one
        biggie = DataFrame(np.zeros((200, 4)), columns=range(4), index=range(200))
        repr(biggie)

    def test_repr_unsortable(self, float_frame):
        # columns are not sortable

        warn_filters = warnings.filters
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*format")

        unsortable = DataFrame(
            {
                "foo": [1] * 50,
                datetime.today(): [1] * 50,
                "bar": ["bar"] * 50,
                datetime.today() + timedelta(1): ["bar"] * 50,
            },
            index=np.arange(50),
        )
        repr(unsortable)

        fmt.set_option("display.precision", 3, "display.column_space", 10)
        repr(float_frame)

        fmt.set_option("display.max_rows", 10, "display.max_columns", 2)
        repr(float_frame)

        fmt.set_option("display.max_rows", 1000, "display.max_columns", 1000)
        repr(float_frame)

        tm.reset_display_options()

        warnings.filters = warn_filters

    def test_repr_unicode(self):
        uval = "\u03c3\u03c3\u03c3\u03c3"

        df = DataFrame({"A": [uval, uval]})

        result = repr(df)
        ex_top = "      A"
        assert result.split("\n")[0].rstrip() == ex_top

        df = DataFrame({"A": [uval, uval]})
        result = repr(df)
        assert result.split("\n")[0].rstrip() == ex_top

    def test_unicode_string_with_unicode(self):
        df = DataFrame({"A": ["\u05d0"]})
        str(df)

    def test_repr_unicode_columns(self):
        df = DataFrame({"\u05d0": [1, 2, 3], "\u05d1": [4, 5, 6], "c": [7, 8, 9]})
        repr(df.columns)  # should not raise UnicodeDecodeError

    def test_str_to_bytes_raises(self):
        # GH 26447
        df = DataFrame({"A": ["abc"]})
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_very_wide_info_repr(self):
        df = DataFrame(np.random.randn(10, 20), columns=tm.rands_array(10, 20))
        repr(df)

    def test_repr_column_name_unicode_truncation_bug(self):
        # #1906
        df = DataFrame(
            {
                "Id": [7117434],
                "StringCol": (
                    "Is it possible to modify drop plot code"
                    "so that the output graph is displayed "
                    "in iphone simulator, Is it possible to "
                    "modify drop plot code so that the "
                    "output graph is \xe2\x80\xa8displayed "
                    "in iphone simulator.Now we are adding "
                    "the CSV file externally. I want to Call "
                    "the File through the code.."
                ),
            }
        )

        with option_context("display.max_columns", 20):
            assert "StringCol" in repr(df)

    def test_latex_repr(self):
        result = r"""\begin{tabular}{llll}
\toprule
{} &         0 &  1 &  2 \\
\midrule
0 &  $\alpha$ &  b &  c \\
1 &         1 &  2 &  3 \\
\bottomrule
\end{tabular}
"""
        with option_context("display.latex.escape", False, "display.latex.repr", True):
            df = DataFrame([[r"$\alpha$", "b", "c"], [1, 2, 3]])
            assert result == df._repr_latex_()

        # GH 12182
        assert df._repr_latex_() is None

    def test_repr_categorical_dates_periods(self):
        # normal DataFrame
        dt = date_range("2011-01-01 09:00", freq="H", periods=5, tz="US/Eastern")
        p = period_range("2011-01", freq="M", periods=5)
        df = DataFrame({"dt": dt, "p": p})
        exp = """                         dt        p
0 2011-01-01 09:00:00-05:00  2011-01
1 2011-01-01 10:00:00-05:00  2011-02
2 2011-01-01 11:00:00-05:00  2011-03
3 2011-01-01 12:00:00-05:00  2011-04
4 2011-01-01 13:00:00-05:00  2011-05"""

        assert repr(df) == exp

        df2 = DataFrame({"dt": Categorical(dt), "p": Categorical(p)})
        assert repr(df2) == exp

    @pytest.mark.parametrize("arg", [np.datetime64, np.timedelta64])
    @pytest.mark.parametrize(
        "box, expected",
        [[Series, "0    NaT\ndtype: object"], [DataFrame, "     0\n0  NaT"]],
    )
    def test_repr_np_nat_with_object(self, arg, box, expected):
        # GH 25445
        result = repr(box([arg("NaT")], dtype=object))
        assert result == expected

    def test_frame_datetime64_pre1900_repr(self):
        df = DataFrame({"year": date_range("1/1/1700", periods=50, freq="A-DEC")})
        # it works!
        repr(df)

    def test_frame_to_string_with_periodindex(self):
        index = PeriodIndex(["2011-1", "2011-2", "2011-3"], freq="M")
        frame = DataFrame(np.random.randn(3, 4), index=index)

        # it works!
        frame.to_string()

    def test_datetime64tz_slice_non_truncate(self):
        # GH 30263
        df = DataFrame({"x": date_range("2019", periods=10, tz="UTC")})
        expected = repr(df)
        df = df.iloc[:, :5]
        result = repr(df)
        assert result == expected
