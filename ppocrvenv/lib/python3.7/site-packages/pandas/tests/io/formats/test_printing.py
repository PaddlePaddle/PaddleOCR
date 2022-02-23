import numpy as np
import pytest

import pandas._config.config as cf

import pandas as pd

import pandas.io.formats.format as fmt
import pandas.io.formats.printing as printing


def test_adjoin():
    data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
    expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

    adjoined = printing.adjoin(2, *data)

    assert adjoined == expected


def test_repr_binary_type():
    import string

    letters = string.ascii_letters
    try:
        raw = bytes(letters, encoding=cf.get_option("display.encoding"))
    except TypeError:
        raw = bytes(letters)
    b = str(raw.decode("utf-8"))
    res = printing.pprint_thing(b, quote_strings=True)
    assert res == repr(b)
    res = printing.pprint_thing(b, quote_strings=False)
    assert res == b


class TestFormattBase:
    def test_adjoin(self):
        data = [["a", "b", "c"], ["dd", "ee", "ff"], ["ggg", "hhh", "iii"]]
        expected = "a  dd  ggg\nb  ee  hhh\nc  ff  iii"

        adjoined = printing.adjoin(2, *data)

        assert adjoined == expected

    def test_adjoin_unicode(self):
        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "hhh", "いいい"]]
        expected = "あ  dd  ggg\nb  ええ  hhh\nc  ff  いいい"
        adjoined = printing.adjoin(2, *data)
        assert adjoined == expected

        adj = fmt.EastAsianTextAdjustment()

        expected = """あ  dd    ggg
b   ええ  hhh
c   ff    いいい"""

        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 13
        assert adj.len(cols[1]) == 13
        assert adj.len(cols[2]) == 16

        expected = """あ       dd         ggg
b        ええ       hhh
c        ff         いいい"""

        adjoined = adj.adjoin(7, *data)
        assert adjoined == expected
        cols = adjoined.split("\n")
        assert adj.len(cols[0]) == 23
        assert adj.len(cols[1]) == 23
        assert adj.len(cols[2]) == 26

    def test_justify(self):
        adj = fmt.EastAsianTextAdjustment()

        def just(x, *args, **kwargs):
            # wrapper to test single str
            return adj.justify([x], *args, **kwargs)[0]

        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"
        assert just("abc", 5, mode="left") == "abc  "
        assert just("abc", 5, mode="center") == " abc "
        assert just("abc", 5, mode="right") == "  abc"

        assert just("パンダ", 5, mode="left") == "パンダ"
        assert just("パンダ", 5, mode="center") == "パンダ"
        assert just("パンダ", 5, mode="right") == "パンダ"

        assert just("パンダ", 10, mode="left") == "パンダ    "
        assert just("パンダ", 10, mode="center") == "  パンダ  "
        assert just("パンダ", 10, mode="right") == "    パンダ"

    def test_east_asian_len(self):
        adj = fmt.EastAsianTextAdjustment()

        assert adj.len("abc") == 3
        assert adj.len("abc") == 3

        assert adj.len("パンダ") == 6
        assert adj.len("ﾊﾟﾝﾀﾞ") == 5
        assert adj.len("パンダpanda") == 11
        assert adj.len("ﾊﾟﾝﾀﾞpanda") == 10

    def test_ambiguous_width(self):
        adj = fmt.EastAsianTextAdjustment()
        assert adj.len("¡¡ab") == 4

        with cf.option_context("display.unicode.ambiguous_as_wide", True):
            adj = fmt.EastAsianTextAdjustment()
            assert adj.len("¡¡ab") == 6

        data = [["あ", "b", "c"], ["dd", "ええ", "ff"], ["ggg", "¡¡ab", "いいい"]]
        expected = "あ  dd    ggg \nb   ええ  ¡¡ab\nc   ff    いいい"
        adjoined = adj.adjoin(2, *data)
        assert adjoined == expected


class TestTableSchemaRepr:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("IPython")

        from IPython.core.interactiveshell import InteractiveShell

        cls.display_formatter = InteractiveShell.instance().display_formatter

    def test_publishes(self):

        df = pd.DataFrame({"A": [1, 2]})
        objects = [df["A"], df, df]  # dataframe / series
        expected_keys = [
            {"text/plain", "application/vnd.dataresource+json"},
            {"text/plain", "text/html", "application/vnd.dataresource+json"},
        ]

        opt = pd.option_context("display.html.table_schema", True)
        for obj, expected in zip(objects, expected_keys):
            with opt:
                formatted = self.display_formatter.format(obj)
            assert set(formatted[0].keys()) == expected

        with_latex = pd.option_context("display.latex.repr", True)

        with opt, with_latex:
            formatted = self.display_formatter.format(obj)

        expected = {
            "text/plain",
            "text/html",
            "text/latex",
            "application/vnd.dataresource+json",
        }
        assert set(formatted[0].keys()) == expected

    def test_publishes_not_implemented(self):
        # column MultiIndex
        # GH 15996
        midx = pd.MultiIndex.from_product([["A", "B"], ["a", "b", "c"]])
        df = pd.DataFrame(np.random.randn(5, len(midx)), columns=midx)

        opt = pd.option_context("display.html.table_schema", True)

        with opt:
            formatted = self.display_formatter.format(df)

        expected = {"text/plain", "text/html"}
        assert set(formatted[0].keys()) == expected

    def test_config_on(self):
        df = pd.DataFrame({"A": [1, 2]})
        with pd.option_context("display.html.table_schema", True):
            result = df._repr_data_resource_()

        assert result is not None

    def test_config_default_off(self):
        df = pd.DataFrame({"A": [1, 2]})
        with pd.option_context("display.html.table_schema", False):
            result = df._repr_data_resource_()

        assert result is None

    def test_enable_data_resource_formatter(self):
        # GH 10491
        formatters = self.display_formatter.formatters
        mimetype = "application/vnd.dataresource+json"

        with pd.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled

        # still there, just disabled
        assert "application/vnd.dataresource+json" in formatters
        assert not formatters[mimetype].enabled

        # able to re-set
        with pd.option_context("display.html.table_schema", True):
            assert "application/vnd.dataresource+json" in formatters
            assert formatters[mimetype].enabled
            # smoke test that it works
            self.display_formatter.format(cf)
