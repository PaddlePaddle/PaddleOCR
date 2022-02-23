import numpy as np
import pytest

import pandas._config.config as cf

from pandas import Index


class TestIndexRendering:
    @pytest.mark.parametrize(
        "index,expected",
        [
            # ASCII
            # short
            (
                Index(["a", "bb", "ccc"]),
                """Index(['a', 'bb', 'ccc'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["a", "bb", "ccc"] * 10),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', "
                "'bb', 'ccc', 'a', 'bb', 'ccc',\n"
                "       'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object')",
            ),
            # truncated
            (
                Index(["a", "bb", "ccc"] * 100),
                "Index(['a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a',\n"
                "       ...\n"
                "       'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc', 'a', 'bb', 'ccc'],\n"
                "      dtype='object', length=300)",
            ),
            # Non-ASCII
            # short
            (
                Index(["あ", "いい", "ううう"]),
                """Index(['あ', 'いい', 'ううう'], dtype='object')""",
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう'],\n"
                    "      dtype='object')"
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', 'ううう', "
                    "'あ', 'いい', 'ううう', 'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr(self, index, expected):
        result = repr(index)
        assert result == expected

    @pytest.mark.parametrize(
        "index,expected",
        [
            # short
            (
                Index(["あ", "いい", "ううう"]),
                ("Index(['あ', 'いい', 'ううう'], dtype='object')"),
            ),
            # multiple lines
            (
                Index(["あ", "いい", "ううう"] * 10),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ', 'いい', 'ううう'],\n"
                    "      dtype='object')"
                    ""
                ),
            ),
            # truncated
            (
                Index(["あ", "いい", "ううう"] * 100),
                (
                    "Index(['あ', 'いい', 'ううう', 'あ', 'いい', "
                    "'ううう', 'あ', 'いい', 'ううう',\n"
                    "       'あ',\n"
                    "       ...\n"
                    "       'ううう', 'あ', 'いい', 'ううう', 'あ', "
                    "'いい', 'ううう', 'あ', 'いい',\n"
                    "       'ううう'],\n"
                    "      dtype='object', length=300)"
                ),
            ),
        ],
    )
    def test_string_index_repr_with_unicode_option(self, index, expected):
        # Enable Unicode option -----------------------------------------
        with cf.option_context("display.unicode.east_asian_width", True):
            result = repr(index)
            assert result == expected

    def test_repr_summary(self):
        with cf.option_context("display.max_seq_items", 10):
            result = repr(Index(np.arange(1000)))
            assert len(result) < 200
            assert "..." in result

    def test_index_repr_bool_nan(self):
        # GH32146
        arr = Index([True, False, np.nan], dtype=object)
        exp1 = arr.format()
        out1 = ["True", "False", "NaN"]
        assert out1 == exp1

        exp2 = repr(arr)
        out2 = "Index([True, False, nan], dtype='object')"
        assert out2 == exp2
