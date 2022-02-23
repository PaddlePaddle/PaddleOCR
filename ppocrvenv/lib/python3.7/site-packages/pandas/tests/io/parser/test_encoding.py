"""
Tests encoding functionality during parsing
for all of the parsers defined in parsers.py
"""

from io import BytesIO
import os
import tempfile

import numpy as np
import pytest

from pandas import (
    DataFrame,
    read_csv,
)
import pandas._testing as tm


def test_bytes_io_input(all_parsers):
    encoding = "cp1255"
    parser = all_parsers

    data = BytesIO("שלום:1234\n562:123".encode(encoding))
    result = parser.read_csv(data, sep=":", encoding=encoding)

    expected = DataFrame([[562, 123]], columns=["שלום", "1234"])
    tm.assert_frame_equal(result, expected)


def test_read_csv_unicode(all_parsers):
    parser = all_parsers
    data = BytesIO("\u0141aski, Jan;1".encode())

    result = parser.read_csv(data, sep=";", encoding="utf-8", header=None)
    expected = DataFrame([["\u0141aski, Jan", 1]])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("sep", [",", "\t"])
@pytest.mark.parametrize("encoding", ["utf-16", "utf-16le", "utf-16be"])
def test_utf16_bom_skiprows(all_parsers, sep, encoding):
    # see gh-2298
    parser = all_parsers
    data = """skip this
skip this too
A,B,C
1,2,3
4,5,6""".replace(
        ",", sep
    )
    path = f"__{tm.rands(10)}__.csv"
    kwargs = {"sep": sep, "skiprows": 2}
    utf8 = "utf-8"

    with tm.ensure_clean(path) as path:
        from io import TextIOWrapper

        bytes_data = data.encode(encoding)

        with open(path, "wb") as f:
            f.write(bytes_data)

        bytes_buffer = BytesIO(data.encode(utf8))
        bytes_buffer = TextIOWrapper(bytes_buffer, encoding=utf8)

        result = parser.read_csv(path, encoding=encoding, **kwargs)
        expected = parser.read_csv(bytes_buffer, encoding=utf8, **kwargs)

        bytes_buffer.close()
        tm.assert_frame_equal(result, expected)


def test_utf16_example(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "utf16_ex.txt")
    parser = all_parsers
    result = parser.read_csv(path, encoding="utf-16", sep="\t")
    assert len(result) == 50


def test_unicode_encoding(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, "unicode_series.csv")
    parser = all_parsers

    result = parser.read_csv(path, header=None, encoding="latin-1")
    result = result.set_index(0)
    got = result[1][1632]

    expected = "\xc1 k\xf6ldum klaka (Cold Fever) (1994)"
    assert got == expected


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # Basic test
        ("a\n1", {}, DataFrame({"a": [1]})),
        # "Regular" quoting
        ('"a"\n1', {"quotechar": '"'}, DataFrame({"a": [1]})),
        # Test in a data row instead of header
        ("b\n1", {"names": ["a"]}, DataFrame({"a": ["b", "1"]})),
        # Test in empty data row with skipping
        ("\n1", {"names": ["a"], "skip_blank_lines": True}, DataFrame({"a": [1]})),
        # Test in empty data row without skipping
        (
            "\n1",
            {"names": ["a"], "skip_blank_lines": False},
            DataFrame({"a": [np.nan, 1]}),
        ),
    ],
)
def test_utf8_bom(all_parsers, data, kwargs, expected):
    # see gh-4793
    parser = all_parsers
    bom = "\ufeff"
    utf8 = "utf-8"

    def _encode_data_with_bom(_data):
        bom_data = (bom + _data).encode(utf8)
        return BytesIO(bom_data)

    result = parser.read_csv(_encode_data_with_bom(data), encoding=utf8, **kwargs)
    tm.assert_frame_equal(result, expected)


def test_read_csv_utf_aliases(all_parsers, utf_value, encoding_fmt):
    # see gh-13549
    expected = DataFrame({"mb_num": [4.8], "multibyte": ["test"]})
    parser = all_parsers

    encoding = encoding_fmt.format(utf_value)
    data = "mb_num,multibyte\n4.8,test".encode(encoding)

    result = parser.read_csv(BytesIO(data), encoding=encoding)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_path,encoding",
    [
        (("io", "data", "csv", "test1.csv"), "utf-8"),
        (("io", "parser", "data", "unicode_series.csv"), "latin-1"),
        (("io", "parser", "data", "sauron.SHIFT_JIS.csv"), "shiftjis"),
    ],
)
def test_binary_mode_file_buffers(
    all_parsers, csv_dir_path, file_path, encoding, datapath
):
    # gh-23779: Python csv engine shouldn't error on files opened in binary.
    # gh-31575: Python csv engine shouldn't error on files opened in raw binary.
    parser = all_parsers

    fpath = datapath(*file_path)
    expected = parser.read_csv(fpath, encoding=encoding)

    with open(fpath, encoding=encoding) as fa:
        result = parser.read_csv(fa)
        assert not fa.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb") as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)

    with open(fpath, mode="rb", buffering=0) as fb:
        result = parser.read_csv(fb, encoding=encoding)
        assert not fb.closed
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("pass_encoding", [True, False])
def test_encoding_temp_file(all_parsers, utf_value, encoding_fmt, pass_encoding):
    # see gh-24130
    parser = all_parsers
    encoding = encoding_fmt.format(utf_value)

    expected = DataFrame({"foo": ["bar"]})

    with tm.ensure_clean(mode="w+", encoding=encoding, return_filelike=True) as f:
        f.write("foo\nbar")
        f.seek(0)

        result = parser.read_csv(f, encoding=encoding if pass_encoding else None)
        tm.assert_frame_equal(result, expected)


def test_encoding_named_temp_file(all_parsers):
    # see gh-31819
    parser = all_parsers
    encoding = "shift-jis"

    if parser.engine == "python":
        pytest.skip("NamedTemporaryFile does not work with Python engine")

    title = "てすと"
    data = "こむ"

    expected = DataFrame({title: [data]})

    with tempfile.NamedTemporaryFile() as f:
        f.write(f"{title}\n{data}".encode(encoding))

        f.seek(0)

        result = parser.read_csv(f, encoding=encoding)
        tm.assert_frame_equal(result, expected)
        assert not f.closed


@pytest.mark.parametrize(
    "encoding", ["utf-8", "utf-16", "utf-16-be", "utf-16-le", "utf-32"]
)
def test_parse_encoded_special_characters(encoding):
    # GH16218 Verify parsing of data with encoded special characters
    # Data contains a Unicode 'FULLWIDTH COLON' (U+FF1A) at position (0,"a")
    data = "a\tb\n：foo\t0\nbar\t1\nbaz\t2"
    encoded_data = BytesIO(data.encode(encoding))
    result = read_csv(encoded_data, delimiter="\t", encoding=encoding)

    expected = DataFrame(data=[["：foo", 0], ["bar", 1], ["baz", 2]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("encoding", ["utf-8", None, "utf-16", "cp1255", "latin-1"])
def test_encoding_memory_map(all_parsers, encoding):
    # GH40986
    parser = all_parsers
    expected = DataFrame(
        {
            "name": ["Raphael", "Donatello", "Miguel Angel", "Leonardo"],
            "mask": ["red", "purple", "orange", "blue"],
            "weapon": ["sai", "bo staff", "nunchunk", "katana"],
        }
    )
    with tm.ensure_clean() as file:
        expected.to_csv(file, index=False, encoding=encoding)
        df = parser.read_csv(file, encoding=encoding, memory_map=True)
    tm.assert_frame_equal(df, expected)


def test_chunk_splits_multibyte_char(all_parsers):
    """
    Chunk splits a multibyte character with memory_map=True

    GH 43540
    """
    parser = all_parsers
    # DEFAULT_CHUNKSIZE = 262144, defined in parsers.pyx
    df = DataFrame(data=["a" * 127] * 2048)

    # Put two-bytes utf-8 encoded character "ą" at the end of chunk
    # utf-8 encoding of "ą" is b'\xc4\x85'
    df.iloc[2047] = "a" * 127 + "ą"
    with tm.ensure_clean("bug-gh43540.csv") as fname:
        df.to_csv(fname, index=False, header=False, encoding="utf-8")
        dfr = parser.read_csv(fname, header=None, memory_map=True, engine="c")
    tm.assert_frame_equal(dfr, df)


def test_not_readable(all_parsers):
    # GH43439
    parser = all_parsers
    if parser.engine in ("python", "pyarrow"):
        pytest.skip("SpooledTemporaryFile does only work with the c-engine")
    with tempfile.SpooledTemporaryFile() as handle:
        handle.write(b"abcd")
        handle.seek(0)
        df = parser.read_csv(handle)
    expected = DataFrame([], columns=["abcd"])
    tm.assert_frame_equal(df, expected)
