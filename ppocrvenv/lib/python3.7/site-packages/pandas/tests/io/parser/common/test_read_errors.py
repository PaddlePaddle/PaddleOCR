"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
import codecs
import csv
from io import StringIO
import os
from pathlib import Path
import warnings

import numpy as np
import pytest

from pandas.errors import (
    EmptyDataError,
    ParserError,
)
import pandas.util._test_decorators as td

from pandas import DataFrame
import pandas._testing as tm


def test_empty_decimal_marker(all_parsers):
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    # Parsers support only length-1 decimals
    msg = "Only length-1 decimal markers supported"
    parser = all_parsers

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), decimal="")


def test_bad_stream_exception(all_parsers, csv_dir_path):
    # see gh-13652
    #
    # This test validates that both the Python engine and C engine will
    # raise UnicodeDecodeError instead of C engine raising ParserError
    # and swallowing the exception that caused read to fail.
    path = os.path.join(csv_dir_path, "sauron.SHIFT_JIS.csv")
    codec = codecs.lookup("utf-8")
    utf8 = codecs.lookup("utf-8")
    parser = all_parsers
    msg = "'utf-8' codec can't decode byte"

    # Stream must be binary UTF8.
    with open(path, "rb") as handle, codecs.StreamRecoder(
        handle, utf8.encode, utf8.decode, codec.streamreader, codec.streamwriter
    ) as stream:

        with pytest.raises(UnicodeDecodeError, match=msg):
            parser.read_csv(stream)


def test_malformed(all_parsers):
    # see gh-6607
    parser = all_parsers
    data = """ignore
A,B,C
1,2,3 # comment
1,2,3,4,5
2,3,4
"""
    msg = "Expected 3 fields in line 4, saw 5"
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), header=1, comment="#")


@pytest.mark.parametrize("nrows", [5, 3, None])
def test_malformed_chunks(all_parsers, nrows):
    data = """ignore
A,B,C
skip
1,2,3
3,5,10 # comment
1,2,3,4,5
2,3,4
"""
    parser = all_parsers
    msg = "Expected 3 fields in line 6, saw 5"
    with parser.read_csv(
        StringIO(data), header=1, comment="#", iterator=True, chunksize=1, skiprows=[2]
    ) as reader:
        with pytest.raises(ParserError, match=msg):
            reader.read(nrows)


def test_catch_too_many_names(all_parsers):
    # see gh-5156
    data = """\
1,2,3
4,,6
7,8,9
10,11,12\n"""
    parser = all_parsers
    msg = (
        "Too many columns specified: expected 4 and found 3"
        if parser.engine == "c"
        else "Number of passed names did not match "
        "number of header fields in the file"
    )

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=0, names=["a", "b", "c", "d"])


@pytest.mark.parametrize("nrows", [0, 1, 2, 3, 4, 5])
def test_raise_on_no_columns(all_parsers, nrows):
    parser = all_parsers
    data = "\n" * nrows

    msg = "No columns to parse from file"
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data))


def test_read_csv_raises_on_header_prefix(all_parsers):
    # gh-27394
    parser = all_parsers
    msg = "Argument prefix must be None if argument header is not None"

    s = StringIO("0,1\n2,3")

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(s, header=0, prefix="_X")


def test_unexpected_keyword_parameter_exception(all_parsers):
    # GH-34976
    parser = all_parsers

    msg = "{}\\(\\) got an unexpected keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg.format("read_csv")):
        parser.read_csv("foo.csv", foo=1)
    with pytest.raises(TypeError, match=msg.format("read_table")):
        parser.read_table("foo.tsv", foo=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"error_bad_lines": False, "warn_bad_lines": False},
            marks=pytest.mark.filterwarnings("ignore"),
        ),
        {"on_bad_lines": "skip"},
    ],
)
def test_suppress_error_output(all_parsers, capsys, kwargs):
    # see gh-15925
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    expected = DataFrame({"a": [1, 4]})

    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

    captured = capsys.readouterr()
    assert captured.err == ""


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"error_bad_lines": True}],  # Default is True.  # Explicitly pass in.
)
@pytest.mark.parametrize(
    "warn_kwargs",
    [{}, {"warn_bad_lines": True}, {"warn_bad_lines": False}],
)
def test_error_bad_lines(all_parsers, kwargs, warn_kwargs):
    # see gh-15925
    parser = all_parsers
    kwargs.update(**warn_kwargs)
    data = "a\n1\n1,2,3\n4\n5,6,7"

    msg = "Expected 1 fields in line 3, saw 3"
    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"error_bad_lines": False, "warn_bad_lines": True},
            marks=pytest.mark.filterwarnings("ignore"),
        ),
        {"on_bad_lines": "warn"},
    ],
)
def test_warn_bad_lines(all_parsers, capsys, kwargs):
    # see gh-15925
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    expected = DataFrame({"a": [1, 4]})

    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

    captured = capsys.readouterr()
    assert "Skipping line 3" in captured.err
    assert "Skipping line 5" in captured.err


def test_read_csv_wrong_num_columns(all_parsers):
    # Too few columns.
    data = """A,B,C,D,E,F
1,2,3,4,5,6
6,7,8,9,10,11,12
11,12,13,14,15,16
"""
    parser = all_parsers
    msg = "Expected 6 fields in line 3, saw 7"

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(data))


def test_null_byte_char(all_parsers):
    # see gh-2741
    data = "\x00,foo"
    names = ["a", "b"]
    parser = all_parsers

    if parser.engine == "c":
        expected = DataFrame([[np.nan, "foo"]], columns=names)
        out = parser.read_csv(StringIO(data), names=names)
        tm.assert_frame_equal(out, expected)
    else:
        msg = "NULL byte detected"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), names=names)


@td.check_file_leaks
def test_open_file(all_parsers):
    # GH 39024
    parser = all_parsers
    if parser.engine == "c":
        pytest.skip()

    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(b"\xe4\na\n1")

        # should not trigger a ResourceWarning
        warnings.simplefilter("always", category=ResourceWarning)
        with warnings.catch_warnings(record=True) as record:
            with pytest.raises(csv.Error, match="Could not determine delimiter"):
                parser.read_csv(file, sep=None, encoding_errors="replace")
            assert len(record) == 0, record[0].message


def test_invalid_on_bad_line(all_parsers):
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    with pytest.raises(ValueError, match="Argument abc is invalid for on_bad_lines"):
        parser.read_csv(StringIO(data), on_bad_lines="abc")


@pytest.mark.parametrize("error_bad_lines", [True, False])
@pytest.mark.parametrize("warn_bad_lines", [True, False])
def test_conflict_on_bad_line(all_parsers, error_bad_lines, warn_bad_lines):
    parser = all_parsers
    data = "a\n1\n1,2,3\n4\n5,6,7"
    kwds = {"error_bad_lines": error_bad_lines, "warn_bad_lines": warn_bad_lines}
    with pytest.raises(
        ValueError,
        match="Both on_bad_lines and error_bad_lines/warn_bad_lines are set. "
        "Please only set on_bad_lines.",
    ):
        parser.read_csv(StringIO(data), on_bad_lines="error", **kwds)
