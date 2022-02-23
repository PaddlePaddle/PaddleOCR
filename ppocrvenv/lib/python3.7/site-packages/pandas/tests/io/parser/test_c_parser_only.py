"""
Tests that apply specifically to the CParser. Unless specifically stated
as a CParser-specific issue, the goal is to eventually move as many of
these tests out of this module as soon as the Python parser can accept
further arguments when parsing.
"""

from io import (
    BytesIO,
    StringIO,
    TextIOWrapper,
)
import mmap
import os
import tarfile

import numpy as np
import pytest

from pandas.compat import IS64
from pandas.errors import ParserError
import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "malformed",
    ["1\r1\r1\r 1\r 1\r", "1\r1\r1\r 1\r 1\r11\r", "1\r1\r1\r 1\r 1\r11\r1\r"],
    ids=["words pointer", "stream pointer", "lines pointer"],
)
def test_buffer_overflow(c_parser_only, malformed):
    # see gh-9205: test certain malformed input files that cause
    # buffer overflows in tokenizer.c
    msg = "Buffer overflow caught - possible malformed input file."
    parser = c_parser_only

    with pytest.raises(ParserError, match=msg):
        parser.read_csv(StringIO(malformed))


def test_buffer_rd_bytes(c_parser_only):
    # see gh-12098: src->buffer in the C parser can be freed twice leading
    # to a segfault if a corrupt gzip file is read with 'read_csv', and the
    # buffer is filled more than once before gzip raises an Exception.

    data = (
        "\x1F\x8B\x08\x00\x00\x00\x00\x00\x00\x03\xED\xC3\x41\x09"
        "\x00\x00\x08\x00\xB1\xB7\xB6\xBA\xFE\xA5\xCC\x21\x6C\xB0"
        "\xA6\x4D" + "\x55" * 267 + "\x7D\xF7\x00\x91\xE0\x47\x97\x14\x38\x04\x00"
        "\x1f\x8b\x08\x00VT\x97V\x00\x03\xed]\xefO"
    )
    parser = c_parser_only

    with tm.assert_produces_warning(RuntimeWarning):
        # compression has no effect when passing a non-binary object as input
        for _ in range(100):
            try:
                parser.read_csv(
                    StringIO(data), compression="gzip", delim_whitespace=True
                )
            except Exception:
                pass


def test_delim_whitespace_custom_terminator(c_parser_only):
    # See gh-12912
    data = "a b c~1 2 3~4 5 6~7 8 9"
    parser = c_parser_only

    df = parser.read_csv(StringIO(data), lineterminator="~", delim_whitespace=True)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
    tm.assert_frame_equal(df, expected)


def test_dtype_and_names_error(c_parser_only):
    # see gh-8833: passing both dtype and names
    # resulting in an error reporting issue
    parser = c_parser_only
    data = """
1.0 1
2.0 2
3.0 3
"""
    # base cases
    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None)
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]])
    tm.assert_frame_equal(result, expected)

    result = parser.read_csv(StringIO(data), sep=r"\s+", header=None, names=["a", "b"])
    expected = DataFrame([[1.0, 1], [2.0, 2], [3.0, 3]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # fallback casting
    result = parser.read_csv(
        StringIO(data), sep=r"\s+", header=None, names=["a", "b"], dtype={"a": np.int32}
    )
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], columns=["a", "b"])
    expected["a"] = expected["a"].astype(np.int32)
    tm.assert_frame_equal(result, expected)

    data = """
1.0 1
nan 2
3.0 3
"""
    # fallback casting, but not castable
    with pytest.raises(ValueError, match="cannot safely convert"):
        parser.read_csv(
            StringIO(data),
            sep=r"\s+",
            header=None,
            names=["a", "b"],
            dtype={"a": np.int32},
        )


@pytest.mark.parametrize(
    "match,kwargs",
    [
        # For each of these cases, all of the dtypes are valid, just unsupported.
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}},
        ),
        (
            (
                "the dtype datetime64 is not supported for parsing, "
                "pass this column using parse_dates instead"
            ),
            {"dtype": {"A": "datetime64", "B": "float64"}, "parse_dates": ["B"]},
        ),
        (
            "the dtype timedelta64 is not supported for parsing",
            {"dtype": {"A": "timedelta64", "B": "float64"}},
        ),
        ("the dtype <U8 is not supported for parsing", {"dtype": {"A": "U8"}}),
    ],
    ids=["dt64-0", "dt64-1", "td64", "<U8"],
)
def test_unsupported_dtype(c_parser_only, match, kwargs):
    parser = c_parser_only
    df = DataFrame(
        np.random.rand(5, 2), columns=list("AB"), index=["1A", "1B", "1C", "1D", "1E"]
    )

    with tm.ensure_clean("__unsupported_dtype__.csv") as path:
        df.to_csv(path)

        with pytest.raises(TypeError, match=match):
            parser.read_csv(path, index_col=0, **kwargs)


@td.skip_if_32bit
@pytest.mark.slow
def test_precise_conversion(c_parser_only):
    from decimal import Decimal

    parser = c_parser_only

    normal_errors = []
    precise_errors = []

    # test numbers between 1 and 2
    for num in np.linspace(1.0, 2.0, num=500):
        # 25 decimal digits of precision
        text = f"a\n{num:.25}"

        normal_val = float(
            parser.read_csv(StringIO(text), float_precision="legacy")["a"][0]
        )
        precise_val = float(
            parser.read_csv(StringIO(text), float_precision="high")["a"][0]
        )
        roundtrip_val = float(
            parser.read_csv(StringIO(text), float_precision="round_trip")["a"][0]
        )
        actual_val = Decimal(text[2:])

        def error(val):
            return abs(Decimal(f"{val:.100}") - actual_val)

        normal_errors.append(error(normal_val))
        precise_errors.append(error(precise_val))

        # round-trip should match float()
        assert roundtrip_val == float(text[2:])

    assert sum(precise_errors) <= sum(normal_errors)
    assert max(precise_errors) <= max(normal_errors)


def test_usecols_dtypes(c_parser_only):
    parser = c_parser_only
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""

    result = parser.read_csv(
        StringIO(data),
        usecols=(0, 1, 2),
        names=("a", "b", "c"),
        header=None,
        converters={"a": str},
        dtype={"b": int, "c": float},
    )
    result2 = parser.read_csv(
        StringIO(data),
        usecols=(0, 2),
        names=("a", "b", "c"),
        header=None,
        converters={"a": str},
        dtype={"b": int, "c": float},
    )

    assert (result.dtypes == [object, int, float]).all()
    assert (result2.dtypes == [object, float]).all()


def test_disable_bool_parsing(c_parser_only):
    # see gh-2090

    parser = c_parser_only
    data = """A,B,C
Yes,No,Yes
No,Yes,Yes
Yes,,Yes
No,No,No"""

    result = parser.read_csv(StringIO(data), dtype=object)
    assert (result.dtypes == object).all()

    result = parser.read_csv(StringIO(data), dtype=object, na_filter=False)
    assert result["B"][2] == ""


def test_custom_lineterminator(c_parser_only):
    parser = c_parser_only
    data = "a,b,c~1,2,3~4,5,6"

    result = parser.read_csv(StringIO(data), lineterminator="~")
    expected = parser.read_csv(StringIO(data.replace("~", "\n")))

    tm.assert_frame_equal(result, expected)


def test_parse_ragged_csv(c_parser_only):
    parser = c_parser_only
    data = """1,2,3
1,2,3,4
1,2,3,4,5
1,2
1,2,3,4"""

    nice_data = """1,2,3,,
1,2,3,4,
1,2,3,4,5
1,2,,,
1,2,3,4,"""
    result = parser.read_csv(
        StringIO(data), header=None, names=["a", "b", "c", "d", "e"]
    )

    expected = parser.read_csv(
        StringIO(nice_data), header=None, names=["a", "b", "c", "d", "e"]
    )

    tm.assert_frame_equal(result, expected)

    # too many columns, cause segfault if not careful
    data = "1,2\n3,4,5"

    result = parser.read_csv(StringIO(data), header=None, names=range(50))
    expected = parser.read_csv(StringIO(data), header=None, names=range(3)).reindex(
        columns=range(50)
    )

    tm.assert_frame_equal(result, expected)


def test_tokenize_CR_with_quoting(c_parser_only):
    # see gh-3453
    parser = c_parser_only
    data = ' a,b,c\r"a,b","e,d","f,f"'

    result = parser.read_csv(StringIO(data), header=None)
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")), header=None)
    tm.assert_frame_equal(result, expected)

    result = parser.read_csv(StringIO(data))
    expected = parser.read_csv(StringIO(data.replace("\r", "\n")))
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_grow_boundary_at_cap(c_parser_only):
    # See gh-12494
    #
    # Cause of error was that the C parser
    # was not increasing the buffer size when
    # the desired space would fill the buffer
    # to capacity, which would later cause a
    # buffer overflow error when checking the
    # EOF terminator of the CSV stream.
    parser = c_parser_only

    def test_empty_header_read(count):
        with StringIO("," * count) as s:
            expected = DataFrame(columns=[f"Unnamed: {i}" for i in range(count + 1)])
            df = parser.read_csv(s)
        tm.assert_frame_equal(df, expected)

    for cnt in range(1, 101):
        test_empty_header_read(cnt)


def test_parse_trim_buffers(c_parser_only):
    # This test is part of a bugfix for gh-13703. It attempts to
    # to stress the system memory allocator, to cause it to move the
    # stream buffer and either let the OS reclaim the region, or let
    # other memory requests of parser otherwise modify the contents
    # of memory space, where it was formally located.
    # This test is designed to cause a `segfault` with unpatched
    # `tokenizer.c`. Sometimes the test fails on `segfault`, other
    # times it fails due to memory corruption, which causes the
    # loaded DataFrame to differ from the expected one.

    parser = c_parser_only

    # Generate a large mixed-type CSV file on-the-fly (one record is
    # approx 1.5KiB).
    record_ = (
        """9999-9,99:99,,,,ZZ,ZZ,,,ZZZ-ZZZZ,.Z-ZZZZ,-9.99,,,9.99,Z"""
        """ZZZZ,,-99,9,ZZZ-ZZZZ,ZZ-ZZZZ,,9.99,ZZZ-ZZZZZ,ZZZ-ZZZZZ,"""
        """ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,9"""
        """99,ZZZ-ZZZZ,,ZZ-ZZZZ,,,,,ZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,,,9,9,"""
        """9,9,99,99,999,999,ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,9,ZZ-ZZZZ,9."""
        """99,ZZ-ZZZZ,ZZ-ZZZZ,,,,ZZZZ,,,ZZ,ZZ,,,,,,,,,,,,,9,,,999."""
        """99,999.99,,,ZZZZZ,,,Z9,,,,,,,ZZZ,ZZZ,,,,,,,,,,,ZZZZZ,ZZ"""
        """ZZZ,ZZZ-ZZZZZZ,ZZZ-ZZZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZ"""
        """ZZ,,,999999,999999,ZZZ,ZZZ,,,ZZZ,ZZZ,999.99,999.99,,,,Z"""
        """ZZ-ZZZ,ZZZ-ZZZ,-9.99,-9.99,9,9,,99,,9.99,9.99,9,9,9.99,"""
        """9.99,,,,9.99,9.99,,99,,99,9.99,9.99,,,ZZZ,ZZZ,,999.99,,"""
        """999.99,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,ZZZZZ,ZZZZZ,ZZZ,ZZZ,9,9,"""
        """,,,,,ZZZ-ZZZZ,ZZZ999Z,,,999.99,,999.99,ZZZ-ZZZZ,,,9.999"""
        """,9.999,9.999,9.999,-9.999,-9.999,-9.999,-9.999,9.999,9."""
        """999,9.999,9.999,9.999,9.999,9.999,9.999,99999,ZZZ-ZZZZ,"""
        """,9.99,ZZZ,,,,,,,,ZZZ,,,,,9,,,,9,,,,,,,,,,ZZZ-ZZZZ,ZZZ-Z"""
        """ZZZ,,ZZZZZ,ZZZZZ,ZZZZZ,ZZZZZ,,,9.99,,ZZ-ZZZZ,ZZ-ZZZZ,ZZ"""
        """,999,,,,ZZ-ZZZZ,ZZZ,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,99.99,99.99"""
        """,,,9.99,9.99,9.99,9.99,ZZZ-ZZZZ,,,ZZZ-ZZZZZ,,,,,-9.99,-"""
        """9.99,-9.99,-9.99,,,,,,,,,ZZZ-ZZZZ,,9,9.99,9.99,99ZZ,,-9"""
        """.99,-9.99,ZZZ-ZZZZ,,,,,,,ZZZ-ZZZZ,9.99,9.99,9999,,,,,,,"""
        """,,,-9.9,Z/Z-ZZZZ,999.99,9.99,,999.99,ZZ-ZZZZ,ZZ-ZZZZ,9."""
        """99,9.99,9.99,9.99,9.99,9.99,,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZ"""
        """ZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ,ZZZ,ZZZ,ZZZ,9.99,,,-9.99,ZZ"""
        """-ZZZZ,-999.99,,-9999,,999.99,,,,999.99,99.99,,,ZZ-ZZZZZ"""
        """ZZZ,ZZ-ZZZZ-ZZZZZZZ,,,,ZZ-ZZ-ZZZZZZZZ,ZZZZZZZZ,ZZZ-ZZZZ"""
        """,9999,999.99,ZZZ-ZZZZ,-9.99,-9.99,ZZZ-ZZZZ,99:99:99,,99"""
        """,99,,9.99,,-99.99,,,,,,9.99,ZZZ-ZZZZ,-9.99,-9.99,9.99,9"""
        """.99,,ZZZ,,,,,,,ZZZ,ZZZ,,,,,"""
    )

    # Set the number of lines so that a call to `parser_trim_buffers`
    # is triggered: after a couple of full chunks are consumed a
    # relatively small 'residual' chunk would cause reallocation
    # within the parser.
    chunksize, n_lines = 128, 2 * 128 + 15
    csv_data = "\n".join([record_] * n_lines) + "\n"

    # We will use StringIO to load the CSV from this text buffer.
    # pd.read_csv() will iterate over the file in chunks and will
    # finally read a residual chunk of really small size.

    # Generate the expected output: manually create the dataframe
    # by splitting by comma and repeating the `n_lines` times.
    row = tuple(val_ if val_ else np.nan for val_ in record_.split(","))
    expected = DataFrame(
        [row for _ in range(n_lines)], dtype=object, columns=None, index=None
    )

    # Iterate over the CSV file in chunks of `chunksize` lines
    with parser.read_csv(
        StringIO(csv_data), header=None, dtype=object, chunksize=chunksize
    ) as chunks_:
        result = concat(chunks_, axis=0, ignore_index=True)

    # Check for data corruption if there was no segfault
    tm.assert_frame_equal(result, expected)

    # This extra test was added to replicate the fault in gh-5291.
    # Force 'utf-8' encoding, so that `_string_convert` would take
    # a different execution branch.
    with parser.read_csv(
        StringIO(csv_data),
        header=None,
        dtype=object,
        chunksize=chunksize,
        encoding="utf_8",
    ) as chunks_:
        result = concat(chunks_, axis=0, ignore_index=True)
    tm.assert_frame_equal(result, expected)


def test_internal_null_byte(c_parser_only):
    # see gh-14012
    #
    # The null byte ('\x00') should not be used as a
    # true line terminator, escape character, or comment
    # character, only as a placeholder to indicate that
    # none was specified.
    #
    # This test should be moved to test_common.py ONLY when
    # Python's csv class supports parsing '\x00'.
    parser = c_parser_only

    names = ["a", "b", "c"]
    data = "1,2,3\n4,\x00,6\n7,8,9"
    expected = DataFrame([[1, 2.0, 3], [4, np.nan, 6], [7, 8, 9]], columns=names)

    result = parser.read_csv(StringIO(data), names=names)
    tm.assert_frame_equal(result, expected)


def test_read_nrows_large(c_parser_only):
    # gh-7626 - Read only nrows of data in for large inputs (>262144b)
    parser = c_parser_only
    header_narrow = "\t".join("COL_HEADER_" + str(i) for i in range(10)) + "\n"
    data_narrow = "\t".join("somedatasomedatasomedata1" for _ in range(10)) + "\n"
    header_wide = "\t".join("COL_HEADER_" + str(i) for i in range(15)) + "\n"
    data_wide = "\t".join("somedatasomedatasomedata2" for _ in range(15)) + "\n"
    test_input = header_narrow + data_narrow * 1050 + header_wide + data_wide * 2

    df = parser.read_csv(StringIO(test_input), sep="\t", nrows=1010)

    assert df.size == 1010 * 10


def test_float_precision_round_trip_with_text(c_parser_only):
    # see gh-15140
    parser = c_parser_only
    df = parser.read_csv(StringIO("a"), header=None, float_precision="round_trip")
    tm.assert_frame_equal(df, DataFrame({0: ["a"]}))


def test_large_difference_in_columns(c_parser_only):
    # see gh-14125
    parser = c_parser_only

    count = 10000
    large_row = ("X," * count)[:-1] + "\n"
    normal_row = "XXXXXX XXXXXX,111111111111111\n"
    test_input = (large_row + normal_row * 6)[:-1]

    result = parser.read_csv(StringIO(test_input), header=None, usecols=[0])
    rows = test_input.split("\n")

    expected = DataFrame([row.split(",")[0] for row in rows])
    tm.assert_frame_equal(result, expected)


def test_data_after_quote(c_parser_only):
    # see gh-15910
    parser = c_parser_only

    data = 'a\n1\n"b"a'
    result = parser.read_csv(StringIO(data))

    expected = DataFrame({"a": ["1", "ba"]})
    tm.assert_frame_equal(result, expected)


def test_comment_whitespace_delimited(c_parser_only, capsys):
    parser = c_parser_only
    test_input = """\
1 2
2 2 3
3 2 3 # 3 fields
4 2 3# 3 fields
5 2 # 2 fields
6 2# 2 fields
7 # 1 field, NaN
8# 1 field, NaN
9 2 3 # skipped line
# comment"""
    df = parser.read_csv(
        StringIO(test_input),
        comment="#",
        header=None,
        delimiter="\\s+",
        skiprows=0,
        on_bad_lines="warn",
    )
    captured = capsys.readouterr()
    # skipped lines 2, 3, 4, 9
    for line_num in (2, 3, 4, 9):
        assert f"Skipping line {line_num}" in captured.err
    expected = DataFrame([[1, 2], [5, 2], [6, 2], [7, np.nan], [8, np.nan]])
    tm.assert_frame_equal(df, expected)


def test_file_like_no_next(c_parser_only):
    # gh-16530: the file-like need not have a "next" or "__next__"
    # attribute despite having an "__iter__" attribute.
    #
    # NOTE: This is only true for the C engine, not Python engine.
    class NoNextBuffer(StringIO):
        def __next__(self):
            raise AttributeError("No next method")

        next = __next__

    parser = c_parser_only
    data = "a\n1"

    expected = DataFrame({"a": [1]})
    result = parser.read_csv(NoNextBuffer(data))

    tm.assert_frame_equal(result, expected)


def test_buffer_rd_bytes_bad_unicode(c_parser_only):
    # see gh-22748
    t = BytesIO(b"\xB0")
    t = TextIOWrapper(t, encoding="ascii", errors="surrogateescape")
    msg = "'utf-8' codec can't encode character"
    with pytest.raises(UnicodeError, match=msg):
        c_parser_only.read_csv(t, encoding="UTF-8")


@pytest.mark.parametrize("tar_suffix", [".tar", ".tar.gz"])
def test_read_tarfile(c_parser_only, csv_dir_path, tar_suffix):
    # see gh-16530
    #
    # Unfortunately, Python's CSV library can't handle
    # tarfile objects (expects string, not bytes when
    # iterating through a file-like).
    parser = c_parser_only
    tar_path = os.path.join(csv_dir_path, "tar_csv" + tar_suffix)

    with tarfile.open(tar_path, "r") as tar:
        data_file = tar.extractfile("tar_data.csv")

        out = parser.read_csv(data_file)
        expected = DataFrame({"a": [1]})
        tm.assert_frame_equal(out, expected)


@pytest.mark.high_memory
def test_bytes_exceed_2gb(c_parser_only):
    # see gh-16798
    #
    # Read from a "CSV" that has a column larger than 2GB.
    parser = c_parser_only

    if parser.low_memory:
        pytest.skip("not a high_memory test")

    csv = StringIO("strings\n" + "\n".join("x" * (1 << 20) for _ in range(2100)))
    df = parser.read_csv(csv)
    assert not df.empty


def test_chunk_whitespace_on_boundary(c_parser_only):
    # see gh-9735: this issue is C parser-specific (bug when
    # parsing whitespace and characters at chunk boundary)
    #
    # This test case has a field too large for the Python parser / CSV library.
    parser = c_parser_only

    chunk1 = "a" * (1024 * 256 - 2) + "\na"
    chunk2 = "\n a"
    result = parser.read_csv(StringIO(chunk1 + chunk2), header=None)

    expected = DataFrame(["a" * (1024 * 256 - 2), "a", " a"])
    tm.assert_frame_equal(result, expected)


def test_file_handles_mmap(c_parser_only, csv1):
    # gh-14418
    #
    # Don't close user provided file handles.
    parser = c_parser_only

    with open(csv1) as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        parser.read_csv(m)

        assert not m.closed
        m.close()


def test_file_binary_mode(c_parser_only):
    # see gh-23779
    parser = c_parser_only
    expected = DataFrame([[1, 2, 3], [4, 5, 6]])

    with tm.ensure_clean() as path:
        with open(path, "w") as f:
            f.write("1,2,3\n4,5,6")

        with open(path, "rb") as f:
            result = parser.read_csv(f, header=None)
            tm.assert_frame_equal(result, expected)


def test_unix_style_breaks(c_parser_only):
    # GH 11020
    parser = c_parser_only
    with tm.ensure_clean() as path:
        with open(path, "w", newline="\n") as f:
            f.write("blah\n\ncol_1,col_2,col_3\n\n")
        result = parser.read_csv(path, skiprows=2, encoding="utf-8", engine="c")
    expected = DataFrame(columns=["col_1", "col_2", "col_3"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("float_precision", [None, "legacy", "high", "round_trip"])
@pytest.mark.parametrize(
    "data,thousands,decimal",
    [
        (
            """A|B|C
1|2,334.01|5
10|13|10.
""",
            ",",
            ".",
        ),
        (
            """A|B|C
1|2.334,01|5
10|13|10,
""",
            ".",
            ",",
        ),
    ],
)
def test_1000_sep_with_decimal(
    c_parser_only, data, thousands, decimal, float_precision
):
    parser = c_parser_only
    expected = DataFrame({"A": [1, 10], "B": [2334.01, 13], "C": [5, 10.0]})

    result = parser.read_csv(
        StringIO(data),
        sep="|",
        thousands=thousands,
        decimal=decimal,
        float_precision=float_precision,
    )
    tm.assert_frame_equal(result, expected)


def test_float_precision_options(c_parser_only):
    # GH 17154, 36228
    parser = c_parser_only
    s = "foo\n243.164\n"
    df = parser.read_csv(StringIO(s))
    df2 = parser.read_csv(StringIO(s), float_precision="high")

    tm.assert_frame_equal(df, df2)

    df3 = parser.read_csv(StringIO(s), float_precision="legacy")

    if IS64:
        assert not df.iloc[0, 0] == df3.iloc[0, 0]
    else:
        assert df.iloc[0, 0] == df3.iloc[0, 0]

    msg = "Unrecognized float_precision option: junk"

    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(s), float_precision="junk")
