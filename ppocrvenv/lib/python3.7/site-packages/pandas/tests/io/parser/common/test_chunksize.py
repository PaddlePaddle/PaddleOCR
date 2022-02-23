"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""
from io import StringIO

import numpy as np
import pytest

from pandas.errors import DtypeWarning

from pandas import (
    DataFrame,
    concat,
)
import pandas._testing as tm


@pytest.mark.parametrize("index_col", [0, "index"])
def test_read_chunksize_with_index(all_parsers, index_col):
    parser = all_parsers
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    expected = DataFrame(
        [
            ["foo", 2, 3, 4, 5],
            ["bar", 7, 8, 9, 10],
            ["baz", 12, 13, 14, 15],
            ["qux", 12, 13, 14, 15],
            ["foo2", 12, 13, 14, 15],
            ["bar2", 12, 13, 14, 15],
        ],
        columns=["index", "A", "B", "C", "D"],
    )
    expected = expected.set_index("index")

    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


@pytest.mark.parametrize("chunksize", [1.3, "foo", 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    msg = r"'chunksize' must be an integer >=1"

    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass


@pytest.mark.parametrize("chunksize", [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    # see gh-15755
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)


def test_read_chunksize_and_nrows_changing_size(all_parsers):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    kwargs = {"index_col": 0, "nrows": 5}

    expected = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])

        with pytest.raises(StopIteration, match=""):
            reader.get_chunk(size=3)


def test_get_chunk_passed_chunksize(all_parsers):
    parser = all_parsers
    data = """A,B,C
1,2,3
4,5,6
7,8,9
1,2,3"""

    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()

    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"index_col": 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    # see gh-12185
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), result)


def test_read_chunksize_jagged_names(all_parsers):
    # see gh-23509
    parser = all_parsers
    data = "\n".join(["0"] * 7 + [",".join(["0"] * 10)])

    expected = DataFrame([[0] + [np.nan] * 9] * 7 + [[0] * 10])
    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    tm.assert_frame_equal(result, expected)


def test_chunk_begins_with_newline_whitespace(all_parsers):
    # see gh-10022
    parser = all_parsers
    data = "\n hello\nworld\n"

    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([" hello", "world"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
@pytest.mark.xfail(reason="GH38630, sometimes gives ResourceWarning", strict=False)
def test_chunks_have_consistent_numerical_type(all_parsers):
    parser = all_parsers
    integers = [str(i) for i in range(499999)]
    data = "a\n" + "\n".join(integers + ["1.0", "2.0"] + integers)

    # Coercions should work without warnings.
    with tm.assert_produces_warning(None):
        result = parser.read_csv(StringIO(data))

    assert type(result.a[0]) is np.float64
    assert result.a.dtype == float


def test_warn_if_chunks_have_mismatched_type(all_parsers, request):
    warning_type = None
    parser = all_parsers
    size = 10000

    # see gh-3866: if chunks are different types and can't
    # be coerced using numerical types, then issue warning.
    if parser.engine == "c" and parser.low_memory:
        warning_type = DtypeWarning
        # Use larger size to hit warning path
        size = 499999

    integers = [str(i) for i in range(size)]
    data = "a\n" + "\n".join(integers + ["a", "b"] + integers)

    buf = StringIO(data)

    try:
        with tm.assert_produces_warning(warning_type):
            df = parser.read_csv(buf)
    except AssertionError as err:
        # 2021-02-21 this occasionally fails on the CI with an unexpected
        #  ResourceWarning that we have been unable to track down,
        #  see GH#38630
        if "ResourceWarning" not in str(err) or parser.engine != "python":
            raise

        # Check the main assertion of the test before re-raising
        assert df.a.dtype == object

        mark = pytest.mark.xfail(
            reason="ResourceWarning for unclosed SSL Socket, GH#38630"
        )
        request.node.add_marker(mark)
        raise

    assert df.a.dtype == object


@pytest.mark.parametrize("iterator", [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    # see gh-9535
    parser = all_parsers
    expected = DataFrame(columns=["foo", "bar"])

    nrows = 10
    data = StringIO("foo,bar\n")

    if iterator:
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        result = parser.read_csv(data, nrows=nrows)

    tm.assert_frame_equal(result, expected)


def test_read_csv_memory_growth_chunksize(all_parsers):
    # see gh-24805
    #
    # Let's just make sure that we don't crash
    # as we iteratively process all chunks.
    parser = all_parsers

    with tm.ensure_clean() as path:
        with open(path, "w") as f:
            for i in range(1000):
                f.write(str(i) + "\n")

        with parser.read_csv(path, chunksize=20) as result:
            for _ in result:
                pass
