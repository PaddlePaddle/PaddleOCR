from io import StringIO
from pathlib import Path

import pytest

import pandas as pd
from pandas import (
    DataFrame,
    read_json,
)
import pandas._testing as tm

from pandas.io.json._json import JsonReader


@pytest.fixture
def lines_json_df():
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return df.to_json(lines=True, orient="records")


def test_read_jsonl():
    # GH9180
    result = read_json('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n', lines=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_unicode_chars():
    # GH15132: non-ascii unicode characters
    # \u201d == RIGHT DOUBLE QUOTATION MARK

    # simulate file handle
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    json = StringIO(json)
    result = read_json(json, lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # simulate string
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    result = read_json(json, lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_to_jsonl():
    # GH9180
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
    assert result == expected

    df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(result, lines=True), df)

    # GH15096: escaped characters in columns and data
    df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
    assert result == expected
    tm.assert_frame_equal(read_json(result, lines=True), df)


def test_to_jsonl_count_new_lines():
    # GH36888
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    actual_new_lines_count = df.to_json(orient="records", lines=True).count("\n")
    expected_new_lines_count = 2
    assert actual_new_lines_count == expected_new_lines_count


@pytest.mark.parametrize("chunksize", [1, 1.0])
def test_readjson_chunks(lines_json_df, chunksize):
    # Basic test that read_json(chunks=True) gives the same result as
    # read_json(chunks=False)
    # GH17048: memory usage when lines=True

    unchunked = read_json(StringIO(lines_json_df), lines=True)
    with read_json(StringIO(lines_json_df), lines=True, chunksize=chunksize) as reader:
        chunked = pd.concat(reader)

    tm.assert_frame_equal(chunked, unchunked)


def test_readjson_chunksize_requires_lines(lines_json_df):
    msg = "chunksize can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        with read_json(StringIO(lines_json_df), lines=False, chunksize=2) as _:
            pass


def test_readjson_chunks_series():
    # Test reading line-format JSON to Series with chunksize param
    s = pd.Series({"A": 1, "B": 2})

    strio = StringIO(s.to_json(lines=True, orient="records"))
    unchunked = read_json(strio, lines=True, typ="Series")

    strio = StringIO(s.to_json(lines=True, orient="records"))
    with read_json(strio, lines=True, typ="Series", chunksize=1) as reader:
        chunked = pd.concat(reader)

    tm.assert_series_equal(chunked, unchunked)


def test_readjson_each_chunk(lines_json_df):
    # Other tests check that the final result of read_json(chunksize=True)
    # is correct. This checks the intermediate chunks.
    with read_json(StringIO(lines_json_df), lines=True, chunksize=2) as reader:
        chunks = list(reader)
    assert chunks[0].shape == (2, 2)
    assert chunks[1].shape == (1, 2)


def test_readjson_chunks_from_file():
    with tm.ensure_clean("test.json") as path:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_json(path, lines=True, orient="records")
        with read_json(path, lines=True, chunksize=1) as reader:
            chunked = pd.concat(reader)
        unchunked = read_json(path, lines=True)
        tm.assert_frame_equal(unchunked, chunked)


@pytest.mark.parametrize("chunksize", [None, 1])
def test_readjson_chunks_closes(chunksize):
    with tm.ensure_clean("test.json") as path:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df.to_json(path, lines=True, orient="records")
        reader = JsonReader(
            path,
            orient=None,
            typ="frame",
            dtype=True,
            convert_axes=True,
            convert_dates=True,
            keep_default_dates=True,
            numpy=False,
            precise_float=False,
            date_unit=None,
            encoding=None,
            lines=True,
            chunksize=chunksize,
            compression=None,
            nrows=None,
        )
        with reader:
            reader.read()
        assert (
            reader.handles.handle.closed
        ), f"didn't close stream with chunksize = {chunksize}"


@pytest.mark.parametrize("chunksize", [0, -1, 2.2, "foo"])
def test_readjson_invalid_chunksize(lines_json_df, chunksize):
    msg = r"'chunksize' must be an integer >=1"

    with pytest.raises(ValueError, match=msg):
        with read_json(StringIO(lines_json_df), lines=True, chunksize=chunksize) as _:
            pass


@pytest.mark.parametrize("chunksize", [None, 1, 2])
def test_readjson_chunks_multiple_empty_lines(chunksize):
    j = """

    {"A":1,"B":4}



    {"A":2,"B":5}







    {"A":3,"B":6}
    """
    orig = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test = read_json(j, lines=True, chunksize=chunksize)
    if chunksize is not None:
        with test:
            test = pd.concat(test)
    tm.assert_frame_equal(orig, test, obj=f"chunksize: {chunksize}")


def test_readjson_unicode(monkeypatch):
    with tm.ensure_clean("test.json") as path:
        monkeypatch.setattr("locale.getpreferredencoding", lambda l: "cp949")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"£©µÀÆÖÞßéöÿ":["АБВГДабвгд가"]}')

        result = read_json(path)
        expected = DataFrame({"£©µÀÆÖÞßéöÿ": ["АБВГДабвгд가"]})
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1, 2])
def test_readjson_nrows(nrows):
    # GH 33916
    # Test reading line-format JSON to Series with nrows param
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    result = read_json(jsonl, lines=True, nrows=nrows)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows,chunksize", [(2, 2), (4, 2)])
def test_readjson_nrows_chunks(nrows, chunksize):
    # GH 33916
    # Test reading line-format JSON to Series with nrows and chunksize param
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    with read_json(jsonl, lines=True, nrows=nrows, chunksize=chunksize) as reader:
        chunked = pd.concat(reader)
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    tm.assert_frame_equal(chunked, expected)


def test_readjson_nrows_requires_lines():
    # GH 33916
    # Test ValuError raised if nrows is set without setting lines in read_json
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    msg = "nrows can only be passed if lines=True"
    with pytest.raises(ValueError, match=msg):
        read_json(jsonl, lines=False, nrows=2)


def test_readjson_lines_chunks_fileurl(datapath):
    # GH 27135
    # Test reading line-format JSON from file url
    df_list_expected = [
        DataFrame([[1, 2]], columns=["a", "b"], index=[0]),
        DataFrame([[3, 4]], columns=["a", "b"], index=[1]),
        DataFrame([[5, 6]], columns=["a", "b"], index=[2]),
    ]
    os_path = datapath("io", "json", "data", "line_delimited.json")
    file_url = Path(os_path).as_uri()
    with read_json(file_url, lines=True, chunksize=1) as url_reader:
        for index, chuck in enumerate(url_reader):
            tm.assert_frame_equal(chuck, df_list_expected[index])


def test_chunksize_is_incremental():
    # See https://github.com/pandas-dev/pandas/issues/34548
    jsonl = (
        """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}\n"""
        * 1000
    )

    class MyReader:
        def __init__(self, contents):
            self.read_count = 0
            self.stringio = StringIO(contents)

        def read(self, *args):
            self.read_count += 1
            return self.stringio.read(*args)

        def __iter__(self):
            self.read_count += 1
            return iter(self.stringio)

    reader = MyReader(jsonl)
    assert len(list(read_json(reader, lines=True, chunksize=100))) > 1
    assert reader.read_count > 10
