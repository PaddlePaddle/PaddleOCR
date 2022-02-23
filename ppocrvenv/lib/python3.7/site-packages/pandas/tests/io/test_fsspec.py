import io

import numpy as np
import pytest

from pandas import (
    DataFrame,
    date_range,
    read_csv,
    read_excel,
    read_feather,
    read_json,
    read_parquet,
    read_pickle,
    read_stata,
)
import pandas._testing as tm
from pandas.util import _test_decorators as td

df1 = DataFrame(
    {
        "int": [1, 3],
        "float": [2.0, np.nan],
        "str": ["t", "s"],
        "dt": date_range("2018-06-18", periods=2),
    }
)
text = str(df1.to_csv(index=False)).encode()


@pytest.fixture
def cleared_fs():
    fsspec = pytest.importorskip("fsspec")

    memfs = fsspec.filesystem("memory")
    yield memfs
    memfs.store.clear()


def test_read_csv(cleared_fs):
    from fsspec.implementations.memory import MemoryFile

    cleared_fs.store["test/test.csv"] = MemoryFile(data=text)
    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"])

    tm.assert_frame_equal(df1, df2)


def test_reasonable_error(monkeypatch, cleared_fs):
    from fsspec import registry
    from fsspec.registry import known_implementations

    registry.target.clear()
    with pytest.raises(ValueError, match="nosuchprotocol"):
        read_csv("nosuchprotocol://test/test.csv")
    err_msg = "test error message"
    monkeypatch.setitem(
        known_implementations,
        "couldexist",
        {"class": "unimportable.CouldExist", "err": err_msg},
    )
    with pytest.raises(ImportError, match=err_msg):
        read_csv("couldexist://test/test.csv")


def test_to_csv(cleared_fs):
    df1.to_csv("memory://test/test.csv", index=True)

    df2 = read_csv("memory://test/test.csv", parse_dates=["dt"], index_col=0)

    tm.assert_frame_equal(df1, df2)


@pytest.mark.parametrize("ext", ["xls", "xlsx"])
def test_to_excel(cleared_fs, ext):
    if ext == "xls":
        pytest.importorskip("xlwt")
    else:
        pytest.importorskip("openpyxl")

    path = f"memory://test/test.{ext}"
    df1.to_excel(path, index=True)

    df2 = read_excel(path, parse_dates=["dt"], index_col=0)

    tm.assert_frame_equal(df1, df2)


@pytest.mark.parametrize("binary_mode", [False, True])
def test_to_csv_fsspec_object(cleared_fs, binary_mode):
    fsspec = pytest.importorskip("fsspec")

    path = "memory://test/test.csv"
    mode = "wb" if binary_mode else "w"
    fsspec_object = fsspec.open(path, mode=mode).open()

    df1.to_csv(fsspec_object, index=True)
    assert not fsspec_object.closed
    fsspec_object.close()

    mode = mode.replace("w", "r")
    fsspec_object = fsspec.open(path, mode=mode).open()

    df2 = read_csv(
        fsspec_object,
        parse_dates=["dt"],
        index_col=0,
    )
    assert not fsspec_object.closed
    fsspec_object.close()

    tm.assert_frame_equal(df1, df2)


def test_csv_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_csv(
        "testmem://test/test.csv", storage_options={"test": "csv_write"}, index=False
    )
    assert fsspectest.test[0] == "csv_write"
    read_csv("testmem://test/test.csv", storage_options={"test": "csv_read"})
    assert fsspectest.test[0] == "csv_read"


@pytest.mark.parametrize("extension", ["xlsx", "xls"])
def test_excel_options(fsspectest, extension):
    if extension == "xls":
        pytest.importorskip("xlwt")
    else:
        pytest.importorskip("openpyxl")

    df = DataFrame({"a": [0]})

    path = f"testmem://test/test.{extension}"

    df.to_excel(path, storage_options={"test": "write"}, index=False)
    assert fsspectest.test[0] == "write"
    read_excel(path, storage_options={"test": "read"})
    assert fsspectest.test[0] == "read"


@td.skip_if_no("fastparquet")
def test_to_parquet_new_file(monkeypatch, cleared_fs):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    df1.to_parquet(
        "memory://test/test.csv", index=True, engine="fastparquet", compression=None
    )


@td.skip_if_no("pyarrow")
def test_arrowparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="pyarrow",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


@td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) fastparquet
@td.skip_if_no("fastparquet")
def test_fastparquet_options(fsspectest):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    df = DataFrame({"a": [0]})
    df.to_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        compression=None,
        storage_options={"test": "parquet_write"},
    )
    assert fsspectest.test[0] == "parquet_write"
    read_parquet(
        "testmem://test/test.csv",
        engine="fastparquet",
        storage_options={"test": "parquet_read"},
    )
    assert fsspectest.test[0] == "parquet_read"


@td.skip_if_no("s3fs")
def test_from_s3_csv(s3_resource, tips_file, s3so):
    tm.assert_equal(
        read_csv("s3://pandas-test/tips.csv", storage_options=s3so), read_csv(tips_file)
    )
    # the following are decompressed by pandas, not fsspec
    tm.assert_equal(
        read_csv("s3://pandas-test/tips.csv.gz", storage_options=s3so),
        read_csv(tips_file),
    )
    tm.assert_equal(
        read_csv("s3://pandas-test/tips.csv.bz2", storage_options=s3so),
        read_csv(tips_file),
    )


@pytest.mark.parametrize("protocol", ["s3", "s3a", "s3n"])
@td.skip_if_no("s3fs")
def test_s3_protocols(s3_resource, tips_file, protocol, s3so):
    tm.assert_equal(
        read_csv("%s://pandas-test/tips.csv" % protocol, storage_options=s3so),
        read_csv(tips_file),
    )


@td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) fastparquet
@td.skip_if_no("s3fs")
@td.skip_if_no("fastparquet")
def test_s3_parquet(s3_resource, s3so):
    fn = "s3://pandas-test/test.parquet"
    df1.to_parquet(
        fn, index=False, engine="fastparquet", compression=None, storage_options=s3so
    )
    df2 = read_parquet(fn, engine="fastparquet", storage_options=s3so)
    tm.assert_equal(df1, df2)


@td.skip_if_installed("fsspec")
def test_not_present_exception():
    msg = "Missing optional dependency 'fsspec'|fsspec library is required"
    with pytest.raises(ImportError, match=msg):
        read_csv("memory://test/test.csv")


@td.skip_if_no("pyarrow")
def test_feather_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_feather("testmem://afile", storage_options={"test": "feather_write"})
    assert fsspectest.test[0] == "feather_write"
    out = read_feather("testmem://afile", storage_options={"test": "feather_read"})
    assert fsspectest.test[0] == "feather_read"
    tm.assert_frame_equal(df, out)


def test_pickle_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_pickle("testmem://afile", storage_options={"test": "pickle_write"})
    assert fsspectest.test[0] == "pickle_write"
    out = read_pickle("testmem://afile", storage_options={"test": "pickle_read"})
    assert fsspectest.test[0] == "pickle_read"
    tm.assert_frame_equal(df, out)


def test_json_options(fsspectest, compression):
    df = DataFrame({"a": [0]})
    df.to_json(
        "testmem://afile",
        compression=compression,
        storage_options={"test": "json_write"},
    )
    assert fsspectest.test[0] == "json_write"
    out = read_json(
        "testmem://afile",
        compression=compression,
        storage_options={"test": "json_read"},
    )
    assert fsspectest.test[0] == "json_read"
    tm.assert_frame_equal(df, out)


def test_stata_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_stata(
        "testmem://afile", storage_options={"test": "stata_write"}, write_index=False
    )
    assert fsspectest.test[0] == "stata_write"
    out = read_stata("testmem://afile", storage_options={"test": "stata_read"})
    assert fsspectest.test[0] == "stata_read"
    tm.assert_frame_equal(df, out.astype("int64"))


@td.skip_if_no("tabulate")
def test_markdown_options(fsspectest):
    df = DataFrame({"a": [0]})
    df.to_markdown("testmem://afile", storage_options={"test": "md_write"})
    assert fsspectest.test[0] == "md_write"
    assert fsspectest.cat("afile")


@td.skip_if_no("pyarrow")
def test_non_fsspec_options():
    with pytest.raises(ValueError, match="storage_options"):
        read_csv("localfile", storage_options={"a": True})
    with pytest.raises(ValueError, match="storage_options"):
        # separate test for parquet, which has a different code path
        read_parquet("localfile", storage_options={"a": True})
    by = io.BytesIO()

    with pytest.raises(ValueError, match="storage_options"):
        read_csv(by, storage_options={"a": True})

    df = DataFrame({"a": [0]})
    with pytest.raises(ValueError, match="storage_options"):
        df.to_parquet("nonfsspecpath", storage_options={"a": True})
