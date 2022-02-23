import io
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_compression_size(obj, method, compression_only):
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression=compression_only)
        compressed_size = os.path.getsize(path)
        getattr(obj, method)(path, compression=None)
        uncompressed_size = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_csv", "to_json"])
def test_compression_size_fh(obj, method, compression_only):
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=compression_only) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        compressed_size = os.path.getsize(path)
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=None) as handles:
            getattr(obj, method)(handles.handle)
            assert not handles.handle.closed
        uncompressed_size = os.path.getsize(path)
        assert uncompressed_size > compressed_size


@pytest.mark.parametrize(
    "write_method, write_kwargs, read_method",
    [
        ("to_csv", {"index": False}, pd.read_csv),
        ("to_json", {}, pd.read_json),
        ("to_pickle", {}, pd.read_pickle),
    ],
)
def test_dataframe_compression_defaults_to_infer(
    write_method, write_kwargs, read_method, compression_only
):
    # GH22004
    input = pd.DataFrame([[1.0, 0, -4], [3.4, 5, 2]], columns=["X", "Y", "Z"])
    extension = icom._compression_to_extension[compression_only]
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        output = read_method(path, compression=compression_only)
    tm.assert_frame_equal(output, input)


@pytest.mark.parametrize(
    "write_method,write_kwargs,read_method,read_kwargs",
    [
        ("to_csv", {"index": False, "header": True}, pd.read_csv, {"squeeze": True}),
        ("to_json", {}, pd.read_json, {"typ": "series"}),
        ("to_pickle", {}, pd.read_pickle, {}),
    ],
)
def test_series_compression_defaults_to_infer(
    write_method, write_kwargs, read_method, read_kwargs, compression_only
):
    # GH22004
    input = pd.Series([0, 5, -2, 10], name="X")
    extension = icom._compression_to_extension[compression_only]
    with tm.ensure_clean("compressed" + extension) as path:
        getattr(input, write_method)(path, **write_kwargs)
        output = read_method(path, compression=compression_only, **read_kwargs)
    tm.assert_series_equal(output, input, check_names=False)


def test_compression_warning(compression_only):
    # Assert that passing a file object to to_csv while explicitly specifying a
    # compression protocol triggers a RuntimeWarning, as per GH21227.
    df = pd.DataFrame(
        100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
        columns=["X", "Y", "Z"],
    )
    with tm.ensure_clean() as path:
        with icom.get_handle(path, "w", compression=compression_only) as handles:
            with tm.assert_produces_warning(RuntimeWarning):
                df.to_csv(handles.handle, compression=compression_only)


def test_compression_binary(compression_only):
    """
    Binary file handles support compression.

    GH22555
    """
    df = tm.makeDataFrame()

    # with a file
    with tm.ensure_clean() as path:
        with open(path, mode="wb") as file:
            df.to_csv(file, mode="wb", compression=compression_only)
            file.seek(0)  # file shouldn't be closed
        tm.assert_frame_equal(
            df, pd.read_csv(path, index_col=0, compression=compression_only)
        )

    # with BytesIO
    file = io.BytesIO()
    df.to_csv(file, mode="wb", compression=compression_only)
    file.seek(0)  # file shouldn't be closed
    tm.assert_frame_equal(
        df, pd.read_csv(file, index_col=0, compression=compression_only)
    )


def test_gzip_reproducibility_file_name():
    """
    Gzip should create reproducible archives with mtime.

    Note: Archives created with different filenames will still be different!

    GH 28103
    """
    df = tm.makeDataFrame()
    compression_options = {"method": "gzip", "mtime": 1}

    # test for filename
    with tm.ensure_clean() as path:
        path = Path(path)
        df.to_csv(path, compression=compression_options)
        time.sleep(2)
        output = path.read_bytes()
        df.to_csv(path, compression=compression_options)
        assert output == path.read_bytes()


def test_gzip_reproducibility_file_object():
    """
    Gzip should create reproducible archives with mtime.

    GH 28103
    """
    df = tm.makeDataFrame()
    compression_options = {"method": "gzip", "mtime": 1}

    # test for file object
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    output = buffer.getvalue()
    time.sleep(2)
    buffer = io.BytesIO()
    df.to_csv(buffer, compression=compression_options, mode="wb")
    assert output == buffer.getvalue()


def test_with_missing_lzma():
    """Tests if import pandas works when lzma is not present."""
    # https://github.com/pandas-dev/pandas/issues/27575
    code = textwrap.dedent(
        """\
        import sys
        sys.modules['lzma'] = None
        import pandas
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


def test_with_missing_lzma_runtime():
    """Tests if RuntimeError is hit when calling lzma without
    having the module available.
    """
    code = textwrap.dedent(
        """
        import sys
        import pytest
        sys.modules['lzma'] = None
        import pandas as pd
        df = pd.DataFrame()
        with pytest.raises(RuntimeError, match='lzma module'):
            df.to_csv('foo.csv', compression='xz')
        """
    )
    subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.PIPE)


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_gzip_compression_level(obj, method):
    # GH33196
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression="gzip")
        compressed_size_default = os.path.getsize(path)
        getattr(obj, method)(path, compression={"method": "gzip", "compresslevel": 1})
        compressed_size_fast = os.path.getsize(path)
        assert compressed_size_default < compressed_size_fast


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            100 * [[0.123456, 0.234567, 0.567567], [12.32112, 123123.2, 321321.2]],
            columns=["X", "Y", "Z"],
        ),
        pd.Series(100 * [0.123456, 0.234567, 0.567567], name="X"),
    ],
)
@pytest.mark.parametrize("method", ["to_pickle", "to_json", "to_csv"])
def test_bzip_compression_level(obj, method):
    """GH33196 bzip needs file size > 100k to show a size difference between
    compression levels, so here we just check if the call works when
    compression is passed as a dict.
    """
    with tm.ensure_clean() as path:
        getattr(obj, method)(path, compression={"method": "bz2", "compresslevel": 1})
