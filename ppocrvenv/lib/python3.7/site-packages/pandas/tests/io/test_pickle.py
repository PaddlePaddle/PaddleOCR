"""
manage legacy pickle tests

How to add pickle tests:

1. Install pandas version intended to output the pickle.

2. Execute "generate_legacy_storage_files.py" to create the pickle.
$ python generate_legacy_storage_files.py <output_dir> pickle

3. Move the created pickle to "data/legacy_pickle/<version>" directory.
"""
import bz2
import datetime
import functools
from functools import partial
import glob
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
from warnings import (
    catch_warnings,
    filterwarnings,
    simplefilter,
)
import zipfile

import numpy as np
import pytest

from pandas.compat import (
    PY38,
    get_lzma_file,
    import_lzma,
    is_platform_little_endian,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Index,
    Series,
    period_range,
)
import pandas._testing as tm

from pandas.tseries.offsets import (
    Day,
    MonthEnd,
)

lzma = import_lzma()


# TODO(ArrayManager) pickling
pytestmark = [
    td.skip_array_manager_not_yet_implemented,
    pytest.mark.filterwarnings("ignore:Timestamp.freq is deprecated:FutureWarning"),
]


@pytest.fixture(scope="module")
def current_pickle_data():
    # our current version pickle data
    from pandas.tests.io.generate_legacy_storage_files import create_pickle_data

    with catch_warnings():
        filterwarnings(
            "ignore", "The 'freq' argument in Timestamp", category=FutureWarning
        )

        return create_pickle_data()


# ---------------------
# comparison functions
# ---------------------
def compare_element(result, expected, typ, version=None):
    if isinstance(expected, Index):
        tm.assert_index_equal(expected, result)
        return

    if typ.startswith("sp_"):
        comparator = tm.assert_equal
        comparator(result, expected)
    elif typ == "timestamp":
        if expected is pd.NaT:
            assert result is pd.NaT
        else:
            assert result == expected
            assert result.freq == expected.freq
    else:
        comparator = getattr(tm, f"assert_{typ}_equal", tm.assert_almost_equal)
        comparator(result, expected)


def compare(data, vf, version):

    data = pd.read_pickle(vf)

    m = globals()
    for typ, dv in data.items():
        for dt, result in dv.items():
            expected = data[typ][dt]

            # use a specific comparator
            # if available
            comparator = f"compare_{typ}_{dt}"

            comparator = m.get(comparator, m["compare_element"])
            comparator(result, expected, typ, version)
    return data


def compare_series_ts(result, expected, typ, version):
    # GH 7748
    tm.assert_series_equal(result, expected)
    assert result.index.freq == expected.index.freq
    assert not result.index.freq.normalize
    tm.assert_series_equal(result > 0, expected > 0)

    # GH 9291
    freq = result.index.freq
    assert freq + Day(1) == Day(2)

    res = freq + pd.Timedelta(hours=1)
    assert isinstance(res, pd.Timedelta)
    assert res == pd.Timedelta(days=1, hours=1)

    res = freq + pd.Timedelta(nanoseconds=1)
    assert isinstance(res, pd.Timedelta)
    assert res == pd.Timedelta(days=1, nanoseconds=1)


def compare_series_dt_tz(result, expected, typ, version):
    tm.assert_series_equal(result, expected)


def compare_series_cat(result, expected, typ, version):
    tm.assert_series_equal(result, expected)


def compare_frame_dt_mixed_tzs(result, expected, typ, version):
    tm.assert_frame_equal(result, expected)


def compare_frame_cat_onecol(result, expected, typ, version):
    tm.assert_frame_equal(result, expected)


def compare_frame_cat_and_float(result, expected, typ, version):
    compare_frame_cat_onecol(result, expected, typ, version)


def compare_index_period(result, expected, typ, version):
    tm.assert_index_equal(result, expected)
    assert isinstance(result.freq, MonthEnd)
    assert result.freq == MonthEnd()
    assert result.freqstr == "M"
    tm.assert_index_equal(result.shift(2), expected.shift(2))


here = os.path.dirname(__file__)
legacy_dirname = os.path.join(here, "data", "legacy_pickle")
files = glob.glob(os.path.join(legacy_dirname, "*", "*.pickle"))


@pytest.fixture(params=files)
def legacy_pickle(request, datapath):
    return datapath(request.param)


# ---------------------
# tests
# ---------------------
def test_pickles(current_pickle_data, legacy_pickle):
    if not is_platform_little_endian():
        pytest.skip("known failure on non-little endian")

    version = os.path.basename(os.path.dirname(legacy_pickle))
    with catch_warnings(record=True):
        simplefilter("ignore")
        compare(current_pickle_data, legacy_pickle, version)


def python_pickler(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=-1)


def python_unpickler(path):
    with open(path, "rb") as fh:
        fh.seek(0)
        return pickle.load(fh)


@pytest.mark.parametrize(
    "pickle_writer",
    [
        pytest.param(python_pickler, id="python"),
        pytest.param(pd.to_pickle, id="pandas_proto_default"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=pickle.HIGHEST_PROTOCOL),
            id="pandas_proto_highest",
        ),
        pytest.param(functools.partial(pd.to_pickle, protocol=4), id="pandas_proto_4"),
        pytest.param(
            functools.partial(pd.to_pickle, protocol=5),
            id="pandas_proto_5",
            marks=pytest.mark.skipif(not PY38, reason="protocol 5 not supported"),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:The 'freq' argument in Timestamp:FutureWarning")
def test_round_trip_current(current_pickle_data, pickle_writer):
    data = current_pickle_data
    for typ, dv in data.items():
        for dt, expected in dv.items():

            for writer in [pd.to_pickle, python_pickler]:
                with tm.ensure_clean() as path:
                    # test writing with each pickler
                    pickle_writer(expected, path)

                    # test reading with each unpickler
                    result = pd.read_pickle(path)
                    compare_element(result, expected, typ)

                    result = python_unpickler(path)
                    compare_element(result, expected, typ)

                    # and the same for file objects (GH 35679)
                    with open(path, mode="wb") as handle:
                        writer(expected, path)
                        handle.seek(0)  # shouldn't close file handle
                    with open(path, mode="rb") as handle:
                        result = pd.read_pickle(handle)
                        handle.seek(0)  # shouldn't close file handle
                    compare_element(result, expected, typ)


def test_pickle_path_pathlib():
    df = tm.makeDataFrame()
    result = tm.round_trip_pathlib(df.to_pickle, pd.read_pickle)
    tm.assert_frame_equal(df, result)


def test_pickle_path_localpath():
    df = tm.makeDataFrame()
    result = tm.round_trip_localpath(df.to_pickle, pd.read_pickle)
    tm.assert_frame_equal(df, result)


def test_legacy_sparse_warning(datapath):
    """

    Generated with

    >>> df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [0, 0, 1, 1]}).to_sparse()
    >>> df.to_pickle("pandas/tests/io/data/pickle/sparseframe-0.20.3.pickle.gz",
    ...              compression="gzip")

    >>> s = df['B']
    >>> s.to_pickle("pandas/tests/io/data/pickle/sparseseries-0.20.3.pickle.gz",
    ...             compression="gzip")
    """
    with tm.assert_produces_warning(FutureWarning):
        simplefilter("ignore", DeprecationWarning)  # from boto
        pd.read_pickle(
            datapath("io", "data", "pickle", "sparseseries-0.20.3.pickle.gz"),
            compression="gzip",
        )

    with tm.assert_produces_warning(FutureWarning):
        simplefilter("ignore", DeprecationWarning)  # from boto
        pd.read_pickle(
            datapath("io", "data", "pickle", "sparseframe-0.20.3.pickle.gz"),
            compression="gzip",
        )


# ---------------------
# test pickle compression
# ---------------------


@pytest.fixture
def get_random_path():
    return f"__{tm.rands(10)}__.pickle"


class TestCompression:

    _compression_to_extension = {
        None: ".none",
        "gzip": ".gz",
        "bz2": ".bz2",
        "zip": ".zip",
        "xz": ".xz",
    }

    def compress_file(self, src_path, dest_path, compression):
        if compression is None:
            shutil.copyfile(src_path, dest_path)
            return

        if compression == "gzip":
            f = gzip.open(dest_path, "w")
        elif compression == "bz2":
            f = bz2.BZ2File(dest_path, "w")
        elif compression == "zip":
            with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
                f.write(src_path, os.path.basename(src_path))
        elif compression == "xz":
            f = get_lzma_file(lzma)(dest_path, "w")
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        if compression != "zip":
            with open(src_path, "rb") as fh, f:
                f.write(fh.read())

    def test_write_explicit(self, compression, get_random_path):
        base = get_random_path
        path1 = base + ".compressed"
        path2 = base + ".raw"

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = tm.makeDataFrame()

            # write to compressed file
            df.to_pickle(p1, compression=compression)

            # decompress
            with tm.decompress_file(p1, compression=compression) as f:
                with open(p2, "wb") as fh:
                    fh.write(f.read())

            # read decompressed file
            df2 = pd.read_pickle(p2, compression=None)

            tm.assert_frame_equal(df, df2)

    @pytest.mark.parametrize("compression", ["", "None", "bad", "7z"])
    def test_write_explicit_bad(self, compression, get_random_path):
        with pytest.raises(ValueError, match="Unrecognized compression type"):
            with tm.ensure_clean(get_random_path) as path:
                df = tm.makeDataFrame()
                df.to_pickle(path, compression=compression)

    @pytest.mark.parametrize("ext", ["", ".gz", ".bz2", ".no_compress", ".xz"])
    def test_write_infer(self, ext, get_random_path):
        base = get_random_path
        path1 = base + ext
        path2 = base + ".raw"
        compression = None
        for c in self._compression_to_extension:
            if self._compression_to_extension[c] == ext:
                compression = c
                break

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = tm.makeDataFrame()

            # write to compressed file by inferred compression method
            df.to_pickle(p1)

            # decompress
            with tm.decompress_file(p1, compression=compression) as f:
                with open(p2, "wb") as fh:
                    fh.write(f.read())

            # read decompressed file
            df2 = pd.read_pickle(p2, compression=None)

            tm.assert_frame_equal(df, df2)

    def test_read_explicit(self, compression, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + ".compressed"

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = tm.makeDataFrame()

            # write to uncompressed file
            df.to_pickle(p1, compression=None)

            # compress
            self.compress_file(p1, p2, compression=compression)

            # read compressed file
            df2 = pd.read_pickle(p2, compression=compression)

            tm.assert_frame_equal(df, df2)

    @pytest.mark.parametrize("ext", ["", ".gz", ".bz2", ".zip", ".no_compress", ".xz"])
    def test_read_infer(self, ext, get_random_path):
        base = get_random_path
        path1 = base + ".raw"
        path2 = base + ext
        compression = None
        for c in self._compression_to_extension:
            if self._compression_to_extension[c] == ext:
                compression = c
                break

        with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
            df = tm.makeDataFrame()

            # write to uncompressed file
            df.to_pickle(p1, compression=None)

            # compress
            self.compress_file(p1, p2, compression=compression)

            # read compressed file by inferred compression method
            df2 = pd.read_pickle(p2)

            tm.assert_frame_equal(df, df2)


# ---------------------
# test pickle compression
# ---------------------


class TestProtocol:
    @pytest.mark.parametrize("protocol", [-1, 0, 1, 2])
    def test_read(self, protocol, get_random_path):
        with tm.ensure_clean(get_random_path) as path:
            df = tm.makeDataFrame()
            df.to_pickle(path, protocol=protocol)
            df2 = pd.read_pickle(path)
            tm.assert_frame_equal(df, df2)


@pytest.mark.parametrize(
    ["pickle_file", "excols"],
    [
        ("test_py27.pkl", Index(["a", "b", "c"])),
        (
            "test_mi_py27.pkl",
            pd.MultiIndex.from_arrays([["a", "b", "c"], ["A", "B", "C"]]),
        ),
    ],
)
def test_unicode_decode_error(datapath, pickle_file, excols):
    # pickle file written with py27, should be readable without raising
    #  UnicodeDecodeError, see GH#28645 and GH#31988
    path = datapath("io", "data", "pickle", pickle_file)
    df = pd.read_pickle(path)

    # just test the columns are correct since the values are random
    tm.assert_index_equal(df.columns, excols)


# ---------------------
# tests for buffer I/O
# ---------------------


def test_pickle_buffer_roundtrip():
    with tm.ensure_clean() as path:
        df = tm.makeDataFrame()
        with open(path, "wb") as fh:
            df.to_pickle(fh)
        with open(path, "rb") as fh:
            result = pd.read_pickle(fh)
        tm.assert_frame_equal(df, result)


# ---------------------
# tests for URL I/O
# ---------------------


@pytest.mark.parametrize(
    "mockurl", ["http://url.com", "ftp://test.com", "http://gzip.com"]
)
def test_pickle_generalurl_read(monkeypatch, mockurl):
    def python_pickler(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=-1)

    class MockReadResponse:
        def __init__(self, path):
            self.file = open(path, "rb")
            if "gzip" in path:
                self.headers = {"Content-Encoding": "gzip"}
            else:
                self.headers = {"Content-Encoding": None}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def read(self):
            return self.file.read()

        def close(self):
            return self.file.close()

    with tm.ensure_clean() as path:

        def mock_urlopen_read(*args, **kwargs):
            return MockReadResponse(path)

        df = tm.makeDataFrame()
        python_pickler(df, path)
        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_read)
        result = pd.read_pickle(mockurl)
        tm.assert_frame_equal(df, result)


@td.skip_if_no("fsspec")
def test_pickle_fsspec_roundtrip():
    with tm.ensure_clean():
        mockurl = "memory://afile"
        df = tm.makeDataFrame()
        df.to_pickle(mockurl)
        result = pd.read_pickle(mockurl)
        tm.assert_frame_equal(df, result)


class MyTz(datetime.tzinfo):
    def __init__(self):
        pass


def test_read_pickle_with_subclass():
    # GH 12163
    expected = Series(dtype=object), MyTz()
    result = tm.round_trip_pickle(expected)

    tm.assert_series_equal(result[0], expected[0])
    assert isinstance(result[1], MyTz)


def test_pickle_binary_object_compression(compression):
    """
    Read/write from binary file-objects w/wo compression.

    GH 26237, GH 29054, and GH 29570
    """
    df = tm.makeDataFrame()

    # reference for compression
    with tm.ensure_clean() as path:
        df.to_pickle(path, compression=compression)
        reference = Path(path).read_bytes()

    # write
    buffer = io.BytesIO()
    df.to_pickle(buffer, compression=compression)
    buffer.seek(0)

    # gzip  and zip safe the filename: cannot compare the compressed content
    assert buffer.getvalue() == reference or compression in ("gzip", "zip")

    # read
    read_df = pd.read_pickle(buffer, compression=compression)
    buffer.seek(0)
    tm.assert_frame_equal(df, read_df)


def test_pickle_dataframe_with_multilevel_index(
    multiindex_year_month_day_dataframe_random_data,
    multiindex_dataframe_random_data,
):
    ymd = multiindex_year_month_day_dataframe_random_data
    frame = multiindex_dataframe_random_data

    def _test_roundtrip(frame):
        unpickled = tm.round_trip_pickle(frame)
        tm.assert_frame_equal(frame, unpickled)

    _test_roundtrip(frame)
    _test_roundtrip(frame.T)
    _test_roundtrip(ymd)
    _test_roundtrip(ymd.T)


def test_pickle_timeseries_periodindex():
    # GH#2891
    prng = period_range("1/1/2011", "1/1/2012", freq="M")
    ts = Series(np.random.randn(len(prng)), prng)
    new_ts = tm.round_trip_pickle(ts)
    assert new_ts.index.freq == "M"


@pytest.mark.parametrize(
    "name", [777, 777.0, "name", datetime.datetime(2001, 11, 11), (1, 2)]
)
def test_pickle_preserve_name(name):

    unpickled = tm.round_trip_pickle(tm.makeTimeSeries(name=name))
    assert unpickled.name == name


def test_pickle_datetimes(datetime_series):
    unp_ts = tm.round_trip_pickle(datetime_series)
    tm.assert_series_equal(unp_ts, datetime_series)


def test_pickle_strings(string_series):
    unp_series = tm.round_trip_pickle(string_series)
    tm.assert_series_equal(unp_series, string_series)


def test_pickle_preserves_block_ndim():
    # GH#37631
    ser = Series(list("abc")).astype("category").iloc[[0]]
    res = tm.round_trip_pickle(ser)

    assert res._mgr.blocks[0].ndim == 1
    assert res._mgr.blocks[0].shape == (1,)

    # GH#37631 OP issue was about indexing, underlying problem was pickle
    tm.assert_series_equal(res[[True]], ser)


@pytest.mark.parametrize("protocol", [pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL])
def test_pickle_big_dataframe_compression(protocol, compression):
    # GH#39002
    df = pd.DataFrame(range(100000))
    result = tm.round_trip_pathlib(
        partial(df.to_pickle, protocol=protocol, compression=compression),
        partial(pd.read_pickle, compression=compression),
    )
    tm.assert_frame_equal(df, result)


def test_pickle_frame_v124_unpickle_130():
    # GH#42345 DataFrame created in 1.2.x, unpickle in 1.3.x
    path = os.path.join(legacy_dirname, "1.2.4", "empty_frame_v1_2_4-GH#42345.pkl")
    with open(path, "rb") as fd:
        df = pickle.load(fd)

    expected = pd.DataFrame()
    tm.assert_frame_equal(df, expected)
