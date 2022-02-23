import os

import numpy as np
import pytest

from pandas.compat import is_platform_little_endian

from pandas import (
    DataFrame,
    HDFStore,
    Series,
    _testing as tm,
    read_hdf,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_path,
    ensure_clean_store,
    tables,
)

from pandas.io import pytables as pytables
from pandas.io.pytables import (
    ClosedFileError,
    PossibleDataLossError,
    Term,
)

pytestmark = pytest.mark.single


def test_mode(setup_path):

    df = tm.makeTimeDataFrame()

    def check(mode):

        msg = r"[\S]* does not exist"
        with ensure_clean_path(setup_path) as path:

            # constructor
            if mode in ["r", "r+"]:
                with pytest.raises(IOError, match=msg):
                    HDFStore(path, mode=mode)

            else:
                store = HDFStore(path, mode=mode)
                assert store._handle.mode == mode
                store.close()

        with ensure_clean_path(setup_path) as path:

            # context
            if mode in ["r", "r+"]:
                with pytest.raises(IOError, match=msg):
                    with HDFStore(path, mode=mode) as store:
                        pass
            else:
                with HDFStore(path, mode=mode) as store:
                    assert store._handle.mode == mode

        with ensure_clean_path(setup_path) as path:

            # conv write
            if mode in ["r", "r+"]:
                with pytest.raises(IOError, match=msg):
                    df.to_hdf(path, "df", mode=mode)
                df.to_hdf(path, "df", mode="w")
            else:
                df.to_hdf(path, "df", mode=mode)

            # conv read
            if mode in ["w"]:
                msg = (
                    "mode w is not allowed while performing a read. "
                    r"Allowed modes are r, r\+ and a."
                )
                with pytest.raises(ValueError, match=msg):
                    read_hdf(path, "df", mode=mode)
            else:
                result = read_hdf(path, "df", mode=mode)
                tm.assert_frame_equal(result, df)

    def check_default_mode():

        # read_hdf uses default mode
        with ensure_clean_path(setup_path) as path:
            df.to_hdf(path, "df", mode="w")
            result = read_hdf(path, "df")
            tm.assert_frame_equal(result, df)

    check("r")
    check("r+")
    check("a")
    check("w")
    check_default_mode()


def test_reopen_handle(setup_path):

    with ensure_clean_path(setup_path) as path:

        store = HDFStore(path, mode="a")
        store["a"] = tm.makeTimeSeries()

        msg = (
            r"Re-opening the file \[[\S]*\] with mode \[a\] will delete the "
            "current file!"
        )
        # invalid mode change
        with pytest.raises(PossibleDataLossError, match=msg):
            store.open("w")

        store.close()
        assert not store.is_open

        # truncation ok here
        store.open("w")
        assert store.is_open
        assert len(store) == 0
        store.close()
        assert not store.is_open

        store = HDFStore(path, mode="a")
        store["a"] = tm.makeTimeSeries()

        # reopen as read
        store.open("r")
        assert store.is_open
        assert len(store) == 1
        assert store._mode == "r"
        store.close()
        assert not store.is_open

        # reopen as append
        store.open("a")
        assert store.is_open
        assert len(store) == 1
        assert store._mode == "a"
        store.close()
        assert not store.is_open

        # reopen as append (again)
        store.open("a")
        assert store.is_open
        assert len(store) == 1
        assert store._mode == "a"
        store.close()
        assert not store.is_open


def test_open_args(setup_path):

    with tm.ensure_clean(setup_path) as path:

        df = tm.makeDataFrame()

        # create an in memory store
        store = HDFStore(
            path, mode="a", driver="H5FD_CORE", driver_core_backing_store=0
        )
        store["df"] = df
        store.append("df2", df)

        tm.assert_frame_equal(store["df"], df)
        tm.assert_frame_equal(store["df2"], df)

        store.close()

    # the file should not have actually been written
    assert not os.path.exists(path)


def test_flush(setup_path):

    with ensure_clean_store(setup_path) as store:
        store["a"] = tm.makeTimeSeries()
        store.flush()
        store.flush(fsync=True)


def test_complibs_default_settings(setup_path):
    # GH15943
    df = tm.makeDataFrame()

    # Set complevel and check if complib is automatically set to
    # default value
    with ensure_clean_path(setup_path) as tmpfile:
        df.to_hdf(tmpfile, "df", complevel=9)
        result = read_hdf(tmpfile, "df")
        tm.assert_frame_equal(result, df)

        with tables.open_file(tmpfile, mode="r") as h5file:
            for node in h5file.walk_nodes(where="/df", classname="Leaf"):
                assert node.filters.complevel == 9
                assert node.filters.complib == "zlib"

    # Set complib and check to see if compression is disabled
    with ensure_clean_path(setup_path) as tmpfile:
        df.to_hdf(tmpfile, "df", complib="zlib")
        result = read_hdf(tmpfile, "df")
        tm.assert_frame_equal(result, df)

        with tables.open_file(tmpfile, mode="r") as h5file:
            for node in h5file.walk_nodes(where="/df", classname="Leaf"):
                assert node.filters.complevel == 0
                assert node.filters.complib is None

    # Check if not setting complib or complevel results in no compression
    with ensure_clean_path(setup_path) as tmpfile:
        df.to_hdf(tmpfile, "df")
        result = read_hdf(tmpfile, "df")
        tm.assert_frame_equal(result, df)

        with tables.open_file(tmpfile, mode="r") as h5file:
            for node in h5file.walk_nodes(where="/df", classname="Leaf"):
                assert node.filters.complevel == 0
                assert node.filters.complib is None

    # Check if file-defaults can be overridden on a per table basis
    with ensure_clean_path(setup_path) as tmpfile:
        store = HDFStore(tmpfile)
        store.append("dfc", df, complevel=9, complib="blosc")
        store.append("df", df)
        store.close()

        with tables.open_file(tmpfile, mode="r") as h5file:
            for node in h5file.walk_nodes(where="/df", classname="Leaf"):
                assert node.filters.complevel == 0
                assert node.filters.complib is None
            for node in h5file.walk_nodes(where="/dfc", classname="Leaf"):
                assert node.filters.complevel == 9
                assert node.filters.complib == "blosc"


def test_complibs(setup_path):
    # GH14478
    df = tm.makeDataFrame()

    # Building list of all complibs and complevels tuples
    all_complibs = tables.filters.all_complibs
    # Remove lzo if its not available on this platform
    if not tables.which_lib_version("lzo"):
        all_complibs.remove("lzo")
    # Remove bzip2 if its not available on this platform
    if not tables.which_lib_version("bzip2"):
        all_complibs.remove("bzip2")

    all_levels = range(0, 10)
    all_tests = [(lib, lvl) for lib in all_complibs for lvl in all_levels]

    for (lib, lvl) in all_tests:
        with ensure_clean_path(setup_path) as tmpfile:
            gname = "foo"

            # Write and read file to see if data is consistent
            df.to_hdf(tmpfile, gname, complib=lib, complevel=lvl)
            result = read_hdf(tmpfile, gname)
            tm.assert_frame_equal(result, df)

            # Open file and check metadata
            # for correct amount of compression
            h5table = tables.open_file(tmpfile, mode="r")
            for node in h5table.walk_nodes(where="/" + gname, classname="Leaf"):
                assert node.filters.complevel == lvl
                if lvl == 0:
                    assert node.filters.complib is None
                else:
                    assert node.filters.complib == lib
            h5table.close()


@pytest.mark.skipif(
    not is_platform_little_endian(), reason="reason platform is not little endian"
)
def test_encoding(setup_path):

    with ensure_clean_store(setup_path) as store:
        df = DataFrame({"A": "foo", "B": "bar"}, index=range(5))
        df.loc[2, "A"] = np.nan
        df.loc[3, "B"] = np.nan
        _maybe_remove(store, "df")
        store.append("df", df, encoding="ascii")
        tm.assert_frame_equal(store["df"], df)

        expected = df.reindex(columns=["A"])
        result = store.select("df", Term("columns=A", encoding="ascii"))
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "val",
    [
        [b"E\xc9, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"a", b"b", b"c"],
        [b"EE, 17", b"", b"a", b"b", b"c"],
        [b"E\xc9, 17", b"\xf8\xfc", b"a", b"b", b"c"],
        [b"", b"a", b"b", b"c"],
        [b"\xf8\xfc", b"a", b"b", b"c"],
        [b"A\xf8\xfc", b"", b"a", b"b", b"c"],
        [np.nan, b"", b"b", b"c"],
        [b"A\xf8\xfc", np.nan, b"", b"b", b"c"],
    ],
)
@pytest.mark.parametrize("dtype", ["category", object])
def test_latin_encoding(setup_path, dtype, val):
    enc = "latin-1"
    nan_rep = ""
    key = "data"

    val = [x.decode(enc) if isinstance(x, bytes) else x for x in val]
    ser = Series(val, dtype=dtype)

    with ensure_clean_path(setup_path) as store:
        ser.to_hdf(store, key, format="table", encoding=enc, nan_rep=nan_rep)
        retr = read_hdf(store, key)

    s_nan = ser.replace(nan_rep, np.nan)

    tm.assert_series_equal(s_nan, retr)


def test_multiple_open_close(setup_path):
    # gh-4409: open & close multiple times

    with ensure_clean_path(setup_path) as path:

        df = tm.makeDataFrame()
        df.to_hdf(path, "df", mode="w", format="table")

        # single
        store = HDFStore(path)
        assert "CLOSED" not in store.info()
        assert store.is_open

        store.close()
        assert "CLOSED" in store.info()
        assert not store.is_open

    with ensure_clean_path(setup_path) as path:

        if pytables._table_file_open_policy_is_strict:
            # multiples
            store1 = HDFStore(path)
            msg = (
                r"The file [\S]* is already opened\.  Please close it before "
                r"reopening in write mode\."
            )
            with pytest.raises(ValueError, match=msg):
                HDFStore(path)

            store1.close()
        else:

            # multiples
            store1 = HDFStore(path)
            store2 = HDFStore(path)

            assert "CLOSED" not in store1.info()
            assert "CLOSED" not in store2.info()
            assert store1.is_open
            assert store2.is_open

            store1.close()
            assert "CLOSED" in store1.info()
            assert not store1.is_open
            assert "CLOSED" not in store2.info()
            assert store2.is_open

            store2.close()
            assert "CLOSED" in store1.info()
            assert "CLOSED" in store2.info()
            assert not store1.is_open
            assert not store2.is_open

            # nested close
            store = HDFStore(path, mode="w")
            store.append("df", df)

            store2 = HDFStore(path)
            store2.append("df2", df)
            store2.close()
            assert "CLOSED" in store2.info()
            assert not store2.is_open

            store.close()
            assert "CLOSED" in store.info()
            assert not store.is_open

            # double closing
            store = HDFStore(path, mode="w")
            store.append("df", df)

            store2 = HDFStore(path)
            store.close()
            assert "CLOSED" in store.info()
            assert not store.is_open

            store2.close()
            assert "CLOSED" in store2.info()
            assert not store2.is_open

    # ops on a closed store
    with ensure_clean_path(setup_path) as path:

        df = tm.makeDataFrame()
        df.to_hdf(path, "df", mode="w", format="table")

        store = HDFStore(path)
        store.close()

        msg = r"[\S]* file is not open!"
        with pytest.raises(ClosedFileError, match=msg):
            store.keys()

        with pytest.raises(ClosedFileError, match=msg):
            "df" in store

        with pytest.raises(ClosedFileError, match=msg):
            len(store)

        with pytest.raises(ClosedFileError, match=msg):
            store["df"]

        with pytest.raises(ClosedFileError, match=msg):
            store.select("df")

        with pytest.raises(ClosedFileError, match=msg):
            store.get("df")

        with pytest.raises(ClosedFileError, match=msg):
            store.append("df2", df)

        with pytest.raises(ClosedFileError, match=msg):
            store.put("df3", df)

        with pytest.raises(ClosedFileError, match=msg):
            store.get_storer("df2")

        with pytest.raises(ClosedFileError, match=msg):
            store.remove("df2")

        with pytest.raises(ClosedFileError, match=msg):
            store.select("df")

        msg = "'HDFStore' object has no attribute 'df'"
        with pytest.raises(AttributeError, match=msg):
            store.df


def test_fspath():
    with tm.ensure_clean("foo.h5") as path:
        with HDFStore(path) as store:
            assert os.fspath(store) == str(path)
