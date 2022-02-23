"""
Tests for the pandas.io.common functionalities
"""
import codecs
import errno
from functools import partial
from io import (
    BytesIO,
    StringIO,
)
import mmap
import os
from pathlib import Path
import tempfile

import pytest

from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

import pandas.io.common as icom


class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return self.path


# Functions that consume a string path and return a string or path-like object
path_types = [str, CustomFSPath, Path]

try:
    from py.path import local as LocalPath

    path_types.append(LocalPath)
except ImportError:
    pass

HERE = os.path.abspath(os.path.dirname(__file__))


# https://github.com/cython/cython/issues/1720
@pytest.mark.filterwarnings("ignore:can't resolve package:ImportWarning")
class TestCommonIOCapabilities:
    data1 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    def test_expand_user(self):
        filename = "~/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name != filename
        assert os.path.isabs(expanded_name)
        assert os.path.expanduser(filename) == expanded_name

    def test_expand_user_normal_path(self):
        filename = "/somefolder/sometest"
        expanded_name = icom._expand_user(filename)

        assert expanded_name == filename
        assert os.path.expanduser(filename) == expanded_name

    def test_stringify_path_pathlib(self):
        rel_path = icom.stringify_path(Path("."))
        assert rel_path == "."
        redundant_path = icom.stringify_path(Path("foo//bar"))
        assert redundant_path == os.path.join("foo", "bar")

    @td.skip_if_no("py.path")
    def test_stringify_path_localpath(self):
        path = os.path.join("foo", "bar")
        abs_path = os.path.abspath(path)
        lpath = LocalPath(path)
        assert icom.stringify_path(lpath) == abs_path

    def test_stringify_path_fspath(self):
        p = CustomFSPath("foo/bar.csv")
        result = icom.stringify_path(p)
        assert result == "foo/bar.csv"

    def test_stringify_file_and_path_like(self):
        # GH 38125: do not stringify file objects that are also path-like
        fsspec = pytest.importorskip("fsspec")
        with tm.ensure_clean() as path:
            with fsspec.open(f"file://{path}", mode="wb") as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)

    @pytest.mark.parametrize(
        "extension,expected",
        [
            ("", None),
            (".gz", "gzip"),
            (".bz2", "bz2"),
            (".zip", "zip"),
            (".xz", "xz"),
            (".GZ", "gzip"),
            (".BZ2", "bz2"),
            (".ZIP", "zip"),
            (".XZ", "xz"),
        ],
    )
    @pytest.mark.parametrize("path_type", path_types)
    def test_infer_compression_from_path(self, extension, expected, path_type):
        path = path_type("foo/bar.csv" + extension)
        compression = icom.infer_compression(path, compression="infer")
        assert compression == expected

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_get_handle_with_path(self, path_type):
        # ignore LocalPath: it creates strange paths: /absolute/~/sometest
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type("~/" + Path(tmp).name + "/sometest")
            with icom.get_handle(filename, "w") as handles:
                assert Path(handles.handle.name).is_absolute()
                assert os.path.expanduser(filename) == handles.handle.name

    def test_get_handle_with_buffer(self):
        input_buffer = StringIO()
        with icom.get_handle(input_buffer, "r") as handles:
            assert handles.handle == input_buffer
        assert not input_buffer.closed
        input_buffer.close()

    def test_iterator(self):
        with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
            result = pd.concat(reader, ignore_index=True)
        expected = pd.read_csv(StringIO(self.data1))
        tm.assert_frame_equal(result, expected)

        # GH12153
        with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
            first = next(it)
            tm.assert_frame_equal(first, expected.iloc[[0]])
            tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", IOError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", ValueError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        pytest.importorskip(module)

        path = os.path.join(HERE, "data", "does_not_exist." + fn_ext)
        msg1 = fr"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = fr"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Expected object or value"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            fr"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            fr"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = fr"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            fr"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = fr"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=fr"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_table, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", IOError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", ValueError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    def test_read_expands_user_home_dir(
        self, reader, module, error_class, fn_ext, monkeypatch
    ):
        pytest.importorskip(module)

        path = os.path.join("~", "does_not_exist." + fn_ext)
        monkeypatch.setattr(icom, "_expand_user", lambda x: os.path.join("foo", x))

        msg1 = fr"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = fr"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            fr"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            fr"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = fr"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            fr"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = fr"Failed to open local file.+does_not_exist\.{fn_ext}"

        with pytest.raises(
            error_class,
            match=fr"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    @pytest.mark.parametrize(
        "reader, module, path",
        [
            (pd.read_csv, "os", ("io", "data", "csv", "iris.csv")),
            (pd.read_table, "os", ("io", "data", "csv", "iris.csv")),
            (
                pd.read_fwf,
                "os",
                ("io", "data", "fixed_width", "fixed_width_format.txt"),
            ),
            (pd.read_excel, "xlrd", ("io", "data", "excel", "test1.xlsx")),
            (
                pd.read_feather,
                "pyarrow",
                ("io", "data", "feather", "feather-0_3_1.feather"),
            ),
            (
                pd.read_hdf,
                "tables",
                ("io", "data", "legacy_hdf", "datetimetz_object.h5"),
            ),
            (pd.read_stata, "os", ("io", "data", "stata", "stata10_115.dta")),
            (pd.read_sas, "os", ("io", "sas", "data", "test1.sas7bdat")),
            (pd.read_json, "os", ("io", "json", "data", "tsframe_v012.json")),
            (
                pd.read_pickle,
                "os",
                ("io", "data", "pickle", "categorical.0.25.0.pickle"),
            ),
        ],
    )
    @pytest.mark.filterwarnings(
        "ignore:CategoricalBlock is deprecated:DeprecationWarning"
    )
    @pytest.mark.filterwarnings(  # pytables np.object usage
        "ignore:`np.object` is a deprecated alias:DeprecationWarning"
    )
    def test_read_fspath_all(self, reader, module, path, datapath):
        pytest.importorskip(module)
        path = datapath(*path)

        mypath = CustomFSPath(path)
        result = reader(mypath)
        expected = reader(path)

        if path.endswith(".pickle"):
            # categorical
            tm.assert_categorical_equal(result, expected)
        else:
            tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "writer_name, writer_kwargs, module",
        [
            ("to_csv", {}, "os"),
            ("to_excel", {"engine": "xlwt"}, "xlwt"),
            ("to_feather", {}, "pyarrow"),
            ("to_html", {}, "os"),
            ("to_json", {}, "os"),
            ("to_latex", {}, "os"),
            ("to_pickle", {}, "os"),
            ("to_stata", {"time_stamp": pd.to_datetime("2019-01-01 00:00")}, "os"),
        ],
    )
    def test_write_fspath_all(self, writer_name, writer_kwargs, module):
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")
        df = pd.DataFrame({"A": [1, 2]})

        with p1 as string, p2 as fspath:
            pytest.importorskip(module)
            mypath = CustomFSPath(fspath)
            writer = getattr(df, writer_name)

            writer(string, **writer_kwargs)
            with open(string, "rb") as f:
                expected = f.read()

            writer(mypath, **writer_kwargs)
            with open(fspath, "rb") as f:
                result = f.read()

            assert result == expected

    @pytest.mark.filterwarnings(  # pytables np.object usage
        "ignore:`np.object` is a deprecated alias:DeprecationWarning"
    )
    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) IO HDF5
    def test_write_fspath_hdf5(self):
        # Same test as write_fspath_all, except HDF5 files aren't
        # necessarily byte-for-byte identical for a given dataframe, so we'll
        # have to read and compare equality
        pytest.importorskip("tables")

        df = pd.DataFrame({"A": [1, 2]})
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")

        with p1 as string, p2 as fspath:
            mypath = CustomFSPath(fspath)
            df.to_hdf(mypath, key="bar")
            df.to_hdf(string, key="bar")

            result = pd.read_hdf(fspath, key="bar")
            expected = pd.read_hdf(string, key="bar")

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def mmap_file(datapath):
    return datapath("io", "data", "csv", "test_mmap.csv")


class TestMMapWrapper:
    def test_constructor_bad_file(self, mmap_file):
        non_file = StringIO("I am not a file")
        non_file.fileno = lambda: -1

        # the error raised is different on Windows
        if is_platform_windows():
            msg = "The parameter is incorrect"
            err = OSError
        else:
            msg = "[Errno 22]"
            err = mmap.error

        with pytest.raises(err, match=msg):
            icom._MMapWrapper(non_file)

        target = open(mmap_file)
        target.close()

        msg = "I/O operation on closed file"
        with pytest.raises(ValueError, match=msg):
            icom._MMapWrapper(target)

    def test_get_attr(self, mmap_file):
        with open(mmap_file) as target:
            wrapper = icom._MMapWrapper(target)

        attrs = dir(wrapper.mmap)
        attrs = [attr for attr in attrs if not attr.startswith("__")]
        attrs.append("__next__")

        for attr in attrs:
            assert hasattr(wrapper, attr)

        assert not hasattr(wrapper, "foo")

    def test_next(self, mmap_file):
        with open(mmap_file) as target:
            wrapper = icom._MMapWrapper(target)
            lines = target.readlines()

        for line in lines:
            next_line = next(wrapper)
            assert next_line.strip() == line.strip()

        with pytest.raises(StopIteration, match=r"^$"):
            next(wrapper)

    def test_unknown_engine(self):
        with tm.ensure_clean() as path:
            df = tm.makeDataFrame()
            df.to_csv(path)
            with pytest.raises(ValueError, match="Unknown engine"):
                pd.read_csv(path, engine="pyt")

    def test_binary_mode(self):
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        with tm.ensure_clean() as path:
            df = tm.makeDataFrame()
            df.to_csv(path, mode="w+b")
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
    @pytest.mark.parametrize("compression_", ["bz2", "xz"])
    def test_warning_missing_utf_bom(self, encoding, compression_):
        """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
        df = tm.makeDataFrame()
        with tm.ensure_clean() as path:
            with tm.assert_produces_warning(UnicodeWarning):
                df.to_csv(path, compression=compression_, encoding=encoding)

            # reading should fail (otherwise we wouldn't need the warning)
            msg = r"UTF-\d+ stream does not start with BOM"
            with pytest.raises(UnicodeError, match=msg):
                pd.read_csv(path, compression=compression_, encoding=encoding)


def test_is_fsspec_url():
    assert icom.is_fsspec_url("gcs://pandas/somethingelse.com")
    assert icom.is_fsspec_url("gs://pandas/somethingelse.com")
    # the following is the only remote URL that is handled without fsspec
    assert not icom.is_fsspec_url("http://pandas/somethingelse.com")
    assert not icom.is_fsspec_url("random:pandas/somethingelse.com")
    assert not icom.is_fsspec_url("/local/path")
    assert not icom.is_fsspec_url("relative/local/path")


@pytest.mark.parametrize("encoding", [None, "utf-8"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_codecs_encoding(encoding, format):
    # GH39247
    expected = tm.makeDataFrame()
    with tm.ensure_clean() as path:
        with codecs.open(path, mode="w", encoding=encoding) as handle:
            getattr(expected, f"to_{format}")(handle)
        with codecs.open(path, mode="r", encoding=encoding) as handle:
            if format == "csv":
                df = pd.read_csv(handle, index_col=0)
            else:
                df = pd.read_json(handle)
    tm.assert_frame_equal(expected, df)


def test_codecs_get_writer_reader():
    # GH39247
    expected = tm.makeDataFrame()
    with tm.ensure_clean() as path:
        with open(path, "wb") as handle:
            with codecs.getwriter("utf-8")(handle) as encoded:
                expected.to_csv(encoded)
        with open(path, "rb") as handle:
            with codecs.getreader("utf-8")(handle) as encoded:
                df = pd.read_csv(encoded, index_col=0)
    tm.assert_frame_equal(expected, df)


@pytest.mark.parametrize(
    "io_class,mode,msg",
    [
        (BytesIO, "t", "a bytes-like object is required, not 'str'"),
        (StringIO, "b", "string argument expected, got 'bytes'"),
    ],
)
def test_explicit_encoding(io_class, mode, msg):
    # GH39247; this test makes sure that if a user provides mode="*t" or "*b",
    # it is used. In the case of this test it leads to an error as intentionally the
    # wrong mode is requested
    expected = tm.makeDataFrame()
    with io_class() as buffer:
        with pytest.raises(TypeError, match=msg):
            expected.to_csv(buffer, mode=f"w{mode}")


@pytest.mark.parametrize("encoding_errors", [None, "strict", "replace"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_encoding_errors(encoding_errors, format):
    # GH39450
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b"\xe4"

    if format == "csv":
        return
        content = bad_encoding + b"\n" + bad_encoding
        reader = pd.read_csv
    else:
        content = (
            b'{"'
            + bad_encoding * 2
            + b'": {"'
            + bad_encoding
            + b'":"'
            + bad_encoding
            + b'"}}'
        )
        reader = partial(pd.read_json, orient="index")
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(content)

        if encoding_errors != "replace":
            with pytest.raises(UnicodeDecodeError, match=msg):
                reader(path, encoding_errors=encoding_errors)
        else:
            df = reader(path, encoding_errors=encoding_errors)
            decoded = bad_encoding.decode(errors=encoding_errors)
            expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
            tm.assert_frame_equal(df, expected)


def test_bad_encdoing_errors():
    # GH 39777
    with tm.ensure_clean() as path:
        with pytest.raises(ValueError, match="Invalid value for `encoding_errors`"):
            icom.get_handle(path, "w", errors="bad")


def test_errno_attribute():
    # GH 13872
    with pytest.raises(FileNotFoundError, match="\\[Errno 2\\]") as err:
        pd.read_csv("doesnt_exist")
        assert err.errno == errno.ENOENT
