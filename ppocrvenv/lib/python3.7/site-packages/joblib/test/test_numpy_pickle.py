"""Test the numpy pickler as a replacement of the standard pickler."""

import copy
import os
import random
import re
import io
import sys
import warnings
import gzip
import zlib
import bz2
import pickle
import socket
from contextlib import closing
import mmap
try:
    import lzma
except ImportError:
    lzma = None
import pytest

from joblib.test.common import np, with_numpy, with_lz4, without_lz4
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import parametrize, raises, SkipTest, warns

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle, register_compressor
from joblib.test import data

from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE
from joblib.numpy_pickle_utils import _detect_compressor
from joblib.numpy_pickle_utils import _is_numpy_array_byte_order_mismatch
from joblib.numpy_pickle_utils import _ensure_native_byte_order
from joblib.compressor import (_COMPRESSORS, _LZ4_PREFIX, CompressorWrapper,
                               LZ4_NOT_INSTALLED_ERROR, BinaryZlibFile)


###############################################################################
# Define a list of standard types.
# Borrowed from dill, initial author: Micheal McKerns:
# http://dev.danse.us/trac/pathos/browser/dill/dill_test2.py

typelist = []

# testing types
_none = None
typelist.append(_none)
_type = type
typelist.append(_type)
_bool = bool(1)
typelist.append(_bool)
_int = int(1)
typelist.append(_int)
_float = float(1)
typelist.append(_float)
_complex = complex(1)
typelist.append(_complex)
_string = str(1)
typelist.append(_string)
_tuple = ()
typelist.append(_tuple)
_list = []
typelist.append(_list)
_dict = {}
typelist.append(_dict)
_builtin = len
typelist.append(_builtin)


def _function(x):
    yield x


class _class:
    def _method(self):
        pass


class _newclass(object):
    def _method(self):
        pass


typelist.append(_function)
typelist.append(_class)
typelist.append(_newclass)  # <type 'type'>
_instance = _class()
typelist.append(_instance)
_object = _newclass()
typelist.append(_object)  # <type 'class'>


###############################################################################
# Tests

@parametrize('compress', [0, 1])
@parametrize('member', typelist)
def test_standard_types(tmpdir, compress, member):
    # Test pickling and saving with standard types.
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(member, filename, compress=compress)
    _member = numpy_pickle.load(filename)
    # We compare the pickled instance to the reloaded one only if it
    # can be compared to a copied one
    if member == copy.deepcopy(member):
        assert member == _member


def test_value_error():
    # Test inverting the input arguments to dump
    with raises(ValueError):
        numpy_pickle.dump('foo', dict())


@parametrize('wrong_compress', [-1, 10, dict()])
def test_compress_level_error(wrong_compress):
    # Verify that passing an invalid compress argument raises an error.
    exception_msg = ('Non valid compress level given: '
                     '"{0}"'.format(wrong_compress))
    with raises(ValueError) as excinfo:
        numpy_pickle.dump('dummy', 'foo', compress=wrong_compress)
    excinfo.match(exception_msg)


@with_numpy
@parametrize('compress', [False, True, 0, 3, 'zlib'])
def test_numpy_persistence(tmpdir, compress):
    filename = tmpdir.join('test.pkl').strpath
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    # We use 'a.T' to have a non C-contiguous array.
    for index, obj in enumerate(((a,), (a.T,), (a, a), [a, a, a])):
        filenames = numpy_pickle.dump(obj, filename, compress=compress)

        # All is cached in one file
        assert len(filenames) == 1
        # Check that only one file was created
        assert filenames[0] == filename
        # Check that this file does exist
        assert os.path.exists(filenames[0])

        # Unpickle the object
        obj_ = numpy_pickle.load(filename)
        # Check that the items are indeed arrays
        for item in obj_:
            assert isinstance(item, np.ndarray)
        # And finally, check that all the values are equal.
        np.testing.assert_array_equal(np.array(obj), np.array(obj_))

    # Now test with array subclasses
    for obj in (np.matrix(np.zeros(10)),
                np.memmap(filename + 'mmap',
                          mode='w+', shape=4, dtype=np.float)):
        filenames = numpy_pickle.dump(obj, filename, compress=compress)
        # All is cached in one file
        assert len(filenames) == 1

        obj_ = numpy_pickle.load(filename)
        if (type(obj) is not np.memmap and
                hasattr(obj, '__array_prepare__')):
            # We don't reconstruct memmaps
            assert isinstance(obj_, type(obj))

        np.testing.assert_array_equal(obj_, obj)

    # Test with an object containing multiple numpy arrays
    obj = ComplexTestObject()
    filenames = numpy_pickle.dump(obj, filename, compress=compress)
    # All is cached in one file
    assert len(filenames) == 1

    obj_loaded = numpy_pickle.load(filename)
    assert isinstance(obj_loaded, type(obj))
    np.testing.assert_array_equal(obj_loaded.array_float, obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int, obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj, obj.array_obj)


@with_numpy
def test_numpy_persistence_bufferred_array_compression(tmpdir):
    big_array = np.ones((_IO_BUFFER_SIZE + 100), dtype=np.uint8)
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(big_array, filename, compress=True)
    arr_reloaded = numpy_pickle.load(filename)

    np.testing.assert_array_equal(big_array, arr_reloaded)


@with_numpy
def test_memmap_persistence(tmpdir):
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    filename = tmpdir.join('test1.pkl').strpath
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')

    assert isinstance(b, np.memmap)

    # Test with an object containing multiple numpy arrays
    filename = tmpdir.join('test2.pkl').strpath
    obj = ComplexTestObject()
    numpy_pickle.dump(obj, filename)
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r')
    assert isinstance(obj_loaded, type(obj))
    assert isinstance(obj_loaded.array_float, np.memmap)
    assert not obj_loaded.array_float.flags.writeable
    assert isinstance(obj_loaded.array_int, np.memmap)
    assert not obj_loaded.array_int.flags.writeable
    # Memory map not allowed for numpy object arrays
    assert not isinstance(obj_loaded.array_obj, np.memmap)
    np.testing.assert_array_equal(obj_loaded.array_float,
                                  obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int,
                                  obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj,
                                  obj.array_obj)

    # Test we can write in memmapped arrays
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r+')
    assert obj_loaded.array_float.flags.writeable
    obj_loaded.array_float[0:10] = 10.0
    assert obj_loaded.array_int.flags.writeable
    obj_loaded.array_int[0:10] = 10

    obj_reloaded = numpy_pickle.load(filename, mmap_mode='r')
    np.testing.assert_array_equal(obj_reloaded.array_float,
                                  obj_loaded.array_float)
    np.testing.assert_array_equal(obj_reloaded.array_int,
                                  obj_loaded.array_int)

    # Test w+ mode is caught and the mode has switched to r+
    numpy_pickle.load(filename, mmap_mode='w+')
    assert obj_loaded.array_int.flags.writeable
    assert obj_loaded.array_int.mode == 'r+'
    assert obj_loaded.array_float.flags.writeable
    assert obj_loaded.array_float.mode == 'r+'


@with_numpy
def test_memmap_persistence_mixed_dtypes(tmpdir):
    # loading datastructures that have sub-arrays with dtype=object
    # should not prevent memmapping on fixed size dtype sub-arrays.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    b = np.array([1, 'b'], dtype=object)
    construct = (a, b)
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(construct, filename)
    a_clone, b_clone = numpy_pickle.load(filename, mmap_mode='r')

    # the floating point array has been memory mapped
    assert isinstance(a_clone, np.memmap)

    # the object-dtype array has been loaded in memory
    assert not isinstance(b_clone, np.memmap)


@with_numpy
def test_masked_array_persistence(tmpdir):
    # The special-case picker fails, because saving masked_array
    # not implemented, but it just delegates to the standard pickler.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    a = np.ma.masked_greater(a, 0.5)
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')
    assert isinstance(b, np.ma.masked_array)


@with_numpy
def test_compress_mmap_mode_warning(tmpdir):
    # Test the warning in case of compress + mmap_mode
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    this_filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(a, this_filename, compress=1)
    with warns(UserWarning) as warninfo:
        numpy_pickle.load(this_filename, mmap_mode='r+')
    assert len(warninfo) == 1
    assert (str(warninfo[0].message) ==
            'mmap_mode "%(mmap_mode)s" is not compatible with compressed '
            'file %(filename)s. "%(mmap_mode)s" flag will be ignored.' %
            {'filename': this_filename, 'mmap_mode': 'r+'})


@with_numpy
@parametrize('cache_size', [None, 0, 10])
def test_cache_size_warning(tmpdir, cache_size):
    # Check deprecation warning raised when cache size is not None
    filename = tmpdir.join('test.pkl').strpath
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))

    warnings.simplefilter("always")
    with warns(None) as warninfo:
        numpy_pickle.dump(a, filename, cache_size=cache_size)
    expected_nb_warnings = 1 if cache_size is not None else 0
    assert len(warninfo) == expected_nb_warnings
    for w in warninfo:
        assert w.category == DeprecationWarning
        assert (str(w.message) ==
                "Please do not set 'cache_size' in joblib.dump, this "
                "parameter has no effect and will be removed. You "
                "used 'cache_size={0}'".format(cache_size))


@with_numpy
@with_memory_profiler
@parametrize('compress', [True, False])
def test_memory_usage(tmpdir, compress):
    # Verify memory stays within expected bounds.
    filename = tmpdir.join('test.pkl').strpath
    small_array = np.ones((10, 10))
    big_array = np.ones(shape=100 * int(1e6), dtype=np.uint8)
    small_matrix = np.matrix(small_array)
    big_matrix = np.matrix(big_array)

    for obj in (small_array, big_array, small_matrix, big_matrix):
        size = obj.nbytes / 1e6
        obj_filename = filename + str(np.random.randint(0, 1000))
        mem_used = memory_used(numpy_pickle.dump,
                               obj, obj_filename, compress=compress)

        # The memory used to dump the object shouldn't exceed the buffer
        # size used to write array chunks (16MB).
        write_buf_size = _IO_BUFFER_SIZE + 16 * 1024 ** 2 / 1e6
        assert mem_used <= write_buf_size

        mem_used = memory_used(numpy_pickle.load, obj_filename)
        # memory used should be less than array size + buffer size used to
        # read the array chunk by chunk.
        read_buf_size = 32 + _IO_BUFFER_SIZE  # MiB
        assert mem_used < size + read_buf_size


@with_numpy
def test_compressed_pickle_dump_and_load(tmpdir):
    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('>i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.arange(5, dtype=np.dtype('>f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     np.arange(256, dtype=np.uint8).tobytes(),
                     # np.matrix is a subclass of np.ndarray, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2], dtype=np.dtype('<i8')),
                     np.matrix([0, 1, 2], dtype=np.dtype('>i8')),
                     u"C'est l'\xe9t\xe9 !"]

    fname = tmpdir.join('temp.pkl.gz').strpath

    dumped_filenames = numpy_pickle.dump(expected_list, fname, compress=1)
    assert len(dumped_filenames) == 1
    result_list = numpy_pickle.load(fname)
    for result, expected in zip(result_list, expected_list):
        if isinstance(expected, np.ndarray):
            expected = _ensure_native_byte_order(expected)
            assert result.dtype == expected.dtype
            np.testing.assert_equal(result, expected)
        else:
            assert result == expected


def _check_pickle(filename, expected_list):
    """Helper function to test joblib pickle content.

    Note: currently only pickles containing an iterable are supported
    by this function.
    """
    version_match = re.match(r'.+py(\d)(\d).+', filename)
    py_version_used_for_writing = int(version_match.group(1))

    py_version_to_default_pickle_protocol = {2: 2, 3: 3}
    pickle_reading_protocol = py_version_to_default_pickle_protocol.get(3, 4)
    pickle_writing_protocol = py_version_to_default_pickle_protocol.get(
        py_version_used_for_writing, 4)
    if pickle_reading_protocol >= pickle_writing_protocol:
        try:
            with warns(None) as warninfo:
                warnings.simplefilter('always')
                warnings.filterwarnings(
                    'ignore', module='numpy',
                    message='The compiler package is deprecated')
                result_list = numpy_pickle.load(filename)
            filename_base = os.path.basename(filename)
            expected_nb_warnings = 1 if ("_0.9" in filename_base or
                                         "_0.8.4" in filename_base) else 0
            assert len(warninfo) == expected_nb_warnings
            for w in warninfo:
                assert w.category == DeprecationWarning
                assert (str(w.message) ==
                        "The file '{0}' has been generated with a joblib "
                        "version less than 0.10. Please regenerate this "
                        "pickle file.".format(filename))
            for result, expected in zip(result_list, expected_list):
                if isinstance(expected, np.ndarray):
                    expected = _ensure_native_byte_order(expected)
                    assert result.dtype == expected.dtype
                    np.testing.assert_equal(result, expected)
                else:
                    assert result == expected
        except Exception as exc:
            # When trying to read with python 3 a pickle generated
            # with python 2 we expect a user-friendly error
            if py_version_used_for_writing == 2:
                assert isinstance(exc, ValueError)
                message = ('You may be trying to read with '
                           'python 3 a joblib pickle generated with python 2.')
                assert message in str(exc)
            elif filename.endswith('.lz4') and with_lz4.args[0]:
                assert isinstance(exc, ValueError)
                assert LZ4_NOT_INSTALLED_ERROR in str(exc)
            else:
                raise
    else:
        # Pickle protocol used for writing is too high. We expect a
        # "unsupported pickle protocol" error message
        try:
            numpy_pickle.load(filename)
            raise AssertionError('Numpy pickle loading should '
                                 'have raised a ValueError exception')
        except ValueError as e:
            message = 'unsupported pickle protocol: {0}'.format(
                pickle_writing_protocol)
            assert message in str(e.args)


@with_numpy
def test_joblib_pickle_across_python_versions():
    # We need to be specific about dtypes in particular endianness
    # because the pickles can be generated on one architecture and
    # the tests run on another one. See
    # https://github.com/joblib/joblib/issues/279.
    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     np.arange(256, dtype=np.uint8).tobytes(),
                     # np.matrix is a subclass of np.ndarray, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2], dtype=np.dtype('<i8')),
                     u"C'est l'\xe9t\xe9 !"]

    # Testing all the compressed and non compressed
    # pickles in joblib/test/data. These pickles were generated by
    # the joblib/test/data/create_numpy_pickle.py script for the
    # relevant python, joblib and numpy versions.
    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))

    pickle_extensions = ('.pkl', '.gz', '.gzip', '.bz2', 'lz4')
    if lzma is not None:
        pickle_extensions += ('.xz', '.lzma')
    pickle_filenames = [os.path.join(test_data_dir, fn)
                        for fn in os.listdir(test_data_dir)
                        if any(fn.endswith(ext) for ext in pickle_extensions)]

    for fname in pickle_filenames:
        _check_pickle(fname, expected_list)


@with_numpy
def test_numpy_array_byte_order_mismatch_detection():
    # List of numpy arrays with big endian byteorder.
    be_arrays = [np.array([(1, 2.0), (3, 4.0)],
                          dtype=[('', '>i8'), ('', '>f8')]),
                 np.arange(3, dtype=np.dtype('>i8')),
                 np.arange(3, dtype=np.dtype('>f8'))]

    # Verify the byteorder mismatch is correctly detected.
    for array in be_arrays:
        if sys.byteorder == 'big':
            assert not _is_numpy_array_byte_order_mismatch(array)
        else:
            assert _is_numpy_array_byte_order_mismatch(array)
        converted = _ensure_native_byte_order(array)
        if converted.dtype.fields:
            for f in converted.dtype.fields.values():
                f[0].byteorder == '='
        else:
            assert converted.dtype.byteorder == "="

    # List of numpy arrays with little endian byteorder.
    le_arrays = [np.array([(1, 2.0), (3, 4.0)],
                          dtype=[('', '<i8'), ('', '<f8')]),
                 np.arange(3, dtype=np.dtype('<i8')),
                 np.arange(3, dtype=np.dtype('<f8'))]

    # Verify the byteorder mismatch is correctly detected.
    for array in le_arrays:
        if sys.byteorder == 'little':
            assert not _is_numpy_array_byte_order_mismatch(array)
        else:
            assert _is_numpy_array_byte_order_mismatch(array)
        converted = _ensure_native_byte_order(array)
        if converted.dtype.fields:
            for f in converted.dtype.fields.values():
                f[0].byteorder == '='
        else:
            assert converted.dtype.byteorder == "="


@parametrize('compress_tuple', [('zlib', 3), ('gzip', 3)])
def test_compress_tuple_argument(tmpdir, compress_tuple):
    # Verify the tuple is correctly taken into account.
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump("dummy", filename,
                      compress=compress_tuple)
    # Verify the file contains the right magic number
    with open(filename, 'rb') as f:
        assert _detect_compressor(f) == compress_tuple[0]


@parametrize('compress_tuple,message',
             [(('zlib', 3, 'extra'),        # wrong compress tuple
               'Compress argument tuple should contain exactly 2 elements'),
              (('wrong', 3),                # wrong compress method
               'Non valid compression method given: "{}"'.format('wrong')),
              (('zlib', 'wrong'),           # wrong compress level
               'Non valid compress level given: "{}"'.format('wrong'))])
def test_compress_tuple_argument_exception(tmpdir, compress_tuple, message):
    filename = tmpdir.join('test.pkl').strpath
    # Verify setting a wrong compress tuple raises a ValueError.
    with raises(ValueError) as excinfo:
        numpy_pickle.dump('dummy', filename, compress=compress_tuple)
    excinfo.match(message)


@parametrize('compress_string', ['zlib', 'gzip'])
def test_compress_string_argument(tmpdir, compress_string):
    # Verify the string is correctly taken into account.
    filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump("dummy", filename,
                      compress=compress_string)
    # Verify the file contains the right magic number
    with open(filename, 'rb') as f:
        assert _detect_compressor(f) == compress_string


@with_numpy
@parametrize('compress', [1, 3, 6])
@parametrize('cmethod', _COMPRESSORS)
def test_joblib_compression_formats(tmpdir, compress, cmethod):
    filename = tmpdir.join('test.pkl').strpath
    objects = (np.ones(shape=(100, 100), dtype='f8'),
               range(10),
               {'a': 1, 2: 'b'}, [], (), {}, 0, 1.0)

    if cmethod in ("lzma", "xz") and lzma is None:
        pytest.skip("lzma is support not available")

    elif cmethod == 'lz4' and with_lz4.args[0]:
        # Skip the test if lz4 is not installed. We here use the with_lz4
        # skipif fixture whose argument is True when lz4 is not installed
        pytest.skip("lz4 is not installed.")

    dump_filename = filename + "." + cmethod
    for obj in objects:
        numpy_pickle.dump(obj, dump_filename, compress=(cmethod, compress))
        # Verify the file contains the right magic number
        with open(dump_filename, 'rb') as f:
            assert _detect_compressor(f) == cmethod
        # Verify the reloaded object is correct
        obj_reloaded = numpy_pickle.load(dump_filename)
        assert isinstance(obj_reloaded, type(obj))
        if isinstance(obj, np.ndarray):
            np.testing.assert_array_equal(obj_reloaded, obj)
        else:
            assert obj_reloaded == obj


def _gzip_file_decompress(source_filename, target_filename):
    """Decompress a gzip file."""
    with closing(gzip.GzipFile(source_filename, "rb")) as fo:
        buf = fo.read()

    with open(target_filename, "wb") as fo:
        fo.write(buf)


def _zlib_file_decompress(source_filename, target_filename):
    """Decompress a zlib file."""
    with open(source_filename, 'rb') as fo:
        buf = zlib.decompress(fo.read())

    with open(target_filename, 'wb') as fo:
        fo.write(buf)


@parametrize('extension,decompress',
             [('.z', _zlib_file_decompress),
              ('.gz', _gzip_file_decompress)])
def test_load_externally_decompressed_files(tmpdir, extension, decompress):
    # Test that BinaryZlibFile generates valid gzip and zlib compressed files.
    obj = "a string to persist"
    filename_raw = tmpdir.join('test.pkl').strpath

    filename_compressed = filename_raw + extension
    # Use automatic extension detection to compress with the right method.
    numpy_pickle.dump(obj, filename_compressed)

    # Decompress with the corresponding method
    decompress(filename_compressed, filename_raw)

    # Test that the uncompressed pickle can be loaded and
    # that the result is correct.
    obj_reloaded = numpy_pickle.load(filename_raw)
    assert obj == obj_reloaded


@parametrize('extension,cmethod',
             # valid compressor extensions
             [('.z', 'zlib'),
              ('.gz', 'gzip'),
              ('.bz2', 'bz2'),
              ('.lzma', 'lzma'),
              ('.xz', 'xz'),
              # invalid compressor extensions
              ('.pkl', 'not-compressed'),
              ('', 'not-compressed')])
def test_compression_using_file_extension(tmpdir, extension, cmethod):
    if cmethod in ("lzma", "xz") and lzma is None:
        pytest.skip("lzma is missing")
    # test that compression method corresponds to the given filename extension.
    filename = tmpdir.join('test.pkl').strpath
    obj = "object to dump"

    dump_fname = filename + extension
    numpy_pickle.dump(obj, dump_fname)
    # Verify the file contains the right magic number
    with open(dump_fname, 'rb') as f:
        assert _detect_compressor(f) == cmethod
    # Verify the reloaded object is correct
    obj_reloaded = numpy_pickle.load(dump_fname)
    assert isinstance(obj_reloaded, type(obj))
    assert obj_reloaded == obj


@with_numpy
def test_file_handle_persistence(tmpdir):
    objs = [np.random.random((10, 10)),
            "some data",
            np.matrix([0, 1, 2])]
    fobjs = [bz2.BZ2File, gzip.GzipFile]
    if lzma is not None:
        fobjs += [lzma.LZMAFile]
    filename = tmpdir.join('test.pkl').strpath

    for obj in objs:
        for fobj in fobjs:
            with fobj(filename, 'wb') as f:
                numpy_pickle.dump(obj, f)

            # using the same decompressor prevents from internally
            # decompress again.
            with fobj(filename, 'rb') as f:
                obj_reloaded = numpy_pickle.load(f)

            # when needed, the correct decompressor should be used when
            # passing a raw file handle.
            with open(filename, 'rb') as f:
                obj_reloaded_2 = numpy_pickle.load(f)

            if isinstance(obj, np.ndarray):
                np.testing.assert_array_equal(obj_reloaded, obj)
                np.testing.assert_array_equal(obj_reloaded_2, obj)
            else:
                assert obj_reloaded == obj
                assert obj_reloaded_2 == obj


@with_numpy
def test_in_memory_persistence():
    objs = [np.random.random((10, 10)),
            "some data",
            np.matrix([0, 1, 2])]
    for obj in objs:
        f = io.BytesIO()
        numpy_pickle.dump(obj, f)
        obj_reloaded = numpy_pickle.load(f)
        if isinstance(obj, np.ndarray):
            np.testing.assert_array_equal(obj_reloaded, obj)
        else:
            assert obj_reloaded == obj


@with_numpy
def test_file_handle_persistence_mmap(tmpdir):
    obj = np.random.random((10, 10))
    filename = tmpdir.join('test.pkl').strpath

    with open(filename, 'wb') as f:
        numpy_pickle.dump(obj, f)

    with open(filename, 'rb') as f:
        obj_reloaded = numpy_pickle.load(f, mmap_mode='r+')

    np.testing.assert_array_equal(obj_reloaded, obj)


@with_numpy
def test_file_handle_persistence_compressed_mmap(tmpdir):
    obj = np.random.random((10, 10))
    filename = tmpdir.join('test.pkl').strpath

    with open(filename, 'wb') as f:
        numpy_pickle.dump(obj, f, compress=('gzip', 3))

    with closing(gzip.GzipFile(filename, 'rb')) as f:
        with warns(UserWarning) as warninfo:
            numpy_pickle.load(f, mmap_mode='r+')
        assert len(warninfo) == 1
        assert (str(warninfo[0].message) ==
                '"%(fileobj)r" is not a raw file, mmap_mode "%(mmap_mode)s" '
                'flag will be ignored.' % {'fileobj': f, 'mmap_mode': 'r+'})


@with_numpy
def test_file_handle_persistence_in_memory_mmap():
    obj = np.random.random((10, 10))
    buf = io.BytesIO()

    numpy_pickle.dump(obj, buf)

    with warns(UserWarning) as warninfo:
        numpy_pickle.load(buf, mmap_mode='r+')
    assert len(warninfo) == 1
    assert (str(warninfo[0].message) ==
            'In memory persistence is not compatible with mmap_mode '
            '"%(mmap_mode)s" flag passed. mmap_mode option will be '
            'ignored.' % {'mmap_mode': 'r+'})


@parametrize('data', [b'a little data as bytes.',
                      # More bytes
                      10000 * "{}".format(
                          random.randint(0, 1000) * 1000).encode('latin-1')],
             ids=["a little data as bytes.", "a large data as bytes."])
@parametrize('compress_level', [1, 3, 9])
def test_binary_zlibfile(tmpdir, data, compress_level):
    filename = tmpdir.join('test.pkl').strpath
    # Regular cases
    with open(filename, 'wb') as f:
        with BinaryZlibFile(f, 'wb',
                            compresslevel=compress_level) as fz:
            assert fz.writable()
            fz.write(data)
            assert fz.fileno() == f.fileno()
            with raises(io.UnsupportedOperation):
                fz._check_can_read()

            with raises(io.UnsupportedOperation):
                fz._check_can_seek()
        assert fz.closed
        with raises(ValueError):
            fz._check_not_closed()

    with open(filename, 'rb') as f:
        with BinaryZlibFile(f) as fz:
            assert fz.readable()
            assert fz.seekable()
            assert fz.fileno() == f.fileno()
            assert fz.read() == data
            with raises(io.UnsupportedOperation):
                fz._check_can_write()
            assert fz.seekable()
            fz.seek(0)
            assert fz.tell() == 0
        assert fz.closed

    # Test with a filename as input
    with BinaryZlibFile(filename, 'wb',
                        compresslevel=compress_level) as fz:
        assert fz.writable()
        fz.write(data)

    with BinaryZlibFile(filename, 'rb') as fz:
        assert fz.read() == data
        assert fz.seekable()

    # Test without context manager
    fz = BinaryZlibFile(filename, 'wb', compresslevel=compress_level)
    assert fz.writable()
    fz.write(data)
    fz.close()

    fz = BinaryZlibFile(filename, 'rb')
    assert fz.read() == data
    fz.close()


@parametrize('bad_value', [-1, 10, 15, 'a', (), {}])
def test_binary_zlibfile_bad_compression_levels(tmpdir, bad_value):
    filename = tmpdir.join('test.pkl').strpath
    with raises(ValueError) as excinfo:
        BinaryZlibFile(filename, 'wb', compresslevel=bad_value)
    pattern = re.escape("'compresslevel' must be an integer between 1 and 9. "
                        "You provided 'compresslevel={}'".format(bad_value))
    excinfo.match(pattern)


@parametrize('bad_mode', ['a', 'x', 'r', 'w', 1, 2])
def test_binary_zlibfile_invalid_modes(tmpdir, bad_mode):
    filename = tmpdir.join('test.pkl').strpath
    with raises(ValueError) as excinfo:
        BinaryZlibFile(filename, bad_mode)
    excinfo.match("Invalid mode")


@parametrize('bad_file', [1, (), {}])
def test_binary_zlibfile_invalid_filename_type(bad_file):
    with raises(TypeError) as excinfo:
        BinaryZlibFile(bad_file, 'rb')
    excinfo.match("filename must be a str or bytes object, or a file")


###############################################################################
# Test dumping array subclasses
if np is not None:

    class SubArray(np.ndarray):

        def __reduce__(self):
            return _load_sub_array, (np.asarray(self), )

    def _load_sub_array(arr):
        d = SubArray(arr.shape)
        d[:] = arr
        return d

    class ComplexTestObject:
        """A complex object containing numpy arrays as attributes."""

        def __init__(self):
            self.array_float = np.arange(100, dtype='float64')
            self.array_int = np.ones(100, dtype='int32')
            self.array_obj = np.array(['a', 10, 20.0], dtype='object')


@with_numpy
def test_numpy_subclass(tmpdir):
    filename = tmpdir.join('test.pkl').strpath
    a = SubArray((10,))
    numpy_pickle.dump(a, filename)
    c = numpy_pickle.load(filename)
    assert isinstance(c, SubArray)
    np.testing.assert_array_equal(c, a)


def test_pathlib(tmpdir):
    try:
        from pathlib import Path
    except ImportError:
        pass
    else:
        filename = tmpdir.join('test.pkl').strpath
        value = 123
        numpy_pickle.dump(value, Path(filename))
        assert numpy_pickle.load(filename) == value
        numpy_pickle.dump(value, filename)
        assert numpy_pickle.load(Path(filename)) == value


@with_numpy
def test_non_contiguous_array_pickling(tmpdir):
    filename = tmpdir.join('test.pkl').strpath

    for array in [  # Array that triggers a contiguousness issue with nditer,
                    # see https://github.com/joblib/joblib/pull/352 and see
                    # https://github.com/joblib/joblib/pull/353
                    np.asfortranarray([[1, 2], [3, 4]])[1:],
                    # Non contiguous array with works fine with nditer
                    np.ones((10, 50, 20), order='F')[:, :1, :]]:
        assert not array.flags.c_contiguous
        assert not array.flags.f_contiguous
        numpy_pickle.dump(array, filename)
        array_reloaded = numpy_pickle.load(filename)
        np.testing.assert_array_equal(array_reloaded, array)


@with_numpy
def test_pickle_highest_protocol(tmpdir):
    # ensure persistence of a numpy array is valid even when using
    # the pickle HIGHEST_PROTOCOL.
    # see https://github.com/joblib/joblib/issues/362

    filename = tmpdir.join('test.pkl').strpath
    test_array = np.zeros(10)

    numpy_pickle.dump(test_array, filename, protocol=pickle.HIGHEST_PROTOCOL)
    array_reloaded = numpy_pickle.load(filename)

    np.testing.assert_array_equal(array_reloaded, test_array)


@with_numpy
def test_pickle_in_socket():
    # test that joblib can pickle in sockets
    test_array = np.arange(10)
    _ADDR = ("localhost", 12345)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(_ADDR)
    listener.listen(1)

    client = socket.create_connection(_ADDR)
    server, client_addr = listener.accept()

    with server.makefile("wb") as sf:
        numpy_pickle.dump(test_array, sf)

    with client.makefile("rb") as cf:
        array_reloaded = numpy_pickle.load(cf)

    np.testing.assert_array_equal(array_reloaded, test_array)


@with_numpy
def test_load_memmap_with_big_offset(tmpdir):
    # Test that numpy memmap offset is set correctly if greater than
    # mmap.ALLOCATIONGRANULARITY, see
    # https://github.com/joblib/joblib/issues/451 and
    # https://github.com/numpy/numpy/pull/8443 for more details.
    fname = tmpdir.join('test.mmap').strpath
    size = mmap.ALLOCATIONGRANULARITY
    obj = [np.zeros(size, dtype='uint8'), np.ones(size, dtype='uint8')]
    numpy_pickle.dump(obj, fname)
    memmaps = numpy_pickle.load(fname, mmap_mode='r')
    assert isinstance(memmaps[1], np.memmap)
    assert memmaps[1].offset > size
    np.testing.assert_array_equal(obj, memmaps)


def test_register_compressor(tmpdir):
    # Check that registering compressor file works.
    compressor_name = 'test-name'
    compressor_prefix = 'test-prefix'

    class BinaryCompressorTestFile(io.BufferedIOBase):
        pass

    class BinaryCompressorTestWrapper(CompressorWrapper):

        def __init__(self):
            CompressorWrapper.__init__(self, obj=BinaryCompressorTestFile,
                                       prefix=compressor_prefix)

    register_compressor(compressor_name, BinaryCompressorTestWrapper())

    assert (_COMPRESSORS[compressor_name].fileobj_factory ==
            BinaryCompressorTestFile)
    assert _COMPRESSORS[compressor_name].prefix == compressor_prefix

    # Remove this dummy compressor file from extra compressors because other
    # tests might fail because of this
    _COMPRESSORS.pop(compressor_name)


@parametrize('invalid_name', [1, (), {}])
def test_register_compressor_invalid_name(invalid_name):
    # Test that registering an invalid compressor name is not allowed.
    with raises(ValueError) as excinfo:
        register_compressor(invalid_name, None)
    excinfo.match("Compressor name should be a string")


def test_register_compressor_invalid_fileobj():
    # Test that registering an invalid file object is not allowed.

    class InvalidFileObject():
        pass

    class InvalidFileObjectWrapper(CompressorWrapper):
        def __init__(self):
            CompressorWrapper.__init__(self, obj=InvalidFileObject,
                                       prefix=b'prefix')

    with raises(ValueError) as excinfo:
        register_compressor('invalid', InvalidFileObjectWrapper())

    excinfo.match("Compressor 'fileobj_factory' attribute should implement "
                  "the file object interface")


class AnotherZlibCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryZlibFile, prefix=b'prefix')


class StandardLibGzipCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=gzip.GzipFile, prefix=b'prefix')


def test_register_compressor_already_registered():
    # Test registration of existing compressor files.
    compressor_name = 'test-name'

    # register a test compressor
    register_compressor(compressor_name, AnotherZlibCompressorWrapper())

    with raises(ValueError) as excinfo:
        register_compressor(compressor_name,
                            StandardLibGzipCompressorWrapper())
    excinfo.match("Compressor '{}' already registered."
                  .format(compressor_name))

    register_compressor(compressor_name, StandardLibGzipCompressorWrapper(),
                        force=True)

    assert compressor_name in _COMPRESSORS
    assert _COMPRESSORS[compressor_name].fileobj_factory == gzip.GzipFile

    # Remove this dummy compressor file from extra compressors because other
    # tests might fail because of this
    _COMPRESSORS.pop(compressor_name)


@with_lz4
def test_lz4_compression(tmpdir):
    # Check that lz4 can be used when dependency is available.
    import lz4.frame
    compressor = 'lz4'
    assert compressor in _COMPRESSORS
    assert _COMPRESSORS[compressor].fileobj_factory == lz4.frame.LZ4FrameFile

    fname = tmpdir.join('test.pkl').strpath
    data = 'test data'
    numpy_pickle.dump(data, fname, compress=compressor)

    with open(fname, 'rb') as f:
        assert f.read(len(_LZ4_PREFIX)) == _LZ4_PREFIX
    assert numpy_pickle.load(fname) == data

    # Test that LZ4 is applied based on file extension
    numpy_pickle.dump(data, fname + '.lz4')
    with open(fname, 'rb') as f:
        assert f.read(len(_LZ4_PREFIX)) == _LZ4_PREFIX
    assert numpy_pickle.load(fname) == data


@without_lz4
def test_lz4_compression_without_lz4(tmpdir):
    # Check that lz4 cannot be used when dependency is not available.
    fname = tmpdir.join('test.nolz4').strpath
    data = 'test data'
    msg = LZ4_NOT_INSTALLED_ERROR
    with raises(ValueError) as excinfo:
        numpy_pickle.dump(data, fname, compress='lz4')
    excinfo.match(msg)

    with raises(ValueError) as excinfo:
        numpy_pickle.dump(data, fname + '.lz4')
    excinfo.match(msg)
