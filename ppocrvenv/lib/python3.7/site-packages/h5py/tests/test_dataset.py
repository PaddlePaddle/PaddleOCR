# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Dataset testing operations.

    Tests all dataset operations, including creation, with the exception of:

    1. Slicing operations for read and write, handled by module test_slicing
    2. Type conversion for read and write (currently untested)
"""

import pathlib
import sys
import numpy as np
import platform
import pytest
import warnings

from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
import h5py
import h5py._hl.selections as sel


class BaseDataset(TestCase):
    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestRepr(BaseDataset):
    """
        Feature: repr(Dataset) behaves sensibly
    """

    def test_repr_open(self):
        """ repr() works on live and dead datasets """
        ds = self.f.create_dataset('foo', (4,))
        self.assertIsInstance(repr(ds), str)
        self.f.close()
        self.assertIsInstance(repr(ds), str)


class TestCreateShape(BaseDataset):

    """
        Feature: Datasets can be created from a shape only
    """

    def test_create_scalar(self):
        """ Create a scalar dataset """
        dset = self.f.create_dataset('foo', ())
        self.assertEqual(dset.shape, ())

    def test_create_simple(self):
        """ Create a size-1 dataset """
        dset = self.f.create_dataset('foo', (1,))
        self.assertEqual(dset.shape, (1,))

    def test_create_integer(self):
        """ Create a size-1 dataset with integer shape"""
        dset = self.f.create_dataset('foo', 1)
        self.assertEqual(dset.shape, (1,))

    def test_create_extended(self):
        """ Create an extended dataset """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.shape, (63,))
        self.assertEqual(dset.size, 63)
        dset = self.f.create_dataset('bar', (6, 10))
        self.assertEqual(dset.shape, (6, 10))
        self.assertEqual(dset.size, (60))

    def test_create_integer_extended(self):
        """ Create an extended dataset """
        dset = self.f.create_dataset('foo', 63)
        self.assertEqual(dset.shape, (63,))
        self.assertEqual(dset.size, 63)
        dset = self.f.create_dataset('bar', (6, 10))
        self.assertEqual(dset.shape, (6, 10))
        self.assertEqual(dset.size, (60))

    def test_default_dtype(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,))
        self.assertEqual(dset.dtype, np.dtype('=f4'))

    def test_missing_shape(self):
        """ Missing shape raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo')

    def test_long_double(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,), dtype=np.longdouble)
        if platform.machine() in ['ppc64le']:
            pytest.xfail("Storage of long double deactivated on %s" % platform.machine())
        self.assertEqual(dset.dtype, np.longdouble)

    @ut.skipIf(not hasattr(np, "complex256"), "No support for complex256")
    def test_complex256(self):
        """ Confirm that the default dtype is float """
        dset = self.f.create_dataset('foo', (63,),
                                     dtype=np.dtype('complex256'))
        self.assertEqual(dset.dtype, np.dtype('complex256'))

    def test_name_bytes(self):
        dset = self.f.create_dataset(b'foo', (1,))
        self.assertEqual(dset.shape, (1,))

        dset2 = self.f.create_dataset(b'bar/baz', (2,))
        self.assertEqual(dset2.shape, (2,))

class TestCreateData(BaseDataset):

    """
        Feature: Datasets can be created from existing data
    """

    def test_create_scalar(self):
        """ Create a scalar dataset from existing array """
        data = np.ones((), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_create_extended(self):
        """ Create an extended dataset from existing data """
        data = np.ones((63,), 'f')
        dset = self.f.create_dataset('foo', data=data)
        self.assertEqual(dset.shape, data.shape)

    def test_dataset_intermediate_group(self):
        """ Create dataset with missing intermediate groups """
        ds = self.f.create_dataset("/foo/bar/baz", shape=(10, 10), dtype='<i4')
        self.assertIsInstance(ds, h5py.Dataset)
        self.assertTrue("/foo/bar/baz" in self.f)

    def test_reshape(self):
        """ Create from existing data, and make it fit a new shape """
        data = np.arange(30, dtype='f')
        dset = self.f.create_dataset('foo', shape=(10, 3), data=data)
        self.assertEqual(dset.shape, (10, 3))
        self.assertArrayEqual(dset[...], data.reshape((10, 3)))

    def test_appropriate_low_level_id(self):
        " Binding Dataset to a non-DatasetID identifier fails with ValueError "
        with self.assertRaises(ValueError):
            Dataset(self.f['/'].id)

    def check_h5_string(self, dset, cset, length):
        tid = dset.id.get_type()
        assert isinstance(tid, h5t.TypeStringID)
        assert tid.get_cset() == cset
        if length is None:
            assert tid.is_variable_str()
        else:
            assert not tid.is_variable_str()
            assert tid.get_size() == length

    def test_create_bytestring(self):
        """ Creating dataset with byte string yields vlen ASCII dataset """
        def check_vlen_ascii(dset):
            self.check_h5_string(dset, h5t.CSET_ASCII, length=None)
        check_vlen_ascii(self.f.create_dataset('a', data=b'abc'))
        check_vlen_ascii(self.f.create_dataset('b', data=[b'abc', b'def']))
        check_vlen_ascii(self.f.create_dataset('c', data=[[b'abc'], [b'def']]))
        check_vlen_ascii(self.f.create_dataset(
            'd', data=np.array([b'abc', b'def'], dtype=object)
        ))

    def test_create_np_s(self):
        dset = self.f.create_dataset('a', data=np.array([b'abc', b'def'], dtype='S3'))
        self.check_h5_string(dset, h5t.CSET_ASCII, length=3)

    def test_create_strings(self):
        def check_vlen_utf8(dset):
            self.check_h5_string(dset, h5t.CSET_UTF8, length=None)
        check_vlen_utf8(self.f.create_dataset('a', data='abc'))
        check_vlen_utf8(self.f.create_dataset('b', data=['abc', 'def']))
        check_vlen_utf8(self.f.create_dataset('c', data=[['abc'], ['def']]))
        check_vlen_utf8(self.f.create_dataset(
            'd', data=np.array(['abc', 'def'], dtype=object)
        ))

    def test_create_np_u(self):
        with self.assertRaises(TypeError):
            self.f.create_dataset('a', data=np.array([b'abc', b'def'], dtype='U3'))

    def test_empty_create_via_None_shape(self):
        self.f.create_dataset('foo', dtype='f')
        self.assertTrue(is_empty_dataspace(self.f['foo'].id))

    def test_empty_create_via_Empty_class(self):
        self.f.create_dataset('foo', data=h5py.Empty(dtype='f'))
        self.assertTrue(is_empty_dataspace(self.f['foo'].id))

    def test_create_incompatible_data(self):
        # Shape tuple is incompatible with data
        with self.assertRaises(ValueError):
            self.f.create_dataset('bar', shape=4, data= np.arange(3))


class TestReadDirectly:

    """
        Feature: Read data directly from Dataset into a Numpy array
    """

    @pytest.mark.parametrize(
        'source_shape,dest_shape,source_sel,dest_sel',
        [
            ((100,), (100,), np.s_[0:10], np.s_[50:60]),
            ((70,), (100,), np.s_[50:60], np.s_[90:]),
            ((30, 10), (20, 20), np.s_[:20, :], np.s_[:, :10]),
            ((5, 7, 9), (6,), np.s_[2, :6, 3], np.s_[:]),
        ])
    def test_read_direct(self, writable_file, source_shape, dest_shape, source_sel, dest_sel):
        source_values = np.arange(np.product(source_shape), dtype="int64").reshape(source_shape)
        dset = writable_file.create_dataset("dset", source_shape, data=source_values)
        arr = np.full(dest_shape, -1, dtype="int64")
        expected = arr.copy()
        expected[dest_sel] = source_values[source_sel]

        dset.read_direct(arr, source_sel, dest_sel)
        np.testing.assert_array_equal(arr, expected)

    def test_no_sel(self, writable_file):
        dset = writable_file.create_dataset("dset", (10,), data=np.arange(10, dtype="int64"))
        arr = np.ones((10,), dtype="int64")
        dset.read_direct(arr)
        np.testing.assert_array_equal(arr, np.arange(10, dtype="int64"))

    def test_empty(self, writable_file):
        empty_dset = writable_file.create_dataset("edset", dtype='int64')
        arr = np.ones((100,), 'int64')
        with pytest.raises(TypeError):
            empty_dset.read_direct(arr, np.s_[0:10], np.s_[50:60])

    def test_wrong_shape(self, writable_file):
        dset = writable_file.create_dataset("dset", (100,), dtype='int64')
        arr = np.ones((200,))
        with pytest.raises(TypeError):
            dset.read_direct(arr)

    def test_not_c_contiguous(self, writable_file):
        dset = writable_file.create_dataset("dset", (10, 10), dtype='int64')
        arr = np.ones((10, 10), order='F')
        with pytest.raises(TypeError):
            dset.read_direct(arr)

class TestWriteDirectly:

    """
        Feature: Write Numpy array directly into Dataset
    """

    @pytest.mark.parametrize(
        'source_shape,dest_shape,source_sel,dest_sel',
        [
            ((100,), (100,), np.s_[0:10], np.s_[50:60]),
            ((70,), (100,), np.s_[50:60], np.s_[90:]),
            ((30, 10), (20, 20), np.s_[:20, :], np.s_[:, :10]),
            ((5, 7, 9), (6,), np.s_[2, :6, 3], np.s_[:]),
        ])
    def test_write_direct(self, writable_file, source_shape, dest_shape, source_sel, dest_sel):
        dset = writable_file.create_dataset('dset', dest_shape, dtype='int32', fillvalue=-1)
        arr = np.arange(np.product(source_shape)).reshape(source_shape)
        expected = np.full(dest_shape, -1, dtype='int32')
        expected[dest_sel] = arr[source_sel]
        dset.write_direct(arr, source_sel, dest_sel)
        np.testing.assert_array_equal(dset[:], expected)

    def test_empty(self, writable_file):
        empty_dset = writable_file.create_dataset("edset", dtype='int64')
        with pytest.raises(TypeError):
            empty_dset.write_direct(np.ones((100,)), np.s_[0:10], np.s_[50:60])

    def test_wrong_shape(self, writable_file):
        dset = writable_file.create_dataset("dset", (100,), dtype='int64')
        arr = np.ones((200,))
        with pytest.raises(TypeError):
            dset.write_direct(arr)

    def test_not_c_contiguous(self, writable_file):
        dset = writable_file.create_dataset("dset", (10, 10), dtype='int64')
        arr = np.ones((10, 10), order='F')
        with pytest.raises(TypeError):
            dset.write_direct(arr)


class TestCreateRequire(BaseDataset):

    """
        Feature: Datasets can be created only if they don't exist in the file
    """

    def test_create(self):
        """ Create new dataset with no conflicts """
        dset = self.f.require_dataset('foo', (10, 3), 'f')
        self.assertIsInstance(dset, Dataset)
        self.assertEqual(dset.shape, (10, 3))

    def test_create_existing(self):
        """ require_dataset yields existing dataset """
        dset = self.f.require_dataset('foo', (10, 3), 'f')
        dset2 = self.f.require_dataset('foo', (10, 3), 'f')
        self.assertEqual(dset, dset2)

    def test_create_1D(self):
        """ require_dataset with integer shape yields existing dataset"""
        dset = self.f.require_dataset('foo', 10, 'f')
        dset2 = self.f.require_dataset('foo', 10, 'f')
        self.assertEqual(dset, dset2)

        dset = self.f.require_dataset('bar', (10,), 'f')
        dset2 = self.f.require_dataset('bar', 10, 'f')
        self.assertEqual(dset, dset2)

        dset = self.f.require_dataset('baz', 10, 'f')
        dset2 = self.f.require_dataset(b'baz', (10,), 'f')
        self.assertEqual(dset, dset2)

    def test_shape_conflict(self):
        """ require_dataset with shape conflict yields TypeError """
        self.f.create_dataset('foo', (10, 3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 4), 'f')

    def test_type_conflict(self):
        """ require_dataset with object type conflict yields TypeError """
        self.f.create_group('foo')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 3), 'f')

    def test_dtype_conflict(self):
        """ require_dataset with dtype conflict (strict mode) yields TypeError
        """
        dset = self.f.create_dataset('foo', (10, 3), 'f')
        with self.assertRaises(TypeError):
            self.f.require_dataset('foo', (10, 3), 'S10')

    def test_dtype_exact(self):
        """ require_dataset with exactly dtype match """

        dset = self.f.create_dataset('foo', (10, 3), 'f')
        dset2 = self.f.require_dataset('foo', (10, 3), 'f', exact=True)
        self.assertEqual(dset, dset2)

    def test_dtype_close(self):
        """ require_dataset with convertible type succeeds (non-strict mode)
        """
        dset = self.f.create_dataset('foo', (10, 3), 'i4')
        dset2 = self.f.require_dataset('foo', (10, 3), 'i2', exact=False)
        self.assertEqual(dset, dset2)
        self.assertEqual(dset2.dtype, np.dtype('i4'))


class TestCreateChunked(BaseDataset):

    """
        Feature: Datasets can be created by manually specifying chunks
    """

    def test_create_chunks(self):
        """ Create via chunks tuple """
        dset = self.f.create_dataset('foo', shape=(100,), chunks=(10,))
        self.assertEqual(dset.chunks, (10,))

    def test_create_chunks_integer(self):
        """ Create via chunks integer """
        dset = self.f.create_dataset('foo', shape=(100,), chunks=10)
        self.assertEqual(dset.chunks, (10,))

    def test_chunks_mismatch(self):
        """ Illegal chunk size raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', shape=(100,), chunks=(200,))

    def test_chunks_false(self):
        """ Chunked format required for given storage options """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', shape=(10,), maxshape=100, chunks=False)

    def test_chunks_scalar(self):
        """ Attempting to create chunked scalar dataset raises TypeError """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo', shape=(), chunks=(50,))

    def test_auto_chunks(self):
        """ Auto-chunking of datasets """
        dset = self.f.create_dataset('foo', shape=(20, 100), chunks=True)
        self.assertIsInstance(dset.chunks, tuple)
        self.assertEqual(len(dset.chunks), 2)

    def test_auto_chunks_abuse(self):
        """ Auto-chunking with pathologically large element sizes """
        dset = self.f.create_dataset('foo', shape=(3,), dtype='S100000000', chunks=True)
        self.assertEqual(dset.chunks, (1,))

    def test_scalar_assignment(self):
        """ Test scalar assignment of chunked dataset """
        dset = self.f.create_dataset('foo', shape=(3, 50, 50),
                                     dtype=np.int32, chunks=(1, 50, 50))
        # test assignment of selection smaller than chunk size
        dset[1, :, 40] = 10
        self.assertTrue(np.all(dset[1, :, 40] == 10))

        # test assignment of selection equal to chunk size
        dset[1] = 11
        self.assertTrue(np.all(dset[1] == 11))

        # test assignment of selection bigger than chunk size
        dset[0:2] = 12
        self.assertTrue(np.all(dset[0:2] == 12))

    def test_auto_chunks_no_shape(self):
        """ Auto-chunking of empty datasets not allowed"""
        with pytest.raises(TypeError, match='Empty') as err:
            self.f.create_dataset('foo', dtype='S100', chunks=True)

        with pytest.raises(TypeError, match='Empty') as err:
            self.f.create_dataset('foo', dtype='S100', maxshape=20)


class TestCreateFillvalue(BaseDataset):

    """
        Feature: Datasets can be created with fill value
    """

    def test_create_fillval(self):
        """ Fill value is reflected in dataset contents """
        dset = self.f.create_dataset('foo', (10,), fillvalue=4.0)
        self.assertEqual(dset[0], 4.0)
        self.assertEqual(dset[7], 4.0)

    def test_property(self):
        """ Fill value is recoverable via property """
        dset = self.f.create_dataset('foo', (10,), fillvalue=3.0)
        self.assertEqual(dset.fillvalue, 3.0)
        self.assertNotIsInstance(dset.fillvalue, np.ndarray)

    def test_property_none(self):
        """ .fillvalue property works correctly if not set """
        dset = self.f.create_dataset('foo', (10,))
        self.assertEqual(dset.fillvalue, 0)

    def test_compound(self):
        """ Fill value works with compound types """
        dt = np.dtype([('a', 'f4'), ('b', 'i8')])
        v = np.ones((1,), dtype=dt)[0]
        dset = self.f.create_dataset('foo', (10,), dtype=dt, fillvalue=v)
        self.assertEqual(dset.fillvalue, v)
        self.assertAlmostEqual(dset[4], v)

    def test_exc(self):
        """ Bogus fill value raises ValueError """
        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (10,),
                    dtype=[('a', 'i'), ('b', 'f')], fillvalue=42)


class TestCreateNamedType(BaseDataset):

    """
        Feature: Datasets created from an existing named type
    """

    def test_named(self):
        """ Named type object works and links the dataset to type """
        self.f['type'] = np.dtype('f8')
        dset = self.f.create_dataset('x', (100,), dtype=self.f['type'])
        self.assertEqual(dset.dtype, np.dtype('f8'))
        self.assertEqual(dset.id.get_type(), self.f['type'].id)
        self.assertTrue(dset.id.get_type().committed())


@ut.skipIf('gzip' not in h5py.filters.encode, "DEFLATE is not installed")
class TestCreateGzip(BaseDataset):

    """
        Feature: Datasets created with gzip compression
    """

    def test_gzip(self):
        """ Create with explicit gzip options """
        dset = self.f.create_dataset('foo', (20, 30), compression='gzip',
                                     compression_opts=9)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 9)

    def test_gzip_implicit(self):
        """ Create with implicit gzip level (level 4) """
        dset = self.f.create_dataset('foo', (20, 30), compression='gzip')
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 4)

    def test_gzip_number(self):
        """ Create with gzip level by specifying integer """
        dset = self.f.create_dataset('foo', (20, 30), compression=7)
        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 7)

        original_compression_vals = h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS
        try:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = tuple()
            with self.assertRaises(ValueError):
                dset = self.f.create_dataset('foo', (20, 30), compression=7)
        finally:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = original_compression_vals

    def test_gzip_exc(self):
        """ Illegal gzip level (explicit or implicit) raises ValueError """
        with self.assertRaises((ValueError, RuntimeError)):
            self.f.create_dataset('foo', (20, 30), compression=14)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression=-4)
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression='gzip',
                                  compression_opts=14)


@ut.skipIf('gzip' not in h5py.filters.encode, "DEFLATE is not installed")
class TestCreateCompressionNumber(BaseDataset):

    """
        Feature: Datasets created with a compression code
    """

    def test_compression_number(self):
        """ Create with compression number of gzip (h5py.h5z.FILTER_DEFLATE) and a compression level of 7"""
        original_compression_vals = h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS
        try:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = tuple()
            dset = self.f.create_dataset('foo', (20, 30), compression=h5py.h5z.FILTER_DEFLATE, compression_opts=(7,))
        finally:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = original_compression_vals

        self.assertEqual(dset.compression, 'gzip')
        self.assertEqual(dset.compression_opts, 7)

    def test_compression_number_invalid(self):
        """ Create with invalid compression numbers  """
        with self.assertRaises(ValueError) as e:
            self.f.create_dataset('foo', (20, 30), compression=-999)
        self.assertIn("Invalid filter", str(e.exception))

        with self.assertRaises(ValueError) as e:
            self.f.create_dataset('foo', (20, 30), compression=100)
        self.assertIn("Unknown compression", str(e.exception))

        original_compression_vals = h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS
        try:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = tuple()

            # Using gzip compression requires a compression level specified in compression_opts
            with self.assertRaises(IndexError):
                self.f.create_dataset('foo', (20, 30), compression=h5py.h5z.FILTER_DEFLATE)
        finally:
            h5py._hl.dataset._LEGACY_GZIP_COMPRESSION_VALS = original_compression_vals


@ut.skipIf('lzf' not in h5py.filters.encode, "LZF is not installed")
class TestCreateLZF(BaseDataset):

    """
        Feature: Datasets created with LZF compression
    """

    def test_lzf(self):
        """ Create with explicit lzf """
        dset = self.f.create_dataset('foo', (20, 30), compression='lzf')
        self.assertEqual(dset.compression, 'lzf')
        self.assertEqual(dset.compression_opts, None)

        testdata = np.arange(100)
        dset = self.f.create_dataset('bar', data=testdata, compression='lzf')
        self.assertEqual(dset.compression, 'lzf')
        self.assertEqual(dset.compression_opts, None)

        self.f.flush()  # Actually write to file

        readdata = self.f['bar'][()]
        self.assertArrayEqual(readdata, testdata)

    def test_lzf_exc(self):
        """ Giving lzf options raises ValueError """
        with self.assertRaises(ValueError):
            self.f.create_dataset('foo', (20, 30), compression='lzf',
                                  compression_opts=4)


@ut.skipIf('szip' not in h5py.filters.encode, "SZIP is not installed")
class TestCreateSZIP(BaseDataset):

    """
        Feature: Datasets created with LZF compression
    """

    def test_szip(self):
        """ Create with explicit szip """
        dset = self.f.create_dataset('foo', (20, 30), compression='szip',
                                     compression_opts=('ec', 16))


@ut.skipIf('shuffle' not in h5py.filters.encode, "SHUFFLE is not installed")
class TestCreateShuffle(BaseDataset):

    """
        Feature: Datasets can use shuffling filter
    """

    def test_shuffle(self):
        """ Enable shuffle filter """
        dset = self.f.create_dataset('foo', (20, 30), shuffle=True)
        self.assertTrue(dset.shuffle)


@ut.skipIf('fletcher32' not in h5py.filters.encode, "FLETCHER32 is not installed")
class TestCreateFletcher32(BaseDataset):
    """
        Feature: Datasets can use the fletcher32 filter
    """

    def test_fletcher32(self):
        """ Enable fletcher32 filter """
        dset = self.f.create_dataset('foo', (20, 30), fletcher32=True)
        self.assertTrue(dset.fletcher32)


@ut.skipIf('scaleoffset' not in h5py.filters.encode, "SCALEOFFSET is not installed")
class TestCreateScaleOffset(BaseDataset):
    """
        Feature: Datasets can use the scale/offset filter
    """

    def test_float_fails_without_options(self):
        """ Ensure that a scale factor is required for scaleoffset compression of floating point data """

        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=float, scaleoffset=True)

    def test_non_integer(self):
        """ Check when scaleoffset is negetive"""

        with self.assertRaises(ValueError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=float, scaleoffset=-0.1)

    def test_unsupport_dtype(self):
        """ Check when dtype is unsupported type"""

        with self.assertRaises(TypeError):
            dset = self.f.create_dataset('foo', (20, 30), dtype=bool, scaleoffset=True)

    def test_float(self):
        """ Scaleoffset filter works for floating point data """

        scalefac = 4
        shape = (100, 300)
        range = 20 * 10 ** scalefac
        testdata = (np.random.rand(*shape) - 0.5) * range

        dset = self.f.create_dataset('foo', shape, dtype=float, scaleoffset=scalefac)

        # Dataset reports that scaleoffset is in use
        assert dset.scaleoffset is not None

        # Dataset round-trips
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]

        # Test that data round-trips to requested precision
        self.assertArrayEqual(readdata, testdata, precision=10 ** (-scalefac))

        # Test that the filter is actually active (i.e. compression is lossy)
        assert not (readdata == testdata).all()

    def test_int(self):
        """ Scaleoffset filter works for integer data with default precision """

        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** nbits - 1, size=shape)

        # Create dataset; note omission of nbits (for library-determined precision)
        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=True)

        # Dataset reports scaleoffset enabled
        assert dset.scaleoffset is not None

        # Data round-trips correctly and identically
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        self.assertArrayEqual(readdata, testdata)

    def test_int_with_minbits(self):
        """ Scaleoffset filter works for integer data with specified precision """

        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** nbits, size=shape)

        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=nbits)

        # Dataset reports scaleoffset enabled with correct precision
        self.assertTrue(dset.scaleoffset == 12)

        # Data round-trips correctly
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]
        self.assertArrayEqual(readdata, testdata)

    def test_int_with_minbits_lossy(self):
        """ Scaleoffset filter works for integer data with specified precision """

        nbits = 12
        shape = (100, 300)
        testdata = np.random.randint(0, 2 ** (nbits + 1) - 1, size=shape)

        dset = self.f.create_dataset('foo', shape, dtype=int, scaleoffset=nbits)

        # Dataset reports scaleoffset enabled with correct precision
        self.assertTrue(dset.scaleoffset == 12)

        # Data can be written and read
        dset[...] = testdata
        filename = self.f.filename
        self.f.close()
        self.f = h5py.File(filename, 'r')
        readdata = self.f['foo'][...]

        # Compression is lossy
        assert not (readdata == testdata).all()


class TestExternal(BaseDataset):
    """
        Feature: Datasets with the external storage property
    """
    def test_contents(self):
        """ Create and access an external dataset """

        shape = (6, 100)
        testdata = np.random.random(shape)

        # create a dataset in an external file and set it
        ext_file = self.mktemp()
        external = [(ext_file, 0, h5f.UNLIMITED)]
        dset = self.f.create_dataset('foo', shape, dtype=testdata.dtype, external=external)
        dset[...] = testdata

        assert dset.external is not None

        # verify file's existence, size, and contents
        with open(ext_file, 'rb') as fid:
            contents = fid.read()
        assert contents == testdata.tobytes()

    def test_name_str(self):
        """ External argument may be a file name str only """

        self.f.create_dataset('foo', (6, 100), external=self.mktemp())

    def test_name_path(self):
        """ External argument may be a file name path only """

        self.f.create_dataset('foo', (6, 100),
                              external=pathlib.Path(self.mktemp()))

    def test_iter_multi(self):
        """ External argument may be an iterable of multiple tuples """

        ext_file = self.mktemp()
        N = 100
        external = iter((ext_file, x * 1000, 1000) for x in range(N))
        dset = self.f.create_dataset('poo', (6, 100), external=external)
        assert len(dset.external) == N

    def test_invalid(self):
        """ Test with invalid external lists """

        shape = (6, 100)
        ext_file = self.mktemp()

        for exc_type, external in [
            (TypeError, [ext_file]),
            (TypeError, [ext_file, 0]),
            (TypeError, [ext_file, 0, h5f.UNLIMITED]),
            (ValueError, [(ext_file,)]),
            (ValueError, [(ext_file, 0)]),
            (ValueError, [(ext_file, 0, h5f.UNLIMITED, 0)]),
            (TypeError, [(ext_file, 0, "h5f.UNLIMITED")]),
        ]:
            with self.assertRaises(exc_type):
                self.f.create_dataset('foo', shape, external=external)


class TestAutoCreate(BaseDataset):

    """
        Feature: Datasets auto-created from data produce the correct types
    """
    def assert_string_type(self, ds, cset, variable=True):
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), cset)
        if variable:
            assert tid.is_variable_str()

    def test_vlen_bytes(self):
        """Assigning byte strings produces a vlen string ASCII dataset """
        self.f['x'] = b"Hello there"
        self.assert_string_type(self.f['x'], h5py.h5t.CSET_ASCII)

        self.f['y'] = [b"a", b"bc"]
        self.assert_string_type(self.f['y'], h5py.h5t.CSET_ASCII)

        self.f['z'] = np.array([b"a", b"bc"], dtype=np.object_)
        self.assert_string_type(self.f['z'], h5py.h5t.CSET_ASCII)

    def test_vlen_unicode(self):
        """Assigning unicode strings produces a vlen string UTF-8 dataset """
        self.f['x'] = "Hello there" + chr(0x2034)
        self.assert_string_type(self.f['x'], h5py.h5t.CSET_UTF8)

        self.f['y'] = ["a", "bc"]
        self.assert_string_type(self.f['y'], h5py.h5t.CSET_UTF8)

        # 2D array; this only works with an array, not nested lists
        self.f['z'] = np.array([["a", "bc"]], dtype=np.object_)
        self.assert_string_type(self.f['z'], h5py.h5t.CSET_UTF8)

    def test_string_fixed(self):
        """ Assignment of fixed-length byte string produces a fixed-length
        ascii dataset """
        self.f['x'] = np.string_("Hello there")
        ds = self.f['x']
        self.assert_string_type(ds, h5py.h5t.CSET_ASCII, variable=False)
        self.assertEqual(ds.id.get_type().get_size(), 11)


class TestCreateLike(BaseDataset):
    def test_no_chunks(self):
        self.f['lol'] = np.arange(25).reshape(5, 5)
        self.f.create_dataset_like('like_lol', self.f['lol'])
        dslike = self.f['like_lol']
        self.assertEqual(dslike.shape, (5, 5))
        self.assertIs(dslike.chunks, None)

    def test_track_times(self):
        orig = self.f.create_dataset('honda', data=np.arange(12),
                                     track_times=True)
        self.assertNotEqual(0, h5py.h5g.get_objinfo(orig._id).mtime)
        similar = self.f.create_dataset_like('hyundai', orig)
        self.assertNotEqual(0, h5py.h5g.get_objinfo(similar._id).mtime)

        orig = self.f.create_dataset('ibm', data=np.arange(12),
                                     track_times=False)
        self.assertEqual(0, h5py.h5g.get_objinfo(orig._id).mtime)
        similar = self.f.create_dataset_like('lenovo', orig)
        self.assertEqual(0, h5py.h5g.get_objinfo(similar._id).mtime)

    def test_maxshape(self):
        """ Test when other.maxshape != other.shape """

        other = self.f.create_dataset('other', (10,), maxshape=20)
        similar = self.f.create_dataset_like('sim', other)
        self.assertEqual(similar.shape, (10,))
        self.assertEqual(similar.maxshape, (20,))

class TestChunkIterator(BaseDataset):
    def test_no_chunks(self):
        dset = self.f.create_dataset("foo", ())
        with self.assertRaises(TypeError):
            dset.iter_chunks()

    def test_1d(self):
        dset = self.f.create_dataset("foo", (100,), chunks=(32,))
        expected = ((slice(0,32,1),), (slice(32,64,1),), (slice(64,96,1),),
            (slice(96,100,1),))
        self.assertEqual(list(dset.iter_chunks()), list(expected))
        expected = ((slice(50,64,1),), (slice(64,96,1),), (slice(96,97,1),))
        self.assertEqual(list(dset.iter_chunks(np.s_[50:97])), list(expected))

    def test_2d(self):
        dset = self.f.create_dataset("foo", (100,100), chunks=(32,64))
        expected = ((slice(0, 32, 1), slice(0, 64, 1)), (slice(0, 32, 1),
        slice(64, 100, 1)), (slice(32, 64, 1), slice(0, 64, 1)),
        (slice(32, 64, 1), slice(64, 100, 1)), (slice(64, 96, 1),
        slice(0, 64, 1)), (slice(64, 96, 1), slice(64, 100, 1)),
        (slice(96, 100, 1), slice(0, 64, 1)), (slice(96, 100, 1),
        slice(64, 100, 1)))
        self.assertEqual(list(dset.iter_chunks()), list(expected))

        expected = ((slice(48, 52, 1), slice(40, 50, 1)),)
        self.assertEqual(list(dset.iter_chunks(np.s_[48:52,40:50])), list(expected))


class TestResize(BaseDataset):

    """
        Feature: Datasets created with "maxshape" may be resized
    """

    def test_create(self):
        """ Create dataset with "maxshape" """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        self.assertIsNot(dset.chunks, None)
        self.assertEqual(dset.maxshape, (20, 60))

    def test_create_1D(self):
        """ Create dataset with "maxshape" using integer maxshape"""
        dset = self.f.create_dataset('foo', (20,), maxshape=20)
        self.assertIsNot(dset.chunks, None)
        self.assertEqual(dset.maxshape, (20,))

        dset = self.f.create_dataset('bar', 20, maxshape=20)
        self.assertEqual(dset.maxshape, (20,))

    def test_resize(self):
        """ Datasets may be resized up to maxshape """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        self.assertEqual(dset.shape, (20, 30))
        dset.resize((20, 50))
        self.assertEqual(dset.shape, (20, 50))
        dset.resize((20, 60))
        self.assertEqual(dset.shape, (20, 60))

    def test_resize_1D(self):
        """ Datasets may be resized up to maxshape using integer maxshape"""
        dset = self.f.create_dataset('foo', 20, maxshape=40)
        self.assertEqual(dset.shape, (20,))
        dset.resize((30,))
        self.assertEqual(dset.shape, (30,))

    def test_resize_over(self):
        """ Resizing past maxshape triggers an exception """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        with self.assertRaises(Exception):
            dset.resize((20, 70))

    def test_resize_nonchunked(self):
        """ Resizing non-chunked dataset raises TypeError """
        dset = self.f.create_dataset("foo", (20, 30))
        with self.assertRaises(TypeError):
            dset.resize((20, 60))

    def test_resize_axis(self):
        """ Resize specified axis """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        dset.resize(50, axis=1)
        self.assertEqual(dset.shape, (20, 50))

    def test_axis_exc(self):
        """ Illegal axis raises ValueError """
        dset = self.f.create_dataset('foo', (20, 30), maxshape=(20, 60))
        with self.assertRaises(ValueError):
            dset.resize(50, axis=2)

    def test_zero_dim(self):
        """ Allow zero-length initial dims for unlimited axes (issue 111) """
        dset = self.f.create_dataset('foo', (15, 0), maxshape=(15, None))
        self.assertEqual(dset.shape, (15, 0))
        self.assertEqual(dset.maxshape, (15, None))


class TestDtype(BaseDataset):

    """
        Feature: Dataset dtype is available as .dtype property
    """

    def test_dtype(self):
        """ Retrieve dtype from dataset """
        dset = self.f.create_dataset('foo', (5,), '|S10')
        self.assertEqual(dset.dtype, np.dtype('|S10'))


class TestLen(BaseDataset):

    """
        Feature: Size of first axis is available via Python's len
    """

    def test_len(self):
        """ Python len() (under 32 bits) """
        dset = self.f.create_dataset('foo', (312, 15))
        self.assertEqual(len(dset), 312)

    def test_len_big(self):
        """ Python len() vs Dataset.len() """
        dset = self.f.create_dataset('foo', (2 ** 33, 15))
        self.assertEqual(dset.shape, (2 ** 33, 15))
        if sys.maxsize == 2 ** 31 - 1:
            with self.assertRaises(OverflowError):
                len(dset)
        else:
            self.assertEqual(len(dset), 2 ** 33)
        self.assertEqual(dset.len(), 2 ** 33)


class TestIter(BaseDataset):

    """
        Feature: Iterating over a dataset yields rows
    """

    def test_iter(self):
        """ Iterating over a dataset yields rows """
        data = np.arange(30, dtype='f').reshape((10, 3))
        dset = self.f.create_dataset('foo', data=data)
        for x, y in zip(dset, data):
            self.assertEqual(len(x), 3)
            self.assertArrayEqual(x, y)

    def test_iter_scalar(self):
        """ Iterating over scalar dataset raises TypeError """
        dset = self.f.create_dataset('foo', shape=())
        with self.assertRaises(TypeError):
            [x for x in dset]


class TestStrings(BaseDataset):

    """
        Feature: Datasets created with vlen and fixed datatypes correctly
        translate to and from HDF5
    """

    def test_vlen_bytes(self):
        """ Vlen bytes dataset maps to vlen ascii in the file """
        dt = h5py.string_dtype(encoding='ascii')
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
        string_info = h5py.check_string_dtype(ds.dtype)
        self.assertEqual(string_info.encoding, 'ascii')

    def test_vlen_unicode(self):
        """ Vlen unicode dataset maps to vlen utf-8 in the file """
        dt = h5py.string_dtype()
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)
        string_info = h5py.check_string_dtype(ds.dtype)
        self.assertEqual(string_info.encoding, 'utf-8')

    def test_fixed_ascii(self):
        """ Fixed-length bytes dataset maps to fixed-length ascii in the file
        """
        dt = np.dtype("|S10")
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertFalse(tid.is_variable_str())
        self.assertEqual(tid.get_size(), 10)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
        string_info = h5py.check_string_dtype(ds.dtype)
        self.assertEqual(string_info.encoding, 'ascii')
        self.assertEqual(string_info.length, 10)

    def test_fixed_utf8(self):
        dt = h5py.string_dtype(encoding='utf-8', length=5)
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        tid = ds.id.get_type()
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)
        s = 'cù'
        ds[0] = s.encode('utf-8')
        ds[1] = s
        ds[2:4] = [s, s]
        ds[4:6] = np.array([s, s], dtype=object)
        ds[6:8] = np.array([s.encode('utf-8')] * 2, dtype=dt)
        with self.assertRaises(TypeError):
            ds[8:10] = np.array([s, s], dtype='U')

        np.testing.assert_array_equal(ds[:8], np.array([s.encode('utf-8')] * 8, dtype='S'))

    def test_fixed_unicode(self):
        """ Fixed-length unicode datasets are unsupported (raise TypeError) """
        dt = np.dtype("|U10")
        with self.assertRaises(TypeError):
            ds = self.f.create_dataset('x', (100,), dtype=dt)

    def test_roundtrip_vlen_bytes(self):
        """ writing and reading to vlen bytes dataset preserves type and content
        """
        dt = h5py.string_dtype(encoding='ascii')
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = b"Hello\xef"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), bytes)
        self.assertEqual(out, data)

    def test_roundtrip_fixed_bytes(self):
        """ Writing to and reading from fixed-length bytes dataset preserves
        type and content """
        dt = np.dtype("|S10")
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = b"Hello\xef"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), np.string_)
        self.assertEqual(out, data)

    def test_retrieve_vlen_unicode(self):
        dt = h5py.string_dtype()
        ds = self.f.create_dataset('x', (10,), dtype=dt)
        data = "fàilte"
        ds[0] = data
        self.assertIsInstance(ds[0], bytes)
        out = ds.asstr()[0]
        self.assertIsInstance(out, str)
        self.assertEqual(out, data)

    def test_asstr(self):
        ds = self.f.create_dataset('x', (10,), dtype=h5py.string_dtype())
        data = "fàilte"
        ds[0] = data

        strwrap1 = ds.asstr('ascii')
        with self.assertRaises(UnicodeDecodeError):
            out = strwrap1[0]

        # Different errors parameter
        self.assertEqual(ds.asstr('ascii', 'ignore')[0], 'filte')

        # latin-1 will decode it but give the wrong text
        self.assertNotEqual(ds.asstr('latin-1')[0], data)

        # len of ds
        self.assertEqual(10, len(ds.asstr()))


        # Array output
        np.testing.assert_array_equal(
            ds.asstr()[:1], np.array([data], dtype=object)
        )

    def test_asstr_fixed(self):
        dt = h5py.string_dtype(length=5)
        ds = self.f.create_dataset('x', (10,), dtype=dt)
        data = 'cù'
        ds[0] = np.array(data.encode('utf-8'), dtype=dt)

        self.assertIsInstance(ds[0], np.bytes_)
        out = ds.asstr()[0]
        self.assertIsInstance(out, str)
        self.assertEqual(out, data)

        # Different errors parameter
        self.assertEqual(ds.asstr('ascii', 'ignore')[0], 'c')

        # latin-1 will decode it but give the wrong text
        self.assertNotEqual(ds.asstr('latin-1')[0], data)

        # Array output
        np.testing.assert_array_equal(
            ds.asstr()[:1], np.array([data], dtype=object)
        )

    def test_unicode_write_error(self):
        """Encoding error when writing a non-ASCII string to an ASCII vlen dataset"""
        dt = h5py.string_dtype('ascii')
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = "fàilte"
        with self.assertRaises(UnicodeEncodeError):
            ds[0] = data

    def test_unicode_write_bytes(self):
        """ Writing valid utf-8 byte strings to a unicode vlen dataset is OK
        """
        dt = h5py.string_dtype()
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = (u"Hello there" + chr(0x2034)).encode('utf8')
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), bytes)
        self.assertEqual(out, data)

    def test_vlen_bytes_write_ascii_str(self):
        """ Writing an ascii str to ascii vlen dataset is OK
        """
        dt = h5py.string_dtype('ascii')
        ds = self.f.create_dataset('x', (100,), dtype=dt)
        data = "ASCII string"
        ds[0] = data
        out = ds[0]
        self.assertEqual(type(out), bytes)
        self.assertEqual(out, data.encode('ascii'))


class TestCompound(BaseDataset):

    """
        Feature: Compound types correctly round-trip
    """

    def test_rt(self):
        """ Compound types are read back in correct order (issue 236)"""

        dt = np.dtype([ ('weight', np.float64),
                             ('cputime', np.float64),
                             ('walltime', np.float64),
                             ('parents_offset', np.uint32),
                             ('n_parents', np.uint32),
                             ('status', np.uint8),
                             ('endpoint_type', np.uint8), ])

        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random((16,)) * 100

        self.f['test'] = testdata
        outdata = self.f['test'][...]
        self.assertTrue(np.all(outdata == testdata))
        self.assertEqual(outdata.dtype, testdata.dtype)

    def test_assign(self):
        dt = np.dtype([ ('weight', (np.float64, 3)),
                         ('endpoint_type', np.uint8), ])

        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random(size=testdata[key].shape) * 100

        ds = self.f.create_dataset('test', (16,), dtype=dt)
        for key in dt.fields:
            ds[key] = testdata[key]

        outdata = self.f['test'][...]

        self.assertTrue(np.all(outdata == testdata))
        self.assertEqual(outdata.dtype, testdata.dtype)

    def test_fields(self):
        dt = np.dtype([
            ('x', np.float64),
            ('y', np.float64),
            ('z', np.float64),
        ])

        testdata = np.ndarray((16,), dtype=dt)
        for key in dt.fields:
            testdata[key] = np.random.random((16,)) * 100

        self.f['test'] = testdata

        # Extract multiple fields
        np.testing.assert_array_equal(
            self.f['test'].fields(['x', 'y'])[:], testdata[['x', 'y']]
        )
        # Extract single field
        np.testing.assert_array_equal(
            self.f['test'].fields('x')[:], testdata['x']
        )

        # Check len() on fields wrapper
        assert len(self.f['test'].fields('x')) == 16


class TestSubarray(BaseDataset):
    def test_write_list(self):
        ds = self.f.create_dataset("a", (1,), dtype="3int8")
        ds[0] = [1, 2, 3]
        np.testing.assert_array_equal(ds[:], [[1, 2, 3]])

        ds[:] = [[4, 5, 6]]
        np.testing.assert_array_equal(ds[:], [[4, 5, 6]])

    def test_write_array(self):
        ds = self.f.create_dataset("a", (1,), dtype="3int8")
        ds[0] = np.array([1, 2, 3])
        np.testing.assert_array_equal(ds[:], [[1, 2, 3]])

        ds[:] = np.array([[4, 5, 6]])
        np.testing.assert_array_equal(ds[:], [[4, 5, 6]])


class TestEnum(BaseDataset):

    """
        Feature: Enum datatype info is preserved, read/write as integer
    """

    EDICT = {'RED': 0, 'GREEN': 1, 'BLUE': 42}

    def test_create(self):
        """ Enum datasets can be created and type correctly round-trips """
        dt = h5py.enum_dtype(self.EDICT, basetype='i')
        ds = self.f.create_dataset('x', (100, 100), dtype=dt)
        dt2 = ds.dtype
        dict2 = h5py.check_enum_dtype(dt2)
        self.assertEqual(dict2, self.EDICT)

    def test_readwrite(self):
        """ Enum datasets can be read/written as integers """
        dt = h5py.enum_dtype(self.EDICT, basetype='i4')
        ds = self.f.create_dataset('x', (100, 100), dtype=dt)
        ds[35, 37] = 42
        ds[1, :] = 1
        self.assertEqual(ds[35, 37], 42)
        self.assertArrayEqual(ds[1, :], np.array((1,) * 100, dtype='i4'))


class TestFloats(BaseDataset):

    """
        Test support for mini and extended-precision floats
    """

    def _exectest(self, dt):
        dset = self.f.create_dataset('x', (100,), dtype=dt)
        self.assertEqual(dset.dtype, dt)
        data = np.ones((100,), dtype=dt)
        dset[...] = data
        self.assertArrayEqual(dset[...], data)

    @ut.skipUnless(hasattr(np, 'float16'), "NumPy float16 support required")
    def test_mini(self):
        """ Mini-floats round trip """
        self._exectest(np.dtype('float16'))

    # TODO: move these tests to test_h5t
    def test_mini_mapping(self):
        """ Test mapping for float16 """
        if hasattr(np, 'float16'):
            self.assertEqual(h5t.IEEE_F16LE.dtype, np.dtype('<f2'))
        else:
            self.assertEqual(h5t.IEEE_F16LE.dtype, np.dtype('<f4'))


class TestTrackTimes(BaseDataset):

    """
        Feature: track_times
    """

    def test_disable_track_times(self):
        """ check that when track_times=False, the time stamp=0 (Jan 1, 1970) """
        ds = self.f.create_dataset('foo', (4,), track_times=False)
        ds_mtime = h5py.h5g.get_objinfo(ds._id).mtime
        self.assertEqual(0, ds_mtime)

    def test_invalid_track_times(self):
        """ check that when give track_times an invalid value """
        with self.assertRaises(TypeError):
            self.f.create_dataset('foo', (4,), track_times='null')


class TestZeroShape(BaseDataset):

    """
        Features of datasets with (0,)-shape axes
    """

    def test_array_conversion(self):
        """ Empty datasets can be converted to NumPy arrays """
        ds = self.f.create_dataset('x', 0, maxshape=None)
        self.assertEqual(ds.shape, np.array(ds).shape)

        ds = self.f.create_dataset('y', (0,), maxshape=(None,))
        self.assertEqual(ds.shape, np.array(ds).shape)

        ds = self.f.create_dataset('z', (0, 0), maxshape=(None, None))
        self.assertEqual(ds.shape, np.array(ds).shape)

    def test_reading(self):
        """ Slicing into empty datasets works correctly """
        dt = [('a', 'f'), ('b', 'i')]
        ds = self.f.create_dataset('x', (0,), dtype=dt, maxshape=(None,))
        arr = np.empty((0,), dtype=dt)

        self.assertEqual(ds[...].shape, arr.shape)
        self.assertEqual(ds[...].dtype, arr.dtype)
        self.assertEqual(ds[()].shape, arr.shape)
        self.assertEqual(ds[()].dtype, arr.dtype)

# https://github.com/h5py/h5py/issues/1492
empty_regionref_xfail = pytest.mark.xfail(
    h5py.version.hdf5_version_tuple == (1, 10, 6),
    reason="Issue with empty region refs in HDF5 1.10.6",
)

class TestRegionRefs(BaseDataset):

    """
        Various features of region references
    """

    def setUp(self):
        BaseDataset.setUp(self)
        self.data = np.arange(100 * 100).reshape((100, 100))
        self.dset = self.f.create_dataset('x', data=self.data)
        self.dset[...] = self.data

    def test_create_ref(self):
        """ Region references can be used as slicing arguments """
        slic = np.s_[25:35, 10:100:5]
        ref = self.dset.regionref[slic]
        self.assertArrayEqual(self.dset[ref], self.data[slic])

    @empty_regionref_xfail
    def test_empty_region(self):
        ref = self.dset.regionref[:0]
        out = self.dset[ref]
        assert out.size == 0
        # Ideally we should preserve shape (0, 100), but it seems this is lost.

    @empty_regionref_xfail
    def test_scalar_dataset(self):
        ds = self.f.create_dataset("scalar", data=1.0, dtype='f4')
        sid = h5py.h5s.create(h5py.h5s.SCALAR)

        # Deselected
        sid.select_none()
        ref = h5py.h5r.create(ds.id, b'.', h5py.h5r.DATASET_REGION, sid)
        assert ds[ref] == h5py.Empty(np.dtype('f4'))

        # Selected
        sid.select_all()
        ref = h5py.h5r.create(ds.id, b'.', h5py.h5r.DATASET_REGION, sid)
        assert ds[ref] == ds[()]

    def test_ref_shape(self):
        """ Region reference shape and selection shape """
        slic = np.s_[25:35, 10:100:5]
        ref = self.dset.regionref[slic]
        self.assertEqual(self.dset.regionref.shape(ref), self.dset.shape)
        self.assertEqual(self.dset.regionref.selection(ref), (10, 18))


class TestAstype(BaseDataset):
    """.astype() wrapper & context manager
    """
    def test_astype_ctx(self):
        dset = self.f.create_dataset('x', (100,), dtype='i2')
        dset[...] = np.arange(100)

        with warnings.catch_warnings(record=True) as warn_rec:
            warnings.simplefilter("always")

            with dset.astype('f8'):
                self.assertArrayEqual(dset[...], np.arange(100, dtype='f8'))

            with dset.astype('f4') as f4ds:
                self.assertArrayEqual(f4ds[...], np.arange(100, dtype='f4'))

        assert [w.category for w in warn_rec] == [H5pyDeprecationWarning] * 2

    def test_astype_wrapper(self):
        dset = self.f.create_dataset('x', (100,), dtype='i2')
        dset[...] = np.arange(100)
        arr = dset.astype('f4')[:]
        self.assertArrayEqual(arr, np.arange(100, dtype='f4'))


    def test_astype_wrapper_len(self):
        dset = self.f.create_dataset('x', (100,), dtype='i2')
        dset[...] = np.arange(100)
        self.assertEqual(100, len(dset.astype('f4')))


class TestScalarCompound(BaseDataset):

    """
        Retrieval of a single field from a scalar compound dataset should
        strip the field info
    """

    def test_scalar_compound(self):

        dt = np.dtype([('a', 'i')])
        dset = self.f.create_dataset('x', (), dtype=dt)
        self.assertEqual(dset['a'].dtype, np.dtype('i'))


class TestVlen(BaseDataset):
    def test_int(self):
        dt = h5py.vlen_dtype(int)
        ds = self.f.create_dataset('vlen', (4,), dtype=dt)
        ds[0] = np.arange(3)
        ds[1] = np.arange(0)
        ds[2] = [1, 2, 3]
        ds[3] = np.arange(1)
        self.assertArrayEqual(ds[0], np.arange(3))
        self.assertArrayEqual(ds[1], np.arange(0))
        self.assertArrayEqual(ds[2], np.array([1, 2, 3]))
        self.assertArrayEqual(ds[1], np.arange(0))
        ds[0:2] = np.array([np.arange(5), np.arange(4)], dtype=object)
        self.assertArrayEqual(ds[0], np.arange(5))
        self.assertArrayEqual(ds[1], np.arange(4))
        ds[0:2] = np.array([np.arange(3), np.arange(3)])
        self.assertArrayEqual(ds[0], np.arange(3))
        self.assertArrayEqual(ds[1], np.arange(3))

    def test_reuse_from_other(self):
        dt = h5py.vlen_dtype(int)
        ds = self.f.create_dataset('vlen', (1,), dtype=dt)
        self.f.create_dataset('vlen2', (1,), ds[()].dtype)

    def test_reuse_struct_from_other(self):
        dt = [('a', int), ('b', h5py.vlen_dtype(int))]
        ds = self.f.create_dataset('vlen', (1,), dtype=dt)
        fname = self.f.filename
        self.f.close()
        self.f = h5py.File(fname, 'a')
        self.f.create_dataset('vlen2', (1,), self.f['vlen']['b'][()].dtype)

    def test_convert(self):
        dt = h5py.vlen_dtype(int)
        ds = self.f.create_dataset('vlen', (3,), dtype=dt)
        ds[0] = np.array([1.4, 1.2])
        ds[1] = np.array([1.2])
        ds[2] = [1.2, 2, 3]
        self.assertArrayEqual(ds[0], np.array([1, 1]))
        self.assertArrayEqual(ds[1], np.array([1]))
        self.assertArrayEqual(ds[2], np.array([1, 2, 3]))
        ds[0:2] = np.array([[0.1, 1.1, 2.1, 3.1, 4], np.arange(4)], dtype=object)
        self.assertArrayEqual(ds[0], np.arange(5))
        self.assertArrayEqual(ds[1], np.arange(4))
        ds[0:2] = np.array([np.array([0.1, 1.2, 2.2]),
                            np.array([0.2, 1.2, 2.2])])
        self.assertArrayEqual(ds[0], np.arange(3))
        self.assertArrayEqual(ds[1], np.arange(3))

    def test_multidim(self):
        dt = h5py.vlen_dtype(int)
        ds = self.f.create_dataset('vlen', (2, 2), dtype=dt)
        ds[0, 0] = np.arange(1)
        ds[:, :] = np.array([[np.arange(3), np.arange(2)],
                            [np.arange(1), np.arange(2)]], dtype=object)
        ds[:, :] = np.array([[np.arange(2), np.arange(2)],
                             [np.arange(2), np.arange(2)]])

    def _help_float_testing(self, np_dt, dataset_name='vlen'):
        """
        Helper for testing various vlen numpy data types.
        :param np_dt: Numpy datatype to test
        :param dataset_name: String name of the dataset to create for testing.
        """
        dt = h5py.vlen_dtype(np_dt)
        ds = self.f.create_dataset(dataset_name, (5,), dtype=dt)

        # Create some arrays, and assign them to the dataset
        array_0 = np.array([1., 2., 30.], dtype=np_dt)
        array_1 = np.array([100.3, 200.4, 98.1, -10.5, -300.0], dtype=np_dt)

        # Test that a numpy array of different type gets cast correctly
        array_2 = np.array([1, 2, 8], dtype=np.dtype('int32'))
        casted_array_2 = array_2.astype(np_dt)

        # Test that we can set a list of floats.
        list_3 = [1., 2., 900., 0., -0.5]
        list_array_3 = np.array(list_3, dtype=np_dt)

        # Test that a list of integers gets casted correctly
        list_4 = [-1, -100, 0, 1, 9999, 70]
        list_array_4 = np.array(list_4, dtype=np_dt)

        ds[0] = array_0
        ds[1] = array_1
        ds[2] = array_2
        ds[3] = list_3
        ds[4] = list_4

        self.assertArrayEqual(array_0, ds[0])
        self.assertArrayEqual(array_1, ds[1])
        self.assertArrayEqual(casted_array_2, ds[2])
        self.assertArrayEqual(list_array_3, ds[3])
        self.assertArrayEqual(list_array_4, ds[4])

        # Test that we can reassign arrays in the dataset
        list_array_3 = np.array([0.3, 2.2], dtype=np_dt)

        ds[0] = list_array_3[:]

        self.assertArrayEqual(list_array_3, ds[0])

        # Make sure we can close the file.
        self.f.flush()
        self.f.close()

    def test_numpy_float16(self):
        np_dt = np.dtype('float16')
        self._help_float_testing(np_dt)

    def test_numpy_float32(self):
        np_dt = np.dtype('float32')
        self._help_float_testing(np_dt)

    def test_numpy_float64_from_dtype(self):
        np_dt = np.dtype('float64')
        self._help_float_testing(np_dt)

    def test_numpy_float64_2(self):
        np_dt = np.float64
        self._help_float_testing(np_dt)

    def test_non_contiguous_arrays(self):
        """Test that non-contiguous arrays are stored correctly"""
        self.f.create_dataset('nc', (10,), dtype=h5py.vlen_dtype('bool'))
        x = np.array([True, False, True, True, False, False, False])
        self.f['nc'][0] = x[::2]

        assert all(self.f['nc'][0] == x[::2]), f"{self.f['nc'][0]} != {x[::2]}"

        self.f.create_dataset('nc2', (10,), dtype=h5py.vlen_dtype('int8'))
        y = np.array([2, 4, 1, 5, -1, 3, 7])
        self.f['nc2'][0] = y[::2]

        assert all(self.f['nc2'][0] == y[::2]), f"{self.f['nc2'][0]} != {y[::2]}"


class TestLowOpen(BaseDataset):

    def test_get_access_list(self):
        """ Test H5Dget_access_plist """
        ds = self.f.create_dataset('foo', (4,))
        p_list = ds.id.get_access_plist()

    def test_dapl(self):
        """ Test the dapl keyword to h5d.open """
        dapl = h5py.h5p.create(h5py.h5p.DATASET_ACCESS)
        dset = self.f.create_dataset('x', (100,))
        del dset
        dsid = h5py.h5d.open(self.f.id, b'x', dapl)
        self.assertIsInstance(dsid, h5py.h5d.DatasetID)


@ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 5),
               "chunk info requires  HDF5 >= 1.10.5")
def test_get_chunk_details():
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        fout.create_dataset('test', shape=(100, 100), chunks=(10, 10), dtype='i4')
        fout['test'][:] = 1

    buf.seek(0)
    with h5py.File(buf, 'r') as fin:
        ds = fin['test'].id

        assert ds.get_num_chunks() == 100
        for j in range(100):
            offset = tuple(np.array(np.unravel_index(j, (10, 10))) * 10)

            si = ds.get_chunk_info(j)
            assert si.chunk_offset == offset
            assert si.filter_mask == 0
            assert si.byte_offset is not None
            assert si.size > 0

        si = ds.get_chunk_info_by_coord((0, 0))
        assert si.chunk_offset == (0, 0)
        assert si.filter_mask == 0
        assert si.byte_offset is not None
        assert si.size > 0


def test_empty_shape(writable_file):
    ds = writable_file.create_dataset('empty', dtype='int32')
    assert ds.shape is None
    assert ds.maxshape is None


def test_zero_storage_size():
    # https://github.com/h5py/h5py/issues/1475
    from io import BytesIO
    buf = BytesIO()
    with h5py.File(buf, 'w') as fout:
        fout.create_dataset('empty', dtype='uint8')

    buf.seek(0)
    with h5py.File(buf, 'r') as fin:
        assert fin['empty'].chunks is None
        assert fin['empty'].id.get_offset() is None
        assert fin['empty'].id.get_storage_size() == 0


def test_python_int_uint64(writable_file):
    # https://github.com/h5py/h5py/issues/1547
    data = [np.iinfo(np.int64).max, np.iinfo(np.int64).max + 1]

    # Check creating a new dataset
    ds = writable_file.create_dataset('x', data=data, dtype=np.uint64)
    assert ds.dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(ds[:], np.array(data, dtype=np.uint64))

    # Check writing to an existing dataset
    ds[:] = data
    np.testing.assert_array_equal(ds[:], np.array(data, dtype=np.uint64))


def test_setitem_fancy_indexing(writable_file):
    # https://github.com/h5py/h5py/issues/1593
    arr = writable_file.create_dataset('data', (5, 1000, 2), dtype=np.uint8)
    block = np.random.randint(255, size=(5, 3, 2))
    arr[:, [0, 2, 4], ...] = block


def test_vlen_spacepad():
    with File(get_data_file_path("vlen_string_dset.h5")) as f:
        assert f["DS1"][0] == b"Parting"


def test_vlen_nullterm():
    with File(get_data_file_path("vlen_string_dset_utc.h5")) as f:
        assert f["ds1"][0] == b"2009-12-20T10:16:18.662409Z"


@pytest.mark.skipif(
    h5py.version.hdf5_version_tuple < (1, 10, 3),
    reason="Appears you cannot pass an unknown filter id for HDF5 < 1.10.3"
)
def test_allow_unknown_filter(writable_file):
    # apparently 256-511 are reserved for testing purposes
    fake_filter_id = 256
    ds = writable_file.create_dataset(
        'data', shape=(10, 10), dtype=np.uint8, compression=fake_filter_id,
        allow_unknown_filter=True
    )
    assert str(fake_filter_id) in ds._filters


class TestCommutative(BaseDataset):
    """
    Test the symmetry of operators, at least with the numpy types.
    Issue: https://github.com/h5py/h5py/issues/1947
    """
    def test_numpy_commutative(self,):
        """
        Create a h5py dataset, extract one element convert to numpy
        Check that it returns symmetric response to == and !=
        """
        shape = (100,1)
        dset = self.f.create_dataset("test", shape, dtype=float,
                                     data=np.random.rand(*shape))

        # grab a value from the elements, ie dset[0]
        # check that mask arrays are commutative wrt ==, !=
        val = np.float64(dset[0])

        assert np.all((val == dset) == (dset == val))
        assert np.all((val != dset) == (dset != val))

        # generate sample not in the dset, ie max(dset)+delta
        # check that mask arrays are commutative wrt ==, !=
        delta = 0.001
        nval = np.nanmax(dset)+delta

        assert np.all((nval == dset) == (dset == nval))
        assert np.all((nval != dset) == (dset != nval))

    def test_basetype_commutative(self,):
        """
        Create a h5py dataset and check basetype compatibility.
        Check that operation is symmetric, even if it is potentially
        not meaningful.
        """
        shape = (100,1)
        dset = self.f.create_dataset("test", shape, dtype=float,
                                     data=np.random.rand(*shape))

        # generate float type, sample float(0.)
        # check that operation is symmetric (but potentially meaningless)
        val = float(0.)
        assert (val == dset) == (dset == val)
        assert (val != dset) == (dset != val)
