# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Attribute data transfer testing module

    Covers all data read/write and type-conversion operations for attributes.
"""

import numpy as np

from .common import TestCase, ut

import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace


class BaseAttrs(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestScalar(BaseAttrs):

    """
        Feature: Scalar types map correctly to array scalars
    """

    def test_int(self):
        """ Integers are read as correct NumPy type """
        self.f.attrs['x'] = np.array(1, dtype=np.int8)
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalars are read as numpy.void """
        dt = np.dtype([('a', 'i'), ('b', 'f')])
        data = np.array((1, 4.2), dtype=dt)
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.void)
        self.assertEqual(out, data)
        self.assertEqual(out['b'], data['b'])


class TestArray(BaseAttrs):

    """
        Feature: Non-scalar types are correctly retrieved as ndarrays
    """

    def test_single(self):
        """ Single-element arrays are correctly recovered """
        data = np.ndarray((1,), dtype='f')
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_multi(self):
        """ Rank-1 arrays are correctly recovered """
        data = np.ndarray((42,), dtype='f')
        data[:] = 42.0
        data[10:35] = -47.0
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (42,))
        self.assertArrayEqual(out, data)


class TestTypes(BaseAttrs):

    """
        Feature: All supported types can be stored in attributes
    """

    def test_int(self):
        """ Storage of integer types """
        dtypes = (np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_float(self):
        """ Storage of floating point types """
        dtypes = tuple(np.dtype(x) for x in ('<f4', '>f4', '>f8', '<f8'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 42.3
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            # TODO: Clean up after issue addressed !
            print("dtype: ", out.dtype, dt)
            print("value: ", out, data)
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_complex(self):
        """ Storage of complex types """
        dtypes = tuple(np.dtype(x) for x in ('<c8', '>c8', '<c16', '>c16'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = -4.2j + 35.9
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertArrayEqual(out, data)

    def test_string(self):
        """ Storage of fixed-length strings """
        dtypes = tuple(np.dtype(x) for x in ('|S1', '|S10'))

        for dt in dtypes:
            data = np.ndarray((1,), dtype=dt)
            data[...] = 'h'
            self.f.attrs['x'] = data
            out = self.f.attrs['x']
            self.assertEqual(out.dtype, dt)
            self.assertEqual(out[0], data[0])

    def test_bool(self):
        """ Storage of NumPy booleans """

        data = np.ndarray((2,), dtype=np.bool_)
        data[...] = True, False
        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertEqual(out.dtype, data.dtype)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_vlen_string_array(self):
        """ Storage of vlen byte string arrays"""
        dt = h5py.string_dtype(encoding='ascii')

        data = np.ndarray((2,), dtype=dt)
        data[...] = "Hello", "Hi there!  This is HDF5!"

        self.f.attrs['x'] = data
        out = self.f.attrs['x']
        self.assertEqual(out.dtype, dt)
        self.assertEqual(out[0], data[0])
        self.assertEqual(out[1], data[1])

    def test_string_scalar(self):
        """ Storage of variable-length byte string scalars (auto-creation) """

        self.f.attrs['x'] = b'Hello'
        out = self.f.attrs['x']

        self.assertEqual(out, 'Hello')
        self.assertEqual(type(out), str)

        aid = h5py.h5a.open(self.f.id, b"x")
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_ASCII)
        self.assertTrue(tid.is_variable_str())

    def test_unicode_scalar(self):
        """ Storage of variable-length unicode strings (auto-creation) """

        self.f.attrs['x'] = u"Hello" + chr(0x2340) + u"!!"
        out = self.f.attrs['x']
        self.assertEqual(out, u"Hello" + chr(0x2340) + u"!!")
        self.assertEqual(type(out), str)

        aid = h5py.h5a.open(self.f.id, b"x")
        tid = aid.get_type()
        self.assertEqual(type(tid), h5py.h5t.TypeStringID)
        self.assertEqual(tid.get_cset(), h5py.h5t.CSET_UTF8)
        self.assertTrue(tid.is_variable_str())


class TestEmpty(BaseAttrs):

    def setUp(self):
        BaseAttrs.setUp(self)
        sid = h5s.create(h5s.NULL)
        tid = h5t.C_S1.copy()
        tid.set_size(10)
        aid = h5a.create(self.f.id, b'x', tid, sid)
        self.empty_obj = h5py.Empty(np.dtype("S10"))

    def test_read(self):
        self.assertEqual(
            self.empty_obj, self.f.attrs['x']
        )

    def test_write(self):
        self.f.attrs["y"] = self.empty_obj
        self.assertTrue(is_empty_dataspace(h5a.open(self.f.id, b'y')))

    def test_modify(self):
        with self.assertRaises(IOError):
            self.f.attrs.modify('x', 1)

    def test_values(self):
        # list() is for Py3 where these are iterators
        values = list(self.f.attrs.values())
        self.assertEqual(
            [self.empty_obj], values
        )

    def test_items(self):
        items = list(self.f.attrs.items())
        self.assertEqual(
            [(u"x", self.empty_obj)], items
        )

    def test_itervalues(self):
        values = list(self.f.attrs.values())
        self.assertEqual(
            [self.empty_obj], values
        )

    def test_iteritems(self):
        items = list(self.f.attrs.items())
        self.assertEqual(
            [(u"x", self.empty_obj)], items
        )


class TestWriteException(BaseAttrs):

    """
        Ensure failed attribute writes don't leave garbage behind.
    """

    def test_write(self):
        """ ValueError on string write wipes out attribute """

        s = b"Hello\x00Hello"

        try:
            self.f.attrs['x'] = s
        except ValueError:
            pass

        with self.assertRaises(KeyError):
            self.f.attrs['x']
