# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Dataset slicing test module.

    Tests all supported slicing operations, including read/write and
    broadcasting operations.  Does not test type conversion except for
    corner cases overlapping with slicing; for example, when selecting
    specific fields of a compound type.
"""

import numpy as np

from .common import ut, TestCase

import h5py
from h5py import h5s, h5t, h5d
from h5py import File, MultiBlockSlice

class BaseSlicing(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestSingleElement(BaseSlicing):

    """
        Feature: Retrieving a single element works with NumPy semantics
    """

    def test_single_index(self):
        """ Single-element selection with [index] yields array scalar """
        dset = self.f.create_dataset('x', (1,), dtype='i1')
        out = dset[0]
        self.assertIsInstance(out, np.int8)

    def test_single_null(self):
        """ Single-element selection with [()] yields ndarray """
        dset = self.f.create_dataset('x', (1,), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1,))

    def test_scalar_index(self):
        """ Slicing with [...] yields scalar ndarray """
        dset = self.f.create_dataset('x', shape=(), dtype='f')
        out = dset[...]
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, ())

    def test_scalar_null(self):
        """ Slicing with [()] yields array scalar """
        dset = self.f.create_dataset('x', shape=(), dtype='i1')
        out = dset[()]
        self.assertIsInstance(out, np.int8)

    def test_compound(self):
        """ Compound scalar is numpy.void, not tuple (issue 135) """
        dt = np.dtype([('a','i4'),('b','f8')])
        v = np.ones((4,), dtype=dt)
        dset = self.f.create_dataset('foo', (4,), data=v)
        self.assertEqual(dset[0], v[0])
        self.assertIsInstance(dset[0], np.void)

class TestObjectIndex(BaseSlicing):

    """
        Feature: numpy.object_ subtypes map to real Python objects
    """

    def test_reference(self):
        """ Indexing a reference dataset returns a h5py.Reference instance """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.ref_dtype)
        dset[0] = self.f.ref
        self.assertEqual(type(dset[0]), h5py.Reference)

    def test_regref(self):
        """ Indexing a region reference dataset returns a h5py.RegionReference
        """
        dset1 = self.f.create_dataset('x', (10,10))
        regref = dset1.regionref[...]
        dset2 = self.f.create_dataset('y', (1,), dtype=h5py.regionref_dtype)
        dset2[0] = regref
        self.assertEqual(type(dset2[0]), h5py.RegionReference)

    def test_reference_field(self):
        """ Compound types of which a reference is an element work right """
        dt = np.dtype([('a', 'i'),('b', h5py.ref_dtype)])

        dset = self.f.create_dataset('x', (1,), dtype=dt)
        dset[0] = (42, self.f['/'].ref)

        out = dset[0]
        self.assertEqual(type(out[1]), h5py.Reference)  # isinstance does NOT work

    def test_scalar(self):
        """ Indexing returns a real Python object on scalar datasets """
        dset = self.f.create_dataset('x', (), dtype=h5py.ref_dtype)
        dset[()] = self.f.ref
        self.assertEqual(type(dset[()]), h5py.Reference)

    def test_bytestr(self):
        """ Indexing a byte string dataset returns a real python byte string
        """
        dset = self.f.create_dataset('x', (1,), dtype=h5py.string_dtype(encoding='ascii'))
        dset[0] = b"Hello there!"
        self.assertEqual(type(dset[0]), bytes)

class TestSimpleSlicing(TestCase):

    """
        Feature: Simple NumPy-style slices (start:stop:step) are supported.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_negative_stop(self):
        """ Negative stop indexes work as they do in NumPy """
        self.assertArrayEqual(self.dset[2:-2], self.arr[2:-2])

    def test_write(self):
        """Assigning to a 1D slice of a 2D dataset
        """
        dset = self.f.create_dataset('x2', (10, 2))

        x = np.zeros((10, 1))
        dset[:, 0] = x[:, 0]
        with self.assertRaises(TypeError):
            dset[:, 1] = x

class TestArraySlicing(BaseSlicing):

    """
        Feature: Array types are handled appropriately
    """

    def test_read(self):
        """ Read arrays tack array dimensions onto end of shape tuple """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x',(10,),dtype=dt)
        self.assertEqual(dset.shape, (10,))
        self.assertEqual(dset.dtype, dt)

        # Full read
        out = dset[...]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (10,3))

        # Single element
        out = dset[0]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3,))

        # Range
        out = dset[2:8:2]
        self.assertEqual(out.dtype, np.dtype('f8'))
        self.assertEqual(out.shape, (3,3))

    def test_write_broadcast(self):
        """ Array fill from constant is not supported (issue 211).
        """
        dt = np.dtype('(3,)i')

        dset = self.f.create_dataset('x', (10,), dtype=dt)

        with self.assertRaises(TypeError):
            dset[...] = 42

    def test_write_element(self):
        """ Write a single element to the array

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)

        data = np.array([1,2,3.0])
        dset[4] = data

        out = dset[4]
        self.assertTrue(np.all(out == data))

    def test_write_slices(self):
        """ Write slices to array type """
        dt = np.dtype('(3,)i')

        data1 = np.ones((2,), dtype=dt)
        data2 = np.ones((4,5), dtype=dt)

        dset = self.f.create_dataset('x', (10,9,11), dtype=dt)

        dset[0,0,2:4] = data1
        self.assertArrayEqual(dset[0,0,2:4], data1)

        dset[3, 1:5, 6:11] = data2
        self.assertArrayEqual(dset[3, 1:5, 6:11], data2)


    def test_roundtrip(self):
        """ Read the contents of an array and write them back

        Issue 211.
        """
        dt = np.dtype('(3,)f8')
        dset = self.f.create_dataset('x', (10,), dtype=dt)

        out = dset[...]
        dset[...] = out

        self.assertTrue(np.all(dset[...] == out))


class TestZeroLengthSlicing(BaseSlicing):

    """
        Slices resulting in empty arrays
    """

    def test_slice_zero_length_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along the zero-length dimension """
        for i, shape in enumerate([(0,), (0, 3), (0, 2, 1)]):
            dset = self.f.create_dataset('x%d'%i, shape, dtype=int, maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[...]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            out = dset[:]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, shape)
            if len(shape) > 1:
                out = dset[:, :1]
                self.assertIsInstance(out, np.ndarray)
                self.assertEqual(out.shape[:2], (0, 1))

    def test_slice_other_dimension(self):
        """ Slice a dataset with a zero in its shape vector
            along a non-zero-length dimension """
        for i, shape in enumerate([(3, 0), (1, 2, 0), (2, 0, 1)]):
            dset = self.f.create_dataset('x%d'%i, shape, dtype=int, maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (1,)+shape[1:])

    def test_slice_of_length_zero(self):
        """ Get a slice of length zero from a non-empty dataset """
        for i, shape in enumerate([(3,), (2, 2,), (2,  1, 5)]):
            dset = self.f.create_dataset('x%d'%i, data=np.zeros(shape, int), maxshape=(None,)*len(shape))
            self.assertEqual(dset.shape, shape)
            out = dset[1:1]
            self.assertIsInstance(out, np.ndarray)
            self.assertEqual(out.shape, (0,)+shape[1:])

class TestFieldNames(BaseSlicing):

    """
        Field names for read & write
    """

    dt = np.dtype([('a', 'f'), ('b', 'i'), ('c', 'f4')])
    data = np.ones((100,), dtype=dt)

    def setUp(self):
        BaseSlicing.setUp(self)
        self.dset = self.f.create_dataset('x', (100,), dtype=self.dt)
        self.dset[...] = self.data

    def test_read(self):
        """ Test read with field selections """
        self.assertArrayEqual(self.dset['a'], self.data['a'])

    def test_unicode_names(self):
        """ Unicode field names for for read and write """
        self.assertArrayEqual(self.dset['a'], self.data['a'])
        self.dset['a'] = 42
        data = self.data.copy()
        data['a'] = 42
        self.assertArrayEqual(self.dset['a'], data['a'])

    def test_write(self):
        """ Test write with field selections """
        data2 = self.data.copy()
        data2['a'] *= 2
        self.dset['a'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['b'] *= 4
        self.dset['b'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))
        data2['a'] *= 3
        data2['c'] *= 3
        self.dset['a','c'] = data2
        self.assertTrue(np.all(self.dset[...] == data2))

    def test_write_noncompound(self):
        """ Test write with non-compound source (single-field) """
        data2 = self.data.copy()
        data2['b'] = 1.0
        self.dset['b'] = 1.0
        self.assertTrue(np.all(self.dset[...] == data2))


class TestMultiBlockSlice(BaseSlicing):

    def setUp(self):
        super(TestMultiBlockSlice, self).setUp()
        self.arr = np.arange(10)
        self.dset = self.f.create_dataset('x', data=self.arr)

    def test_default(self):
        # Default selects entire dataset as one block
        mbslice = MultiBlockSlice()

        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_default_explicit(self):
        mbslice = MultiBlockSlice(start=0, count=10, stride=1, block=1)

        self.assertEqual(mbslice.indices(10), (0, 1, 10, 1))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_start(self):
        mbslice = MultiBlockSlice(start=4)

        self.assertEqual(mbslice.indices(10), (4, 1, 6, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([4, 5, 6, 7, 8, 9]))

    def test_count(self):
        mbslice = MultiBlockSlice(count=7)

        self.assertEqual(mbslice.indices(10), (0, 1, 7, 1))
        np.testing.assert_array_equal(
            self.dset[mbslice], np.array([0, 1, 2, 3, 4, 5, 6])
        )

    def test_count_more_than_length_error(self):
        mbslice = MultiBlockSlice(count=11)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_stride(self):
        mbslice = MultiBlockSlice(stride=2)

        self.assertEqual(mbslice.indices(10), (0, 2, 5, 1))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 2, 4, 6, 8]))

    def test_stride_zero_error(self):
        with self.assertRaises(ValueError):
            # This would cause a ZeroDivisionError if not caught
            MultiBlockSlice(stride=0, block=0).indices(10)

    def test_stride_block_equal(self):
        mbslice = MultiBlockSlice(stride=2, block=2)

        self.assertEqual(mbslice.indices(10), (0, 2, 5, 2))
        np.testing.assert_array_equal(self.dset[mbslice], self.arr)

    def test_block_more_than_stride_error(self):
        with self.assertRaises(ValueError):
            MultiBlockSlice(block=3)

        with self.assertRaises(ValueError):
            MultiBlockSlice(stride=2, block=3)

    def test_stride_more_than_block(self):
        mbslice = MultiBlockSlice(stride=3, block=2)

        self.assertEqual(mbslice.indices(10), (0, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([0, 1, 3, 4, 6, 7]))

    def test_block_overruns_extent_error(self):
        # If fully described then must fit within extent
        mbslice = MultiBlockSlice(start=2, count=2, stride=5, block=4)
        with self.assertRaises(ValueError):
            mbslice.indices(10)

    def test_fully_described(self):
        mbslice = MultiBlockSlice(start=1, count=2, stride=5, block=4)

        self.assertEqual(mbslice.indices(10), (1, 5, 2, 4))
        np.testing.assert_array_equal(
            self.dset[mbslice], np.array([1, 2, 3, 4, 6, 7, 8, 9])
        )

    def test_count_calculated(self):
        # If not given, count should be calculated to select as many full blocks as possible
        mbslice = MultiBlockSlice(start=1, stride=3, block=2)

        self.assertEqual(mbslice.indices(10), (1, 3, 3, 2))
        np.testing.assert_array_equal(self.dset[mbslice], np.array([1, 2, 4, 5, 7, 8]))

    def test_zero_count_calculated_error(self):
        # In this case, there is no possible count to select even one block, so error
        mbslice = MultiBlockSlice(start=8, stride=4, block=3)

        with self.assertRaises(ValueError):
            mbslice.indices(10)
