# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

import sys

import numpy as np

from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py


class BaseDataset(TestCase):

    """
    data is a 3-dimensional dataset with dimensions [z, y, x]

    The z dimension is labeled. It does not have any attached scales.
    The y dimension is not labeled. It has one attached scale.
    The x dimension is labeled. It has two attached scales.

    data2 is a 3-dimensional dataset with no associated dimension scales.
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.f['data'] = np.ones((4, 3, 2), 'f')
        self.f['data2'] = np.ones((4, 3, 2), 'f')
        self.f['x1'] = np.ones((2), 'f')
        h5py.h5ds.set_scale(self.f['x1'].id)
        h5py.h5ds.attach_scale(self.f['data'].id, self.f['x1'].id, 2)
        self.f['x2'] = np.ones((2), 'f')
        h5py.h5ds.set_scale(self.f['x2'].id, b'x2 name')
        h5py.h5ds.attach_scale(self.f['data'].id, self.f['x2'].id, 2)
        self.f['y1'] = np.ones((3), 'f')
        h5py.h5ds.set_scale(self.f['y1'].id, b'y1 name')
        h5py.h5ds.attach_scale(self.f['data'].id, self.f['y1'].id, 1)
        self.f['z1'] = np.ones((4), 'f')

        h5py.h5ds.set_label(self.f['data'].id, 0, b'z')
        h5py.h5ds.set_label(self.f['data'].id, 2, b'x')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestH5DSBindings(BaseDataset):

    """
        Feature: Datasets can be created from existing data
    """

    def test_create_dimensionscale(self):
        """ Create a dimension scale from existing dataset """
        self.assertTrue(h5py.h5ds.is_scale(self.f['x1'].id))
        self.assertEqual(h5py.h5ds.get_scale_name(self.f['x1'].id), b'')
        self.assertEqual(self.f['x1'].attrs['CLASS'], b"DIMENSION_SCALE")
        self.assertEqual(h5py.h5ds.get_scale_name(self.f['x2'].id), b'x2 name')

    def test_attach_dimensionscale(self):
        self.assertTrue(
            h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2)
            )
        self.assertFalse(
            h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 1))
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 0), 0)
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 1), 1)
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 2), 2)

    def test_detach_dimensionscale(self):
        self.assertTrue(
            h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2)
            )
        h5py.h5ds.detach_scale(self.f['data'].id, self.f['x1'].id, 2)
        self.assertFalse(
            h5py.h5ds.is_attached(self.f['data'].id, self.f['x1'].id, 2)
            )
        self.assertEqual(h5py.h5ds.get_num_scales(self.f['data'].id, 2), 1)

    # TODO: update condition once the bug is fixed upstream
    @ut.skipUnless(
        h5py.version.hdf5_version_tuple > (2, 0, 0),
        "Reading non-existent label segfaults"
        )
    def test_label_dimensionscale(self):
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 0), b'z')
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 1), b'')
        self.assertEqual(h5py.h5ds.get_label(self.f['data'].id, 2), b'x')

    def test_iter_dimensionscales(self):
        def func(dsid):
            res = h5py.h5ds.get_scale_name(dsid)
            if res == b'x2 name':
                return dsid

        res = h5py.h5ds.iterate(self.f['data'].id, 2, func, 0)
        self.assertEqual(h5py.h5ds.get_scale_name(res), b'x2 name')


class TestDimensionManager(BaseDataset):

    def test_make_scale(self):
        # test recreating or renaming an existing scale:
        self.f['x1'].make_scale(b'foobar')
        self.assertEqual(self.f['data'].dims[2]['foobar'], self.f['x1'])
        # test creating entirely new scale:
        self.f['data2'].make_scale(b'foobaz')
        self.f['data'].dims[2].attach_scale(self.f['data2'])
        self.assertEqual(self.f['data'].dims[2]['foobaz'], self.f['data2'])

    def test_get_dimension(self):
        with self.assertRaises(IndexError):
            self.f['data'].dims[3]

    def test_len(self):
        self.assertEqual(len(self.f['data'].dims), 3)
        self.assertEqual(len(self.f['data2'].dims), 3)

    def test_iter(self):
        dims = self.f['data'].dims
        self.assertEqual(
            [d for d in dims],
            [dims[0], dims[1], dims[2]]
            )

    def test_repr(self):
        ds = self.f.create_dataset('x', (2,3))
        self.assertIsInstance(repr(ds.dims), str)
        self.f.close()
        self.assertIsInstance(repr(ds.dims), str)


class TestDimensionsHighLevel(BaseDataset):

    def test_len(self):
        self.assertEqual(len(self.f['data'].dims[0]), 0)
        self.assertEqual(len(self.f['data'].dims[1]), 1)
        self.assertEqual(len(self.f['data'].dims[2]), 2)
        self.assertEqual(len(self.f['data2'].dims[0]), 0)
        self.assertEqual(len(self.f['data2'].dims[1]), 0)
        self.assertEqual(len(self.f['data2'].dims[2]), 0)

    def test_get_label(self):
        self.assertEqual(self.f['data'].dims[2].label, 'x')
        self.assertEqual(self.f['data'].dims[1].label, '')
        self.assertEqual(self.f['data'].dims[0].label, 'z')
        self.assertEqual(self.f['data2'].dims[2].label, '')
        self.assertEqual(self.f['data2'].dims[1].label, '')
        self.assertEqual(self.f['data2'].dims[0].label, '')

    def test_set_label(self):
        self.f['data'].dims[0].label = 'foo'
        self.assertEqual(self.f['data'].dims[2].label, 'x')
        self.assertEqual(self.f['data'].dims[1].label, '')
        self.assertEqual(self.f['data'].dims[0].label, 'foo')

    def test_detach_scale(self):
        self.f['data'].dims[2].detach_scale(self.f['x1'])
        self.assertEqual(len(self.f['data'].dims[2]), 1)
        self.assertEqual(self.f['data'].dims[2][0], self.f['x2'])
        self.f['data'].dims[2].detach_scale(self.f['x2'])
        self.assertEqual(len(self.f['data'].dims[2]), 0)

    def test_attach_scale(self):
        self.f['x3'] = self.f['x2'][...]
        self.f['data'].dims[2].attach_scale(self.f['x3'])
        self.assertEqual(len(self.f['data'].dims[2]), 3)
        self.assertEqual(self.f['data'].dims[2][2], self.f['x3'])

    def test_get_dimension_scale(self):
        self.assertEqual(self.f['data'].dims[2][0], self.f['x1'])
        with self.assertRaises(RuntimeError):
            self.f['data2'].dims[2][0], self.f['x2']
        self.assertEqual(self.f['data'].dims[2][''], self.f['x1'])
        self.assertEqual(self.f['data'].dims[2]['x2 name'], self.f['x2'])

    def test_get_items(self):
        self.assertEqual(
            self.f['data'].dims[2].items(),
            [('', self.f['x1']), ('x2 name', self.f['x2'])]
            )

    def test_get_keys(self):
        self.assertEqual(self.f['data'].dims[2].keys(), ['', 'x2 name'])

    def test_get_values(self):
        self.assertEqual(
            self.f['data'].dims[2].values(),
            [self.f['x1'], self.f['x2']]
            )

    def test_iter(self):
        self.assertEqual([i for i in self.f['data'].dims[2]], ['', 'x2 name'])

    def test_repr(self):
        ds = self.f["data"]
        self.assertEqual(repr(ds.dims[2])[1:16], '"x" dimension 2')
        self.f.close()
        self.assertIsInstance(repr(ds.dims), str)

    def test_attributes(self):
        self.f["data2"].attrs["DIMENSION_LIST"] = self.f["data"].attrs[
            "DIMENSION_LIST"]
        self.assertEqual(len(self.f['data2'].dims[0]), 0)
        self.assertEqual(len(self.f['data2'].dims[1]), 1)
        self.assertEqual(len(self.f['data2'].dims[2]), 2)
