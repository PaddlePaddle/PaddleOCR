# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Common high-level operations test

    Tests features common to all high-level objects, like the .name property.
"""

from h5py import File
from h5py._hl.base import is_hdf5, Empty
from .common import ut, TestCase, UNICODE_FILENAMES

import numpy as np
import os
import tempfile

class BaseTest(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()


class TestName(BaseTest):

    """
        Feature: .name attribute returns the object name
    """

    def test_anonymous(self):
        """ Anonymous objects have name None """
        grp = self.f.create_group(None)
        self.assertIs(grp.name, None)

class TestParent(BaseTest):

    """
        test the parent group of the high-level interface objects
    """

    def test_object_parent(self):
        # Anonymous objects
        grp = self.f.create_group(None)
        # Parent of an anonymous object is undefined
        with self.assertRaises(ValueError):
            grp.parent

        # Named objects
        grp = self.f.create_group("bar")
        sub_grp = grp.create_group("foo")
        parent = sub_grp.parent.name
        self.assertEqual(parent, "/bar")

class TestMapping(BaseTest):

    """
        Test if the registration of Group as a
        Mapping behaves as expected
    """

    def setUp(self):
        super().setUp()
        data = ('a', 'b')
        self.grp = self.f.create_group('bar')
        self.attr = self.f.attrs.create('x', data)

    def test_keys(self):
        key_1 = self.f.keys()
        self.assertIsInstance(repr(key_1), str)
        key_2 = self.grp.keys()
        self.assertIsInstance(repr(key_2), str)

    def test_values(self):
        value_1 = self.f.values()
        self.assertIsInstance(repr(value_1), str)
        value_2 = self.grp.values()
        self.assertIsInstance(repr(value_2), str)

    def test_items(self):
        item_1 = self.f.items()
        self.assertIsInstance(repr(item_1), str)
        item_2 = self.grp.items()
        self.assertIsInstance(repr(item_1), str)


class TestRepr(BaseTest):

    """
        repr() works correctly with Unicode names
    """

    USTRING = chr(0xfc) + chr(0xdf)

    def _check_type(self, obj):
        self.assertIsInstance(repr(obj), str)

    def test_group(self):
        """ Group repr() with unicode """
        grp = self.f.create_group(self.USTRING)
        self._check_type(grp)

    def test_dataset(self):
        """ Dataset repr() with unicode """
        dset = self.f.create_dataset(self.USTRING, (1,))
        self._check_type(dset)

    def test_namedtype(self):
        """ Named type repr() with unicode """
        self.f['type'] = np.dtype('f')
        typ = self.f['type']
        self._check_type(typ)

    def test_empty(self):
        data = Empty(dtype='f')
        self.assertNotEqual(Empty(dtype='i'), data)
        self._check_type(data)

    @ut.skipIf(not UNICODE_FILENAMES, "Filesystem unicode support required")
    def test_file(self):
        """ File object repr() with unicode """
        fname = tempfile.mktemp(self.USTRING+'.hdf5')
        try:
            with File(fname,'w') as f:
                self._check_type(f)
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass

def test_is_hdf5():
    filename = File(tempfile.mktemp(), "w").filename
    assert is_hdf5(filename)
    # non-existing HDF5 file
    filename = tempfile.mktemp()
    assert not is_hdf5(filename)
