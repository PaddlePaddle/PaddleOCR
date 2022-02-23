# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Attributes testing module

    Covers all operations which access the .attrs property, with the
    exception of data read/write and type conversion.  Those operations
    are tested by module test_attrs_data.
"""

import numpy as np

from collections.abc import MutableMapping

from .common import TestCase, ut

import h5py
from h5py import File
from h5py import h5a,  h5t
from h5py import AttributeManager


class BaseAttrs(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestRepr(TestCase):

    """ Feature: AttributeManager provide a helpful
        __repr__ string
    """

    def test_repr(self):
        grp = self.f.create_group('grp')
        grp.attrs.create('att', 1)
        self.assertIsInstance(repr(grp.attrs), str)
        grp.id.close()
        self.assertIsInstance(repr(grp.attrs), str)


class TestAccess(BaseAttrs):

    """
        Feature: Attribute creation/retrieval via special methods
    """

    def test_create(self):
        """ Attribute creation by direct assignment """
        self.f.attrs['a'] = 4.0
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 4.0)

    def test_create_2(self):
        """ Attribute creation by create() method """
        self.f.attrs.create('a', 4.0)
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 4.0)

    def test_modify(self):
        """ Attributes are modified by direct assignment"""
        self.f.attrs['a'] = 3
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 3)
        self.f.attrs['a'] = 4
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 4)

    def test_modify_2(self):
        """ Attributes are modified by modify() method """
        self.f.attrs.modify('a',3)
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 3)

        self.f.attrs.modify('a', 4)
        self.assertEqual(list(self.f.attrs.keys()), ['a'])
        self.assertEqual(self.f.attrs['a'], 4)

        # If the attribute doesn't exist, create new
        self.f.attrs.modify('b', 5)
        self.assertEqual(list(self.f.attrs.keys()), ['a', 'b'])
        self.assertEqual(self.f.attrs['a'], 4)
        self.assertEqual(self.f.attrs['b'], 5)

        # Shape of new value is incompatible with the previous
        new_value = np.arange(5)
        with self.assertRaises(TypeError):
            self.f.attrs.modify('b', new_value)

    def test_overwrite(self):
        """ Attributes are silently overwritten """
        self.f.attrs['a'] = 4.0
        self.f.attrs['a'] = 5.0
        self.assertEqual(self.f.attrs['a'], 5.0)

    def test_rank(self):
        """ Attribute rank is preserved """
        self.f.attrs['a'] = (4.0, 5.0)
        self.assertEqual(self.f.attrs['a'].shape, (2,))
        self.assertArrayEqual(self.f.attrs['a'], np.array((4.0,5.0)))

    def test_single(self):
        """ Attributes of shape (1,) don't become scalars """
        self.f.attrs['a'] = np.ones((1,))
        out = self.f.attrs['a']
        self.assertEqual(out.shape, (1,))
        self.assertEqual(out[()], 1)

    def test_access_exc(self):
        """ Attempt to access missing item raises KeyError """
        with self.assertRaises(KeyError):
            self.f.attrs['a']

    def test_get_id(self):
        self.f.attrs['a'] = 4.0
        aid = self.f.attrs.get_id('a')
        assert isinstance(aid, h5a.AttrID)

        with self.assertRaises(KeyError):
            self.f.attrs.get_id('b')

class TestDelete(BaseAttrs):

    """
        Feature: Deletion of attributes using __delitem__
    """

    def test_delete(self):
        """ Deletion via "del" """
        self.f.attrs['a'] = 4.0
        self.assertIn('a', self.f.attrs)
        del self.f.attrs['a']
        self.assertNotIn('a', self.f.attrs)

    def test_delete_exc(self):
        """ Attempt to delete missing item raises KeyError """
        with self.assertRaises(KeyError):
            del self.f.attrs['a']


class TestUnicode(BaseAttrs):

    """
        Feature: Attributes can be accessed via Unicode or byte strings
    """

    def test_ascii(self):
        """ Access via pure-ASCII byte string """
        self.f.attrs[b"ascii"] = 42
        out = self.f.attrs[b"ascii"]
        self.assertEqual(out, 42)

    def test_raw(self):
        """ Access via non-ASCII byte string """
        name = b"non-ascii\xfe"
        self.f.attrs[name] = 42
        out = self.f.attrs[name]
        self.assertEqual(out, 42)

    def test_unicode(self):
        """ Access via Unicode string with non-ascii characters """
        name = "Omega" + chr(0x03A9)
        self.f.attrs[name] = 42
        out = self.f.attrs[name]
        self.assertEqual(out, 42)


class TestCreate(BaseAttrs):

    """
        Options for explicit attribute creation
    """

    def test_named(self):
        """ Attributes created from named types link to the source type object
        """
        self.f['type'] = np.dtype('u8')
        self.f.attrs.create('x', 42, dtype=self.f['type'])
        self.assertEqual(self.f.attrs['x'], 42)
        aid = h5a.open(self.f.id, b'x')
        htype = aid.get_type()
        htype2 = self.f['type'].id
        self.assertEqual(htype, htype2)
        self.assertTrue(htype.committed())

    def test_empty(self):
        # https://github.com/h5py/h5py/issues/1540
        """ Create attribute with h5py.Empty value
        """
        self.f.attrs.create('empty', h5py.Empty('f'))
        self.assertEqual(self.f.attrs['empty'], h5py.Empty('f'))

        self.f.attrs.create('empty', h5py.Empty(None))
        self.assertEqual(self.f.attrs['empty'], h5py.Empty(None))

class TestMutableMapping(BaseAttrs):
    '''Tests if the registration of AttributeManager as a MutableMapping
    behaves as expected
    '''
    def test_resolution(self):
        assert issubclass(AttributeManager, MutableMapping)
        assert isinstance(self.f.attrs, MutableMapping)

    def test_validity(self):
        '''
        Test that the required functions are implemented.
        '''
        AttributeManager.__getitem__
        AttributeManager.__setitem__
        AttributeManager.__delitem__
        AttributeManager.__iter__
        AttributeManager.__len__

class TestVlen(BaseAttrs):
    def test_vlen(self):
        a = np.array([np.arange(3), np.arange(4)],
            dtype=h5t.vlen_dtype(int))
        self.f.attrs['a'] = a
        self.assertArrayEqual(self.f.attrs['a'][0], a[0])

    def test_vlen_s1(self):
        dt = h5py.vlen_dtype(np.dtype('S1'))
        a = np.empty((1,), dtype=dt)
        a[0] = np.array([b'a', b'b'], dtype='S1')

        self.f.attrs.create('test', a)
        self.assertArrayEqual(self.f.attrs['test'][0], a[0])


class TestTrackOrder(BaseAttrs):
    def fill_attrs(self, track_order):
        attrs = self.f.create_group('test', track_order=track_order).attrs
        for i in range(100):
            attrs[str(i)] = i
        return attrs

    @ut.skipUnless(h5py.version.hdf5_version_tuple >= (1, 10, 6), 'HDF5 1.10.6 required')
    # https://forum.hdfgroup.org/t/bug-h5arename-fails-unexpectedly/4881
    def test_track_order(self):
        attrs = self.fill_attrs(track_order=True)  # creation order
        self.assertEqual(list(attrs),
                         [str(i) for i in range(100)])

    def test_no_track_order(self):
        attrs = self.fill_attrs(track_order=False)  # name alphanumeric
        self.assertEqual(list(attrs),
                         sorted([str(i) for i in range(100)]))


class TestDatatype(BaseAttrs):

    def test_datatype(self):
        self.f['foo'] = np.dtype('f')
        dt = self.f['foo']
        self.assertEqual(list(dt.attrs.keys()), [])
        dt.attrs.create('a', 4.0)
        self.assertEqual(list(dt.attrs.keys()), ['a'])
        self.assertEqual(list(dt.attrs.values()), [4.0])

def test_python_int_uint64(writable_file):
    f = writable_file
    data = [np.iinfo(np.int64).max, np.iinfo(np.int64).max + 1]

    # Check creating a new attribute
    f.attrs.create('a', data, dtype=np.uint64)
    assert f.attrs['a'].dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(f.attrs['a'], np.array(data, dtype=np.uint64))

    # Check modifying an existing attribute
    f.attrs.modify('a', data)
    np.testing.assert_array_equal(f.attrs['a'], np.array(data, dtype=np.uint64))
