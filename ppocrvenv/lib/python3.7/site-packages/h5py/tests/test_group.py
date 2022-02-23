# -*- coding: utf-8 -*-
# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Group test module.

    Tests all methods and properties of Group objects, with the following
    exceptions:

    1. Method create_dataset is tested in module test_dataset
"""

import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp

from collections.abc import MutableMapping

from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode

# If we can't encode unicode filenames, there's not much point failing tests
# which must fail
try:
    filename_encode(u"α")
except UnicodeEncodeError:
    NO_FS_UNICODE = True
else:
    NO_FS_UNICODE = False


class BaseGroup(TestCase):

    def setUp(self):
        self.f = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f:
            self.f.close()

class TestCreate(BaseGroup):

    """
        Feature: New groups can be created via .create_group method
    """

    def test_create(self):
        """ Simple .create_group call """
        grp = self.f.create_group('foo')
        self.assertIsInstance(grp, Group)

        grp2 = self.f.create_group(b'bar')
        self.assertIsInstance(grp, Group)

    def test_create_intermediate(self):
        """ Intermediate groups can be created automatically """
        grp = self.f.create_group('foo/bar/baz')
        self.assertEqual(grp.name, '/foo/bar/baz')

        grp2 = self.f.create_group(b'boo/bar/baz')
        self.assertEqual(grp2.name, '/boo/bar/baz')

    def test_create_exception(self):
        """ Name conflict causes group creation to fail with ValueError """
        self.f.create_group('foo')
        with self.assertRaises(ValueError):
            self.f.create_group('foo')

    def test_unicode(self):
        """ Unicode names are correctly stored """
        name = u"/Name" + chr(0x4500)
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_UTF8)

    def test_unicode_default(self):
        """ Unicode names convertible to ASCII are stored as ASCII (issue 239)
        """
        name = u"/Hello, this is a name"
        group = self.f.create_group(name)
        self.assertEqual(group.name, name)
        self.assertEqual(group.id.links.get_info(name.encode('utf8')).cset, h5t.CSET_ASCII)

    def test_appropriate_low_level_id(self):
        " Binding a group to a non-group identifier fails with ValueError "
        dset = self.f.create_dataset('foo', [1])
        with self.assertRaises(ValueError):
            Group(dset.id)

class TestDatasetAssignment(BaseGroup):

    """
        Feature: Datasets can be created by direct assignment of data
    """

    def test_ndarray(self):
        """ Dataset auto-creation by direct assignment """
        data = np.ones((4,4),dtype='f')
        self.f['a'] = data
        self.assertIsInstance(self.f['a'], Dataset)
        self.assertArrayEqual(self.f['a'][...], data)

    def test_name_bytes(self):
        data = np.ones((4, 4), dtype='f')
        self.f[b'b'] = data
        self.assertIsInstance(self.f[b'b'], Dataset)

class TestDtypeAssignment(BaseGroup):

    """
        Feature: Named types can be created by direct assignment of dtypes
    """

    def test_dtype(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        self.f['a'] = dtype
        self.assertIsInstance(self.f['a'], Datatype)
        self.assertEqual(self.f['a'].dtype, dtype)

    def test_name_bytes(self):
        """ Named type creation """
        dtype = np.dtype('|S10')
        self.f[b'b'] = dtype
        self.assertIsInstance(self.f[b'b'], Datatype)


class TestRequire(BaseGroup):

    """
        Feature: Groups can be auto-created, or opened via .require_group
    """

    def test_open_existing(self):
        """ Existing group is opened and returned """
        grp = self.f.create_group('foo')
        grp2 = self.f.require_group('foo')
        self.assertEqual(grp2, grp)

        grp3 = self.f.require_group(b'foo')
        self.assertEqual(grp3, grp)

    def test_create(self):
        """ Group is created if it doesn't exist """
        grp = self.f.require_group('foo')
        self.assertIsInstance(grp, Group)
        self.assertEqual(grp.name, '/foo')

    def test_require_exception(self):
        """ Opening conflicting object results in TypeError """
        self.f.create_dataset('foo', (1,), 'f')
        with self.assertRaises(TypeError):
            self.f.require_group('foo')

    def test_intermediate_create_dataset(self):
        """ Intermediate is created if it doesn't exist """
        dt = h5py.string_dtype()
        self.f.require_dataset("foo/bar/baz", (1,), dtype=dt)
        group = self.f.get('foo')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar')
        assert isinstance(group, Group)

    def test_intermediate_create_group(self):
        dt = h5py.string_dtype()
        self.f.require_group("foo/bar/baz")
        group = self.f.get('foo')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar')
        assert isinstance(group, Group)
        group = self.f.get('foo/bar/baz')
        assert isinstance(group, Group)

class TestDelete(BaseGroup):

    """
        Feature: Objects can be unlinked via "del" operator
    """

    def test_delete(self):
        """ Object deletion via "del" """
        self.f.create_group('foo')
        self.assertIn('foo', self.f)
        del self.f['foo']
        self.assertNotIn('foo', self.f)

    def test_nonexisting(self):
        """ Deleting non-existent object raises KeyError """
        with self.assertRaises(KeyError):
            del self.f['foo']

    def test_readonly_delete_exception(self):
        """ Deleting object in readonly file raises KeyError """
        # Note: it is impossible to restore the old behavior (ValueError)
        # without breaking the above test (non-existing objects)
        fname = self.mktemp()
        hfile = File(fname, 'w')
        try:
            hfile.create_group('foo')
        finally:
            hfile.close()

        hfile = File(fname, 'r')
        try:
            with self.assertRaises(KeyError):
                del hfile['foo']
        finally:
            hfile.close()

class TestOpen(BaseGroup):

    """
        Feature: Objects can be opened via indexing syntax obj[name]
    """

    def test_open(self):
        """ Simple obj[name] opening """
        grp = self.f.create_group('foo')
        grp2 = self.f['foo']
        grp3 = self.f['/foo']
        self.assertEqual(grp, grp2)
        self.assertEqual(grp, grp3)

    def test_nonexistent(self):
        """ Opening missing objects raises KeyError """
        with self.assertRaises(KeyError):
            self.f['foo']

    def test_reference(self):
        """ Objects can be opened by HDF5 object reference """
        grp = self.f.create_group('foo')
        grp2 = self.f[grp.ref]
        self.assertEqual(grp2, grp)

    def test_reference_numpyobj(self):
        """ Object can be opened by numpy.object_ containing object ref

        Test for issue 181, issue 202.
        """
        g = self.f.create_group('test')

        dt = np.dtype([('a', 'i'),('b', h5py.ref_dtype)])
        dset = self.f.create_dataset('test_dset', (1,), dt)

        dset[0] =(42,g.ref)
        data = dset[0]
        self.assertEqual(self.f[data[1]], g)

    def test_invalid_ref(self):
        """ Invalid region references should raise an exception """

        ref = h5py.h5r.Reference()

        with self.assertRaises(ValueError):
            self.f[ref]

        self.f.create_group('x')
        ref = self.f['x'].ref
        del self.f['x']

        with self.assertRaises(Exception):
            self.f[ref]

    def test_path_type_validation(self):
        """ Access with non bytes or str types should raise an exception """
        self.f.create_group('group')

        with self.assertRaises(TypeError):
            self.f[0]

        with self.assertRaises(TypeError):
            self.f[...]

    # TODO: check that regionrefs also work with __getitem__

class TestRepr(BaseGroup):
    """Opened and closed groups provide a useful __repr__ string"""

    def test_repr(self):
        """ Opened and closed groups provide a useful __repr__ string """
        g = self.f.create_group('foo')
        self.assertIsInstance(repr(g), str)
        g.id._close()
        self.assertIsInstance(repr(g), str)
        g = self.f['foo']
        # Closing the file shouldn't break it
        self.f.close()
        self.assertIsInstance(repr(g), str)

class BaseMapping(BaseGroup):

    """
        Base class for mapping tests
    """
    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = ('a', 'b', 'c', 'd')
        for x in self.groups:
            self.f.create_group(x)
        self.f['x'] = h5py.SoftLink('/mongoose')
        self.groups = self.groups + ('x',)

    def tearDown(self):
        if self.f:
            self.f.close()

class TestLen(BaseMapping):

    """
        Feature: The Python len() function returns the number of groups
    """

    def test_len(self):
        """ len() returns number of group members """
        self.assertEqual(len(self.f), len(self.groups))
        self.f.create_group('e')
        self.assertEqual(len(self.f), len(self.groups)+1)


class TestContains(BaseGroup):

    """
        Feature: The Python "in" builtin tests for membership
    """

    def test_contains(self):
        """ "in" builtin works for membership (byte and Unicode) """
        self.f.create_group('a')
        self.assertIn(b'a', self.f)
        self.assertIn('a', self.f)
        self.assertIn(b'/a', self.f)
        self.assertIn('/a', self.f)
        self.assertNotIn(b'mongoose', self.f)
        self.assertNotIn('mongoose', self.f)

    def test_exc(self):
        """ "in" on closed group returns False (see also issue 174) """
        self.f.create_group('a')
        self.f.close()
        self.assertFalse(b'a' in self.f)
        self.assertFalse('a' in self.f)

    def test_empty(self):
        """ Empty strings work properly and aren't contained """
        self.assertNotIn('', self.f)
        self.assertNotIn(b'', self.f)

    def test_dot(self):
        """ Current group "." is always contained """
        self.assertIn(b'.', self.f)
        self.assertIn('.', self.f)

    def test_root(self):
        """ Root group (by itself) is contained """
        self.assertIn(b'/', self.f)
        self.assertIn('/', self.f)

    def test_trailing_slash(self):
        """ Trailing slashes are unconditionally ignored """
        self.f.create_group('group')
        self.f['dataset'] = 42
        self.assertIn('/group/', self.f)
        self.assertIn('group/', self.f)
        self.assertIn('/dataset/', self.f)
        self.assertIn('dataset/', self.f)

    def test_softlinks(self):
        """ Broken softlinks are contained, but their members are not """
        self.f.create_group('grp')
        self.f['/grp/soft'] = h5py.SoftLink('/mongoose')
        self.f['/grp/external'] = h5py.ExternalLink('mongoose.hdf5', '/mongoose')
        self.assertIn('/grp/soft', self.f)
        self.assertNotIn('/grp/soft/something', self.f)
        self.assertIn('/grp/external', self.f)
        self.assertNotIn('/grp/external/something', self.f)

    def test_oddball_paths(self):
        """ Technically legitimate (but odd-looking) paths """
        self.f.create_group('x/y/z')
        self.f['dset'] = 42
        self.assertIn('/', self.f)
        self.assertIn('//', self.f)
        self.assertIn('///', self.f)
        self.assertIn('.///', self.f)
        self.assertIn('././/', self.f)
        grp = self.f['x']
        self.assertIn('.//x/y/z', self.f)
        self.assertNotIn('.//x/y/z', grp)
        self.assertIn('x///', self.f)
        self.assertIn('./x///', self.f)
        self.assertIn('dset///', self.f)
        self.assertIn('/dset//', self.f)

class TestIter(BaseMapping):

    """
        Feature: You can iterate over group members via "for x in y", etc.
    """

    def test_iter(self):
        """ "for x in y" iteration """
        lst = [x for x in self.f]
        self.assertSameElements(lst, self.groups)

    def test_iter_zero(self):
        """ Iteration works properly for the case with no group members """
        hfile = File(self.mktemp(), 'w')
        try:
            lst = [x for x in hfile]
            self.assertEqual(lst, [])
        finally:
            hfile.close()

class TestTrackOrder(BaseGroup):
    def populate(self, g):
        for i in range(100):
            # Mix group and dataset creation.
            if i % 10 == 0:
                g.create_group(str(i))
            else:
                g[str(i)] = [i]

    def test_track_order(self):
        g = self.f.create_group('order', track_order=True)  # creation order
        self.populate(g)

        ref = [str(i) for i in range(100)]
        self.assertEqual(list(g), ref)
        self.assertEqual(list(reversed(g)), list(reversed(ref)))

    def test_no_track_order(self):
        g = self.f.create_group('order', track_order=False)  # name alphanumeric
        self.populate(g)

        ref = sorted([str(i) for i in range(100)])
        self.assertEqual(list(g), ref)
        self.assertEqual(list(reversed(g)), list(reversed(ref)))

class TestPy3Dict(BaseMapping):

    def test_keys(self):
        """ .keys provides a key view """
        kv = getattr(self.f, 'keys')()
        ref = self.groups
        self.assertSameElements(list(kv), ref)
        self.assertSameElements(list(reversed(kv)), list(reversed(ref)))

        for x in self.groups:
            self.assertIn(x, kv)
        self.assertEqual(len(kv), len(self.groups))

    def test_values(self):
        """ .values provides a value view """
        vv = getattr(self.f, 'values')()
        ref = [self.f.get(x) for x in self.groups]
        self.assertSameElements(list(vv), ref)
        self.assertSameElements(list(reversed(vv)), list(reversed(ref)))

        self.assertEqual(len(vv), len(self.groups))
        for x in self.groups:
            self.assertIn(self.f.get(x), vv)

    def test_items(self):
        """ .items provides an item view """
        iv = getattr(self.f, 'items')()
        ref = [(x,self.f.get(x)) for x in self.groups]
        self.assertSameElements(list(iv), ref)
        self.assertSameElements(list(reversed(iv)), list(reversed(ref)))

        self.assertEqual(len(iv), len(self.groups))
        for x in self.groups:
            self.assertIn((x, self.f.get(x)), iv)

class TestAdditionalMappingFuncs(BaseMapping):
    """
    Feature: Other dict methods (pop, pop_item, clear, update, setdefault) are
    available.
    """
    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        for x in ('/test/a', '/test/b', '/test/c', '/test/d'):
            self.f.create_group(x)
        self.group = self.f['test']

    def tearDown(self):
        if self.f:
            self.f.close()

    def test_pop_item(self):
        """.pop_item exists and removes item"""
        key, val = self.group.popitem()
        self.assertNotIn(key, self.group)

    def test_pop(self):
        """.pop exists and removes specified item"""
        self.group.pop('a')
        self.assertNotIn('a', self.group)

    def test_pop_default(self):
        """.pop falls back to default"""
        # e shouldn't exist as a group
        value = self.group.pop('e', None)
        self.assertEqual(value, None)

    def test_pop_raises(self):
        """.pop raises KeyError for non-existence"""
        # e shouldn't exist as a group
        with self.assertRaises(KeyError):
            key = self.group.pop('e')

    def test_clear(self):
        """.clear removes groups"""
        self.group.clear()
        self.assertEqual(len(self.group), 0)

    def test_update_dict(self):
        """.update works with dict"""
        new_items = {'e': np.array([42])}
        self.group.update(new_items)
        self.assertIn('e', self.group)

    def test_update_iter(self):
        """.update works with list"""
        new_items = [
            ('e', np.array([42])),
            ('f', np.array([42]))
        ]
        self.group.update(new_items)
        self.assertIn('e', self.group)

    def test_update_kwargs(self):
        """.update works with kwargs"""
        new_items = {'e': np.array([42])}
        self.group.update(**new_items)
        self.assertIn('e', self.group)

    def test_setdefault(self):
        """.setdefault gets group if it exists"""
        value = self.group.setdefault('a')
        self.assertEqual(value, self.group.get('a'))

    def test_setdefault_with_default(self):
        """.setdefault gets default if group doesn't exist"""
        # e shouldn't exist as a group
        # 42 used as groups should be strings
        value = self.group.setdefault('e', np.array([42]))
        self.assertEqual(value, 42)

    def test_setdefault_no_default(self):
        """
        .setdefault gets None if group doesn't exist, but as None isn't defined
        as data for a dataset, this should raise a TypeError.
        """
        # e shouldn't exist as a group
        with self.assertRaises(TypeError):
            self.group.setdefault('e')


class TestGet(BaseGroup):

    """
        Feature: The .get method allows access to objects and metadata
    """

    def test_get_default(self):
        """ Object is returned, or default if it doesn't exist """
        default = object()
        out = self.f.get('mongoose', default)
        self.assertIs(out, default)

        grp = self.f.create_group('a')
        out = self.f.get(b'a')
        self.assertEqual(out, grp)

    def test_get_class(self):
        """ Object class is returned with getclass option """
        self.f.create_group('foo')
        out = self.f.get('foo', getclass=True)
        self.assertEqual(out, Group)

        self.f.create_dataset('bar', (4,))
        out = self.f.get('bar', getclass=True)
        self.assertEqual(out, Dataset)

        self.f['baz'] = np.dtype('|S10')
        out = self.f.get('baz', getclass=True)
        self.assertEqual(out, Datatype)

    def test_get_link_class(self):
        """ Get link classes """
        default = object()

        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el

        out_hl = self.f.get('hard', default, getlink=True, getclass=True)
        out_sl = self.f.get('soft', default, getlink=True, getclass=True)
        out_el = self.f.get('external', default, getlink=True, getclass=True)

        self.assertEqual(out_hl, HardLink)
        self.assertEqual(out_sl, SoftLink)
        self.assertEqual(out_el, ExternalLink)

    def test_get_link(self):
        """ Get link values """
        sl = SoftLink('/mongoose')
        el = ExternalLink('somewhere.hdf5', 'mongoose')

        self.f.create_group('hard')
        self.f['soft'] = sl
        self.f['external'] = el

        out_hl = self.f.get('hard', getlink=True)
        out_sl = self.f.get('soft', getlink=True)
        out_el = self.f.get('external', getlink=True)

        #TODO: redo with SoftLink/ExternalLink built-in equality
        self.assertIsInstance(out_hl, HardLink)
        self.assertIsInstance(out_sl, SoftLink)
        self.assertEqual(out_sl._path, sl._path)
        self.assertIsInstance(out_el, ExternalLink)
        self.assertEqual(out_el._path, el._path)
        self.assertEqual(out_el._filename, el._filename)

class TestVisit(TestCase):

    """
        Feature: The .visit and .visititems methods allow iterative access to
        group and subgroup members
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.groups = [
            'grp1', 'grp1/sg1', 'grp1/sg2', 'grp2', 'grp2/sg1', 'grp2/sg1/ssg1'
            ]
        for x in self.groups:
            self.f.create_group(x)

    def tearDown(self):
        self.f.close()

    def test_visit(self):
        """ All subgroups are visited """
        l = []
        self.f.visit(l.append)
        self.assertSameElements(l, self.groups)

    def test_visititems(self):
        """ All subgroups and contents are visited """
        l = []
        comp = [(x, self.f[x]) for x in self.groups]
        self.f.visititems(lambda x, y: l.append((x,y)))
        self.assertSameElements(comp, l)

    def test_bailout(self):
        """ Returning a non-None value immediately aborts iteration """
        x = self.f.visit(lambda x: x)
        self.assertEqual(x, self.groups[0])
        x = self.f.visititems(lambda x, y: (x,y))
        self.assertEqual(x, (self.groups[0], self.f[self.groups[0]]))

class TestSoftLinks(BaseGroup):

    """
        Feature: Create and manage soft links with the high-level interface
    """

    def test_spath(self):
        """ SoftLink path attribute """
        sl = SoftLink('/foo')
        self.assertEqual(sl.path, '/foo')

    def test_srepr(self):
        """ SoftLink path repr """
        sl = SoftLink('/foo')
        self.assertIsInstance(repr(sl), str)

    def test_create(self):
        """ Create new soft link by assignment """
        g = self.f.create_group('new')
        sl = SoftLink('/new')
        self.f['alias'] = sl
        g2 = self.f['alias']
        self.assertEqual(g, g2)

    def test_exc(self):
        """ Opening dangling soft link results in KeyError """
        self.f['alias'] = SoftLink('new')
        with self.assertRaises(KeyError):
            self.f['alias']

class TestExternalLinks(TestCase):

    """
        Feature: Create and manage external links
    """

    def setUp(self):
        self.f = File(self.mktemp(), 'w')
        self.ename = self.mktemp()
        self.ef = File(self.ename, 'w')
        self.ef.create_group('external')
        self.ef.close()

    def tearDown(self):
        if self.f:
            self.f.close()
        if self.ef:
            self.ef.close()

    def test_epath(self):
        """ External link paths attributes """
        el = ExternalLink('foo.hdf5', '/foo')
        self.assertEqual(el.filename, 'foo.hdf5')
        self.assertEqual(el.path, '/foo')

    def test_erepr(self):
        """ External link repr """
        el = ExternalLink('foo.hdf5','/foo')
        self.assertIsInstance(repr(el), str)

    def test_create(self):
        """ Creating external links """
        self.f['ext'] = ExternalLink(self.ename, '/external')
        grp = self.f['ext']
        self.ef = grp.file
        self.assertNotEqual(self.ef, self.f)
        self.assertEqual(grp.name, '/external')

    def test_exc(self):
        """ KeyError raised when attempting to open broken link """
        self.f['ext'] = ExternalLink(self.ename, '/missing')
        with self.assertRaises(KeyError):
            self.f['ext']

    # I would prefer IOError but there's no way to fix this as the exception
    # class is determined by HDF5.
    def test_exc_missingfile(self):
        """ KeyError raised when attempting to open missing file """
        self.f['ext'] = ExternalLink('mongoose.hdf5','/foo')
        with self.assertRaises(KeyError):
            self.f['ext']

    def test_close_file(self):
        """ Files opened by accessing external links can be closed

        Issue 189.
        """
        self.f['ext'] = ExternalLink(self.ename, '/')
        grp = self.f['ext']
        f2 = grp.file
        f2.close()
        self.assertFalse(f2)

    @ut.skipIf(NO_FS_UNICODE, "No unicode filename support")
    def test_unicode_encode(self):
        """
        Check that external links encode unicode filenames properly
        Testing issue #732
        """
        ext_filename = os.path.join(mkdtemp(), u"α.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('external')
        self.f['ext'] = ExternalLink(ext_filename, '/external')

    @ut.skipIf(NO_FS_UNICODE, "No unicode filename support")
    def test_unicode_decode(self):
        """
        Check that external links decode unicode filenames properly
        Testing issue #732
        """
        ext_filename = os.path.join(mkdtemp(), u"α.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('external')
            ext_file["external"].attrs["ext_attr"] = "test"
        self.f['ext'] = ExternalLink(ext_filename, '/external')
        self.assertEqual(self.f["ext"].attrs["ext_attr"], "test")

    def test_unicode_hdf5_path(self):
        """
        Check that external links handle unicode hdf5 paths properly
        Testing issue #333
        """
        ext_filename = os.path.join(mkdtemp(), "external.hdf5")
        with File(ext_filename, "w") as ext_file:
            ext_file.create_group('α')
            ext_file["α"].attrs["ext_attr"] = "test"
        self.f['ext'] = ExternalLink(ext_filename, '/α')
        self.assertEqual(self.f["ext"].attrs["ext_attr"], "test")

class TestExtLinkBugs(TestCase):

    """
        Bugs: Specific regressions for external links
    """

    def test_issue_212(self):
        """ Issue 212

        Fails with:

        AttributeError: 'SharedConfig' object has no attribute 'lapl'
        """
        def closer(x):
            def w():
                try:
                    if x:
                        x.close()
                except IOError:
                    pass
            return w
        orig_name = self.mktemp()
        new_name = self.mktemp()
        f = File(orig_name, 'w')
        self.addCleanup(closer(f))
        f.create_group('a')
        f.close()

        g = File(new_name, 'w')
        self.addCleanup(closer(g))
        g['link'] = ExternalLink(orig_name, '/')  # note root group
        g.close()

        h = File(new_name, 'r')
        self.addCleanup(closer(h))
        self.assertIsInstance(h['link']['a'], Group)


class TestCopy(TestCase):

    def setUp(self):
        self.f1 = File(self.mktemp(), 'w')
        self.f2 = File(self.mktemp(), 'w')

    def tearDown(self):
        if self.f1:
            self.f1.close()
        if self.f2:
            self.f2.close()

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_path_to_path(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]

        self.f1.copy('foo', 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_path_to_group(self):
        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]
        baz = self.f1.create_group('baz')

        self.f1.copy('foo', baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1,2,3]))

        self.f1.copy('foo', self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_group_to_path(self):

        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]

        self.f1.copy(foo, 'baz')
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['bar'], np.array([1,2,3]))

        self.f2.copy(foo, 'foo')
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_group_to_group(self):

        foo = self.f1.create_group('foo')
        foo['bar'] = [1,2,3]
        baz = self.f1.create_group('baz')

        self.f1.copy(foo, baz)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertArrayEqual(baz['foo/bar'], np.array([1,2,3]))

        self.f1.copy(foo, self.f2['/'])
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_dataset(self):
        self.f1['foo'] = [1,2,3]
        foo = self.f1['foo']
        grp = self.f1.create_group("grp")

        self.f1.copy(foo, 'bar')
        self.assertArrayEqual(self.f1['bar'], np.array([1,2,3]))

        self.f1.copy('foo', 'baz')
        self.assertArrayEqual(self.f1['baz'], np.array([1,2,3]))

        self.f1.copy(foo, grp)
        self.assertArrayEqual(self.f1['/grp/foo'], np.array([1,2,3]))

        self.f1.copy('foo', self.f2)
        self.assertArrayEqual(self.f2['foo'], np.array([1,2,3]))

        self.f2.copy(self.f1['foo'], self.f2, 'bar')
        self.assertArrayEqual(self.f2['bar'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_shallow(self):

        foo = self.f1.create_group('foo')
        bar = foo.create_group('bar')
        foo['qux'] = [1,2,3]
        bar['quux'] = [4,5,6]

        self.f1.copy(foo, 'baz', shallow=True)
        baz = self.f1['baz']
        self.assertIsInstance(baz, Group)
        self.assertIsInstance(baz['bar'], Group)
        self.assertEqual(len(baz['bar']), 0)
        self.assertArrayEqual(baz['qux'], np.array([1,2,3]))

        self.f2.copy(foo, 'foo', shallow=True)
        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertIsInstance(self.f2['foo/bar'], Group)
        self.assertEqual(len(self.f2['foo/bar']), 0)
        self.assertArrayEqual(self.f2['foo/qux'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_without_attributes(self):

        self.f1['foo'] = [1,2,3]
        foo = self.f1['foo']
        foo.attrs['bar'] = [4,5,6]

        self.f1.copy(foo, 'baz', without_attrs=True)
        self.assertArrayEqual(self.f1['baz'], np.array([1,2,3]))
        assert 'bar' not in self.f1['baz'].attrs

        self.f2.copy(foo, 'baz', without_attrs=True)
        self.assertArrayEqual(self.f2['baz'], np.array([1,2,3]))
        assert 'bar' not in self.f2['baz'].attrs

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_soft_links(self):

        self.f1['bar'] = [1, 2, 3]
        foo = self.f1.create_group('foo')
        foo['baz'] = SoftLink('/bar')

        self.f1.copy(foo, 'qux', expand_soft=True)
        self.f2.copy(foo, 'foo', expand_soft=True)
        del self.f1['bar']

        self.assertIsInstance(self.f1['qux'], Group)
        self.assertArrayEqual(self.f1['qux/baz'], np.array([1, 2, 3]))

        self.assertIsInstance(self.f2['/foo'], Group)
        self.assertArrayEqual(self.f2['foo/baz'], np.array([1, 2, 3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_external_links(self):

        filename = self.f1.filename
        self.f1['foo'] = [1,2,3]
        self.f2['bar'] = ExternalLink(filename, 'foo')
        self.f1.close()
        self.f1 = None

        self.assertArrayEqual(self.f2['bar'], np.array([1,2,3]))

        self.f2.copy('bar', 'baz', expand_external=True)
        os.unlink(filename)
        self.assertArrayEqual(self.f2['baz'], np.array([1,2,3]))

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1,8,9),
               "Bug in HDF5<1.8.8 prevents copying open dataset")
    def test_copy_refs(self):

        self.f1['foo'] = [1,2,3]
        self.f1['bar'] = [4,5,6]
        foo = self.f1['foo']
        bar = self.f1['bar']
        foo.attrs['bar'] = bar.ref

        self.f1.copy(foo, 'baz', expand_refs=True)
        self.assertArrayEqual(self.f1['baz'], np.array([1,2,3]))
        baz_bar = self.f1['baz'].attrs['bar']
        self.assertArrayEqual(self.f1[baz_bar], np.array([4,5,6]))
        # The reference points to a copy of bar, not to bar itself.
        self.assertNotEqual(self.f1[baz_bar].name, bar.name)

        self.f1.copy('foo', self.f2, 'baz', expand_refs=True)
        self.assertArrayEqual(self.f2['baz'], np.array([1,2,3]))
        baz_bar = self.f2['baz'].attrs['bar']
        self.assertArrayEqual(self.f2[baz_bar], np.array([4,5,6]))

        self.f1.copy('/', self.f2, 'root', expand_refs=True)
        self.assertArrayEqual(self.f2['root/foo'], np.array([1,2,3]))
        self.assertArrayEqual(self.f2['root/bar'], np.array([4,5,6]))
        foo_bar = self.f2['root/foo'].attrs['bar']
        self.assertArrayEqual(self.f2[foo_bar], np.array([4,5,6]))
        # There's only one copy of bar, which the reference points to.
        self.assertEqual(self.f2[foo_bar], self.f2['root/bar'])


class TestMove(BaseGroup):

    """
        Feature: Group.move moves links in a file
    """

    def test_move_hardlink(self):
        """ Moving an object """
        grp = self.f.create_group("X")
        self.f.move("X", "Y")
        self.assertEqual(self.f["Y"], grp)
        self.f.move("Y", "new/nested/path")
        self.assertEqual(self.f['new/nested/path'], grp)

    def test_move_softlink(self):
        """ Moving a soft link """
        self.f['soft'] = h5py.SoftLink("relative/path")
        self.f.move('soft', 'new_soft')
        lnk = self.f.get('new_soft', getlink=True)
        self.assertEqual(lnk.path, "relative/path")

    def test_move_conflict(self):
        """ Move conflict raises ValueError """
        self.f.create_group("X")
        self.f.create_group("Y")
        with self.assertRaises(ValueError):
            self.f.move("X", "Y")

    def test_short_circuit(self):
        ''' Test that a null-move works '''
        self.f.create_group("X")
        self.f.move("X", "X")


class TestMutableMapping(BaseGroup):
    '''Tests if the registration of Group as a MutableMapping
    behaves as expected
    '''
    def test_resolution(self):
        assert issubclass(Group, MutableMapping)
        grp = self.f.create_group("K")
        assert isinstance(grp, MutableMapping)

    def test_validity(self):
        '''
        Test that the required functions are implemented.
        '''
        Group.__getitem__
        Group.__setitem__
        Group.__delitem__
        Group.__iter__
        Group.__len__
