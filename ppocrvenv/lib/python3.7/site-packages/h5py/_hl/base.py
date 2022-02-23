# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements operations common to all high-level objects (File, etc.).
"""

from collections.abc import (
    Mapping, MutableMapping, KeysView, ValuesView, ItemsView
)
import os
import posixpath

import numpy as np

# The high-level interface is serialized; every public API function & method
# is wrapped in a lock.  We re-use the low-level lock because (1) it's fast,
# and (2) it eliminates the possibility of deadlocks due to out-of-order
# lock acquisition.
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode


def is_hdf5(fname):
    """ Determine if a file is valid HDF5 (False if it doesn't exist). """
    with phil:
        fname = os.path.abspath(fspath(fname))

        if os.path.isfile(fname):
            return h5f.is_hdf5(filename_encode(fname))
        return False


def find_item_type(data):
    """Find the item type of a simple object or collection of objects.

    E.g. [[['a']]] -> str

    The focus is on collections where all items have the same type; we'll return
    None if that's not the case.

    The aim is to treat numpy arrays of Python objects like normal Python
    collections, while treating arrays with specific dtypes differently.
    We're also only interested in array-like collections - lists and tuples,
    possibly nested - not things like sets or dicts.
    """
    if isinstance(data, np.ndarray):
        if (
            data.dtype.kind == 'O'
            and not h5t.check_string_dtype(data.dtype)
            and not h5t.check_vlen_dtype(data.dtype)
        ):
            item_types = {type(e) for e in data.flat}
        else:
            return None
    elif isinstance(data, (list, tuple)):
        item_types = {find_item_type(e) for e in data}
    else:
        return type(data)

    if len(item_types) != 1:
        return None
    return item_types.pop()


def guess_dtype(data):
    """ Attempt to guess an appropriate dtype for the object, returning None
    if nothing is appropriate (or if it should be left up the the array
    constructor to figure out)
    """
    with phil:
        if isinstance(data, h5r.RegionReference):
            return h5t.regionref_dtype
        if isinstance(data, h5r.Reference):
            return h5t.ref_dtype

        item_type = find_item_type(data)

        if item_type is bytes:
            return h5t.string_dtype(encoding='ascii')
        if item_type is str:
            return h5t.string_dtype()

        return None


def is_float16_dtype(dt):
    if dt is None:
        return False

    dt = np.dtype(dt)  # normalize strings -> np.dtype objects
    return dt.kind == 'f' and dt.itemsize == 2


def array_for_new_object(data, specified_dtype=None):
    """Prepare an array from data used to create a new dataset or attribute"""

    # We mostly let HDF5 convert data as necessary when it's written.
    # But if we are going to a float16 datatype, pre-convert in python
    # to workaround a bug in the conversion.
    # https://github.com/h5py/h5py/issues/819
    if is_float16_dtype(specified_dtype):
        as_dtype = specified_dtype
    elif not isinstance(data, np.ndarray) and (specified_dtype is not None):
        # If we need to convert e.g. a list to an array, don't leave numpy
        # to guess a dtype we already know.
        as_dtype = specified_dtype
    else:
        as_dtype = guess_dtype(data)

    data = np.asarray(data, order="C", dtype=as_dtype)

    # In most cases, this does nothing. But if data was already an array,
    # and as_dtype is a tagged h5py dtype (e.g. for an object array of strings),
    # asarray() doesn't replace its dtype object. This gives it the tagged dtype:
    if as_dtype is not None:
        data = data.view(dtype=as_dtype)

    return data


def default_lapl():
    """ Default link access property list """
    lapl = h5p.create(h5p.LINK_ACCESS)
    fapl = h5p.create(h5p.FILE_ACCESS)
    fapl.set_fclose_degree(h5f.CLOSE_STRONG)
    lapl.set_elink_fapl(fapl)
    return lapl


def default_lcpl():
    """ Default link creation property list """
    lcpl = h5p.create(h5p.LINK_CREATE)
    lcpl.set_create_intermediate_group(True)
    return lcpl

dlapl = default_lapl()
dlcpl = default_lcpl()


def is_empty_dataspace(obj):
    """ Check if an object's dataspace is empty """
    if obj.get_space().get_simple_extent_type() == h5s.NULL:
        return True
    return False


class CommonStateObject(object):

    """
        Mixin class that allows sharing information between objects which
        reside in the same HDF5 file.  Requires that the host class have
        a ".id" attribute which returns a low-level ObjectID subclass.

        Also implements Unicode operations.
    """

    @property
    def _lapl(self):
        """ Fetch the link access property list appropriate for this object
        """
        return dlapl

    @property
    def _lcpl(self):
        """ Fetch the link creation property list appropriate for this object
        """
        return dlcpl

    def _e(self, name, lcpl=None):
        """ Encode a name according to the current file settings.

        Returns name, or 2-tuple (name, lcpl) if lcpl is True

        - Binary strings are always passed as-is, h5t.CSET_ASCII
        - Unicode strings are encoded utf8, h5t.CSET_UTF8

        If name is None, returns either None or (None, None) appropriately.
        """
        def get_lcpl(coding):
            """ Create an appropriate link creation property list """
            lcpl = self._lcpl.copy()
            lcpl.set_char_encoding(coding)
            return lcpl

        if name is None:
            return (None, None) if lcpl else None

        if isinstance(name, bytes):
            coding = h5t.CSET_ASCII
        else:
            try:
                name = name.encode('ascii')
                coding = h5t.CSET_ASCII
            except UnicodeEncodeError:
                name = name.encode('utf8')
                coding = h5t.CSET_UTF8

        if lcpl:
            return name, get_lcpl(coding)
        return name

    def _d(self, name):
        """ Decode a name according to the current file settings.

        - Try to decode utf8
        - Failing that, return the byte string

        If name is None, returns None.
        """
        if name is None:
            return None

        try:
            return name.decode('utf8')
        except UnicodeDecodeError:
            pass
        return name


class _RegionProxy(object):

    """
        Proxy object which handles region references.

        To create a new region reference (datasets only), use slicing syntax:

            >>> newref = obj.regionref[0:10:2]

        To determine the target dataset shape from an existing reference:

            >>> shape = obj.regionref.shape(existingref)

        where <obj> may be any object in the file. To determine the shape of
        the selection in use on the target dataset:

            >>> selection_shape = obj.regionref.selection(existingref)
    """

    def __init__(self, obj):
        self.obj = obj
        self.id = obj.id

    def __getitem__(self, args):
        if not isinstance(self.id, h5d.DatasetID):
            raise TypeError("Region references can only be made to datasets")
        from . import selections
        with phil:
            selection = selections.select(self.id.shape, args, dataset=self.obj)
            return h5r.create(self.id, b'.', h5r.DATASET_REGION, selection.id)

    def shape(self, ref):
        """ Get the shape of the target dataspace referred to by *ref*. """
        with phil:
            sid = h5r.get_region(ref, self.id)
            return sid.shape

    def selection(self, ref):
        """ Get the shape of the target dataspace selection referred to by *ref*
        """
        from . import selections
        with phil:
            sid = h5r.get_region(ref, self.id)
            return selections.guess_shape(sid)


class HLObject(CommonStateObject):

    """
        Base class for high-level interface objects.
    """

    @property
    def file(self):
        """ Return a File instance associated with this object """
        from . import files
        with phil:
            return files.File(self.id)

    @property
    @with_phil
    def name(self):
        """ Return the full name of this object.  None if anonymous. """
        return self._d(h5i.get_name(self.id))

    @property
    @with_phil
    def parent(self):
        """Return the parent group of this object.

        This is always equivalent to obj.file[posixpath.dirname(obj.name)].
        ValueError if this object is anonymous.
        """
        if self.name is None:
            raise ValueError("Parent of an anonymous object is undefined")
        return self.file[posixpath.dirname(self.name)]

    @property
    @with_phil
    def id(self):
        """ Low-level identifier appropriate for this object """
        return self._id

    @property
    @with_phil
    def ref(self):
        """ An (opaque) HDF5 reference to this object """
        return h5r.create(self.id, b'.', h5r.OBJECT)

    @property
    @with_phil
    def regionref(self):
        """Create a region reference (Datasets only).

        The syntax is regionref[<slices>]. For example, dset.regionref[...]
        creates a region reference in which the whole dataset is selected.

        Can also be used to determine the shape of the referenced dataset
        (via .shape property), or the shape of the selection (via the
        .selection property).
        """
        return _RegionProxy(self)

    @property
    def attrs(self):
        """ Attributes attached to this object """
        from . import attrs
        with phil:
            return attrs.AttributeManager(self)

    @with_phil
    def __init__(self, oid):
        """ Setup this object, given its low-level identifier """
        self._id = oid

    @with_phil
    def __hash__(self):
        return hash(self.id)

    @with_phil
    def __eq__(self, other):
        if hasattr(other, 'id'):
            return self.id == other.id
        return NotImplemented

    def __bool__(self):
        with phil:
            return bool(self.id)
    __nonzero__ = __bool__

    def __getnewargs__(self):
        """Disable pickle.

        Handles for HDF5 objects can't be reliably deserialised, because the
        recipient may not have access to the same files. So we do this to
        fail early.

        If you really want to pickle h5py objects and can live with some
        limitations, look at the h5pickle project on PyPI.
        """
        raise TypeError("h5py objects cannot be pickled")

    def __getstate__(self):
        # Pickle protocols 0 and 1 use this instead of __getnewargs__
        raise TypeError("h5py objects cannot be pickled")

# --- Dictionary-style interface ----------------------------------------------

# To implement the dictionary-style interface from groups and attributes,
# we inherit from the appropriate abstract base classes in collections.
#
# All locking is taken care of by the subclasses.
# We have to override ValuesView and ItemsView here because Group and
# AttributeManager can only test for key names.


class KeysViewHDF5(KeysView):
    def __str__(self):
        return "<KeysViewHDF5 {}>".format(list(self))

    def __reversed__(self):
        yield from reversed(self._mapping)

    __repr__ = __str__

class ValuesViewHDF5(ValuesView):

    """
        Wraps e.g. a Group or AttributeManager to provide a value view.

        Note that __contains__ will have poor performance as it has
        to scan all the links or attributes.
    """

    def __contains__(self, value):
        with phil:
            for key in self._mapping:
                if value == self._mapping.get(key):
                    return True
            return False

    def __iter__(self):
        with phil:
            for key in self._mapping:
                yield self._mapping.get(key)

    def __reversed__(self):
        with phil:
            for key in reversed(self._mapping):
                yield self._mapping.get(key)


class ItemsViewHDF5(ItemsView):

    """
        Wraps e.g. a Group or AttributeManager to provide an items view.
    """

    def __contains__(self, item):
        with phil:
            key, val = item
            if key in self._mapping:
                return val == self._mapping.get(key)
            return False

    def __iter__(self):
        with phil:
            for key in self._mapping:
                yield (key, self._mapping.get(key))

    def __reversed__(self):
        with phil:
            for key in reversed(self._mapping):
                yield (key, self._mapping.get(key))


class MappingHDF5(Mapping):

    """
        Wraps a Group, AttributeManager or DimensionManager object to provide
        an immutable mapping interface.

        We don't inherit directly from MutableMapping because certain
        subclasses, for example DimensionManager, are read-only.
    """
    def keys(self):
        """ Get a view object on member names """
        return KeysViewHDF5(self)

    def values(self):
        """ Get a view object on member objects """
        return ValuesViewHDF5(self)

    def items(self):
        """ Get a view object on member items """
        return ItemsViewHDF5(self)

    def _ipython_key_completions_(self):
        """ Custom tab completions for __getitem__ in IPython >=5.0. """
        return sorted(self.keys())


class MutableMappingHDF5(MappingHDF5, MutableMapping):

    """
        Wraps a Group or AttributeManager object to provide a mutable
        mapping interface, in contrast to the read-only mapping of
        MappingHDF5.
    """

    pass


class Empty(object):

    """
        Proxy object to represent empty/null dataspaces (a.k.a H5S_NULL).

        This can have an associated dtype, but has no shape or data. This is not
        the same as an array with shape (0,).
    """
    shape = None
    size = None

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        if isinstance(other, Empty) and self.dtype == other.dtype:
            return True
        return False

    def __repr__(self):
        return "Empty(dtype={0!r})".format(self.dtype)


def product(nums):
    """Calculate a numeric product

    For small amounts of data (e.g. shape tuples), this simple code is much
    faster than calling numpy.prod().
    """
    prod = 1
    for n in nums:
        prod *= n
    return prod
