# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements high-level operations for attributes.

    Provides the AttributeManager class, available on high-level objects
    as <obj>.attrs.
"""

import numpy
import uuid

from .. import h5, h5s, h5t, h5a, h5p
from . import base
from .base import phil, with_phil, Empty, is_empty_dataspace, product
from .datatype import Datatype


class AttributeManager(base.MutableMappingHDF5, base.CommonStateObject):

    """
        Allows dictionary-style access to an HDF5 object's attributes.

        These are created exclusively by the library and are available as
        a Python attribute at <object>.attrs

        Like Group objects, attributes provide a minimal dictionary-
        style interface.  Anything which can be reasonably converted to a
        Numpy array or Numpy scalar can be stored.

        Attributes are automatically created on assignment with the
        syntax <obj>.attrs[name] = value, with the HDF5 type automatically
        deduced from the value.  Existing attributes are overwritten.

        To modify an existing attribute while preserving its type, use the
        method modify().  To specify an attribute of a particular type and
        shape, use create().
    """

    def __init__(self, parent):
        """ Private constructor.
        """
        self._id = parent.id

    @with_phil
    def __getitem__(self, name):
        """ Read the value of an attribute.
        """
        attr = h5a.open(self._id, self._e(name))

        if is_empty_dataspace(attr):
            return Empty(attr.dtype)

        dtype = attr.dtype
        shape = attr.shape

        # Do this first, as we'll be fiddling with the dtype for top-level
        # array types
        htype = h5t.py_create(dtype)

        # NumPy doesn't support top-level array types, so we have to "fake"
        # the correct type and shape for the array.  For example, consider
        # attr.shape == (5,) and attr.dtype == '(3,)f'. Then:
        if dtype.subdtype is not None:
            subdtype, subshape = dtype.subdtype
            shape = attr.shape + subshape   # (5, 3)
            dtype = subdtype                # 'f'

        arr = numpy.ndarray(shape, dtype=dtype, order='C')
        attr.read(arr, mtype=htype)

        string_info = h5t.check_string_dtype(dtype)
        if string_info and (string_info.length is None):
            # Vlen strings: convert bytes to Python str
            arr = numpy.array([
                b.decode('utf-8', 'surrogateescape') for b in arr.flat
            ], dtype=dtype).reshape(arr.shape)

        if len(arr.shape) == 0:
            return arr[()]
        return arr

    def get_id(self, name):
        """Get a low-level AttrID object for the named attribute.
        """
        return h5a.open(self._id, self._e(name))

    @with_phil
    def __setitem__(self, name, value):
        """ Set a new attribute, overwriting any existing attribute.

        The type and shape of the attribute are determined from the data.  To
        use a specific type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        self.create(name, data=value)

    @with_phil
    def __delitem__(self, name):
        """ Delete an attribute (which must already exist). """
        h5a.delete(self._id, self._e(name))

    def create(self, name, data, shape=None, dtype=None):
        """ Create a new attribute, overwriting any existing attribute.

        name
            Name of the new attribute (required)
        data
            An array to initialize the attribute (required)
        shape
            Shape of the attribute.  Overrides data.shape if both are
            given, in which case the total number of points must be unchanged.
        dtype
            Data type of the attribute.  Overrides data.dtype if both
            are given.
        """

        with phil:
            # First, make sure we have a NumPy array.  We leave the data type
            # conversion for HDF5 to perform.
            if not isinstance(data, Empty):
                data = base.array_for_new_object(data, specified_dtype=dtype)

            if shape is None:
                shape = data.shape
            elif isinstance(shape, int):
                shape = (shape,)

            use_htype = None    # If a committed type is given, we must use it
                                # in the call to h5a.create.

            if isinstance(dtype, Datatype):
                use_htype = dtype.id
                dtype = dtype.dtype
            elif dtype is None:
                dtype = data.dtype
            else:
                dtype = numpy.dtype(dtype) # In case a string, e.g. 'i8' is passed

            original_dtype = dtype  # We'll need this for top-level array types

            # Where a top-level array type is requested, we have to do some
            # fiddling around to present the data as a smaller array of
            # subarrays.
            if dtype.subdtype is not None:

                subdtype, subshape = dtype.subdtype

                # Make sure the subshape matches the last N axes' sizes.
                if shape[-len(subshape):] != subshape:
                    raise ValueError("Array dtype shape %s is incompatible with data shape %s" % (subshape, shape))

                # New "advertised" shape and dtype
                shape = shape[0:len(shape)-len(subshape)]
                dtype = subdtype

            # Not an array type; make sure to check the number of elements
            # is compatible, and reshape if needed.
            else:

                if shape is not None and numpy.product(shape, dtype=numpy.ulonglong) != numpy.product(data.shape, dtype=numpy.ulonglong):
                    raise ValueError("Shape of new attribute conflicts with shape of data")

                if shape != data.shape:
                    data = data.reshape(shape)

            # We need this to handle special string types.
            if not isinstance(data, Empty):
                data = numpy.asarray(data, dtype=dtype)

            # Make HDF5 datatype and dataspace for the H5A calls
            if use_htype is None:
                htype = h5t.py_create(original_dtype, logical=True)
                htype2 = h5t.py_create(original_dtype)  # Must be bit-for-bit representation rather than logical
            else:
                htype = use_htype
                htype2 = None

            if isinstance(data, Empty):
                space = h5s.create(h5s.NULL)
            else:
                space = h5s.create_simple(shape)

            # This mess exists because you can't overwrite attributes in HDF5.
            # So we write to a temporary attribute first, and then rename.

            tempname = uuid.uuid4().hex

            attr = h5a.create(self._id, self._e(tempname), htype, space)
            try:
                if not isinstance(data, Empty):
                    attr.write(data, mtype=htype2)
            except:
                attr.close()
                h5a.delete(self._id, self._e(tempname))
                raise
            else:
                try:
                    # No atomic rename in HDF5 :(
                    if h5a.exists(self._id, self._e(name)):
                        h5a.delete(self._id, self._e(name))
                    h5a.rename(self._id, self._e(tempname), self._e(name))
                except:
                    attr.close()
                    h5a.delete(self._id, self._e(tempname))
                    raise
            finally:
                attr.close()

    def modify(self, name, value):
        """ Change the value of an attribute while preserving its type.

        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.

        If the attribute doesn't exist, it will be automatically created.
        """
        with phil:
            if not name in self:
                self[name] = value
            else:
                attr = h5a.open(self._id, self._e(name))

                if is_empty_dataspace(attr):
                    raise OSError("Empty attributes can't be modified")

                # If the input data is already an array, let HDF5 do the conversion.
                # If it's a list or similar, don't make numpy guess a dtype for it.
                dt = None if isinstance(value, numpy.ndarray) else attr.dtype
                value = numpy.asarray(value, order='C', dtype=dt)

                # Allow the case of () <-> (1,)
                if (value.shape != attr.shape) and not \
                   (value.size == 1 and product(attr.shape) == 1):
                    raise TypeError("Shape of data is incompatible with existing attribute")
                attr.write(value)

    @with_phil
    def __len__(self):
        """ Number of attributes attached to the object. """
        # I expect we will not have more than 2**32 attributes
        return h5a.get_num_attrs(self._id)

    def __iter__(self):
        """ Iterate over the names of attributes. """
        with phil:

            attrlist = []
            def iter_cb(name, *args):
                """ Callback to gather attribute names """
                attrlist.append(self._d(name))

            cpl = self._id.get_create_plist()
            crt_order = cpl.get_attr_creation_order()
            cpl.close()
            if crt_order & h5p.CRT_ORDER_TRACKED:
                idx_type = h5.INDEX_CRT_ORDER
            else:
                idx_type = h5.INDEX_NAME

            h5a.iterate(self._id, iter_cb, index_type=idx_type)

        for name in attrlist:
            yield name

    @with_phil
    def __contains__(self, name):
        """ Determine if an attribute exists, by name. """
        return h5a.exists(self._id, self._e(name))

    @with_phil
    def __repr__(self):
        if not self._id:
            return "<Attributes of closed HDF5 object>"
        return "<Attributes of HDF5 object at %s>" % id(self._id)
