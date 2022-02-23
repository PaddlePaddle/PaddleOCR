# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    High-level interface for creating HDF5 virtual datasets
"""

from copy import deepcopy as copy
from collections import namedtuple

import numpy as np

from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t, h5
from .. import version


class VDSmap(namedtuple('VDSmap', ('vspace', 'file_name',
                                   'dset_name', 'src_space'))):
    '''Defines a region in a virtual dataset mapping to part of a source dataset
    '''


vds_support = False
hdf5_version = version.hdf5_version_tuple[0:3]

if hdf5_version >= h5.get_config().vds_min_hdf5_version:
    vds_support = True


def _convert_space_for_key(space, key):
    """
    Converts the space with the given key. Mainly used to allow unlimited
    dimensions in virtual space selection.
    """
    key = key if isinstance(key, tuple) else (key,)
    type_code = space.get_select_type()

    # check for unlimited selections in case where selection is regular
    # hyperslab, which is the only allowed case for h5s.UNLIMITED to be
    # in the selection
    if type_code == h5s.SEL_HYPERSLABS and space.is_regular_hyperslab():
        rank = space.get_simple_extent_ndims()
        nargs = len(key)

        idx_offset = 0
        start, stride, count, block = space.get_regular_hyperslab()
        # iterate through keys. we ignore numeral indices. if we get a
        # slice, we check for an h5s.UNLIMITED value as the stop
        # if we get an ellipsis, we offset index by (rank - nargs)
        for i, sl in enumerate(key):
            if isinstance(sl, slice):
                if sl.stop == h5s.UNLIMITED:
                    counts = list(count)
                    idx = i + idx_offset
                    counts[idx] = h5s.UNLIMITED
                    count = tuple(counts)
            elif sl is Ellipsis:
                idx_offset = rank - nargs

        space.select_hyperslab(start, count, stride, block)


class VirtualSource(object):
    """Source definition for virtual data sets.

    Instantiate this class to represent an entire source dataset, and then
    slice it to indicate which regions should be used in the virtual dataset.

    path_or_dataset
        The path to a file, or an h5py dataset. If a dataset is given,
        no other parameters are allowed, as the relevant values are taken from
        the dataset instead.
    name
        The name of the source dataset within the file.
    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The source dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    """
    def __init__(self, path_or_dataset, name=None,
                 shape=None, dtype=None, maxshape=None):
        from .dataset import Dataset
        if isinstance(path_or_dataset, Dataset):
            failed = {k: v
                      for k, v in
                      {'name': name, 'shape': shape,
                       'dtype': dtype, 'maxshape': maxshape}.items()
                      if v is not None}
            if failed:
                raise TypeError("If a Dataset is passed as the first argument "
                                "then no other arguments may be passed.  You "
                                "passed {failed}".format(failed=failed))
            ds = path_or_dataset
            path = ds.file.filename
            name = ds.name
            shape = ds.shape
            dtype = ds.dtype
            maxshape = ds.maxshape
        else:
            path = path_or_dataset
            if name is None:
                raise TypeError("The name parameter is required when "
                                "specifying a source by path")
            if shape is None:
                raise TypeError("The shape parameter is required when "
                                "specifying a source by path")
            elif isinstance(shape, int):
                shape = (shape,)

            if isinstance(maxshape, int):
                maxshape = (maxshape,)

        self.path = path
        self.name = name
        self.dtype = dtype

        if maxshape is None:
            self.maxshape = shape
        else:
            self.maxshape = tuple([h5s.UNLIMITED if ix is None else ix
                                   for ix in maxshape])
        self.sel = SimpleSelection(shape)

    @property
    def shape(self):
        return self.sel.array_shape

    def __getitem__(self, key):
        tmp = copy(self)
        tmp.sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(tmp.sel.id, key)
        return tmp

class VirtualLayout(object):
    """Object for building a virtual dataset.

    Instantiate this class to define a virtual dataset, assign to slices of it
    (using VirtualSource objects), and then pass it to
    group.create_virtual_dataset() to add the virtual dataset to a file.

    This class does not allow access to the data; the virtual dataset must
    be created in a file before it can be used.

    shape
        A tuple giving the shape of the dataset.
    dtype
        Numpy dtype or string.
    maxshape
        The virtual dataset is resizable up to this shape. Use None for
        axes you want to be unlimited.
    filename
        The name of the destination file, if known in advance. Mappings from
        data in the same file will be stored with filename '.', allowing the
        file to be renamed later.
    """
    def __init__(self, shape, dtype, maxshape=None, filename=None):
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.dtype = dtype
        self.maxshape = (maxshape,) if isinstance(maxshape, int) else maxshape
        self._filename = filename
        self._src_filenames = set()
        self.dcpl = h5p.create(h5p.DATASET_CREATE)

    def __setitem__(self, key, source):
        sel = select(self.shape, key, dataset=None)
        _convert_space_for_key(sel.id, key)
        src_filename = self._source_file_name(source.path, self._filename)

        self.dcpl.set_virtual(
            sel.id, src_filename, source.name.encode('utf-8'), source.sel.id
        )
        if self._filename is None:
            self._src_filenames.add(src_filename)

    @staticmethod
    def _source_file_name(src_filename, dst_filename) -> bytes:
        src_filename = filename_encode(src_filename)
        if dst_filename and (src_filename == filename_encode(dst_filename)):
            # use relative path if the source dataset is in the same
            # file, in order to keep the virtual dataset valid in case
            # the file is renamed.
            return b'.'
        return filename_encode(src_filename)

    def _get_dcpl(self, dst_filename):
        """Get the property list containing virtual dataset mappings

        If the destination filename wasn't known when the VirtualLayout was
        created, it is handled here.
        """
        dst_filename = filename_encode(dst_filename)
        if self._filename is not None:
            # filename was known in advance; check dst_filename matches
            if dst_filename != filename_encode(self._filename):
                raise Exception(f"{dst_filename!r} != {self._filename!r}")
            return self.dcpl

        # destination file not known in advance
        if dst_filename in self._src_filenames:
            # At least 1 source file is the same as the destination file,
            # but we didn't know this when making the mapping. Copy the mappings
            # to a new property list, replacing the dest filename with '.'
            new_dcpl = h5p.create(h5p.DATASET_CREATE)
            for i in range(self.dcpl.get_virtual_count()):
                src_filename = self.dcpl.get_virtual_filename(i)
                new_dcpl.set_virtual(
                    self.dcpl.get_virtual_vspace(i),
                    self._source_file_name(src_filename, dst_filename),
                    self.dcpl.get_virtual_dsetname(i).encode('utf-8'),
                    self.dcpl.get_virtual_srcspace(i),
                )
            return new_dcpl
        else:
            return self.dcpl  # Mappings are all from other files

    def make_dataset(self, parent, name, fillvalue=None):
        """ Return a new low-level dataset identifier for a virtual dataset """
        dcpl = self._get_dcpl(parent.file.filename)

        if fillvalue is not None:
            dcpl.set_fill_value(np.array([fillvalue]))

        maxshape = self.maxshape
        if maxshape is not None:
            maxshape = tuple(m if m is not None else h5s.UNLIMITED for m in maxshape)

        virt_dspace = h5s.create_simple(self.shape, maxshape)

        if isinstance(self.dtype, Datatype):
            # Named types are used as-is
            tid = self.dtype.id
        else:
            dtype = np.dtype(self.dtype)
            tid = h5t.py_create(dtype, logical=1)

        return h5d.create(parent.id, name=name, tid=tid, space=virt_dspace,
                          dcpl=dcpl)
