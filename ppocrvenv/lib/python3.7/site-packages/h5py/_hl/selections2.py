# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Implements a portion of the selection operations.
"""

import numpy as np
from .. import h5s

def read_dtypes(dataset_dtype, names):
    """ Returns a 2-tuple containing:

    1. Output dataset dtype
    2. Dtype containing HDF5-appropriate description of destination
    """

    if len(names) == 0:     # Not compound, or all fields needed
        format_dtype = dataset_dtype

    elif dataset_dtype.names is None:
        raise ValueError("Field names only allowed for compound types")

    elif any(x not in dataset_dtype.names for x in names):
        raise ValueError("Field does not appear in this type.")

    else:
        format_dtype = np.dtype([(name, dataset_dtype.fields[name][0]) for name in names])

    if len(names) == 1:
        # We don't preserve the field information if only one explicitly selected.
        output_dtype = format_dtype.fields[names[0]][0]

    else:
        output_dtype = format_dtype

    return output_dtype, format_dtype


def read_selections_scalar(dsid, args):
    """ Returns a 2-tuple containing:

    1. Output dataset shape
    2. HDF5 dataspace containing source selection.

    Works for scalar datasets.
    """

    if dsid.shape != ():
        raise RuntimeError("Illegal selection function for non-scalar dataset")

    if args == ():
        # This is a signal that an array scalar should be returned instead
        # of an ndarray with shape ()
        out_shape = None

    elif args == (Ellipsis,):
        out_shape = ()

    else:
        raise ValueError("Illegal slicing argument for scalar dataspace")

    source_space = dsid.get_space()
    source_space.select_all()

    return out_shape, source_space

class ScalarReadSelection(object):

    """
        Implements slicing for scalar datasets.
    """

    def __init__(self, fspace, args):
        if args == ():
            self.mshape = None
        elif args == (Ellipsis,):
            self.mshape = ()
        else:
            raise ValueError("Illegal slicing argument for scalar dataspace")

        self.mspace = h5s.create(h5s.SCALAR)
        self.fspace = fspace

    def __iter__(self):
        self.mspace.select_all()
        yield self.fspace, self.mspace

def select_read(fspace, args):
    """ Top-level dispatch function for reading.

    At the moment, only supports reading from scalar datasets.
    """
    if fspace.shape == ():
        return ScalarReadSelection(fspace, args)

    raise NotImplementedError()
