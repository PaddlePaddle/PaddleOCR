# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    This module contains the warning classes for h5py. These classes are part of
    the public API of h5py, and should be imported from this module.
"""


class H5pyWarning(UserWarning):
    pass


class H5pyDeprecationWarning(H5pyWarning):
    pass
