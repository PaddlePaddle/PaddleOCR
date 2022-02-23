# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    Versioning module for h5py.
"""

from collections import namedtuple
from . import h5 as _h5
import sys
import numpy

# All should be integers, except pre, as validating versions is more than is
# needed for our use case
_H5PY_VERSION_CLS = namedtuple("_H5PY_VERSION_CLS",
                               "major minor bugfix pre post dev")

hdf5_built_version_tuple = _h5.HDF5_VERSION_COMPILED_AGAINST

version_tuple = _H5PY_VERSION_CLS(3, 6, 0, None, None, None)

version = "{0.major:d}.{0.minor:d}.{0.bugfix:d}".format(version_tuple)
if version_tuple.pre is not None:
    version += version_tuple.pre
if version_tuple.post is not None:
    version += ".post{0.post:d}".format(version_tuple)
if version_tuple.dev is not None:
    version += ".dev{0.dev:d}".format(version_tuple)

hdf5_version_tuple = _h5.get_libversion()
hdf5_version = "%d.%d.%d" % hdf5_version_tuple

api_version_tuple = (1,8)
api_version = "%d.%d" % api_version_tuple

info = """\
Summary of the h5py configuration
---------------------------------

h5py    %(h5py)s
HDF5    %(hdf5)s
Python  %(python)s
sys.platform    %(platform)s
sys.maxsize     %(maxsize)s
numpy   %(numpy)s
cython (built with) %(cython_version)s
numpy (built against) %(numpy_build_version)s
HDF5 (built against) %(hdf5_build_version)s
""" % {
    'h5py': version,
    'hdf5': hdf5_version,
    'python': sys.version,
    'platform': sys.platform,
    'maxsize': sys.maxsize,
    'numpy': numpy.__version__,
    'cython_version': _h5.CYTHON_VERSION_COMPILED_WITH,
    'numpy_build_version': _h5.NUMPY_VERSION_COMPILED_AGAINST,
    'hdf5_build_version': "%d.%d.%d" % hdf5_built_version_tuple,
}
