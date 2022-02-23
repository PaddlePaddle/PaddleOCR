# This file is part of h5py, a Python interface to the HDF5 library.
#
# http://www.h5py.org
#
# Copyright 2008-2013 Andrew Collette and contributors
#
# License:  Standard 3-clause BSD; see "license.txt" for full license terms
#           and contributor agreement.

"""
    This is the h5py package, a Python interface to the HDF5
    scientific data format.
"""

from warnings import warn as _warn
import atexit


# --- Library setup -----------------------------------------------------------

# When importing from the root of the unpacked tarball or git checkout,
# Python sees the "h5py" source directory and tries to load it, which fails.
# We tried working around this by using "package_dir" but that breaks Cython.
try:
    from . import _errors
except ImportError:
    import os.path as _op
    if _op.exists(_op.join(_op.dirname(__file__), '..', 'setup.py')):
        raise ImportError("You cannot import h5py from inside the install directory.\nChange to another directory first.")
    else:
        raise

from . import version

if version.hdf5_version_tuple != version.hdf5_built_version_tuple:
    _warn(("h5py is running against HDF5 {0} when it was built against {1}, "
           "this may cause problems").format(
            '{0}.{1}.{2}'.format(*version.hdf5_version_tuple),
            '{0}.{1}.{2}'.format(*version.hdf5_built_version_tuple)
    ))


_errors.silence_errors()

from ._conv import register_converters as _register_converters, \
                   unregister_converters as _unregister_converters
_register_converters()
atexit.register(_unregister_converters)

from .h5z import _register_lzf
_register_lzf()


# --- Public API --------------------------------------------------------------

from . import h5a, h5d, h5ds, h5f, h5fd, h5g, h5r, h5s, h5t, h5p, h5z, h5pl

from ._hl import filters
from ._hl.base import is_hdf5, HLObject, Empty
from ._hl.files import (
    File,
    register_driver,
    unregister_driver,
    registered_drivers,
)
from ._hl.group import Group, SoftLink, ExternalLink, HardLink
from ._hl.dataset import Dataset
from ._hl.datatype import Datatype
from ._hl.attrs import AttributeManager

from ._selector import MultiBlockSlice
from .h5 import get_config
from .h5r import Reference, RegionReference
from .h5t import (special_dtype, check_dtype,
    vlen_dtype, string_dtype, enum_dtype, ref_dtype, regionref_dtype,
    opaque_dtype,
    check_vlen_dtype, check_string_dtype, check_enum_dtype, check_ref_dtype,
    check_opaque_dtype,
)
from .h5s import UNLIMITED

from .version import version as __version__


if version.hdf5_version_tuple[:3] >= get_config().vds_min_hdf5_version:
    from ._hl.vds import VirtualSource, VirtualLayout


def run_tests(args=''):
    """Run tests with pytest and returns the exit status as an int.
    """
    # Lazy-loading of tests package to avoid strong dependency on test
    # requirements, e.g. pytest
    from .tests import run_tests
    return run_tests(args)


def enable_ipython_completer():
    """ Call this from an interactive IPython session to enable tab-completion
    of group and attribute names.
    """
    import sys
    if 'IPython' in sys.modules:
        ip_running = False
        try:
            from IPython.core.interactiveshell import InteractiveShell
            ip_running = InteractiveShell.initialized()
        except ImportError:
            # support <ipython-0.11
            from IPython import ipapi as _ipapi
            ip_running = _ipapi.get() is not None
        except Exception:
            pass
        if ip_running:
            from . import ipy_completer
            return ipy_completer.load_ipython_extension()

    raise RuntimeError('Completer must be enabled in active ipython session')
