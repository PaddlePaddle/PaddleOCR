import os
import pytest

import numpy as np
from numpy.testing import assert_, assert_raises, assert_equal, assert_string_equal

from . import util


def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))


class TestIntentInOut(util.F2PyTest):
    # Check that intent(in out) translates as intent(inout)
    sources = [_path('src', 'regression', 'inout.f90')]

    @pytest.mark.slow
    def test_inout(self):
        # non-contiguous should raise error
        x = np.arange(6, dtype=np.float32)[::2]
        assert_raises(ValueError, self.module.foo, x)

        # check values with contiguous array
        x = np.arange(3, dtype=np.float32)
        self.module.foo(x)
        assert_equal(x, [3, 1, 2])


class TestNumpyVersionAttribute(util.F2PyTest):
    # Check that th attribute __f2py_numpy_version__ is present
    # in the compiled module and that has the value np.__version__.
    sources = [_path('src', 'regression', 'inout.f90')]

    @pytest.mark.slow
    def test_numpy_version_attribute(self):

        # Check that self.module has an attribute named "__f2py_numpy_version__"
        assert_(hasattr(self.module, "__f2py_numpy_version__"),
                msg="Fortran module does not have __f2py_numpy_version__")

        # Check that the attribute __f2py_numpy_version__ is a string
        assert_(isinstance(self.module.__f2py_numpy_version__, str),
                msg="__f2py_numpy_version__ is not a string")

        # Check that __f2py_numpy_version__ has the value numpy.__version__
        assert_string_equal(np.__version__, self.module.__f2py_numpy_version__)


def test_include_path():
    incdir = np.f2py.get_include()
    fnames_in_dir = os.listdir(incdir)
    for fname in ('fortranobject.c', 'fortranobject.h'):
        assert fname in fnames_in_dir

