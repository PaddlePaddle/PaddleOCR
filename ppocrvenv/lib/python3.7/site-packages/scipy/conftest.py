# Pytest customization
import os
import pytest
import warnings

from distutils.version import LooseVersion
import numpy as np
from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning


def pytest_configure(config):
    config.addinivalue_line("markers",
        "slow: Tests that are very slow.")
    config.addinivalue_line("markers",
        "xslow: mark test as extremely slow (not run unless explicitly requested)")
    config.addinivalue_line("markers",
        "xfail_on_32bit: mark test as failing on 32-bit platforms")


def _get_mark(item, name):
    if LooseVersion(pytest.__version__) >= LooseVersion("3.6.0"):
        mark = item.get_closest_marker(name)
    else:
        mark = item.get_marker(name)
    return mark


def pytest_runtest_setup(item):
    mark = _get_mark(item, "xslow")
    if mark is not None:
        try:
            v = int(os.environ.get('SCIPY_XSLOW', '0'))
        except ValueError:
            v = False
        if not v:
            pytest.skip("very slow test; set environment variable SCIPY_XSLOW=1 to run it")
    mark = _get_mark(item, 'xfail_on_32bit')
    if mark is not None and np.intp(0).itemsize < 8:
        pytest.xfail('Fails on our 32-bit test platform(s): %s' % (mark.args[0],))


@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()

    if old_mode != new_mode:
        warnings.warn("FPU mode changed from {0:#x} to {1:#x} during "
                      "the test".format(old_mode, new_mode),
                      category=FPUModeChangeWarning, stacklevel=0)
