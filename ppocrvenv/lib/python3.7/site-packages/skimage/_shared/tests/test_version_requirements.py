"""Tests for the version requirement functions.

"""
import numpy as np
from numpy.testing import assert_equal
from skimage._shared import version_requirements as version_req
from skimage._shared import testing


def test_get_module_version():
    assert version_req.get_module_version('numpy')
    assert version_req.get_module_version('scipy')
    with testing.raises(ImportError):
        version_req.get_module_version('fakenumpy')


def test_is_installed():
    assert version_req.is_installed('python', '>=2.7')
    assert not version_req.is_installed('numpy', '<1.0')


def test_require():
    # A function that only runs on Python >2.7 and numpy > 1.5 (should pass)
    @version_req.require('python', '>2.7')
    @version_req.require('numpy', '>1.5')
    def foo():
        return 1

    assert_equal(foo(), 1)

    # function that requires scipy < 0.1 (should fail)
    @version_req.require('scipy', '<0.1')
    def bar():
        return 0

    with testing.raises(ImportError):
        bar()


def test_get_module():
    assert version_req.get_module("numpy") is np
