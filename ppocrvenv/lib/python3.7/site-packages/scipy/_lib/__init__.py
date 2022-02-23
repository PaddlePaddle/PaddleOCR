"""
Module containing private utility functions
===========================================

The ``scipy._lib`` namespace is empty (for now). Tests for all
utilities in submodules of ``_lib`` can be run with::

    from scipy import _lib
    _lib.test()

"""
from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
