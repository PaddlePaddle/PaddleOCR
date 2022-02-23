""" Unit tests for nonnegative least squares
Author: Uwe Schmitt
Sep 2008
"""
import numpy as np

from numpy.testing import assert_
from pytest import raises as assert_raises

from scipy.optimize import nnls
from numpy import arange, dot
from numpy.linalg import norm


class TestNNLS:

    def test_nnls(self):
        a = arange(25.0).reshape(-1,5)
        x = arange(5.0)
        y = dot(a,x)
        x, res = nnls(a,y)
        assert_(res < 1e-7)
        assert_(norm(dot(a,x)-y) < 1e-7)

    def test_maxiter(self):
        # test that maxiter argument does stop iterations
        # NB: did not manage to find a test case where the default value
        # of maxiter is not sufficient, so use a too-small value
        rndm = np.random.RandomState(1234)
        a = rndm.uniform(size=(100, 100))
        b = rndm.uniform(size=100)
        with assert_raises(RuntimeError):
            nnls(a, b, maxiter=1)

