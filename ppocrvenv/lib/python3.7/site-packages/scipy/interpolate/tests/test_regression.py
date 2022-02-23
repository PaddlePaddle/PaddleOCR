import numpy as np
import scipy.interpolate as interp
from numpy.testing import assert_almost_equal


class TestRegression:
    def test_spalde_scalar_input(self):
        """Ticket #629"""
        x = np.linspace(0,10)
        y = x**3
        tck = interp.splrep(x, y, k=3, t=[5])
        res = interp.spalde(np.float64(1), tck)
        des = np.array([1., 3., 6., 6.])
        assert_almost_equal(res, des)
