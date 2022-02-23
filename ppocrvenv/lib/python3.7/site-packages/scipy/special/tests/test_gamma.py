import numpy as np
import scipy.special as sc


class TestRgamma:

    def test_gh_11315(self):
        assert sc.rgamma(-35) == 0

    def test_rgamma_zeros(self):
        x = np.array([0, -10, -100, -1000, -10000])
        assert np.all(sc.rgamma(x) == 0)
