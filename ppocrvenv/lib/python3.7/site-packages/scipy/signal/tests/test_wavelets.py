import numpy as np
from numpy.testing import assert_equal, \
    assert_array_equal, assert_array_almost_equal, assert_array_less, assert_

from scipy.signal import wavelets


class TestWavelets:
    def test_qmf(self):
        assert_array_equal(wavelets.qmf([1, 1]), [1, -1])

    def test_daub(self):
        for i in range(1, 15):
            assert_equal(len(wavelets.daub(i)), i * 2)

    def test_cascade(self):
        for J in range(1, 7):
            for i in range(1, 5):
                lpcoef = wavelets.daub(i)
                k = len(lpcoef)
                x, phi, psi = wavelets.cascade(lpcoef, J)
                assert_(len(x) == len(phi) == len(psi))
                assert_equal(len(x), (k - 1) * 2 ** J)

    def test_morlet(self):
        x = wavelets.morlet(50, 4.1, complete=True)
        y = wavelets.morlet(50, 4.1, complete=False)
        # Test if complete and incomplete wavelet have same lengths:
        assert_equal(len(x), len(y))
        # Test if complete wavelet is less than incomplete wavelet:
        assert_array_less(x, y)

        x = wavelets.morlet(10, 50, complete=False)
        y = wavelets.morlet(10, 50, complete=True)
        # For large widths complete and incomplete wavelets should be
        # identical within numerical precision:
        assert_equal(x, y)

        # miscellaneous tests:
        x = np.array([1.73752399e-09 + 9.84327394e-25j,
                      6.49471756e-01 + 0.00000000e+00j,
                      1.73752399e-09 - 9.84327394e-25j])
        y = wavelets.morlet(3, w=2, complete=True)
        assert_array_almost_equal(x, y)

        x = np.array([2.00947715e-09 + 9.84327394e-25j,
                      7.51125544e-01 + 0.00000000e+00j,
                      2.00947715e-09 - 9.84327394e-25j])
        y = wavelets.morlet(3, w=2, complete=False)
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, s=4, complete=True)
        y = wavelets.morlet(20000, s=8, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, s=4, complete=False)
        assert_array_almost_equal(y, x, decimal=2)
        y = wavelets.morlet(20000, s=8, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, w=3, s=5, complete=True)
        y = wavelets.morlet(20000, w=3, s=10, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, w=3, s=5, complete=False)
        assert_array_almost_equal(y, x, decimal=2)
        y = wavelets.morlet(20000, w=3, s=10, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, w=7, s=10, complete=True)
        y = wavelets.morlet(20000, w=7, s=20, complete=True)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

        x = wavelets.morlet(10000, w=7, s=10, complete=False)
        assert_array_almost_equal(x, y, decimal=2)
        y = wavelets.morlet(20000, w=7, s=20, complete=False)[5000:15000]
        assert_array_almost_equal(x, y, decimal=2)

    def test_morlet2(self):
        w = wavelets.morlet2(1.0, 0.5)
        expected = (np.pi**(-0.25) * np.sqrt(1/0.5)).astype(complex)
        assert_array_equal(w, expected)

        lengths = [5, 11, 15, 51, 101]
        for length in lengths:
            w = wavelets.morlet2(length, 1.0)
            assert_(len(w) == length)
            max_loc = np.argmax(w)
            assert_(max_loc == (length // 2))

        points = 100
        w = abs(wavelets.morlet2(points, 2.0))
        half_vec = np.arange(0, points // 2)
        assert_array_almost_equal(w[half_vec], w[-(half_vec + 1)])

        x = np.array([5.03701224e-09 + 2.46742437e-24j,
                      1.88279253e+00 + 0.00000000e+00j,
                      5.03701224e-09 - 2.46742437e-24j])
        y = wavelets.morlet2(3, s=1/(2*np.pi), w=2)
        assert_array_almost_equal(x, y)

    def test_ricker(self):
        w = wavelets.ricker(1.0, 1)
        expected = 2 / (np.sqrt(3 * 1.0) * (np.pi ** 0.25))
        assert_array_equal(w, expected)

        lengths = [5, 11, 15, 51, 101]
        for length in lengths:
            w = wavelets.ricker(length, 1.0)
            assert_(len(w) == length)
            max_loc = np.argmax(w)
            assert_(max_loc == (length // 2))

        points = 100
        w = wavelets.ricker(points, 2.0)
        half_vec = np.arange(0, points // 2)
        #Wavelet should be symmetric
        assert_array_almost_equal(w[half_vec], w[-(half_vec + 1)])

        #Check zeros
        aas = [5, 10, 15, 20, 30]
        points = 99
        for a in aas:
            w = wavelets.ricker(points, a)
            vec = np.arange(0, points) - (points - 1.0) / 2
            exp_zero1 = np.argmin(np.abs(vec - a))
            exp_zero2 = np.argmin(np.abs(vec + a))
            assert_array_almost_equal(w[exp_zero1], 0)
            assert_array_almost_equal(w[exp_zero2], 0)

    def test_cwt(self):
        widths = [1.0]
        delta_wavelet = lambda s, t: np.array([1])
        len_data = 100
        test_data = np.sin(np.pi * np.arange(0, len_data) / 10.0)

        #Test delta function input gives same data as output
        cwt_dat = wavelets.cwt(test_data, delta_wavelet, widths)
        assert_(cwt_dat.shape == (len(widths), len_data))
        assert_array_almost_equal(test_data, cwt_dat.flatten())

        #Check proper shape on output
        widths = [1, 3, 4, 5, 10]
        cwt_dat = wavelets.cwt(test_data, wavelets.ricker, widths)
        assert_(cwt_dat.shape == (len(widths), len_data))

        widths = [len_data * 10]
        #Note: this wavelet isn't defined quite right, but is fine for this test
        flat_wavelet = lambda l, w: np.full(w, 1 / w)
        cwt_dat = wavelets.cwt(test_data, flat_wavelet, widths)
        assert_array_almost_equal(cwt_dat, np.mean(test_data))

