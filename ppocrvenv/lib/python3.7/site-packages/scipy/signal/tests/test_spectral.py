import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
                           assert_allclose, assert_array_equal, assert_equal,
                           assert_array_almost_equal_nulp, suppress_warnings)
import pytest
from pytest import raises as assert_raises

from scipy import signal
from scipy.fft import fftfreq
from scipy.signal import (periodogram, welch, lombscargle, csd, coherence,
                          spectrogram, stft, istft, check_COLA, check_NOLA)
from scipy.signal.spectral import _spectral_helper


class TestPeriodogram:
    def test_real_onesided_even(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_real_onesided_odd(self):
        x = np.zeros(15)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0)/15.0)
        q = np.ones(8)
        q[0] = 0
        q *= 2.0/15.0
        assert_allclose(p, q, atol=1e-15)

    def test_real_twosided(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1/16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_real_spectrum(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, scaling='spectrum')
        g, q = periodogram(x, scaling='density')
        assert_allclose(f, np.linspace(0, 0.5, 9))
        assert_allclose(p, q/16.0)

    def test_integer_even(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_integer_odd(self):
        x = np.zeros(15, dtype=int)
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0)/15.0)
        q = np.ones(8)
        q[0] = 0
        q *= 2.0/15.0
        assert_allclose(p, q, atol=1e-15)

    def test_integer_twosided(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1/16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_complex(self):
        x = np.zeros(16, np.complex128)
        x[0] = 1.0 + 2.0j
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 5.0/16.0)
        q[0] = 0
        assert_allclose(p, q)

    def test_unk_scaling(self):
        assert_raises(ValueError, periodogram, np.zeros(4, np.complex128),
                scaling='foo')

    def test_nd_axis_m1(self):
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((2,1,10))
        x[:,:,0] = 1.0
        f, p = periodogram(x)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_array_almost_equal_nulp(p[0,0,:], p[1,0,:], 60)
        f0, p0 = periodogram(x[0,0,:])
        assert_array_almost_equal_nulp(p0[np.newaxis,:], p[1,:], 60)

    def test_nd_axis_0(self):
        x = np.zeros(20, dtype=np.float64)
        x = x.reshape((10,2,1))
        x[0,:,:] = 1.0
        f, p = periodogram(x, axis=0)
        assert_array_equal(p.shape, (6,2,1))
        assert_array_almost_equal_nulp(p[:,0,0], p[:,1,0], 60)
        f0, p0 = periodogram(x[:,0,0])
        assert_array_almost_equal_nulp(p0, p[:,1,0])

    def test_window_external(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, 10, 'hann')
        win = signal.get_window('hann', 16)
        fe, pe = periodogram(x, 10, win)
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        win_err = signal.get_window('hann', 32)
        assert_raises(ValueError, periodogram, x,
                      10, win_err)  # win longer than signal

    def test_padded_fft(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x)
        fp, pp = periodogram(x, nfft=32)
        assert_allclose(f, fp[::2])
        assert_allclose(p, pp[::2])
        assert_array_equal(pp.shape, (17,))

    def test_empty_input(self):
        f, p = periodogram([])
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))
        for shape in [(0,), (3,0), (0,5,2)]:
            f, p = periodogram(np.empty(shape))
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_empty_input_other_axis(self):
        for shape in [(3,0), (0,5,2)]:
            f, p = periodogram(np.empty(shape), axis=1)
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_short_nfft(self):
        x = np.zeros(18)
        x[0] = 1
        f, p = periodogram(x, nfft=16)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_nfft_is_xshape(self):
        x = np.zeros(16)
        x[0] = 1
        f, p = periodogram(x, nfft=16)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9)
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.linspace(0, 0.5, 9))
        q = np.ones(9, 'f')
        q[0] = 0
        q[-1] /= 2.0
        q /= 8
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(15, 'f')
        x[0] = 1
        f, p = periodogram(x)
        assert_allclose(f, np.arange(8.0)/15.0)
        q = np.ones(8, 'f')
        q[0] = 0
        q *= 2.0/15.0
        assert_allclose(p, q, atol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 1/16.0, 'f')
        q[0] = 0
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)

    def test_complex_32(self):
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        f, p = periodogram(x, return_onesided=False)
        assert_allclose(f, fftfreq(16, 1.0))
        q = np.full(16, 5.0/16.0, 'f')
        q[0] = 0
        assert_allclose(p, q)
        assert_(p.dtype == q.dtype)


class TestWelch:
    def test_real_onesided_even(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_onesided_odd(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_twosided(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_spectrum(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8, scaling='spectrum')
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.015625, 0.02864583, 0.04166667, 0.04166667,
                      0.02083333])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_onesided_even(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_onesided_odd(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_twosided(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_complex(self):
        x = np.zeros(16, np.complex128)
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = welch(x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666667, 0.38194444, 0.55555556, 0.55555556,
                      0.55555556, 0.55555556, 0.55555556, 0.38194444])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_unk_scaling(self):
        assert_raises(ValueError, welch, np.zeros(4, np.complex128),
                      scaling='foo', nperseg=4)

    def test_detrend_linear(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = welch(x, nperseg=10, detrend='linear')
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_no_detrending(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f1, p1 = welch(x, nperseg=10, detrend=False)
        f2, p2 = welch(x, nperseg=10, detrend=lambda x: x)
        assert_allclose(f1, f2, atol=1e-15)
        assert_allclose(p1, p2, atol=1e-15)

    def test_detrend_external(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = welch(x, nperseg=10,
                     detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_m1(self):
        x = np.arange(40, dtype=np.float64) + 0.04
        x = x.reshape((2,2,10))
        f, p = welch(x, nperseg=10,
                     detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2,1,10))
        x = np.rollaxis(x, 2, 0)
        f, p = welch(x, nperseg=10, axis=0,
                     detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_nd_axis_m1(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2,1,10))
        f, p = welch(x, nperseg=10)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_allclose(p[0,0,:], p[1,0,:], atol=1e-13, rtol=1e-13)
        f0, p0 = welch(x[0,0,:], nperseg=10)
        assert_allclose(p0[np.newaxis,:], p[1,:], atol=1e-13, rtol=1e-13)

    def test_nd_axis_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((10,2,1))
        f, p = welch(x, nperseg=10, axis=0)
        assert_array_equal(p.shape, (6,2,1))
        assert_allclose(p[:,0,0], p[:,1,0], atol=1e-13, rtol=1e-13)
        f0, p0 = welch(x[:,0,0], nperseg=10)
        assert_allclose(p0, p[:,1,0], atol=1e-13, rtol=1e-13)

    def test_window_external(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, 10, 'hann', nperseg=8)
        win = signal.get_window('hann', 8)
        fe, pe = welch(x, 10, win, nperseg=None)
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        assert_array_equal(fe.shape, (5,))  # because win length used as nperseg
        assert_array_equal(pe.shape, (5,))
        assert_raises(ValueError, welch, x,
                      10, win, nperseg=4)  # because nperseg != win.shape[-1]
        win_err = signal.get_window('hann', 32)
        assert_raises(ValueError, welch, x,
                      10, win_err, nperseg=None)  # win longer than signal

    def test_empty_input(self):
        f, p = welch([])
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))
        for shape in [(0,), (3,0), (0,5,2)]:
            f, p = welch(np.empty(shape))
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_empty_input_other_axis(self):
        for shape in [(3,0), (0,5,2)]:
            f, p = welch(np.empty(shape), axis=1)
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

    def test_short_data(self):
        x = np.zeros(8)
        x[0] = 1
        #for string-like window, input signal length < nperseg value gives
        #UserWarning, sets nperseg to x.shape[-1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "nperseg = 256 is greater than input length  = 8, using nperseg = 8")
            f, p = welch(x,window='hann')  # default nperseg
            f1, p1 = welch(x,window='hann', nperseg=256)  # user-specified nperseg
        f2, p2 = welch(x, nperseg=8)  # valid nperseg, doesn't give warning
        assert_allclose(f, f2)
        assert_allclose(p, p2)
        assert_allclose(f1, f2)
        assert_allclose(p1, p2)

    def test_window_long_or_nd(self):
        assert_raises(ValueError, welch, np.zeros(4), 1, np.array([1,1,1,1,1]))
        assert_raises(ValueError, welch, np.zeros(4), 1,
                      np.arange(6).reshape((2,3)))

    def test_nondefault_noverlap(self):
        x = np.zeros(64)
        x[::8] = 1
        f, p = welch(x, nperseg=16, noverlap=4)
        q = np.array([0, 1./12., 1./3., 1./5., 1./3., 1./5., 1./3., 1./5.,
                      1./6.])
        assert_allclose(p, q, atol=1e-12)

    def test_bad_noverlap(self):
        assert_raises(ValueError, welch, np.zeros(4), 1, 'hann', 2, 7)

    def test_nfft_too_short(self):
        assert_raises(ValueError, welch, np.ones(12), nfft=3, nperseg=4)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477458, 0.23430935, 0.17072113, 0.17072116,
                      0.17072113], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.11111111,
                      0.07638889], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_complex_32(self):
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = welch(x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552,
                      0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype,
                'dtype mismatch, %s, %s' % (p.dtype, q.dtype))

    def test_padded_freqs(self):
        x = np.zeros(12)

        nfft = 24
        f = fftfreq(nfft, 1.0)[:nfft//2+1]
        f[-1] *= -1
        fodd, _ = welch(x, nperseg=5, nfft=nfft)
        feven, _ = welch(x, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

        nfft = 25
        f = fftfreq(nfft, 1.0)[:(nfft + 1)//2]
        fodd, _ = welch(x, nperseg=5, nfft=nfft)
        feven, _ = welch(x, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

    def test_window_correction(self):
        A = 20
        fs = 1e4
        nperseg = int(fs//10)
        fsig = 300
        ii = int(fsig*nperseg//fs)  # Freq index of fsig

        tt = np.arange(fs)/fs
        x = A*np.sin(2*np.pi*fsig*tt)

        for window in ['hann', 'bartlett', ('tukey', 0.1), 'flattop']:
            _, p_spec = welch(x, fs=fs, nperseg=nperseg, window=window,
                              scaling='spectrum')
            freq, p_dens = welch(x, fs=fs, nperseg=nperseg, window=window,
                                 scaling='density')

            # Check peak height at signal frequency for 'spectrum'
            assert_allclose(p_spec[ii], A**2/2.0)
            # Check integrated spectrum RMS for 'density'
            assert_allclose(np.sqrt(np.trapz(p_dens, freq)), A*np.sqrt(2)/2,
                            rtol=1e-3)

    def test_axis_rolling(self):
        np.random.seed(1234)

        x_flat = np.random.randn(1024)
        _, p_flat = welch(x_flat)

        for a in range(3):
            newshape = [1,]*3
            newshape[a] = -1
            x = x_flat.reshape(newshape)

            _, p_plus = welch(x, axis=a)  # Positive axis index
            _, p_minus = welch(x, axis=a-x.ndim)  # Negative axis index

            assert_equal(p_flat, p_plus.squeeze(), err_msg=a)
            assert_equal(p_flat, p_minus.squeeze(), err_msg=a-x.ndim)

    def test_average(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = welch(x, nperseg=8, average='median')
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([.1, .05, 0., 1.54074396e-33, 0.])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

        assert_raises(ValueError, welch, x, nperseg=8,
                      average='unrecognised-average')


class TestCSD:
    def test_pad_shorter_x(self):
        x = np.zeros(8)
        y = np.zeros(12)

        f = np.linspace(0, 0.5, 7)
        c = np.zeros(7,dtype=np.complex128)
        f1, c1 = csd(x, y, nperseg=12)

        assert_allclose(f, f1)
        assert_allclose(c, c1)

    def test_pad_shorter_y(self):
        x = np.zeros(12)
        y = np.zeros(8)

        f = np.linspace(0, 0.5, 7)
        c = np.zeros(7,dtype=np.complex128)
        f1, c1 = csd(x, y, nperseg=12)

        assert_allclose(f, f1)
        assert_allclose(c, c1)

    def test_real_onesided_even(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_onesided_odd(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_twosided(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_real_spectrum(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, scaling='spectrum')
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.015625, 0.02864583, 0.04166667, 0.04166667,
                      0.02083333])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_onesided_even(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_onesided_odd(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477455, 0.23430933, 0.17072113, 0.17072113,
                      0.17072113])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_integer_twosided(self):
        x = np.zeros(16, dtype=int)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.07638889])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_complex(self):
        x = np.zeros(16, np.complex128)
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666667, 0.38194444, 0.55555556, 0.55555556,
                      0.55555556, 0.55555556, 0.55555556, 0.38194444])
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)

    def test_unk_scaling(self):
        assert_raises(ValueError, csd, np.zeros(4, np.complex128),
                      np.ones(4, np.complex128), scaling='foo', nperseg=4)

    def test_detrend_linear(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = csd(x, x, nperseg=10, detrend='linear')
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_no_detrending(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f1, p1 = csd(x, x, nperseg=10, detrend=False)
        f2, p2 = csd(x, x, nperseg=10, detrend=lambda x: x)
        assert_allclose(f1, f2, atol=1e-15)
        assert_allclose(p1, p2, atol=1e-15)

    def test_detrend_external(self):
        x = np.arange(10, dtype=np.float64) + 0.04
        f, p = csd(x, x, nperseg=10,
                   detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_m1(self):
        x = np.arange(40, dtype=np.float64) + 0.04
        x = x.reshape((2,2,10))
        f, p = csd(x, x, nperseg=10,
                   detrend=lambda seg: signal.detrend(seg, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_detrend_external_nd_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2,1,10))
        x = np.rollaxis(x, 2, 0)
        f, p = csd(x, x, nperseg=10, axis=0,
                   detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
        assert_allclose(p, np.zeros_like(p), atol=1e-15)

    def test_nd_axis_m1(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((2,1,10))
        f, p = csd(x, x, nperseg=10)
        assert_array_equal(p.shape, (2, 1, 6))
        assert_allclose(p[0,0,:], p[1,0,:], atol=1e-13, rtol=1e-13)
        f0, p0 = csd(x[0,0,:], x[0,0,:], nperseg=10)
        assert_allclose(p0[np.newaxis,:], p[1,:], atol=1e-13, rtol=1e-13)

    def test_nd_axis_0(self):
        x = np.arange(20, dtype=np.float64) + 0.04
        x = x.reshape((10,2,1))
        f, p = csd(x, x, nperseg=10, axis=0)
        assert_array_equal(p.shape, (6,2,1))
        assert_allclose(p[:,0,0], p[:,1,0], atol=1e-13, rtol=1e-13)
        f0, p0 = csd(x[:,0,0], x[:,0,0], nperseg=10)
        assert_allclose(p0, p[:,1,0], atol=1e-13, rtol=1e-13)

    def test_window_external(self):
        x = np.zeros(16)
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, 10, 'hann', 8)
        win = signal.get_window('hann', 8)
        fe, pe = csd(x, x, 10, win, nperseg=None)
        assert_array_almost_equal_nulp(p, pe)
        assert_array_almost_equal_nulp(f, fe)
        assert_array_equal(fe.shape, (5,))  # because win length used as nperseg
        assert_array_equal(pe.shape, (5,))
        assert_raises(ValueError, csd, x, x,
                      10, win, nperseg=256)  # because nperseg != win.shape[-1]
        win_err = signal.get_window('hann', 32)
        assert_raises(ValueError, csd, x, x,
              10, win_err, nperseg=None)  # because win longer than signal

    def test_empty_input(self):
        f, p = csd([],np.zeros(10))
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))

        f, p = csd(np.zeros(10),[])
        assert_array_equal(f.shape, (0,))
        assert_array_equal(p.shape, (0,))

        for shape in [(0,), (3,0), (0,5,2)]:
            f, p = csd(np.empty(shape), np.empty(shape))
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

        f, p = csd(np.ones(10), np.empty((5,0)))
        assert_array_equal(f.shape, (5,0))
        assert_array_equal(p.shape, (5,0))

        f, p = csd(np.empty((5,0)), np.ones(10))
        assert_array_equal(f.shape, (5,0))
        assert_array_equal(p.shape, (5,0))

    def test_empty_input_other_axis(self):
        for shape in [(3,0), (0,5,2)]:
            f, p = csd(np.empty(shape), np.empty(shape), axis=1)
            assert_array_equal(f.shape, shape)
            assert_array_equal(p.shape, shape)

        f, p = csd(np.empty((10,10,3)), np.zeros((10,0,1)), axis=1)
        assert_array_equal(f.shape, (10,0,3))
        assert_array_equal(p.shape, (10,0,3))

        f, p = csd(np.empty((10,0,1)), np.zeros((10,10,3)), axis=1)
        assert_array_equal(f.shape, (10,0,3))
        assert_array_equal(p.shape, (10,0,3))

    def test_short_data(self):
        x = np.zeros(8)
        x[0] = 1

        #for string-like window, input signal length < nperseg value gives
        #UserWarning, sets nperseg to x.shape[-1]
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "nperseg = 256 is greater than input length  = 8, using nperseg = 8")
            f, p = csd(x, x, window='hann')  # default nperseg
            f1, p1 = csd(x, x, window='hann', nperseg=256)  # user-specified nperseg
        f2, p2 = csd(x, x, nperseg=8)  # valid nperseg, doesn't give warning
        assert_allclose(f, f2)
        assert_allclose(p, p2)
        assert_allclose(f1, f2)
        assert_allclose(p1, p2)

    def test_window_long_or_nd(self):
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1,
                      np.array([1,1,1,1,1]))
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1,
                      np.arange(6).reshape((2,3)))

    def test_nondefault_noverlap(self):
        x = np.zeros(64)
        x[::8] = 1
        f, p = csd(x, x, nperseg=16, noverlap=4)
        q = np.array([0, 1./12., 1./3., 1./5., 1./3., 1./5., 1./3., 1./5.,
                      1./6.])
        assert_allclose(p, q, atol=1e-12)

    def test_bad_noverlap(self):
        assert_raises(ValueError, csd, np.zeros(4), np.ones(4), 1, 'hann',
                      2, 7)

    def test_nfft_too_short(self):
        assert_raises(ValueError, csd, np.ones(12), np.zeros(12), nfft=3,
                      nperseg=4)

    def test_real_onesided_even_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8)
        assert_allclose(f, np.linspace(0, 0.5, 5))
        q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222,
                      0.11111111], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_real_onesided_odd_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=9)
        assert_allclose(f, np.arange(5.0)/9.0)
        q = np.array([0.12477458, 0.23430935, 0.17072113, 0.17072116,
                      0.17072113], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_real_twosided_32(self):
        x = np.zeros(16, 'f')
        x[0] = 1
        x[8] = 1
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.08333333, 0.07638889, 0.11111111,
                      0.11111111, 0.11111111, 0.11111111, 0.11111111,
                      0.07638889], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype)

    def test_complex_32(self):
        x = np.zeros(16, 'F')
        x[0] = 1.0 + 2.0j
        x[8] = 1.0 + 2.0j
        f, p = csd(x, x, nperseg=8, return_onesided=False)
        assert_allclose(f, fftfreq(8, 1.0))
        q = np.array([0.41666666, 0.38194442, 0.55555552, 0.55555552,
                      0.55555558, 0.55555552, 0.55555552, 0.38194442], 'f')
        assert_allclose(p, q, atol=1e-7, rtol=1e-7)
        assert_(p.dtype == q.dtype,
                'dtype mismatch, %s, %s' % (p.dtype, q.dtype))

    def test_padded_freqs(self):
        x = np.zeros(12)
        y = np.ones(12)

        nfft = 24
        f = fftfreq(nfft, 1.0)[:nfft//2+1]
        f[-1] *= -1
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

        nfft = 25
        f = fftfreq(nfft, 1.0)[:(nfft + 1)//2]
        fodd, _ = csd(x, y, nperseg=5, nfft=nfft)
        feven, _ = csd(x, y, nperseg=6, nfft=nfft)
        assert_allclose(f, fodd)
        assert_allclose(f, feven)

class TestCoherence:
    def test_identical_input(self):
        x = np.random.randn(20)
        y = np.copy(x)  # So `y is x` -> False

        f = np.linspace(0, 0.5, 6)
        C = np.ones(6)
        f1, C1 = coherence(x, y, nperseg=10)

        assert_allclose(f, f1)
        assert_allclose(C, C1)

    def test_phase_shifted_input(self):
        x = np.random.randn(20)
        y = -x

        f = np.linspace(0, 0.5, 6)
        C = np.ones(6)
        f1, C1 = coherence(x, y, nperseg=10)

        assert_allclose(f, f1)
        assert_allclose(C, C1)


class TestSpectrogram:
    def test_average_all_segments(self):
        x = np.random.randn(1024)

        fs = 1.0
        window = ('tukey', 0.25)
        nperseg = 16
        noverlap = 2

        f, _, P = spectrogram(x, fs, window, nperseg, noverlap)
        fw, Pw = welch(x, fs, window, nperseg, noverlap)
        assert_allclose(f, fw)
        assert_allclose(np.mean(P, axis=-1), Pw)

    def test_window_external(self):
        x = np.random.randn(1024)

        fs = 1.0
        window = ('tukey', 0.25)
        nperseg = 16
        noverlap = 2
        f, _, P = spectrogram(x, fs, window, nperseg, noverlap)

        win = signal.get_window(('tukey', 0.25), 16)
        fe, _, Pe = spectrogram(x, fs, win, nperseg=None, noverlap=2)
        assert_array_equal(fe.shape, (9,))  # because win length used as nperseg
        assert_array_equal(Pe.shape, (9,73))
        assert_raises(ValueError, spectrogram, x,
                      fs, win, nperseg=8)  # because nperseg != win.shape[-1]
        win_err = signal.get_window(('tukey', 0.25), 2048)
        assert_raises(ValueError, spectrogram, x,
                      fs, win_err, nperseg=None)  # win longer than signal

    def test_short_data(self):
        x = np.random.randn(1024)
        fs = 1.0

        #for string-like window, input signal length < nperseg value gives
        #UserWarning, sets nperseg to x.shape[-1]
        f, _, p = spectrogram(x, fs, window=('tukey',0.25))  # default nperseg
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "nperseg = 1025 is greater than input length  = 1024, using nperseg = 1024")
            f1, _, p1 = spectrogram(x, fs, window=('tukey',0.25),
                                    nperseg=1025)  # user-specified nperseg
        f2, _, p2 = spectrogram(x, fs, nperseg=256)  # to compare w/default
        f3, _, p3 = spectrogram(x, fs, nperseg=1024)  # compare w/user-spec'd
        assert_allclose(f, f2)
        assert_allclose(p, p2)
        assert_allclose(f1, f3)
        assert_allclose(p1, p3)

class TestLombscargle:
    def test_frequency(self):
        """Test if frequency location of peak corresponds to frequency of
        generated input signal.
        """

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram
        P = lombscargle(t, x, f)

        # Check if difference between found frequency maximum and input
        # frequency is less than accuracy
        delta = f[1] - f[0]
        assert_(w - f[np.argmax(P)] < (delta/2.))

    def test_amplitude(self):
        # Test if height of peak in normalized Lomb-Scargle periodogram
        # corresponds to amplitude of the generated input signal.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram
        pgram = lombscargle(t, x, f)

        # Normalize
        pgram = np.sqrt(4 * pgram / t.shape[0])

        # Check if difference between found frequency maximum and input
        # frequency is less than accuracy
        assert_approx_equal(np.max(pgram), ampl, significant=2)

    def test_precenter(self):
        # Test if precenter gives the same result as manually precentering.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select
        offset = 0.15  # Offset to be subtracted in pre-centering

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi) + offset

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram
        pgram = lombscargle(t, x, f, precenter=True)
        pgram2 = lombscargle(t, x - x.mean(), f, precenter=False)

        # check if centering worked
        assert_allclose(pgram, pgram2)

    def test_normalize(self):
        # Test normalize option of Lomb-Scarge.

        # Input parameters
        ampl = 2.
        w = 1.
        phi = 0.5 * np.pi
        nin = 100
        nout = 1000
        p = 0.7  # Fraction of points to select

        # Randomly select a fraction of an array with timesteps
        np.random.seed(2353425)
        r = np.random.rand(nin)
        t = np.linspace(0.01*np.pi, 10.*np.pi, nin)[r >= p]

        # Plot a sine wave for the selected times
        x = ampl * np.sin(w*t + phi)

        # Define the array of frequencies for which to compute the periodogram
        f = np.linspace(0.01, 10., nout)

        # Calculate Lomb-Scargle periodogram
        pgram = lombscargle(t, x, f)
        pgram2 = lombscargle(t, x, f, normalize=True)

        # check if normalization works as expected
        assert_allclose(pgram * 2 / np.dot(x, x), pgram2)
        assert_approx_equal(np.max(pgram2), 1.0, significant=2)

    def test_wrong_shape(self):
        t = np.linspace(0, 1, 1)
        x = np.linspace(0, 1, 2)
        f = np.linspace(0, 1, 3)
        assert_raises(ValueError, lombscargle, t, x, f)

    def test_zero_division(self):
        t = np.zeros(1)
        x = np.zeros(1)
        f = np.zeros(1)
        assert_raises(ZeroDivisionError, lombscargle, t, x, f)

    def test_lombscargle_atan_vs_atan2(self):
        # https://github.com/scipy/scipy/issues/3787
        # This raised a ZeroDivisionError.
        t = np.linspace(0, 10, 1000, endpoint=False)
        x = np.sin(4*t)
        f = np.linspace(0, 50, 500, endpoint=False) + 0.1
        lombscargle(t, x, f*2*np.pi)


class TestSTFT:
    def test_input_validation(self):
        assert_raises(ValueError, check_COLA, 'hann', -10, 0)
        assert_raises(ValueError, check_COLA, 'hann', 10, 20)
        assert_raises(ValueError, check_COLA, np.ones((2,2)), 10, 0)
        assert_raises(ValueError, check_COLA, np.ones(20), 10, 0)

        assert_raises(ValueError, check_NOLA, 'hann', -10, 0)
        assert_raises(ValueError, check_NOLA, 'hann', 10, 20)
        assert_raises(ValueError, check_NOLA, np.ones((2,2)), 10, 0)
        assert_raises(ValueError, check_NOLA, np.ones(20), 10, 0)
        assert_raises(ValueError, check_NOLA, 'hann', 64, -32)

        x = np.zeros(1024)
        z = np.array(stft(x), dtype=object)

        assert_raises(ValueError, stft, x, window=np.ones((2,2)))
        assert_raises(ValueError, stft, x, window=np.ones(10), nperseg=256)
        assert_raises(ValueError, stft, x, nperseg=-256)
        assert_raises(ValueError, stft, x, nperseg=256, noverlap=1024)
        assert_raises(ValueError, stft, x, nperseg=256, nfft=8)

        assert_raises(ValueError, istft, x)  # Not 2d
        assert_raises(ValueError, istft, z, window=np.ones((2,2)))
        assert_raises(ValueError, istft, z, window=np.ones(10), nperseg=256)
        assert_raises(ValueError, istft, z, nperseg=-256)
        assert_raises(ValueError, istft, z, nperseg=256, noverlap=1024)
        assert_raises(ValueError, istft, z, nperseg=256, nfft=8)
        assert_raises(ValueError, istft, z, nperseg=256, noverlap=0,
                      window='hann')  # Doesn't meet COLA
        assert_raises(ValueError, istft, z, time_axis=0, freq_axis=0)

        assert_raises(ValueError, _spectral_helper, x, x, mode='foo')
        assert_raises(ValueError, _spectral_helper, x[:512], x[512:],
                      mode='stft')
        assert_raises(ValueError, _spectral_helper, x, x, boundary='foo')

    def test_check_COLA(self):
        settings = [
                    ('boxcar', 10, 0),
                    ('boxcar', 10, 9),
                    ('bartlett', 51, 26),
                    ('hann', 256, 128),
                    ('hann', 256, 192),
                    ('blackman', 300, 200),
                    (('tukey', 0.5), 256, 64),
                    ('hann', 256, 255),
                    ]

        for setting in settings:
            msg = '{0}, {1}, {2}'.format(*setting)
            assert_equal(True, check_COLA(*setting), err_msg=msg)

    def test_check_NOLA(self):
        settings_pass = [
                    ('boxcar', 10, 0),
                    ('boxcar', 10, 9),
                    ('boxcar', 10, 7),
                    ('bartlett', 51, 26),
                    ('bartlett', 51, 10),
                    ('hann', 256, 128),
                    ('hann', 256, 192),
                    ('hann', 256, 37),
                    ('blackman', 300, 200),
                    ('blackman', 300, 123),
                    (('tukey', 0.5), 256, 64),
                    (('tukey', 0.5), 256, 38),
                    ('hann', 256, 255),
                    ('hann', 256, 39),
                    ]
        for setting in settings_pass:
            msg = '{0}, {1}, {2}'.format(*setting)
            assert_equal(True, check_NOLA(*setting), err_msg=msg)

        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings_fail = [
                    (w_fail, len(w_fail), len(w_fail) // 2),
                    ('hann', 64, 0),
        ]
        for setting in settings_fail:
            msg = '{0}, {1}, {2}'.format(*setting)
            assert_equal(False, check_NOLA(*setting), err_msg=msg)

    def test_average_all_segments(self):
        np.random.seed(1234)
        x = np.random.randn(1024)

        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8

        # Compare twosided, because onesided welch doubles non-DC terms to
        # account for power at negative frequencies. stft doesn't do this,
        # because it breaks invertibility.
        f, _, Z = stft(x, fs, window, nperseg, noverlap, padded=False,
                       return_onesided=False, boundary=None)
        fw, Pw = welch(x, fs, window, nperseg, noverlap, return_onesided=False,
                       scaling='spectrum', detrend=False)

        assert_allclose(f, fw)
        assert_allclose(np.mean(np.abs(Z)**2, axis=-1), Pw)

    def test_permute_axes(self):
        np.random.seed(1234)
        x = np.random.randn(1024)

        fs = 1.0
        window = 'hann'
        nperseg = 16
        noverlap = 8

        f1, t1, Z1 = stft(x, fs, window, nperseg, noverlap)
        f2, t2, Z2 = stft(x.reshape((-1, 1, 1)), fs, window, nperseg, noverlap,
                          axis=0)

        t3, x1 = istft(Z1, fs, window, nperseg, noverlap)
        t4, x2 = istft(Z2.T, fs, window, nperseg, noverlap, time_axis=0,
                       freq_axis=-1)

        assert_allclose(f1, f2)
        assert_allclose(t1, t2)
        assert_allclose(t3, t4)
        assert_allclose(Z1, Z2[:, 0, 0, :])
        assert_allclose(x1, x2[:, 0, 0])

    def test_roundtrip_real(self):
        np.random.seed(1234)

        settings = [
                    ('boxcar', 100, 10, 0),           # Test no overlap
                    ('boxcar', 100, 10, 9),           # Test high overlap
                    ('bartlett', 101, 51, 26),        # Test odd nperseg
                    ('hann', 1024, 256, 128),         # Test defaults
                    (('tukey', 0.5), 1152, 256, 64),  # Test Tukey
                    ('hann', 1024, 256, 255),         # Test overlapped hann
                    ]

        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False)

            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window)

            msg = '{0}, {1}'.format(window, noverlap)
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)

    def test_roundtrip_not_nola(self):
        np.random.seed(1234)

        w_fail = np.ones(16)
        w_fail[::2] = 0
        settings = [
                    (w_fail, 256, len(w_fail), len(w_fail) // 2),
                    ('hann', 256, 64, 0),
        ]

        for window, N, nperseg, noverlap in settings:
            msg = '{0}, {1}, {2}, {3}'.format(window, N, nperseg, noverlap)
            assert not check_NOLA(window, nperseg, noverlap), msg

            t = np.arange(N)
            x = 10 * np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True,
                            boundary='zeros')
            with pytest.warns(UserWarning, match='NOLA'):
                tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                               window=window, boundary=True)

            assert np.allclose(t, tr[:len(t)]), msg
            assert not np.allclose(x, xr[:len(x)]), msg

    def test_roundtrip_nola_not_cola(self):
        np.random.seed(1234)

        settings = [
                    ('boxcar', 100, 10, 3),           # NOLA True, COLA False
                    ('bartlett', 101, 51, 37),        # NOLA True, COLA False
                    ('hann', 1024, 256, 127),         # NOLA True, COLA False
                    (('tukey', 0.5), 1152, 256, 14),  # NOLA True, COLA False
                    ('hann', 1024, 256, 5),           # NOLA True, COLA False
                    ]

        for window, N, nperseg, noverlap in settings:
            msg = '{0}, {1}, {2}'.format(window, nperseg, noverlap)
            assert check_NOLA(window, nperseg, noverlap), msg
            assert not check_COLA(window, nperseg, noverlap), msg

            t = np.arange(N)
            x = 10 * np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True,
                            boundary='zeros')

            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window, boundary=True)

            msg = '{0}, {1}'.format(window, noverlap)
            assert_allclose(t, tr[:len(t)], err_msg=msg)
            assert_allclose(x, xr[:len(x)], err_msg=msg)

    def test_roundtrip_float32(self):
        np.random.seed(1234)

        settings = [('hann', 1024, 256, 128)]

        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)
            x = x.astype(np.float32)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False)

            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window)

            msg = '{0}, {1}'.format(window, noverlap)
            assert_allclose(t, t, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg, rtol=1e-4, atol=1e-5)
            assert_(x.dtype == xr.dtype)

    def test_roundtrip_complex(self):
        np.random.seed(1234)

        settings = [
                    ('boxcar', 100, 10, 0),           # Test no overlap
                    ('boxcar', 100, 10, 9),           # Test high overlap
                    ('bartlett', 101, 51, 26),        # Test odd nperseg
                    ('hann', 1024, 256, 128),         # Test defaults
                    (('tukey', 0.5), 1152, 256, 64),  # Test Tukey
                    ('hann', 1024, 256, 255),         # Test overlapped hann
                    ]

        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size) + 10j*np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False,
                            return_onesided=False)

            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                           window=window, input_onesided=False)

            msg = '{0}, {1}, {2}'.format(window, nperseg, noverlap)
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)

        # Check that asking for onesided switches to twosided
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "Input data is complex, switching to return_onesided=False")
            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=False,
                            return_onesided=True)

        tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap,
                       window=window, input_onesided=False)

        msg = '{0}, {1}, {2}'.format(window, nperseg, noverlap)
        assert_allclose(t, tr, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg)

    def test_roundtrip_boundary_extension(self):
        np.random.seed(1234)

        # Test against boxcar, since window is all ones, and thus can be fully
        # recovered with no boundary extension

        settings = [
                    ('boxcar', 100, 10, 0),           # Test no overlap
                    ('boxcar', 100, 10, 9),           # Test high overlap
                    ]

        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                           window=window, detrend=None, padded=True,
                           boundary=None)

            _, xr = istft(zz, noverlap=noverlap, window=window, boundary=False)

            for boundary in ['even', 'odd', 'constant', 'zeros']:
                _, _, zz_ext = stft(x, nperseg=nperseg, noverlap=noverlap,
                                window=window, detrend=None, padded=True,
                                boundary=boundary)

                _, xr_ext = istft(zz_ext, noverlap=noverlap, window=window,
                                boundary=True)

                msg = '{0}, {1}, {2}'.format(window, noverlap, boundary)
                assert_allclose(x, xr, err_msg=msg)
                assert_allclose(x, xr_ext, err_msg=msg)

    def test_roundtrip_padded_signal(self):
        np.random.seed(1234)

        settings = [
                    ('boxcar', 101, 10, 0),
                    ('hann', 1000, 256, 128),
                    ]

        for window, N, nperseg, noverlap in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)

            _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap,
                            window=window, detrend=None, padded=True)

            tr, xr = istft(zz, noverlap=noverlap, window=window)

            msg = '{0}, {1}'.format(window, noverlap)
            # Account for possible zero-padding at the end
            assert_allclose(t, tr[:t.size], err_msg=msg)
            assert_allclose(x, xr[:x.size], err_msg=msg)

    def test_roundtrip_padded_FFT(self):
        np.random.seed(1234)

        settings = [
                    ('hann', 1024, 256, 128, 512),
                    ('hann', 1024, 256, 128, 501),
                    ('boxcar', 100, 10, 0, 33),
                    (('tukey', 0.5), 1152, 256, 64, 1024),
                    ]

        for window, N, nperseg, noverlap, nfft in settings:
            t = np.arange(N)
            x = 10*np.random.randn(t.size)
            xc = x*np.exp(1j*np.pi/4)

            # real signal
            _, _, z = stft(x, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, detrend=None, padded=True)

            # complex signal
            _, _, zc = stft(xc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, detrend=None, padded=True,
                            return_onesided=False)

            tr, xr = istft(z, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                           window=window)

            tr, xcr = istft(zc, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                            window=window, input_onesided=False)

            msg = '{0}, {1}'.format(window, noverlap)
            assert_allclose(t, tr, err_msg=msg)
            assert_allclose(x, xr, err_msg=msg)
            assert_allclose(xc, xcr, err_msg=msg)

    def test_axis_rolling(self):
        np.random.seed(1234)

        x_flat = np.random.randn(1024)
        _, _, z_flat = stft(x_flat)

        for a in range(3):
            newshape = [1,]*3
            newshape[a] = -1
            x = x_flat.reshape(newshape)

            _, _, z_plus = stft(x, axis=a)  # Positive axis index
            _, _, z_minus = stft(x, axis=a-x.ndim)  # Negative axis index

            assert_equal(z_flat, z_plus.squeeze(), err_msg=a)
            assert_equal(z_flat, z_minus.squeeze(), err_msg=a-x.ndim)

        # z_flat has shape [n_freq, n_time]

        # Test vs. transpose
        _, x_transpose_m = istft(z_flat.T, time_axis=-2, freq_axis=-1)
        _, x_transpose_p = istft(z_flat.T, time_axis=0, freq_axis=1)

        assert_allclose(x_flat, x_transpose_m, err_msg='istft transpose minus')
        assert_allclose(x_flat, x_transpose_p, err_msg='istft transpose plus')
