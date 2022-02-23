# Created by Pearu Peterson, September 2002

__usage__ = """
Build fftpack:
  python setup_fftpack.py build
Run tests if scipy is installed:
  python -c 'import scipy;scipy.fftpack.test(<level>)'
Run tests if fftpack is not installed:
  python tests/test_pseudo_diffs.py [<level>]
"""

from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
                           ihilbert, shift, fftfreq, cs_diff, sc_diff,
                           ss_diff, cc_diff)

import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random


def direct_diff(x,k=1,period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2*pi
    w = fftfreq(n)*2j*pi/period*n
    if k < 0:
        w = 1 / w**k
        w[0] = 0.0
    else:
        w = w**k
    if n > 2000:
        w[250:n-250] = 0.0
    return ifft(w*fx).real


def direct_tilbert(x,h=1,period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2*pi
    w = fftfreq(n)*h*2*pi/period*n
    w[0] = 1
    w = 1j/tanh(w)
    w[0] = 0j
    return ifft(w*fx)


def direct_itilbert(x,h=1,period=None):
    fx = fft(x)
    n = len(fx)
    if period is None:
        period = 2*pi
    w = fftfreq(n)*h*2*pi/period*n
    w = -1j*tanh(w)
    return ifft(w*fx)


def direct_hilbert(x):
    fx = fft(x)
    n = len(fx)
    w = fftfreq(n)*n
    w = 1j*sign(w)
    return ifft(w*fx)


def direct_ihilbert(x):
    return -direct_hilbert(x)


def direct_shift(x,a,period=None):
    n = len(x)
    if period is None:
        k = fftfreq(n)*1j*n
    else:
        k = fftfreq(n)*2j*pi/period*n
    return ifft(fft(x)*exp(k*a)).real


class TestDiff:

    def test_definition(self):
        for n in [16,17,64,127,32]:
            x = arange(n)*2*pi/n
            assert_array_almost_equal(diff(sin(x)),direct_diff(sin(x)))
            assert_array_almost_equal(diff(sin(x),2),direct_diff(sin(x),2))
            assert_array_almost_equal(diff(sin(x),3),direct_diff(sin(x),3))
            assert_array_almost_equal(diff(sin(x),4),direct_diff(sin(x),4))
            assert_array_almost_equal(diff(sin(x),5),direct_diff(sin(x),5))
            assert_array_almost_equal(diff(sin(2*x),3),direct_diff(sin(2*x),3))
            assert_array_almost_equal(diff(sin(2*x),4),direct_diff(sin(2*x),4))
            assert_array_almost_equal(diff(cos(x)),direct_diff(cos(x)))
            assert_array_almost_equal(diff(cos(x),2),direct_diff(cos(x),2))
            assert_array_almost_equal(diff(cos(x),3),direct_diff(cos(x),3))
            assert_array_almost_equal(diff(cos(x),4),direct_diff(cos(x),4))
            assert_array_almost_equal(diff(cos(2*x)),direct_diff(cos(2*x)))
            assert_array_almost_equal(diff(sin(x*n/8)),direct_diff(sin(x*n/8)))
            assert_array_almost_equal(diff(cos(x*n/8)),direct_diff(cos(x*n/8)))
            for k in range(5):
                assert_array_almost_equal(diff(sin(4*x),k),direct_diff(sin(4*x),k))
                assert_array_almost_equal(diff(cos(4*x),k),direct_diff(cos(4*x),k))

    def test_period(self):
        for n in [17,64]:
            x = arange(n)/float(n)
            assert_array_almost_equal(diff(sin(2*pi*x),period=1),
                                      2*pi*cos(2*pi*x))
            assert_array_almost_equal(diff(sin(2*pi*x),3,period=1),
                                      -(2*pi)**3*cos(2*pi*x))

    def test_sin(self):
        for n in [32,64,77]:
            x = arange(n)*2*pi/n
            assert_array_almost_equal(diff(sin(x)),cos(x))
            assert_array_almost_equal(diff(cos(x)),-sin(x))
            assert_array_almost_equal(diff(sin(x),2),-sin(x))
            assert_array_almost_equal(diff(sin(x),4),sin(x))
            assert_array_almost_equal(diff(sin(4*x)),4*cos(4*x))
            assert_array_almost_equal(diff(sin(sin(x))),cos(x)*cos(sin(x)))

    def test_expr(self):
        for n in [64,77,100,128,256,512,1024,2048,4096,8192][:5]:
            x = arange(n)*2*pi/n
            f = sin(x)*cos(4*x)+exp(sin(3*x))
            df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
            ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
                 - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
            d1 = diff(f)
            assert_array_almost_equal(d1,df)
            assert_array_almost_equal(diff(df),ddf)
            assert_array_almost_equal(diff(f,2),ddf)
            assert_array_almost_equal(diff(ddf,-1),df)

    def test_expr_large(self):
        for n in [2048,4096]:
            x = arange(n)*2*pi/n
            f = sin(x)*cos(4*x)+exp(sin(3*x))
            df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
            ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
                 - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
            assert_array_almost_equal(diff(f),df)
            assert_array_almost_equal(diff(df),ddf)
            assert_array_almost_equal(diff(ddf,-1),df)
            assert_array_almost_equal(diff(f,2),ddf)

    def test_int(self):
        n = 64
        x = arange(n)*2*pi/n
        assert_array_almost_equal(diff(sin(x),-1),-cos(x))
        assert_array_almost_equal(diff(sin(x),-2),-sin(x))
        assert_array_almost_equal(diff(sin(x),-4),sin(x))
        assert_array_almost_equal(diff(2*cos(2*x),-1),sin(2*x))

    def test_random_even(self):
        for k in [0,2,4,6]:
            for n in [60,32,64,56,55]:
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                # zeroing Nyquist mode:
                f = diff(diff(f,1),-1)
                assert_almost_equal(sum(f,axis=0),0.0)
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                assert_array_almost_equal(diff(diff(f,-k),k),f)

    def test_random_odd(self):
        for k in [0,1,2,3,4,5,6]:
            for n in [33,65,55]:
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                assert_almost_equal(sum(f,axis=0),0.0)
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                assert_array_almost_equal(diff(diff(f,-k),k),f)

    def test_zero_nyquist(self):
        for k in [0,1,2,3,4,5,6]:
            for n in [32,33,64,56,55]:
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                # zeroing Nyquist mode:
                f = diff(diff(f,1),-1)
                assert_almost_equal(sum(f,axis=0),0.0)
                assert_array_almost_equal(diff(diff(f,k),-k),f)
                assert_array_almost_equal(diff(diff(f,-k),k),f)


class TestTilbert:

    def test_definition(self):
        for h in [0.1,0.5,1,5.5,10]:
            for n in [16,17,64,127]:
                x = arange(n)*2*pi/n
                y = tilbert(sin(x),h)
                y1 = direct_tilbert(sin(x),h)
                assert_array_almost_equal(y,y1)
                assert_array_almost_equal(tilbert(sin(x),h),
                                          direct_tilbert(sin(x),h))
                assert_array_almost_equal(tilbert(sin(2*x),h),
                                          direct_tilbert(sin(2*x),h))

    def test_random_even(self):
        for h in [0.1,0.5,1,5.5,10]:
            for n in [32,64,56]:
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                assert_almost_equal(sum(f,axis=0),0.0)
                assert_array_almost_equal(direct_tilbert(direct_itilbert(f,h),h),f)

    def test_random_odd(self):
        for h in [0.1,0.5,1,5.5,10]:
            for n in [33,65,55]:
                f = random((n,))
                af = sum(f,axis=0)/n
                f = f-af
                assert_almost_equal(sum(f,axis=0),0.0)
                assert_array_almost_equal(itilbert(tilbert(f,h),h),f)
                assert_array_almost_equal(tilbert(itilbert(f,h),h),f)


class TestITilbert:

    def test_definition(self):
        for h in [0.1,0.5,1,5.5,10]:
            for n in [16,17,64,127]:
                x = arange(n)*2*pi/n
                y = itilbert(sin(x),h)
                y1 = direct_itilbert(sin(x),h)
                assert_array_almost_equal(y,y1)
                assert_array_almost_equal(itilbert(sin(x),h),
                                          direct_itilbert(sin(x),h))
                assert_array_almost_equal(itilbert(sin(2*x),h),
                                          direct_itilbert(sin(2*x),h))


class TestHilbert:

    def test_definition(self):
        for n in [16,17,64,127]:
            x = arange(n)*2*pi/n
            y = hilbert(sin(x))
            y1 = direct_hilbert(sin(x))
            assert_array_almost_equal(y,y1)
            assert_array_almost_equal(hilbert(sin(2*x)),
                                      direct_hilbert(sin(2*x)))

    def test_tilbert_relation(self):
        for n in [16,17,64,127]:
            x = arange(n)*2*pi/n
            f = sin(x)+cos(2*x)*sin(x)
            y = hilbert(f)
            y1 = direct_hilbert(f)
            assert_array_almost_equal(y,y1)
            y2 = tilbert(f,h=10)
            assert_array_almost_equal(y,y2)

    def test_random_odd(self):
        for n in [33,65,55]:
            f = random((n,))
            af = sum(f,axis=0)/n
            f = f-af
            assert_almost_equal(sum(f,axis=0),0.0)
            assert_array_almost_equal(ihilbert(hilbert(f)),f)
            assert_array_almost_equal(hilbert(ihilbert(f)),f)

    def test_random_even(self):
        for n in [32,64,56]:
            f = random((n,))
            af = sum(f,axis=0)/n
            f = f-af
            # zeroing Nyquist mode:
            f = diff(diff(f,1),-1)
            assert_almost_equal(sum(f,axis=0),0.0)
            assert_array_almost_equal(direct_hilbert(direct_ihilbert(f)),f)
            assert_array_almost_equal(hilbert(ihilbert(f)),f)


class TestIHilbert:

    def test_definition(self):
        for n in [16,17,64,127]:
            x = arange(n)*2*pi/n
            y = ihilbert(sin(x))
            y1 = direct_ihilbert(sin(x))
            assert_array_almost_equal(y,y1)
            assert_array_almost_equal(ihilbert(sin(2*x)),
                                      direct_ihilbert(sin(2*x)))

    def test_itilbert_relation(self):
        for n in [16,17,64,127]:
            x = arange(n)*2*pi/n
            f = sin(x)+cos(2*x)*sin(x)
            y = ihilbert(f)
            y1 = direct_ihilbert(f)
            assert_array_almost_equal(y,y1)
            y2 = itilbert(f,h=10)
            assert_array_almost_equal(y,y2)


class TestShift:

    def test_definition(self):
        for n in [18,17,64,127,32,2048,256]:
            x = arange(n)*2*pi/n
            for a in [0.1,3]:
                assert_array_almost_equal(shift(sin(x),a),direct_shift(sin(x),a))
                assert_array_almost_equal(shift(sin(x),a),sin(x+a))
                assert_array_almost_equal(shift(cos(x),a),cos(x+a))
                assert_array_almost_equal(shift(cos(2*x)+sin(x),a),
                                          cos(2*(x+a))+sin(x+a))
                assert_array_almost_equal(shift(exp(sin(x)),a),exp(sin(x+a)))
            assert_array_almost_equal(shift(sin(x),2*pi),sin(x))
            assert_array_almost_equal(shift(sin(x),pi),-sin(x))
            assert_array_almost_equal(shift(sin(x),pi/2),cos(x))


class TestOverwrite:
    """Check input overwrite behavior """

    real_dtypes = (np.float32, np.float64)
    dtypes = real_dtypes + (np.complex64, np.complex128)

    def _check(self, x, routine, *args, **kwargs):
        x2 = x.copy()
        routine(x2, *args, **kwargs)
        sig = routine.__name__
        if args:
            sig += repr(args)
        if kwargs:
            sig += repr(kwargs)
        assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)

    def _check_1d(self, routine, dtype, shape, *args, **kwargs):
        np.random.seed(1234)
        if np.issubdtype(dtype, np.complexfloating):
            data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
        else:
            data = np.random.randn(*shape)
        data = data.astype(dtype)
        self._check(data, routine, *args, **kwargs)

    def test_diff(self):
        for dtype in self.dtypes:
            self._check_1d(diff, dtype, (16,))

    def test_tilbert(self):
        for dtype in self.dtypes:
            self._check_1d(tilbert, dtype, (16,), 1.6)

    def test_itilbert(self):
        for dtype in self.dtypes:
            self._check_1d(itilbert, dtype, (16,), 1.6)

    def test_hilbert(self):
        for dtype in self.dtypes:
            self._check_1d(hilbert, dtype, (16,))

    def test_cs_diff(self):
        for dtype in self.dtypes:
            self._check_1d(cs_diff, dtype, (16,), 1.0, 4.0)

    def test_sc_diff(self):
        for dtype in self.dtypes:
            self._check_1d(sc_diff, dtype, (16,), 1.0, 4.0)

    def test_ss_diff(self):
        for dtype in self.dtypes:
            self._check_1d(ss_diff, dtype, (16,), 1.0, 4.0)

    def test_cc_diff(self):
        for dtype in self.dtypes:
            self._check_1d(cc_diff, dtype, (16,), 1.0, 4.0)

    def test_shift(self):
        for dtype in self.dtypes:
            self._check_1d(shift, dtype, (16,), 1.0)
