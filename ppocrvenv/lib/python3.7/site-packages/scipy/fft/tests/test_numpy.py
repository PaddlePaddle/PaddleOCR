import queue
import threading
import multiprocessing
import numpy as np
import pytest
from numpy.random import random
from numpy.testing import (
        assert_array_almost_equal, assert_array_equal, assert_allclose
        )
from pytest import raises as assert_raises
import scipy.fft as fft

def fft1(x):
    L = len(x)
    phase = -2j*np.pi*(np.arange(L)/float(L))
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x*np.exp(phase), axis=1)


class TestFFTShift:

    def test_fft_n(self):
        assert_raises(ValueError, fft.fft, [1, 2, 3], 0)


class TestFFT1D:

    def test_identity(self):
        maxlen = 512
        x = random(maxlen) + 1j*random(maxlen)
        xr = random(maxlen)
        for i in range(1,maxlen):
            assert_array_almost_equal(fft.ifft(fft.fft(x[0:i])), x[0:i],
                                      decimal=12)
            assert_array_almost_equal(fft.irfft(fft.rfft(xr[0:i]),i),
                                      xr[0:i], decimal=12)

    def test_fft(self):
        x = random(30) + 1j*random(30)
        expect = fft1(x)
        assert_array_almost_equal(expect, fft.fft(x))
        assert_array_almost_equal(expect, fft.fft(x, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30),
                                  fft.fft(x, norm="ortho"))
        assert_array_almost_equal(expect / 30, fft.fft(x, norm="forward"))

    def test_ifft(self):
        x = random(30) + 1j*random(30)
        assert_array_almost_equal(x, fft.ifft(fft.fft(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.ifft(fft.fft(x, norm=norm), norm=norm))

    def test_fft2(self):
        x = random((30, 20)) + 1j*random((30, 20))
        expect = fft.fft(fft.fft(x, axis=1), axis=0)
        assert_array_almost_equal(expect, fft.fft2(x))
        assert_array_almost_equal(expect, fft.fft2(x, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30 * 20),
                                  fft.fft2(x, norm="ortho"))
        assert_array_almost_equal(expect / (30 * 20),
                                  fft.fft2(x, norm="forward"))

    def test_ifft2(self):
        x = random((30, 20)) + 1j*random((30, 20))
        expect = fft.ifft(fft.ifft(x, axis=1), axis=0)
        assert_array_almost_equal(expect, fft.ifft2(x))
        assert_array_almost_equal(expect, fft.ifft2(x, norm="backward"))
        assert_array_almost_equal(expect * np.sqrt(30 * 20),
                                  fft.ifft2(x, norm="ortho"))
        assert_array_almost_equal(expect * (30 * 20),
                                  fft.ifft2(x, norm="forward"))

    def test_fftn(self):
        x = random((30, 20, 10)) + 1j*random((30, 20, 10))
        expect = fft.fft(fft.fft(fft.fft(x, axis=2), axis=1), axis=0)
        assert_array_almost_equal(expect, fft.fftn(x))
        assert_array_almost_equal(expect, fft.fftn(x, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30 * 20 * 10),
                                  fft.fftn(x, norm="ortho"))
        assert_array_almost_equal(expect / (30 * 20 * 10),
                                  fft.fftn(x, norm="forward"))

    def test_ifftn(self):
        x = random((30, 20, 10)) + 1j*random((30, 20, 10))
        expect = fft.ifft(fft.ifft(fft.ifft(x, axis=2), axis=1), axis=0)
        assert_array_almost_equal(expect, fft.ifftn(x))
        assert_array_almost_equal(expect, fft.ifftn(x, norm="backward"))
        assert_array_almost_equal(fft.ifftn(x) * np.sqrt(30 * 20 * 10),
                                  fft.ifftn(x, norm="ortho"))
        assert_array_almost_equal(expect * (30 * 20 * 10),
                                  fft.ifftn(x, norm="forward"))

    def test_rfft(self):
        x = random(29)
        for n in [x.size, 2*x.size]:
            for norm in [None, "backward", "ortho", "forward"]:
                assert_array_almost_equal(
                    fft.fft(x, n=n, norm=norm)[:(n//2 + 1)],
                    fft.rfft(x, n=n, norm=norm))
            assert_array_almost_equal(fft.rfft(x, n=n) / np.sqrt(n),
                                      fft.rfft(x, n=n, norm="ortho"))

    def test_irfft(self):
        x = random(30)
        assert_array_almost_equal(x, fft.irfft(fft.rfft(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.irfft(fft.rfft(x, norm=norm), norm=norm))

    def test_rfft2(self):
        x = random((30, 20))
        expect = fft.fft2(x)[:, :11]
        assert_array_almost_equal(expect, fft.rfft2(x))
        assert_array_almost_equal(expect, fft.rfft2(x, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30 * 20),
                                  fft.rfft2(x, norm="ortho"))
        assert_array_almost_equal(expect / (30 * 20),
                                  fft.rfft2(x, norm="forward"))

    def test_irfft2(self):
        x = random((30, 20))
        assert_array_almost_equal(x, fft.irfft2(fft.rfft2(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.irfft2(fft.rfft2(x, norm=norm), norm=norm))

    def test_rfftn(self):
        x = random((30, 20, 10))
        expect = fft.fftn(x)[:, :, :6]
        assert_array_almost_equal(expect, fft.rfftn(x))
        assert_array_almost_equal(expect, fft.rfftn(x, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30 * 20 * 10),
                                  fft.rfftn(x, norm="ortho"))
        assert_array_almost_equal(expect / (30 * 20 * 10),
                                  fft.rfftn(x, norm="forward"))

    def test_irfftn(self):
        x = random((30, 20, 10))
        assert_array_almost_equal(x, fft.irfftn(fft.rfftn(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.irfftn(fft.rfftn(x, norm=norm), norm=norm))

    def test_hfft(self):
        x = random(14) + 1j*random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        expect = fft.fft(x)
        assert_array_almost_equal(expect, fft.hfft(x_herm))
        assert_array_almost_equal(expect, fft.hfft(x_herm, norm="backward"))
        assert_array_almost_equal(expect / np.sqrt(30),
                                  fft.hfft(x_herm, norm="ortho"))
        assert_array_almost_equal(expect / 30,
                                  fft.hfft(x_herm, norm="forward"))

    def test_ihfft(self):
        x = random(14) + 1j*random(14)
        x_herm = np.concatenate((random(1), x, random(1)))
        x = np.concatenate((x_herm, x[::-1].conj()))
        assert_array_almost_equal(x_herm, fft.ihfft(fft.hfft(x_herm)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x_herm, fft.ihfft(fft.hfft(x_herm, norm=norm), norm=norm))

    def test_hfft2(self):
        x = random((30, 20))
        assert_array_almost_equal(x, fft.hfft2(fft.ihfft2(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.hfft2(fft.ihfft2(x, norm=norm), norm=norm))

    def test_ihfft2(self):
        x = random((30, 20))
        expect = fft.ifft2(x)[:, :11]
        assert_array_almost_equal(expect, fft.ihfft2(x))
        assert_array_almost_equal(expect, fft.ihfft2(x, norm="backward"))
        assert_array_almost_equal(expect * np.sqrt(30 * 20),
                                  fft.ihfft2(x, norm="ortho"))
        assert_array_almost_equal(expect * (30 * 20),
                                  fft.ihfft2(x, norm="forward"))

    def test_hfftn(self):
        x = random((30, 20, 10))
        assert_array_almost_equal(x, fft.hfftn(fft.ihfftn(x)))
        for norm in ["backward", "ortho", "forward"]:
            assert_array_almost_equal(
                x, fft.hfftn(fft.ihfftn(x, norm=norm), norm=norm))

    def test_ihfftn(self):
        x = random((30, 20, 10))
        expect = fft.ifftn(x)[:, :, :6]
        assert_array_almost_equal(expect, fft.ihfftn(x))
        assert_array_almost_equal(expect, fft.ihfftn(x, norm="backward"))
        assert_array_almost_equal(expect * np.sqrt(30 * 20 * 10),
                                  fft.ihfftn(x, norm="ortho"))
        assert_array_almost_equal(expect * (30 * 20 * 10),
                                  fft.ihfftn(x, norm="forward"))

    @pytest.mark.parametrize("op", [fft.fftn, fft.ifftn,
                                    fft.rfftn, fft.irfftn,
                                    fft.hfftn, fft.ihfftn])
    def test_axes(self, op):
        x = random((30, 20, 10))
        axes = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for a in axes:
            op_tr = op(np.transpose(x, a))
            tr_op = np.transpose(op(x, axes=a), a)
            assert_array_almost_equal(op_tr, tr_op)

    @pytest.mark.parametrize("op", [fft.fft2, fft.ifft2,
                                    fft.rfft2, fft.irfft2,
                                    fft.hfft2, fft.ihfft2,
                                    fft.fftn, fft.ifftn,
                                    fft.rfftn, fft.irfftn,
                                    fft.hfftn, fft.ihfftn])
    def test_axes_subset_with_shape(self, op):
        x = random((16, 8, 4))
        axes = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        for a in axes:
            # different shape on the first two axes
            shape = tuple([2*x.shape[ax] if ax in a[:2] else x.shape[ax]
                           for ax in range(x.ndim)])
            # transform only the first two axes
            op_tr = op(np.transpose(x, a), s=shape[:2], axes=(0, 1))
            tr_op = np.transpose(op(x, s=shape[:2], axes=a[:2]), a)
            assert_array_almost_equal(op_tr, tr_op)

    def test_all_1d_norm_preserving(self):
        # verify that round-trip transforms are norm-preserving
        x = random(30)
        x_norm = np.linalg.norm(x)
        n = x.size * 2
        func_pairs = [(fft.fft, fft.ifft),
                      (fft.rfft, fft.irfft),
                      # hfft: order so the first function takes x.size samples
                      #       (necessary for comparison to x_norm above)
                      (fft.ihfft, fft.hfft),
                      ]
        for forw, back in func_pairs:
            for n in [x.size, 2*x.size]:
                for norm in ['backward', 'ortho', 'forward']:
                    tmp = forw(x, n=n, norm=norm)
                    tmp = back(tmp, n=n, norm=norm)
                    assert_array_almost_equal(x_norm,
                                              np.linalg.norm(tmp))

    @pytest.mark.parametrize("dtype", [np.half, np.single, np.double,
                                       np.longdouble])
    def test_dtypes(self, dtype):
        # make sure that all input precisions are accepted
        x = random(30).astype(dtype)
        assert_array_almost_equal(fft.ifft(fft.fft(x)), x)
        assert_array_almost_equal(fft.irfft(fft.rfft(x)), x)
        assert_array_almost_equal(fft.hfft(fft.ihfft(x), len(x)), x)


@pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float64, np.longfloat,
         np.complex64, np.complex128, np.longcomplex])
@pytest.mark.parametrize("order", ["F", 'non-contiguous'])
@pytest.mark.parametrize(
        "fft",
        [fft.fft, fft.fft2, fft.fftn,
         fft.ifft, fft.ifft2, fft.ifftn])
def test_fft_with_order(dtype, order, fft):
    # Check that FFT/IFFT produces identical results for C, Fortran and
    # non contiguous arrays
    rng = np.random.RandomState(42)
    X = rng.rand(8, 7, 13).astype(dtype, copy=False)
    if order == 'F':
        Y = np.asfortranarray(X)
    else:
        # Make a non contiguous array
        Y = X[::-1]
        X = np.ascontiguousarray(X[::-1])

    if fft.__name__.endswith('fft'):
        for axis in range(3):
            X_res = fft(X, axis=axis)
            Y_res = fft(Y, axis=axis)
            assert_array_almost_equal(X_res, Y_res)
    elif fft.__name__.endswith(('fft2', 'fftn')):
        axes = [(0, 1), (1, 2), (0, 2)]
        if fft.__name__.endswith('fftn'):
            axes.extend([(0,), (1,), (2,), None])
        for ax in axes:
            X_res = fft(X, axes=ax)
            Y_res = fft(Y, axes=ax)
            assert_array_almost_equal(X_res, Y_res)
    else:
        raise ValueError


class TestFFTThreadSafe:
    threads = 16
    input_shape = (800, 200)

    def _test_mtsame(self, func, *args):
        def worker(args, q):
            q.put(func(*args))

        q = queue.Queue()
        expected = func(*args)

        # Spin off a bunch of threads to call the same function simultaneously
        t = [threading.Thread(target=worker, args=(args, q))
             for i in range(self.threads)]
        [x.start() for x in t]

        [x.join() for x in t]
        # Make sure all threads returned the correct value
        for i in range(self.threads):
            assert_array_equal(q.get(timeout=5), expected,
                'Function returned wrong value in multithreaded context')

    def test_fft(self):
        a = np.ones(self.input_shape, dtype=np.complex128)
        self._test_mtsame(fft.fft, a)

    def test_ifft(self):
        a = np.full(self.input_shape, 1+0j)
        self._test_mtsame(fft.ifft, a)

    def test_rfft(self):
        a = np.ones(self.input_shape)
        self._test_mtsame(fft.rfft, a)

    def test_irfft(self):
        a = np.full(self.input_shape, 1+0j)
        self._test_mtsame(fft.irfft, a)

    def test_hfft(self):
        a = np.ones(self.input_shape, np.complex64)
        self._test_mtsame(fft.hfft, a)

    def test_ihfft(self):
        a = np.ones(self.input_shape)
        self._test_mtsame(fft.ihfft, a)


@pytest.mark.parametrize("func", [fft.fft, fft.ifft, fft.rfft, fft.irfft])
def test_multiprocess(func):
    # Test that fft still works after fork (gh-10422)

    with multiprocessing.Pool(2) as p:
        res = p.map(func, [np.ones(100) for _ in range(4)])

    expect = func(np.ones(100))
    for x in res:
        assert_allclose(x, expect)


class TestIRFFTN:

    def test_not_last_axis_success(self):
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j*ai

        axes = (-2,)

        # Should not raise error
        fft.irfftn(a, axes=axes)
