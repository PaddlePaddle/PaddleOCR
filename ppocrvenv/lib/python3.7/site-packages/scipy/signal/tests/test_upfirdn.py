# Code adapted from "upfirdn" python library with permission:
#
# Copyright (c) 2009, Motorola, Inc
#
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of Motorola nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from itertools import product

from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest

from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test


def upfirdn_naive(x, h, up=1, down=1):
    """Naive upfirdn processing in Python.

    Note: arg order (x, h) differs to facilitate apply_along_axis use.
    """
    h = np.asarray(h)
    out = np.zeros(len(x) * up, x.dtype)
    out[::up] = x
    out = np.convolve(h, out)[::down][:_output_len(len(h), len(x), up, down)]
    return out


class UpFIRDnCase:
    """Test _UpFIRDn object"""
    def __init__(self, up, down, h, x_dtype):
        self.up = up
        self.down = down
        self.h = np.atleast_1d(h)
        self.x_dtype = x_dtype
        self.rng = np.random.RandomState(17)

    def __call__(self):
        # tiny signal
        self.scrub(np.ones(1, self.x_dtype))
        # ones
        self.scrub(np.ones(10, self.x_dtype))  # ones
        # randn
        x = self.rng.randn(10).astype(self.x_dtype)
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(10)
        self.scrub(x)
        # ramp
        self.scrub(np.arange(10).astype(self.x_dtype))
        # 3D, random
        size = (2, 3, 5)
        x = self.rng.randn(*size).astype(self.x_dtype)
        if self.x_dtype in (np.complex64, np.complex128):
            x += 1j * self.rng.randn(*size)
        for axis in range(len(size)):
            self.scrub(x, axis=axis)
        x = x[:, ::2, 1::3].T
        for axis in range(len(size)):
            self.scrub(x, axis=axis)

    def scrub(self, x, axis=-1):
        yr = np.apply_along_axis(upfirdn_naive, axis, x,
                                 self.h, self.up, self.down)
        want_len = _output_len(len(self.h), x.shape[axis], self.up, self.down)
        assert yr.shape[axis] == want_len
        y = upfirdn(self.h, x, self.up, self.down, axis=axis)
        assert y.shape[axis] == want_len
        assert y.shape == yr.shape
        dtypes = (self.h.dtype, x.dtype)
        if all(d == np.complex64 for d in dtypes):
            assert_equal(y.dtype, np.complex64)
        elif np.complex64 in dtypes and np.float32 in dtypes:
            assert_equal(y.dtype, np.complex64)
        elif all(d == np.float32 for d in dtypes):
            assert_equal(y.dtype, np.float32)
        elif np.complex128 in dtypes or np.complex64 in dtypes:
            assert_equal(y.dtype, np.complex128)
        else:
            assert_equal(y.dtype, np.float64)
        assert_allclose(yr, y)


_UPFIRDN_TYPES = (int, np.float32, np.complex64, float, complex)


class TestUpfirdn:

    def test_valid_input(self):
        assert_raises(ValueError, upfirdn, [1], [1], 1, 0)  # up or down < 1
        assert_raises(ValueError, upfirdn, [], [1], 1, 1)  # h.ndim != 1
        assert_raises(ValueError, upfirdn, [[1]], [1], 1, 1)

    @pytest.mark.parametrize('len_h', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('len_x', [1, 2, 3, 4, 5])
    def test_singleton(self, len_h, len_x):
        # gh-9844: lengths producing expected outputs
        h = np.zeros(len_h)
        h[len_h // 2] = 1.  # make h a delta
        x = np.ones(len_x)
        y = upfirdn(h, x, 1, 1)
        want = np.pad(x, (len_h // 2, (len_h - 1) // 2), 'constant')
        assert_allclose(y, want)

    def test_shift_x(self):
        # gh-9844: shifted x can change values?
        y = upfirdn([1, 1], [1.], 1, 1)
        assert_allclose(y, [1, 1])  # was [0, 1] in the issue
        y = upfirdn([1, 1], [0., 1.], 1, 1)
        assert_allclose(y, [0, 1, 1])

    # A bunch of lengths/factors chosen because they exposed differences
    # between the "old way" and new way of computing length, and then
    # got `expected` from MATLAB
    @pytest.mark.parametrize('len_h, len_x, up, down, expected', [
        (2, 2, 5, 2, [1, 0, 0, 0]),
        (2, 3, 6, 3, [1, 0, 1, 0, 1]),
        (2, 4, 4, 3, [1, 0, 0, 0, 1]),
        (3, 2, 6, 2, [1, 0, 0, 1, 0]),
        (4, 11, 3, 5, [1, 0, 0, 1, 0, 0, 1]),
    ])
    def test_length_factors(self, len_h, len_x, up, down, expected):
        # gh-9844: weird factors
        h = np.zeros(len_h)
        h[0] = 1.
        x = np.ones(len_x)
        y = upfirdn(h, x, up, down)
        assert_allclose(y, expected)

    @pytest.mark.parametrize('down, want_len', [  # lengths from MATLAB
        (2, 5015),
        (11, 912),
        (79, 127),
    ])
    def test_vs_convolve(self, down, want_len):
        # Check that up=1.0 gives same answer as convolve + slicing
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            h = firwin(31, 1. / down, window='hamming')
            yl = upfirdn_naive(x, h, 1, down)
            y = upfirdn(h, x, up=1, down=down)
            assert y.shape == (want_len,)
            assert yl.shape[0] == y.shape[0]
            assert_allclose(yl, y, atol=1e-7, rtol=1e-7)

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h', (1., 1j))
    @pytest.mark.parametrize('up, down', [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_vs_naive_delta(self, x_dtype, h, up, down):
        UpFIRDnCase(up, down, h, x_dtype)()

    @pytest.mark.parametrize('x_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('h_dtype', _UPFIRDN_TYPES)
    @pytest.mark.parametrize('p_max, q_max',
                             list(product((10, 100), (10, 100))))
    def test_vs_naive(self, x_dtype, h_dtype, p_max, q_max):
        tests = self._random_factors(p_max, q_max, h_dtype, x_dtype)
        for test in tests:
            test()

    def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
        n_rep = 3
        longest_h = 25
        random_state = np.random.RandomState(17)
        tests = []

        for _ in range(n_rep):
            # Randomize the up/down factors somewhat
            p_add = q_max if p_max > q_max else 1
            q_add = p_max if q_max > p_max else 1
            p = random_state.randint(p_max) + p_add
            q = random_state.randint(q_max) + q_add

            # Generate random FIR coefficients
            len_h = random_state.randint(longest_h) + 1
            h = np.atleast_1d(random_state.randint(len_h))
            h = h.astype(h_dtype)
            if h_dtype == complex:
                h += 1j * random_state.randint(len_h)

            tests.append(UpFIRDnCase(p, q, h, x_dtype))

        return tests

    @pytest.mark.parametrize('mode', _upfirdn_modes)
    def test_extensions(self, mode):
        """Test vs. manually computed results for modes not in numpy's pad."""
        x = np.array([1, 2, 3, 1], dtype=float)
        npre, npost = 6, 6
        y = _pad_test(x, npre=npre, npost=npost, mode=mode)
        if mode == 'antisymmetric':
            y_expected = np.asarray(
                [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2])
        elif mode == 'antireflect':
            y_expected = np.asarray(
                [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1])
        elif mode == 'smooth':
            y_expected = np.asarray(
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11])
        elif mode == "line":
            lin_slope = (x[-1] - x[0]) / (len(x) - 1)
            left = x[0] + np.arange(-npre, 0, 1) * lin_slope
            right = x[-1] + np.arange(1, npost + 1) * lin_slope
            y_expected = np.concatenate((left, x, right))
        else:
            y_expected = np.pad(x, (npre, npost), mode=mode)
        assert_allclose(y, y_expected)

    @pytest.mark.parametrize(
        'size, h_len, mode, dtype',
        product(
            [8],
            [4, 5, 26],  # include cases with h_len > 2*size
            _upfirdn_modes,
            [np.float32, np.float64, np.complex64, np.complex128],
        )
    )
    def test_modes(self, size, h_len, mode, dtype):
        random_state = np.random.RandomState(5)
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        h = np.arange(1, 1 + h_len, dtype=x.real.dtype)

        y = upfirdn(h, x, up=1, down=1, mode=mode)
        # expected result: pad the input, filter with zero padding, then crop
        npad = h_len - 1
        if mode in ['antisymmetric', 'antireflect', 'smooth', 'line']:
            # use _pad_test test function for modes not supported by np.pad.
            xpad = _pad_test(x, npre=npad, npost=npad, mode=mode)
        else:
            xpad = np.pad(x, npad, mode=mode)
        ypad = upfirdn(h, xpad, up=1, down=1, mode='constant')
        y_expected = ypad[npad:-npad]

        atol = rtol = np.finfo(dtype).eps * 1e2
        assert_allclose(y, y_expected, atol=atol, rtol=rtol)
