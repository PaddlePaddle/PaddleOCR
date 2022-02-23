#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

from itertools import product
from functools import reduce
import operator
import numpy as np
from numpy.testing import (assert_allclose, assert_, assert_raises,
                           assert_equal)

import pywt


def test_traversing_tree_nd():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, dtype=np.float64)
    wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric')

    assert_(np.all(wp.data == x))
    assert_(wp.path == '')
    assert_(wp.level == 0)
    assert_(wp.maxlevel == 3)

    assert_allclose(wp['aa'].data, np.array([[3., 7., 11., 15.]] * 4),
                    rtol=1e-12)
    assert_allclose(wp['da'].data, np.zeros((4, 4)), rtol=1e-12, atol=1e-14)
    assert_allclose(wp['ad'].data, -np.ones((4, 4)), rtol=1e-12, atol=1e-14)
    assert_allclose(wp['dd'].data, np.zeros((4, 4)), rtol=1e-12, atol=1e-14)

    assert_allclose(wp['aa'*2].data, np.array([[10., 26.]] * 2), rtol=1e-12)
    # __getitem__ using a tuple acces instead
    assert_allclose(wp[('aa', 'aa')].data, np.array([[10., 26.]] * 2),
                    rtol=1e-12)

    assert_(wp['aa']['aa'].data is wp['aa'*2].data)
    assert_allclose(wp['aa'*3].data, np.array([[36.]]), rtol=1e-12)

    assert_raises(IndexError, lambda: wp['aa'*(wp.maxlevel+1)])
    assert_raises(ValueError, lambda: wp['f'])

    # getitem input must be a string or tuple of strings
    assert_raises(TypeError, wp.__getitem__, (5, 3))
    assert_raises(TypeError, wp.__getitem__, 5)


def test_accessing_node_attributes_nd():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, dtype=np.float64)
    wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric')

    assert_allclose(wp['aa'+'ad'].data, np.zeros((2, 2)) - 4, rtol=1e-12)
    assert_(wp['aa'+'ad'].path == 'aa'+'ad')
    assert_(wp['aa'+'ad'].node_name == 'ad')
    assert_(wp['aa'+'ad'].parent.path == 'aa')

    assert_allclose(wp['aa'+'ad'].parent.data,
                    np.array([[3., 7., 11., 15.]] * 4), rtol=1e-12)
    # can also index via a tuple instead of concatenated strings
    assert_(wp[('aa', 'ad')].level == 2)
    assert_(wp[('aa', 'ad')].maxlevel == 3)
    assert_(wp[('aa', 'ad')].mode == 'symmetric')

    # can access a node's path as either a single string or in tuple form
    node = wp[('ad', 'dd')]
    assert_(node.path == 'addd')
    assert_(node.path_tuple == ('ad', 'dd'))


def test_collecting_nodes_nd():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, dtype=np.float64)
    wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric')

    assert_(len(wp.get_level(0)) == 1)
    assert_(wp.get_level(0)[0].path == '')

    # First level
    assert_(len(wp.get_level(1)) == 4)
    assert_(
        [node.path for node in wp.get_level(1)] == ['aa', 'ad', 'da', 'dd'])

    # Second and third levels
    for lev in [2, 3]:
        assert_(len(wp.get_level(lev)) == (2**x.ndim)**lev)
        paths = [node.path for node in wp.get_level(lev)]
        expected_paths = [
            reduce(operator.add, p) for
            p in sorted(product(['aa', 'ad', 'da', 'dd'], repeat=lev))]
        assert_(paths == expected_paths)


def test_data_reconstruction_delete_nodes_nd():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, dtype=np.float64)
    wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric')

    # The user must supply either data or axes
    assert_raises(ValueError, pywt.WaveletPacketND, data=None, wavelet='db1',
                  axes=None)

    new_wp = pywt.WaveletPacketND(data=None, wavelet='db1', mode='symmetric',
                                  axes=range(x.ndim))

    new_wp['ad'+'da'] = wp['ad'+'da'].data
    new_wp['ad'*2] = wp['ad'+'da'].data
    new_wp['ad'+'dd'] = np.zeros((2, 2), dtype=np.float64)
    new_wp['aa'] = [[3.0, 7.0, 11.0, 15.0]] * 4
    new_wp['dd'] = np.zeros((4, 4), dtype=np.float64)
    new_wp['da'] = wp['da']       # all zeros

    assert_allclose(new_wp.reconstruct(update=False),
                    np.array([[1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5]] * 8),
                    rtol=1e-12)

    new_wp['ad'+'aa'] = wp['ad'+'aa'].data
    assert_allclose(new_wp.reconstruct(update=False), x, rtol=1e-12)

    del(new_wp['ad'+'aa'])
    # TypeError on accessing deleted node
    assert_raises(TypeError, lambda: new_wp['ad'+'aa'])

    new_wp['ad'+'aa'] = wp['ad'+'aa'].data
    assert_(new_wp.data is None)

    assert_allclose(new_wp.reconstruct(update=True), x, rtol=1e-12)
    assert_allclose(new_wp.data, x, rtol=1e-12)

    # TODO: decompose=True


def test_wavelet_packet_dtypes():
    shape = (16, 8, 8)
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        x = np.random.randn(*shape).astype(dtype)
        if np.iscomplexobj(x):
            x = x + 1j*np.random.randn(*shape).astype(x.real.dtype)
        wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric')
        # no unnecessary copy made
        assert_(wp.data is x)

        # full decomposition
        wp.get_level(wp.maxlevel)

        # reconstruction from coefficients should preserve dtype
        r = wp.reconstruct(False)
        assert_equal(r.dtype, x.dtype)
        assert_allclose(r, x, atol=1e-6, rtol=1e-6)


def test_wavelet_packet_axes():
    rstate = np.random.RandomState(0)
    shape = (32, 16, 8)
    x = rstate.standard_normal(shape)
    for axes in [(0, 1), 1, (-3, -2, -1), (0, 2), (1, )]:
        wp = pywt.WaveletPacketND(data=x, wavelet='db1', mode='symmetric',
                                  axes=axes)

        # partial decomposition
        nodes = wp.get_level(1)
        # size along the transformed axes has changed
        for ax2 in range(x.ndim):
            if ax2 in tuple(np.atleast_1d(axes) % x.ndim):
                nodes[0].data.shape[ax2] < x.shape[ax2]
            else:
                nodes[0].data.shape[ax2] == x.shape[ax2]

        # recontsruction from coefficients should preserve dtype
        r = wp.reconstruct(False)
        assert_equal(r.dtype, x.dtype)
        assert_allclose(r, x, atol=1e-12, rtol=1e-12)

    # must have non-duplicate axes
    assert_raises(ValueError, pywt.WaveletPacketND, data=x, wavelet='db1',
                  axes=(0, 0))
