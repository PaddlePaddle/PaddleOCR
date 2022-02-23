#!/usr/bin/env python

import os
import pickle

import numpy as np
from numpy.testing import (assert_allclose, assert_, assert_raises,
                           assert_equal)

import pywt


def test_wavelet_packet_structure():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    assert_(wp.data == [1, 2, 3, 4, 5, 6, 7, 8])
    assert_(wp.path == '')
    assert_(wp.level == 0)
    assert_(wp['ad'].maxlevel == 3)


def test_traversing_wp_tree():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    assert_(wp.maxlevel == 3)

    # First level
    assert_allclose(wp['a'].data, np.array([2.12132034356, 4.949747468306,
                                           7.778174593052, 10.606601717798]),
                    rtol=1e-12)

    # Second level
    assert_allclose(wp['aa'].data, np.array([5., 13.]), rtol=1e-12)

    # Third level
    assert_allclose(wp['aaa'].data, np.array([12.727922061358]), rtol=1e-12)


def test_acess_path():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    assert_(wp['a'].path == 'a')
    assert_(wp['aa'].path == 'aa')
    assert_(wp['aaa'].path == 'aaa')

    # Maximum level reached:
    assert_raises(IndexError, lambda: wp['aaaa'].path)

    # Wrong path
    assert_raises(ValueError, lambda: wp['ac'].path)


def test_access_node_attributes():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    assert_allclose(wp['ad'].data, np.array([-2., -2.]), rtol=1e-12)
    assert_(wp['ad'].path == 'ad')
    assert_(wp['ad'].node_name == 'd')
    assert_(wp['ad'].parent.path == 'a')
    assert_(wp['ad'].level == 2)
    assert_(wp['ad'].maxlevel == 3)
    assert_(wp['ad'].mode == 'symmetric')

    # tuple-based access is also supported
    node = wp[('a', 'd')]
    # can access a node's path as either a single string or in tuple form
    assert_(node.path == 'ad')
    assert_(node.path_tuple == ('a', 'd'))


def test_collecting_nodes():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    # All nodes in natural order
    assert_([node.path for node in wp.get_level(3, 'natural')] ==
            ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'])

    # and in frequency order.
    assert_([node.path for node in wp.get_level(3, 'freq')] ==
            ['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa'])

    assert_raises(ValueError, wp.get_level, 3, 'invalid_order')


def test_reconstructing_data():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')

    # Create another Wavelet Packet and feed it with some data.
    new_wp = pywt.WaveletPacket(data=None, wavelet='db1', mode='symmetric')
    new_wp['aa'] = wp['aa'].data
    new_wp['ad'] = [-2., -2.]

    # For convenience, :attr:`Node.data` gets automatically extracted
    # from the :class:`Node` object:
    new_wp['d'] = wp['d']

    # Reconstruct data from aa, ad, and d packets.
    assert_allclose(new_wp.reconstruct(update=False), x, rtol=1e-12)

    # The node's :attr:`~Node.data` will not be updated
    assert_(new_wp.data is None)

    # When `update` is True:
    assert_allclose(new_wp.reconstruct(update=True), x, rtol=1e-12)
    assert_allclose(new_wp.data, np.arange(1, 9), rtol=1e-12)

    assert_([n.path for n in new_wp.get_leaf_nodes(False)] ==
            ['aa', 'ad', 'd'])
    assert_([n.path for n in new_wp.get_leaf_nodes(True)] ==
            ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'])


def test_removing_nodes():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
    wp.get_level(2)

    dataleafs = [n.data for n in wp.get_leaf_nodes(False)]
    expected = np.array([[5., 13.], [-2, -2], [-1, -1], [0, 0]])

    for i in range(4):
        assert_allclose(dataleafs[i], expected[i, :], atol=1e-12)

    node = wp['ad']
    del(wp['ad'])
    dataleafs = [n.data for n in wp.get_leaf_nodes(False)]
    expected = np.array([[5., 13.], [-1, -1], [0, 0]])

    for i in range(3):
        assert_allclose(dataleafs[i], expected[i, :], atol=1e-12)

    wp.reconstruct()
    # The reconstruction is:
    assert_allclose(wp.reconstruct(),
                    np.array([2., 3., 2., 3., 6., 7., 6., 7.]), rtol=1e-12)

    # Restore the data
    wp['ad'].data = node.data

    dataleafs = [n.data for n in wp.get_leaf_nodes(False)]
    expected = np.array([[5., 13.], [-2, -2], [-1, -1], [0, 0]])

    for i in range(4):
        assert_allclose(dataleafs[i], expected[i, :], atol=1e-12)

    assert_allclose(wp.reconstruct(), np.arange(1, 9), rtol=1e-12)


def test_wavelet_packet_dtypes():
    rstate = np.random.RandomState(0)
    N = 32
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        x = rstate.randn(N).astype(dtype)
        if np.iscomplexobj(x):
            x = x + 1j*np.random.randn(N).astype(x.real.dtype)
        wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
        # no unnecessary copy made
        assert_(wp.data is x)

        # assiging to a node should not change supported dtypes
        wp['d'] = wp['d'].data
        assert_equal(wp['d'].data.dtype, x.dtype)

        # full decomposition
        wp.get_level(wp.maxlevel)

        # reconstruction from coefficients should preserve dtype
        r = wp.reconstruct(False)
        assert_equal(r.dtype, x.dtype)
        assert_allclose(r, x, atol=1e-5, rtol=1e-5)

    # first element of the tuple is the input dtype
    # second element of the tuple is the transform dtype
    dtype_pairs = [(np.uint8, np.float64),
                   (np.intp, np.float64), ]
    if hasattr(np, "complex256"):
        dtype_pairs += [(np.complex256, np.complex128), ]
    if hasattr(np, "half"):
        dtype_pairs += [(np.half, np.float32), ]
    for (dtype, transform_dtype) in dtype_pairs:
        x = np.arange(N, dtype=dtype)
        wp = pywt.WaveletPacket(x, wavelet='db1', mode='symmetric')

        # no unnecessary copy made of top-level data
        assert_(wp.data is x)

        # full decomposition
        wp.get_level(wp.maxlevel)

        # reconstructed data will have modified dtype
        r = wp.reconstruct(False)
        assert_equal(r.dtype, transform_dtype)
        assert_allclose(r, x.astype(transform_dtype), atol=1e-5, rtol=1e-5)


def test_db3_roundtrip():
    original = np.arange(512)
    wp = pywt.WaveletPacket(data=original, wavelet='db3', mode='smooth',
                            maxlevel=3)
    r = wp.reconstruct()
    assert_allclose(original, r, atol=1e-12, rtol=1e-12)


def test_wavelet_packet_axis():
    rstate = np.random.RandomState(0)
    shape = (32, 16)
    x = rstate.standard_normal(shape)
    for axis in [0, 1, -1]:
        wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric',
                                axis=axis)

        # partial decomposition
        nodes = wp.get_level(2)
        # size along the transformed axis has changed
        for ax2 in range(x.ndim):
            if ax2 == (axis % x.ndim):
                nodes[0].data.shape[ax2] < x.shape[ax2]
            else:
                nodes[0].data.shape[ax2] == x.shape[ax2]

        # recontsruction from coefficients should preserve dtype
        r = wp.reconstruct(False)
        assert_equal(r.dtype, x.dtype)
        assert_allclose(r, x, atol=1e-12, rtol=1e-12)

    # ValueError if axis is out of range
    assert_raises(ValueError, pywt.WaveletPacket, data=x, wavelet='db1',
                  axis=x.ndim)


def test_wavelet_packet_pickle(tmpdir):
    packet = pywt.WaveletPacket(np.arange(16), 'sym4')
    filename = os.path.join(tmpdir, 'wp.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(packet, f)
    with open(filename, 'rb') as f:
        packet2 = pickle.load(f)
    assert isinstance(packet2, pywt.WaveletPacket)
