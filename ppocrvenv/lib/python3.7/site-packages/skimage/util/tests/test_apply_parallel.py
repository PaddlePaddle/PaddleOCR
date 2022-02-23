import numpy as np

from skimage._shared.testing import (assert_array_almost_equal, assert_equal,
                                     expected_warnings)
from skimage import color, data, img_as_float
from skimage.filters import threshold_local, gaussian
from skimage.util.apply_parallel import apply_parallel

import pytest
da = pytest.importorskip('dask.array')


def test_apply_parallel():
    # data
    a = np.arange(144).reshape(12, 12).astype(float)

    # apply the filter
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'})

    assert_array_almost_equal(result1, expected1)

    def wrapped_gauss(arr):
        return gaussian(arr, 1, mode='reflect')

    expected2 = gaussian(a, 1, mode='reflect')
    result2 = apply_parallel(wrapped_gauss, a, chunks=(6, 6), depth=5)

    assert_array_almost_equal(result2, expected2)

    expected3 = gaussian(a, 1, mode='reflect')
    result3 = apply_parallel(
        wrapped_gauss, da.from_array(a, chunks=(6, 6)), depth=5, compute=True
    )

    assert isinstance(result3, np.ndarray)
    assert_array_almost_equal(result3, expected3)


def test_apply_parallel_lazy():
    # data
    a = np.arange(144).reshape(12, 12).astype(float)
    d = da.from_array(a, chunks=(6, 6))

    # apply the filter
    expected1 = threshold_local(a, 3)
    result1 = apply_parallel(threshold_local, a, chunks=(6, 6), depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'},
                             compute=False)

    # apply the filter on a Dask Array
    result2 = apply_parallel(threshold_local, d, depth=5,
                             extra_arguments=(3,),
                             extra_keywords={'mode': 'reflect'})

    assert isinstance(result1, da.Array)

    assert_array_almost_equal(result1.compute(), expected1)

    assert isinstance(result2, da.Array)

    assert_array_almost_equal(result2.compute(), expected1)


def test_no_chunks():
    a = np.ones(1 * 4 * 8 * 9).reshape(1, 4, 8, 9)

    def add_42(arr):
        return arr + 42

    expected = add_42(a)
    result = apply_parallel(add_42, a)

    assert_array_almost_equal(result, expected)


def test_apply_parallel_wrap():
    def wrapped(arr):
        return gaussian(arr, 1, mode='wrap')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian(a, 1, mode='wrap')
    result = apply_parallel(wrapped, a, chunks=(6, 6), depth=5, mode='wrap')

    assert_array_almost_equal(result, expected)


def test_apply_parallel_nearest():
    def wrapped(arr):
        return gaussian(arr, 1, mode='nearest')
    a = np.arange(144).reshape(12, 12).astype(float)
    expected = gaussian(a, 1, mode='nearest')
    result = apply_parallel(wrapped, a, chunks=(6, 6), depth={0: 5, 1: 5},
                            mode='nearest')

    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('chunks', (None, (128, 128, 3)))
@pytest.mark.parametrize('depth', (0, 8, (8, 8, 0)))
def test_apply_parallel_rgb(depth, chunks, dtype):
    cat = data.chelsea().astype(dtype) / 255.

    func = color.rgb2ycbcr
    cat_ycbcr_expected = func(cat)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        cat_ycbcr = apply_parallel(func, cat, chunks=chunks, depth=depth,
                                   dtype=dtype, multichannel=True)

    assert_equal(cat_ycbcr.dtype, cat.dtype)

    assert_array_almost_equal(cat_ycbcr_expected, cat_ycbcr)


@pytest.mark.parametrize('chunks', (None, (128, 256), 'ndim'))
@pytest.mark.parametrize('depth', (0, 8, (8, 16), 'ndim'))
@pytest.mark.parametrize('channel_axis', (0, 1, 2, -1, -2, -3))
def test_apply_parallel_rgb_channel_axis(depth, chunks, channel_axis):
    """Test channel_axis combinations.

    For depth and chunks, test in three ways:
    1.) scalar (to be applied over all axes)
    2.) tuple of length ``image.ndim - 1`` corresponding to spatial axes
    3.) tuple of length ``image.ndim`` corresponding to all axes
    """
    cat = img_as_float(data.chelsea())

    func = color.rgb2ycbcr
    cat_ycbcr_expected = func(cat, channel_axis=-1)

    # move channel axis to another position
    cat = np.moveaxis(cat, -1, channel_axis)
    if chunks == 'ndim':
        # explicitly specify the chunksize for the channel axis
        chunks = [128, 128]
        chunks.insert(channel_axis % cat.ndim, cat.shape[channel_axis])
    if depth == 'ndim':
        # explicitly specify the depth for the channel axis
        depth = [8, 8]
        depth.insert(channel_axis % cat.ndim, 0)
    cat_ycbcr = apply_parallel(func, cat, chunks=chunks, depth=depth,
                               dtype=cat.dtype, channel_axis=channel_axis,
                               extra_keywords=dict(channel_axis=channel_axis))
    # move channels of output back to the last dimension
    cat_ycbcr = np.moveaxis(cat_ycbcr, channel_axis, -1)

    assert_array_almost_equal(cat_ycbcr_expected, cat_ycbcr)
