import numpy as np
from skimage.measure import block_reduce

from skimage._shared import testing
from skimage._shared.testing import assert_equal


def test_block_reduce_sum():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3))
    expected1 = np.array([[ 24,  42],
                          [ 96, 114]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (3, 3))
    expected2 = np.array([[ 81, 108,  87],
                          [174, 192, 138]])
    assert_equal(expected2, out2)


def test_block_reduce_mean():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.mean)
    expected1 = np.array([[  4.,   7.],
                          [ 16.,  19.]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.mean)
    expected2 = np.array([[14. , 10.8],
                          [ 8.5,  5.7]])
    assert_equal(expected2, out2)


def test_block_reduce_median():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.median)
    expected1 = np.array([[  4.,   7.],
                          [ 16.,  19.]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.median)
    expected2 = np.array([[ 14.,  6.5],
                          [  0.,  0. ]])
    assert_equal(expected2, out2)

    image3 = np.array([[1, 5, 5, 5], [5, 5, 5, 1000]])
    out3 = block_reduce(image3, (2, 4), func=np.median)
    assert_equal(5, out3)


def test_block_reduce_min():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.min)
    expected1 = np.array([[ 0, 3],
                          [12, 15]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.min)
    expected2 = np.array([[0, 0],
                          [0, 0]])
    assert_equal(expected2, out2)


def test_block_reduce_max():
    image1 = np.arange(4 * 6).reshape(4, 6)
    out1 = block_reduce(image1, (2, 3), func=np.max)
    expected1 = np.array([[ 8, 11],
                          [20, 23]])
    assert_equal(expected1, out1)

    image2 = np.arange(5 * 8).reshape(5, 8)
    out2 = block_reduce(image2, (4, 5), func=np.max)
    expected2 = np.array([[28, 31],
                          [36, 39]])
    assert_equal(expected2, out2)


def test_invalid_block_size():
    image = np.arange(4 * 6).reshape(4, 6)

    with testing.raises(ValueError):
        block_reduce(image, [1, 2, 3])
    with testing.raises(ValueError):
        block_reduce(image, [1, 0.5])


def test_default_block_size():
    image = np.arange(4 * 6).reshape(4, 6)
    out = block_reduce(image, func=np.min)
    expected = np.array([[0, 2, 4],
                         [12, 14, 16]])
    assert_equal(expected, out)


def test_scalar_block_size():
    image = np.arange(6 * 6).reshape(6, 6)
    out = block_reduce(image, 3, func=np.min)
    expected1 = np.array([[0, 3],
                         [18, 21]])
    assert_equal(expected1, out)
    expected2 = block_reduce(image, (3, 3), func=np.min)
    assert_equal(expected2, out)


def test_func_kwargs_same_dtype():
    image = np.array([[97, 123, 173, 227],
                     [217, 241, 221, 214],
                     [211,  11, 170,  53],
                     [214, 205, 101,  57]], dtype=np.uint8)

    out = block_reduce(image, (2, 2), func=np.mean,
                       func_kwargs={'dtype': np.uint8})
    expected = np.array([[41, 16], [32, 31]], dtype=np.uint8)

    assert_equal(out, expected)
    assert out.dtype == expected.dtype


def test_func_kwargs_different_dtype():
    image = np.array([[0.45745366, 0.67479345, 0.20949775, 0.3147348],
                      [0.7209286, 0.88915504, 0.66153409, 0.07919526],
                      [0.04640037, 0.54008495, 0.34664343, 0.56152301],
                      [0.58085003, 0.80144708, 0.87844473, 0.29811511]],
                     dtype=np.float64)

    out = block_reduce(image, (2, 2), func=np.mean,
                       func_kwargs={'dtype': np.float16})
    expected = np.array([[0.6855, 0.3164], [0.4922, 0.521]], dtype=np.float16)

    assert_equal(out, expected)
    assert out.dtype == expected.dtype
