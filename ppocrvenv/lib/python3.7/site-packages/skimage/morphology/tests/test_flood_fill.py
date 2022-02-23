import numpy as np
import pytest

from skimage._shared.testing import expected_warnings
from skimage.morphology import flood, flood_fill

eps = 1e-12


def test_empty_input():
    # Test shortcut
    output = flood_fill(np.empty(0), (), 2)
    assert output.size == 0

    # Boolean output type
    assert flood(np.empty(0), ()).dtype == bool

    # Maintain shape, even with zero size present
    assert flood(np.empty((20, 0, 4)), ()).shape == (20, 0, 4)


def test_selem_kwarg_deprecation():
    with expected_warnings(["`selem` is a deprecated argument name"]):
        output = flood_fill(np.empty(0), (), 2, selem=None)
    assert output.size == 0


def test_float16():
    image = np.array([9., 0.1, 42], dtype=np.float16)
    with pytest.raises(TypeError, match="dtype of `image` is float16"):
        flood_fill(image, 0, 1)


def test_overrange_tolerance_int():
    image = np.arange(256, dtype=np.uint8).reshape((8, 8, 4))
    expected = np.zeros_like(image)

    output = flood_fill(image, (7, 7, 3), 0, tolerance=379)

    np.testing.assert_equal(output, expected)


def test_overrange_tolerance_float():
    max_value = np.finfo(np.float32).max

    image = np.random.uniform(size=(64, 64), low=-1., high=1.).astype(
        np.float32)
    image *= max_value

    expected = np.ones_like(image)
    output = flood_fill(image, (0, 1), 1., tolerance=max_value * 10)

    np.testing.assert_equal(output, expected)


def test_inplace_int():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]])

    flood_fill(image, (0, 0), 5, in_place=True)

    expected = np.array([[5, 5, 5, 5, 5, 5, 5],
                         [5, 1, 1, 5, 2, 2, 5],
                         [5, 1, 1, 5, 2, 2, 5],
                         [1, 5, 5, 5, 5, 5, 3],
                         [5, 1, 1, 1, 3, 3, 4]])

    np.testing.assert_array_equal(image, expected)


def test_inplace_float():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]], dtype=np.float32)

    flood_fill(image, (0, 0), 5, in_place=True)

    expected = np.array([[5., 5., 5., 5., 5., 5., 5.],
                         [5., 1., 1., 5., 2., 2., 5.],
                         [5., 1., 1., 5., 2., 2., 5.],
                         [1., 5., 5., 5., 5., 5., 3.],
                         [5., 1., 1., 1., 3., 3., 4.]], dtype=np.float32)

    np.testing.assert_allclose(image, expected)


def test_inplace_noncontiguous():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]])

    # Transpose is noncontiguous
    image2 = image[::2, ::2]

    flood_fill(image2, (0, 0), 5, in_place=True)

    # The inplace modified result
    expected2 = np.array([[5, 5, 5, 5],
                          [5, 1, 2, 5],
                          [5, 1, 3, 4]])

    np.testing.assert_allclose(image2, expected2)

    # Projected back through the view, `image` also modified
    expected = np.array([[5, 0, 5, 0, 5, 0, 5],
                         [0, 1, 1, 0, 2, 2, 0],
                         [5, 1, 1, 0, 2, 2, 5],
                         [1, 0, 0, 0, 0, 0, 3],
                         [5, 1, 1, 1, 3, 3, 4]])

    np.testing.assert_allclose(image, expected)


def test_1d():
    image = np.arange(11)
    expected = np.array([0, 1, -20, -20, -20, -20, -20, -20, -20, 9, 10])

    output = flood_fill(image, 5, -20, tolerance=3)
    output2 = flood_fill(image, (5,), -20, tolerance=3)

    np.testing.assert_equal(output, expected)
    np.testing.assert_equal(output, output2)


def test_wraparound():
    # If the borders (or neighbors) aren't correctly accounted for, this fails,
    # because the algorithm uses an ravelled array.
    test = np.zeros((5, 7), dtype=np.float64)
    test[:, 3] = 100

    expected = np.array([[-1., -1., -1., 100., 0., 0., 0.],
                         [-1., -1., -1., 100., 0., 0., 0.],
                         [-1., -1., -1., 100., 0., 0., 0.],
                         [-1., -1., -1., 100., 0., 0., 0.],
                         [-1., -1., -1., 100., 0., 0., 0.]])

    np.testing.assert_equal(flood_fill(test, (0, 0), -1), expected)


def test_neighbors():
    # This test will only pass if the neighbors are exactly correct
    test = np.zeros((5, 7), dtype=np.float64)
    test[:, 3] = 100

    expected = np.array([[0, 0, 0, 255, 0, 0, 0],
                         [0, 0, 0, 255, 0, 0, 0],
                         [0, 0, 0, 255, 0, 0, 0],
                         [0, 0, 0, 255, 0, 0, 0],
                         [0, 0, 0, 255, 0, 0, 0]])
    output = flood_fill(test, (0, 3), 255)

    np.testing.assert_equal(output, expected)

    test[2] = 100
    expected[2] = 255

    output2 = flood_fill(test, (2, 3), 255)

    np.testing.assert_equal(output2, expected)


def test_footprint():
    # Basic tests for nonstandard footprints
    footprint = np.array([[0, 1, 1],
                          [0, 1, 1],
                          [0, 0, 0]])  # Cannot grow left or down

    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (3, 1), 255,
                        footprint=footprint)

    expected = np.array([[0, 255, 255, 255, 255, 255],
                         [0, 255, 255, 255, 255, 255],
                         [0, 255, 255, 255, 255, 255],
                         [0, 255, 255, 255, 255, 255],
                         [0,   0,   0,   0,   0,   0]], dtype=np.uint8)

    np.testing.assert_equal(output, expected)

    footprint = np.array([[0, 0, 0],
                          [1, 1, 0],
                          [1, 1, 0]])  # Cannot grow right or up

    output = flood_fill(np.zeros((5, 6), dtype=np.uint8), (1, 4), 255,
                        footprint=footprint)

    expected = np.array([[  0,   0,   0,   0,   0,   0],
                         [255, 255, 255, 255, 255,   0],
                         [255, 255, 255, 255, 255,   0],
                         [255, 255, 255, 255, 255,   0],
                         [255, 255, 255, 255, 255,   0]], dtype=np.uint8)

    np.testing.assert_equal(output, expected)


def test_basic_nd():
    for dimension in (3, 4, 5):
        shape = (5,) * dimension
        hypercube = np.zeros(shape)
        slice_mid = tuple(slice(1, -1, None) for dim in range(dimension))
        hypercube[slice_mid] = 1  # sum is 3**dimension
        filled = flood_fill(hypercube, (2,) * dimension, 2)

        # Test that the middle sum is correct
        assert filled.sum() == 3**dimension * 2

        # Test that the entire array is as expected
        np.testing.assert_equal(
            filled, np.pad(np.ones((3,) * dimension) * 2, 1, 'constant'))


@pytest.mark.parametrize("tolerance", [None, 0])
def test_f_order(tolerance):
    image = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], order="F")
    expected = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=bool)

    mask = flood(image, seed_point=(1, 0), tolerance=tolerance)
    np.testing.assert_array_equal(expected, mask)

    mask = flood(image, seed_point=(2, 1), tolerance=tolerance)
    np.testing.assert_array_equal(expected, mask)


def test_negative_indexing_seed_point():
    image = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [0, 1, 1, 0, 2, 2, 0],
                      [1, 0, 0, 0, 0, 0, 3],
                      [0, 1, 1, 1, 3, 3, 4]], dtype=np.float32)

    expected = np.array([[5., 5., 5., 5., 5., 5., 5.],
                         [5., 1., 1., 5., 2., 2., 5.],
                         [5., 1., 1., 5., 2., 2., 5.],
                         [1., 5., 5., 5., 5., 5., 3.],
                         [5., 1., 1., 1., 3., 3., 4.]], dtype=np.float32)

    image = flood_fill(image, (0, -1), 5)

    np.testing.assert_allclose(image, expected)
