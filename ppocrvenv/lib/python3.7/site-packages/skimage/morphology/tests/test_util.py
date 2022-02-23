"""Tests for `_util`."""


import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skimage.morphology import _util


@pytest.mark.parametrize("image_shape", [
    (111,), (33, 44), (22, 55, 11), (6, 5, 4, 3)
])
@pytest.mark.parametrize("order", ["C", "F"])
def test_offsets_to_raveled_neighbors_highest_connectivity(image_shape, order):
    """
    Check scenarios where footprint is always of the highest connectivity
    and all dimensions are > 2.
    """
    footprint = np.ones((3,) * len(image_shape), dtype=bool)
    center = (1,) * len(image_shape)
    offsets = _util._offsets_to_raveled_neighbors(
        image_shape, footprint, center, order
    )

    # Assert only neighbors are present, center was removed
    assert len(offsets) == footprint.sum() - 1
    assert 0 not in offsets
    # Assert uniqueness
    assert len(set(offsets)) == offsets.size
    # offsets form pairs of with same value but different signs
    # if footprint is symmetric around center
    assert all(-x in offsets for x in offsets)

    # Construct image whose values are the Manhattan distance to its center
    image_center = tuple(s // 2 for s in image_shape)
    coords = [
        np.abs(np.arange(s, dtype=np.intp) - c)
        for s, c in zip(image_shape, image_center)
    ]
    grid = np.meshgrid(*coords, indexing="ij")
    image = np.sum(grid, axis=0)

    image_raveled = image.ravel(order)
    image_center_raveled = np.ravel_multi_index(
        image_center, image_shape, order=order
    )

    # Sample raveled image around its center
    samples = []
    for offset in offsets:
        index = image_center_raveled + offset
        samples.append(image_raveled[index])

    # Assert that center with value 0 wasn't selected
    assert np.min(samples) == 1
    # Assert that only neighbors where selected
    # (highest value == connectivity)
    assert np.max(samples) == len(image_shape)
    # Assert that nearest neighbors are selected first
    assert list(sorted(samples)) == samples


@pytest.mark.parametrize("image_shape", [
    (2,), (2, 2), (2, 1, 2), (2, 2, 1, 2), (0, 2, 1, 2)
])
@pytest.mark.parametrize("order", ["C", "F"])
def test_offsets_to_raveled_neighbors_footprint_smaller_image(image_shape,
                                                              order):
    """
    Test if a dimension indicated by `image_shape` is smaller than in
    `footprint`.
    """
    footprint = np.ones((3,) * len(image_shape), dtype=bool)
    center = (1,) * len(image_shape)
    offsets = _util._offsets_to_raveled_neighbors(
        image_shape, footprint, center, order
    )

    # Assert only neighbors are present, center and duplicates (possible
    # for this scenario) where removed
    assert len(offsets) <= footprint.sum() - 1
    assert 0 not in offsets
    # Assert uniqueness
    assert len(set(offsets)) == offsets.size
    # offsets form pairs of with same value but different signs
    # if footprint is symmetric around center
    assert all(-x in offsets for x in offsets)


def test_offsets_to_raveled_neighbors_explicit_0():
    """Check reviewed example."""
    image_shape = (100, 200, 3)
    footprint = np.ones((3, 3, 3), dtype=bool)
    center = (1, 1, 1)
    offsets = _util._offsets_to_raveled_neighbors(
        image_shape, footprint, center
    )

    desired = np.array([
          3, -600,    1,   -1,  600,   -3,    4,    2,  603,   -2,   -4,
        -597,  601, -599, -601, -603,  599,  597,  602, -604,  596, -596,
        -598, -602,  598,  604
    ])
    assert_array_equal(offsets, desired)


def test_offsets_to_raveled_neighbors_explicit_1():
    """Check reviewed example where footprint is larger in last dimension."""
    image_shape = (10, 9, 8, 3)
    footprint = np.ones((3, 3, 3, 4), dtype=bool)
    center = (1, 1, 1, 1)
    offsets = _util._offsets_to_raveled_neighbors(
        image_shape, footprint, center
    )

    desired = np.array([
            216, 24, -24, 3, -216, 1, -1, -3, 215, -27, -25, -23, -21, -2,
            -192, 192, 2, 4, 21, 23, 25, 27, -4, 217, 213, -219, 219, -217,
            -213, -215, 240, -240, 193, 239, -237, 241, -239, 218, -220, 22,
            -241, 243, 189, 26, -243, 191, 20, -218, 195, -193, 220, -191,
            -212, -189, 214, 28, -195, -214, -28, 212, -22, 237, -20, -26, 236,
            196, 190, 242, 238, 194, 188, -244, -188, -196, -194, -190, -238,
            -236, 244, -242, 5, 221, -211, -19, 29, -235, -187, 197, 245
            ])
    assert_array_equal(offsets, desired)
