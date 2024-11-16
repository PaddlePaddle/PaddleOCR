import os
import sys
import pytest
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from ppocr.data.imaug.iaa_augment import IaaAugment

# Set a fixed random seed to ensure test reproducibility
np.random.seed(42)
random.seed(42)


# Fixture to provide a sample image for tests
@pytest.fixture
def sample_image():
    # Create a 100x100 pixel dummy image with 3 color channels (RGB)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


# Fixture to provide sample polygons for tests
@pytest.fixture
def sample_polys():
    # Create dummy polygons as sample data
    polys = [
        np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
        np.array([[30, 30], [40, 30], [40, 40], [30, 40]], dtype=np.float32),
    ]
    return polys


# Helper function to create a data dictionary for testing
def create_data(sample_image, sample_polys):
    return {
        "image": sample_image.copy(),
        "polys": [poly.copy() for poly in sample_polys],
    }


# Test the default behavior of the augmenter (without specified arguments)
def test_iaa_augment_default(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment()
    transformed_data = augmenter(data)

    # Check the data types and structure of the transformed image and polygons
    assert isinstance(
        transformed_data["image"], np.ndarray
    ), "Image should be a numpy array"
    assert isinstance(
        transformed_data["polys"], np.ndarray
    ), "Polys should be a numpy array"
    assert transformed_data["image"].ndim == 3, "Image should be 3-dimensional"

    # Verify that the polygons have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test the augmenter with empty arguments, meaning no transformations should occur
def test_iaa_augment_none(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment(augmenter_args=[])
    transformed_data = augmenter(data)

    # Check that the image and polygons remain unchanged
    assert np.array_equal(
        data["image"], transformed_data["image"]
    ), "Image should be unchanged"
    for orig_poly, transformed_poly in zip(data["polys"], transformed_data["polys"]):
        assert np.array_equal(
            orig_poly, transformed_poly
        ), "Polygons should be unchanged"


# Parameterized test to check various augmenter arguments and expected image shapes
@pytest.mark.parametrize(
    "augmenter_args, expected_shape",
    [
        ([], (100, 100, 3)),
        ([{"type": "Resize", "args": {"size": [0.5, 0.5]}}], (50, 50, 3)),
        ([{"type": "Resize", "args": {"size": [2.0, 2.0]}}], (200, 200, 3)),
    ],
)
def test_iaa_augment_resize(sample_image, sample_polys, augmenter_args, expected_shape):
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Verify that the transformed image has the expected shape
    assert (
        transformed_data["image"].shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {transformed_data['image'].shape}"


# Test custom augmenter arguments with specific transformations
def test_iaa_augment_custom(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter_args = [
        {"type": "Affine", "args": {"rotate": [45, 45]}},  # Apply 45-degree rotation
        {"type": "Resize", "args": {"size": [0.5, 0.5]}},
    ]
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Check the expected image dimensions after resizing
    expected_height = int(sample_image.shape[0] * 0.5)
    expected_width = int(sample_image.shape[1] * 0.5)
    assert (
        transformed_data["image"].shape[0] == expected_height
    ), "Image height should be scaled by 0.5"
    assert (
        transformed_data["image"].shape[1] == expected_width
    ), "Image width should be scaled by 0.5"

    # Verify that the polygons have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test that an unknown transformation type raises an AttributeError
def test_iaa_augment_unknown_transform():
    augmenter_args = [{"type": "UnknownTransform", "args": {}}]
    with pytest.raises(AttributeError):
        IaaAugment(augmenter_args=augmenter_args)


# Test that an invalid resize size parameter raises a ValueError
def test_iaa_augment_invalid_resize_size(sample_image, sample_polys):
    augmenter_args = [{"type": "Resize", "args": {"size": "invalid_size"}}]
    with pytest.raises(ValueError) as exc_info:
        IaaAugment(augmenter_args=augmenter_args)
    assert "'size' must be a list or tuple of two numbers" in str(exc_info.value)


# Test that polygons are transformed as expected
def test_iaa_augment_polys_transformation(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter_args = [
        {"type": "Affine", "args": {"rotate": [90, 90]}},  # Apply 90-degree rotation
    ]
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Verify that the polygons have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test multiple transformations applied to the augmenter
def test_iaa_augment_multiple_transforms(sample_image, sample_polys):
    augmenter_args = [
        {"type": "Fliplr", "args": {"p": 1.0}},  # Always apply horizontal flip
        {"type": "Affine", "args": {"shear": 10}},
    ]
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Ensure the image has been transformed
    images_different = not np.array_equal(transformed_data["image"], sample_image)
    assert images_different, "Image should be transformed"

    # Ensure the polygons have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"
