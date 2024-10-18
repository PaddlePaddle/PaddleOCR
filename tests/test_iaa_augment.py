import os
import sys
import pytest
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from ppocr.data.imaug.iaa_augment import IaaAugment

# Set a fixed random seed for reproducibility
np.random.seed(42)
random.seed(42)


# Fixtures for common test inputs
@pytest.fixture
def sample_image():
    # Create a dummy image of size 100x100 with 3 channels
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_polys():
    # Create some dummy polygons
    polys = [
        np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
        np.array([[30, 30], [40, 30], [40, 40], [30, 40]], dtype=np.float32),
    ]
    return polys


# Helper function to create data dictionary
def create_data(sample_image, sample_polys):
    return {
        "image": sample_image.copy(),
        "polys": [poly.copy() for poly in sample_polys],
    }


# Test default augmenter (with default augmenter_args)
def test_iaa_augment_default(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment()
    transformed_data = augmenter(data)

    assert isinstance(
        transformed_data["image"], np.ndarray
    ), "Image should be a numpy array"
    assert isinstance(transformed_data["polys"], list), "Polys should be a list"
    assert transformed_data["image"].ndim == 3, "Image should be 3-dimensional"

    # Check that the polys have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test augmenter with empty augmenter_args (no augmentation)
def test_iaa_augment_none(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment(augmenter_args=[])
    transformed_data = augmenter(data)

    assert np.array_equal(
        data["image"], transformed_data["image"]
    ), "Image should be unchanged"
    for orig_poly, transformed_poly in zip(data["polys"], transformed_data["polys"]):
        assert np.array_equal(
            orig_poly, transformed_poly
        ), "Polygons should be unchanged"


# Parameterize tests to cover multiple augmenter_args scenarios
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

    assert (
        transformed_data["image"].shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {transformed_data['image'].shape}"


# Test with custom augmenter_args
def test_iaa_augment_custom(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter_args = [
        {"type": "Affine", "args": {"rotate": [45, 45]}},  # Fixed rotation angle
        {"type": "Resize", "args": {"size": [0.5, 0.5]}},
    ]
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    expected_height = int(sample_image.shape[0] * 0.5)
    expected_width = int(sample_image.shape[1] * 0.5)

    assert (
        transformed_data["image"].shape[0] == expected_height
    ), "Image height should be scaled by 0.5"
    assert (
        transformed_data["image"].shape[1] == expected_width
    ), "Image width should be scaled by 0.5"

    # Check that the polys have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test unknown transform type raises NotImplementedError
def test_iaa_augment_unknown_transform():
    augmenter_args = [{"type": "UnknownTransform", "args": {}}]
    with pytest.raises(NotImplementedError):
        IaaAugment(augmenter_args=augmenter_args)


# Test invalid size parameter raises ValueError
def test_iaa_augment_invalid_resize_size():
    augmenter_args = [{"type": "Resize", "args": {"size": "invalid_size"}}]
    with pytest.raises(ValueError):
        IaaAugment(augmenter_args=augmenter_args)


# Test that polys are transformed appropriately
def test_iaa_augment_polys_transformation(sample_image, sample_polys):
    data = create_data(sample_image, sample_polys)
    augmenter_args = [
        {"type": "Affine", "args": {"rotate": [90, 90]}},  # Fixed rotation angle
    ]
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Check that the polygons have changed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"


# Test with multiple transforms in augmenter_args
def test_iaa_augment_multiple_transforms(sample_image, sample_polys):
    augmenter_args = [
        {"type": "Fliplr", "args": {"p": 1.0}},  # Always flip
        {"type": "Affine", "args": {"shear": 10}},
    ]
    data = create_data(sample_image, sample_polys)
    augmenter = IaaAugment(augmenter_args=augmenter_args)
    transformed_data = augmenter(data)

    # Check that the image has been transformed
    images_different = not np.array_equal(transformed_data["image"], sample_image)
    assert images_different, "Image should be transformed"

    # Check that the polys have been transformed
    polys_changed = any(
        not np.allclose(orig_poly, trans_poly)
        for orig_poly, trans_poly in zip(sample_polys, transformed_data["polys"])
    )
    assert polys_changed, "Polygons should have been transformed"
