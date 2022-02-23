from random import shuffle

import pytest

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature


def test_haar_like_feature_error():
    img = np.ones((5, 5), dtype=np.float32)
    img_ii = integral_image(img)

    feature_type = 'unknown_type'
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
        haar_like_feature_coord(5, 5, feature_type=feature_type)
        draw_haar_like_feature(img, 0, 0, 5, 5, feature_type=feature_type)

    feat_coord, feat_type = haar_like_feature_coord(5, 5, 'type-2-x')
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feat_type[:3],
                          feature_coord=feat_coord)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.float32, np.float64])
@pytest.mark.parametrize("feature_type,shape_feature,expected_feature_value",
                         [('type-2-x', (84,), [0.]),
                          ('type-2-y', (84,), [0.]),
                          ('type-3-x', (42,), [-5, -4., -3., -2., -1.]),
                          ('type-3-y', (42,), [-5, -4., -3., -2., -1.]),
                          ('type-4', (36,), [0.])])
def test_haar_like_feature(feature_type, shape_feature,
                           expected_feature_value, dtype):
    # test Haar-like feature on a basic one image
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5,
                                     feature_type=feature_type)
    assert_allclose(np.sort(np.unique(haar_feature)), expected_feature_value)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.float32, np.float64])
@pytest.mark.parametrize("feature_type", ['type-2-x', 'type-2-y',
                                          'type-3-x', 'type-3-y',
                                          'type-4'])
def test_haar_like_feature_fused_type(dtype, feature_type):
    # check that the input type is kept
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    expected_dtype = img_ii.dtype
    # to avoid overflow, unsigned type are converted to signed
    if 'uint' in expected_dtype.name:
        expected_dtype = np.dtype(expected_dtype.name.replace('u', ''))
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5,
                                     feature_type=feature_type)
    assert haar_feature.dtype == expected_dtype


def test_haar_like_feature_list():
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    haar_list = haar_like_feature(img_ii, 0, 0, 5, 5,
                                  feature_type=feature_type)
    haar_all = haar_like_feature(img_ii, 0, 0, 5, 5)
    assert_array_equal(haar_list, haar_all)


@pytest.mark.parametrize("feature_type", ['type-2-x', 'type-2-y',
                                          'type-3-x', 'type-3-y',
                                          'type-4',
                                          ['type-2-y', 'type-3-x',
                                           'type-4']])
def test_haar_like_feature_precomputed(feature_type):
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    if isinstance(feature_type, list):
        # shuffle the index of the feature to be sure that we are output
        # the features in the same order
        shuffle(feature_type)
        feat_coord, feat_type = zip(*[haar_like_feature_coord(5, 5, feat_t)
                                      for feat_t in feature_type])
        feat_coord = np.concatenate(feat_coord)
        feat_type = np.concatenate(feat_type)
    else:
        feat_coord, feat_type = haar_like_feature_coord(5, 5, feature_type)
    haar_feature_precomputed = haar_like_feature(img_ii, 0, 0, 5, 5,
                                                 feature_type=feat_type,
                                                 feature_coord=feat_coord)
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type)
    assert_array_equal(haar_feature_precomputed, haar_feature)


@pytest.mark.parametrize("feature_type,height,width,expected_coord",
                         [('type-2-x', 2, 2,
                           [[[(0, 0), (0, 0)], [(0, 1), (0, 1)]],
                            [[(0, 0), (1, 0)], [(0, 1), (1, 1)]],
                            [[(1, 0), (1, 0)], [(1, 1), (1, 1)]]]),
                          ('type-2-y', 2, 2,
                           [[[(0, 0), (0, 0)], [(1, 0), (1, 0)]],
                            [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
                            [[(0, 1), (0, 1)], [(1, 1), (1, 1)]]]),
                          ('type-3-x', 3, 3,
                           [[[(0, 0), (0, 0)], [(0, 1), (0, 1)],
                             [(0, 2), (0, 2)]],
                            [[(0, 0), (1, 0)], [(0, 1), (1, 1)],
                             [(0, 2), (1, 2)]],
                            [[(0, 0), (2, 0)], [(0, 1), (2, 1)],
                             [(0, 2), (2, 2)]],
                            [[(1, 0), (1, 0)], [(1, 1), (1, 1)],
                             [(1, 2), (1, 2)]],
                            [[(1, 0), (2, 0)], [(1, 1), (2, 1)],
                             [(1, 2), (2, 2)]],
                            [[(2, 0), (2, 0)], [(2, 1), (2, 1)],
                             [(2, 2), (2, 2)]]]),
                          ('type-3-y', 3, 3,
                           [[[(0, 0), (0, 0)], [(1, 0), (1, 0)],
                             [(2, 0), (2, 0)]],
                            [[(0, 0), (0, 1)], [(1, 0), (1, 1)],
                             [(2, 0), (2, 1)]],
                            [[(0, 0), (0, 2)], [(1, 0), (1, 2)],
                             [(2, 0), (2, 2)]],
                            [[(0, 1), (0, 1)], [(1, 1), (1, 1)],
                             [(2, 1), (2, 1)]],
                            [[(0, 1), (0, 2)], [(1, 1), (1, 2)],
                             [(2, 1), (2, 2)]],
                            [[(0, 2), (0, 2)], [(1, 2), (1, 2)],
                             [(2, 2), (2, 2)]]]),
                          ('type-4', 2, 2,
                           [[[(0, 0), (0, 0)], [(0, 1), (0, 1)],
                             [(1, 1), (1, 1)], [(1, 0), (1, 0)]]])])
def test_haar_like_feature_coord(feature_type, height, width, expected_coord):
    feat_coord, feat_type = haar_like_feature_coord(width, height,
                                                    feature_type)
    # convert the output to a full numpy array just for comparison
    feat_coord = np.array([hf for hf in feat_coord])
    assert_array_equal(feat_coord, expected_coord)
    assert np.all(feat_type == feature_type)


@pytest.mark.parametrize("max_n_features,nnz_values", [(None, 46),
                                                       (1, 4)])
def test_draw_haar_like_feature(max_n_features, nnz_values):
    img = np.zeros((5, 5), dtype=np.float32)
    coord, _ = haar_like_feature_coord(5, 5, 'type-4')
    image = draw_haar_like_feature(img, 0, 0, 5, 5, coord,
                                   max_n_features=max_n_features,
                                   random_state=0)
    assert image.shape == (5, 5, 3)
    assert np.count_nonzero(image) == nnz_values
