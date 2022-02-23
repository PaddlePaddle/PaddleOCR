import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from skimage import data
from skimage._shared.testing import test_parallel, xfail, arch32
from skimage.feature import ORB
from skimage.util.dtype import _convert


img = data.coins()


@test_parallel()
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64']
)
def test_keypoints_orb_desired_no_of_keypoints(dtype):
    _img = _convert(img, dtype)
    detector_extractor = ORB(n_keypoints=10, fast_n=12, fast_threshold=0.20)
    detector_extractor.detect(_img)

    exp_rows = np.array([141., 108., 214.56, 131., 214.272, 67.,
                         206., 177., 108., 141.])
    exp_cols = np.array([323., 328., 282.24, 292., 281.664, 85.,
                         260., 284., 328.8, 267.])

    exp_scales = np.array([1,  1,  1.44,  1,  1.728, 1, 1, 1, 1.2, 1])

    exp_orientations = np.array([-53.97446153, 59.5055285, -96.01885186,
                                 -149.70789506, -94.70171899, -45.76429535,
                                 -51.49752849, 113.57081195, 63.30428063,
                                 -79.56091118])
    exp_response = np.array([1.01168357, 0.82934145, 0.67784179, 0.57176438,
                             0.56637459, 0.52248355, 0.43696175, 0.42992376,
                             0.37700486, 0.36126832])

    if np.dtype(dtype) == np.float32:
        assert detector_extractor.scales.dtype == np.float32
        assert detector_extractor.responses.dtype == np.float32
        assert detector_extractor.orientations.dtype == np.float32
    else:
        assert detector_extractor.scales.dtype == np.float64
        assert detector_extractor.responses.dtype == np.float64
        assert detector_extractor.orientations.dtype == np.float64

    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses, 5)
    assert_almost_equal(exp_orientations,
                        np.rad2deg(detector_extractor.orientations), 4)

    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])


@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64']
)
def test_keypoints_orb_less_than_desired_no_of_keypoints(dtype):
    _img = _convert(img, dtype)
    detector_extractor = ORB(n_keypoints=15, fast_n=12,
                             fast_threshold=0.33, downscale=2, n_scales=2)
    detector_extractor.detect(_img)

    exp_rows = np.array([108., 203., 140.,  65.,  58.])
    exp_cols = np.array([293., 267., 202., 130., 291.])

    exp_scales = np.array([1., 1., 1., 1., 1.])

    exp_orientations = np.array([151.93906, -56.90052, -79.46341,
                                 -59.42996, -158.26941])

    exp_response = np.array([-0.1764169, 0.2652126, -0.0324343,
                             0.0400902, 0.2667641])

    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])
    assert_almost_equal(exp_scales, detector_extractor.scales)
    assert_almost_equal(exp_response, detector_extractor.responses)
    assert_almost_equal(exp_orientations,
                        np.rad2deg(detector_extractor.orientations), 3)

    detector_extractor.detect_and_extract(img)
    assert_almost_equal(exp_rows, detector_extractor.keypoints[:, 0])
    assert_almost_equal(exp_cols, detector_extractor.keypoints[:, 1])


@xfail(condition=arch32,
       reason=('Known test failure on 32-bit platforms. See links for '
               'details: '
               'https://github.com/scikit-image/scikit-image/issues/3091 '
               'https://github.com/scikit-image/scikit-image/issues/2529'))
def test_descriptor_orb():
    detector_extractor = ORB(fast_n=12, fast_threshold=0.20)
    exp_descriptors = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                                [1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                                [0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                                [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                                [1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                                [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                                [0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                                [0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                                [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=bool)

    detector_extractor.detect(img)
    detector_extractor.extract(img, detector_extractor.keypoints,
                               detector_extractor.scales,
                               detector_extractor.orientations)

    assert_equal(exp_descriptors,
                 detector_extractor.descriptors[100:120, 10:20])

    detector_extractor.detect_and_extract(img)
    assert_equal(exp_descriptors,
                 detector_extractor.descriptors[100:120, 10:20])
    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.responses.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]


def test_no_descriptors_extracted_orb():
    img = np.ones((128, 128))
    detector_extractor = ORB()
    with pytest.raises(RuntimeError):
        detector_extractor.detect_and_extract(img)
