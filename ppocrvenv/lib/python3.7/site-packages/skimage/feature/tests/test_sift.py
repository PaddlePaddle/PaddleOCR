import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from skimage import data
from skimage._shared.testing import test_parallel
from skimage.feature import SIFT
from skimage.util.dtype import _convert

img = data.coins()


@test_parallel()
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'uint8', 'uint16', 'int64']
)
def test_keypoints_sift(dtype):
    _img = _convert(img, dtype)
    detector_extractor = SIFT()
    detector_extractor.detect_and_extract(_img)

    exp_keypoint_rows = np.array([18, 18, 19, 22, 26, 26, 30, 31, 31, 32])
    exp_keypoint_cols = np.array([331, 331, 325, 330, 310, 330, 205, 323, 149,
                                  338])

    exp_octaves = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    exp_position_rows = np.array([17.81909936, 17.81909936, 19.05454661,
                                  21.85933727, 25.54800708, 26.25710504,
                                  29.90826307, 30.78713806, 30.87953572,
                                  31.72557969])

    exp_position_cols = np.array([331.49693187, 331.49693187, 325.24476016,
                                  330.44616424, 310.33932904, 330.46155224,
                                  204.74535177, 322.84100812, 149.43192282,
                                  337.89643013])

    exp_orientations = np.array([0.26391655, 0.26391655, 0.39134262,
                                 1.77063053, 0.98637565, 1.37997279, 0.4919992,
                                 1.48615988, 0.33753212, 1.64859617])

    exp_scales = np.array([2, 2, 1, 3, 3, 1, 2, 1, 1, 1])

    exp_sigmas = np.array([1.35160379, 1.35160379, 0.94551567, 1.52377498,
                           1.55173233, 0.93973722, 1.37594124, 1.06663786,
                           1.04827034, 1.0378916])

    exp_scalespace_sigmas = np.array([[0.8, 1.00793684, 1.26992084, 1.6,
                                       2.01587368, 2.53984168],
                                      [1.6, 2.01587368, 2.53984168, 3.2,
                                       4.03174736, 5.07968337],
                                      [3.2, 4.03174736, 5.07968337, 6.4,
                                       8.06349472, 10.15936673],
                                      [6.4, 8.06349472, 10.15936673, 12.8,
                                       16.12698944, 20.31873347],
                                      [12.8, 16.12698944, 20.31873347, 25.6,
                                       32.25397888, 40.63746693],
                                      [25.6, 32.25397888, 40.63746693, 51.2,
                                       64.50795775, 81.27493386]])

    assert_almost_equal(exp_keypoint_rows,
                        detector_extractor.keypoints[:10, 0])
    assert_almost_equal(exp_keypoint_cols,
                        detector_extractor.keypoints[:10, 1])
    assert_almost_equal(exp_octaves,
                        detector_extractor.octaves[:10])
    assert_almost_equal(exp_position_rows,
                        detector_extractor.positions[:10, 0], decimal=4)
    assert_almost_equal(exp_position_cols,
                        detector_extractor.positions[:10, 1], decimal=4)
    assert_almost_equal(exp_orientations,
                        detector_extractor.orientations[:10], decimal=4)
    assert_almost_equal(exp_scales,
                        detector_extractor.scales[:10])
    assert_almost_equal(exp_sigmas,
                        detector_extractor.sigmas[:10], decimal=4)
    assert_almost_equal(exp_scalespace_sigmas,
                        detector_extractor.scalespace_sigmas, decimal=4)

    detector_extractor2 = SIFT()
    detector_extractor2.detect(img)
    detector_extractor2.extract(img)
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])
    assert_almost_equal(detector_extractor.keypoints[:10, 0],
                        detector_extractor2.keypoints[:10, 0])


def test_descriptor_sift():
    detector_extractor = SIFT(n_hist=2, n_ori=4)
    exp_descriptors = np.array([[173, 30, 55, 32, 173, 16, 45, 82, 173, 154,
                                 170, 173, 173, 169, 65, 110],
                                [173, 30, 55, 32, 173, 16, 45, 82, 173, 154,
                                 170, 173, 173, 169, 65, 110],
                                [189, 52, 18, 18, 189, 11, 21, 55, 189, 75,
                                 173, 91, 189, 65, 189, 162],
                                [172, 156, 185, 66, 92, 76, 78, 185, 185, 87,
                                 88, 82, 98, 56, 96, 185],
                                [216, 19, 40, 9, 196, 7, 57, 36, 216, 56, 158,
                                 29, 216, 42, 144, 154],
                                [169, 120, 169, 91, 129, 108, 169, 67, 169,
                                 142, 111, 95, 169, 120, 69, 41],
                                [199, 10, 138, 44, 178, 11, 161, 34, 199, 113,
                                 73, 64, 199, 82, 31, 178],
                                [154, 56, 154, 49, 144, 154, 154, 78, 154, 51,
                                 154, 83, 154, 154, 154, 72],
                                [230, 46, 47, 21, 230, 15, 65, 95, 230, 52, 72,
                                 51, 230, 19, 59, 130],
                                [155, 117, 154, 102, 155, 155, 90, 110, 145,
                                 127, 155, 50, 57, 155, 155, 70]],
                               dtype=np.uint8
                               )

    detector_extractor.detect_and_extract(img)

    assert_equal(exp_descriptors, detector_extractor.descriptors[:10])

    keypoints_count = detector_extractor.keypoints.shape[0]
    assert keypoints_count == detector_extractor.descriptors.shape[0]
    assert keypoints_count == detector_extractor.orientations.shape[0]
    assert keypoints_count == detector_extractor.octaves.shape[0]
    assert keypoints_count == detector_extractor.positions.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]
    assert keypoints_count == detector_extractor.scales.shape[0]


def test_no_descriptors_extracted_sift():
    img = np.ones((128, 128))
    detector_extractor = SIFT()
    with pytest.raises(RuntimeError):
        detector_extractor.detect_and_extract(img)
