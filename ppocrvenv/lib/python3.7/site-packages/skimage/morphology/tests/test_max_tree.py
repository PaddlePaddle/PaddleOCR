import numpy as np
from skimage.morphology import max_tree, area_closing, area_opening
from skimage.morphology import max_tree_local_maxima, diameter_opening
from skimage.morphology import diameter_closing
from skimage.util import invert

from skimage._shared.testing import assert_array_equal, TestCase

eps = 1e-12


def _full_type_test(img, param, expected, func, param_scale=False,
                    **keywords):

    # images as they are
    out = func(img, param, **keywords)
    assert_array_equal(out, expected)

    # unsigned int
    for dt in [np.uint32, np.uint64]:
        img_cast = img.astype(dt)
        out = func(img_cast, param, **keywords)
        exp_cast = expected.astype(dt)
        assert_array_equal(out, exp_cast)

    # float
    data_float = img.astype(np.float64)
    data_float = data_float / 255.0
    expected_float = expected.astype(np.float64)
    expected_float = expected_float / 255.0
    if param_scale:
        param_cast = param / 255.0
    else:
        param_cast = param
    for dt in [np.float32, np.float64]:
        data_cast = data_float.astype(dt)
        out = func(data_cast, param_cast, **keywords)
        exp_cast = expected_float.astype(dt)
        error_img = 255.0 * exp_cast - 255.0 * out
        error = (error_img >= 1.0).sum()
        assert error < eps

    # signed images
    img_signed = img.astype(np.int16)
    img_signed = img_signed - 128
    exp_signed = expected.astype(np.int16)
    exp_signed = exp_signed - 128
    for dt in [np.int8, np.int16, np.int32, np.int64]:
        img_s = img_signed.astype(dt)
        out = func(img_s, param, **keywords)
        exp_s = exp_signed.astype(dt)
        assert_array_equal(out, exp_s)


class TestMaxtree(TestCase):

    def test_max_tree(self):
        "Test for max tree"
        img_type = np.uint8
        img = np.array([[10, 8, 8, 9],
                        [7, 7, 9, 9],
                        [8, 7, 10, 10],
                        [9, 9, 10, 10]], dtype=img_type)

        P_exp = np.array([[1, 4, 1, 1],
                          [4, 4, 3, 3],
                          [1, 4, 3, 10],
                          [3, 3, 10, 10]], dtype=np.int64)

        S_exp = np.array([4, 5, 9, 1, 2, 8, 3, 6, 7,
                          12, 13, 0, 10, 11, 14, 15],
                         dtype=np.int64)

        for img_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
            img = img.astype(img_type)
            P, S = max_tree(img, connectivity=2)
            assert_array_equal(P, P_exp)
            assert_array_equal(S, S_exp)

        for img_type in [np.int8, np.int16, np.int32, np.int64]:
            img = img.astype(img_type)
            img_shifted = img - 9
            P, S = max_tree(img_shifted, connectivity=2)
            assert_array_equal(P, P_exp)
            assert_array_equal(S, S_exp)

        img_float = img.astype(float)
        img_float = (img_float - 8) / 2.0
        for img_type in [np.float32, np.float64]:
            img_float = img_float.astype(img_type)
            P, S = max_tree(img_float, connectivity=2)
            assert_array_equal(P, P_exp)
            assert_array_equal(S, S_exp)

        return

    def test_area_closing(self):
        "Test for Area Closing (2 thresholds, all types)"

        # original image
        img = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 200, 240, 200, 200, 240, 240, 200, 240],
             [240, 200, 40, 240, 240, 240, 240, 240, 240, 240, 40, 240],
             [240, 240, 240, 240, 100, 240, 100, 100, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 40],
             [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 40, 200, 240, 240, 100, 255, 255],
             [200, 40, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # expected area closing with area 2
        expected_2 = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 200, 200, 240, 240, 200, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 100, 100, 240, 240, 200, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 100, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 40, 200, 240, 240, 200, 255, 255],
             [200, 200, 255, 255, 255, 40, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # expected diameter closing with diameter 4
        expected_4 = np.array(
            [[240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 200, 200, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240],
             [200, 200, 200, 200, 200, 200, 200, 240, 240, 240, 255, 255],
             [200, 255, 200, 200, 200, 255, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 255, 255, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 240, 240, 200, 255, 255],
             [200, 200, 255, 255, 255, 200, 200, 255, 200, 200, 255, 255],
             [200, 200, 200, 200, 200, 200, 200, 255, 255, 255, 255, 255]],
            dtype=np.uint8)

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, expected_2, area_closing, connectivity=2)
        _full_type_test(img, 4, expected_4, area_closing, connectivity=2)

        P, S = max_tree(invert(img), connectivity=2)
        _full_type_test(img, 4, expected_4, area_closing,
                        parent=P, tree_traverser=S)

    def test_area_opening(self):
        "Test for Area Opening (2 thresholds, all types)"

        # original image
        img = np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                        [15, 55, 55, 15, 55, 15, 55, 55, 15, 15, 55, 15],
                        [15, 55, 215, 15, 15, 15, 15, 15, 15, 15, 215, 15],
                        [15, 15, 15, 15, 155, 15, 155, 155, 15, 15, 55, 15],
                        [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                        [55, 55, 55, 55, 55, 55, 55, 15, 55, 55, 0, 0],
                        [55, 0, 55, 55, 55, 0, 55, 15, 0, 0, 0, 215],
                        [55, 55, 55, 155, 55, 55, 55, 15, 0, 0, 0, 0],
                        [55, 55, 55, 155, 55, 55, 55, 15, 55, 55, 0, 0],
                        [55, 55, 55, 55, 55, 215, 55, 15, 15, 155, 0, 0],
                        [55, 215, 0, 0, 0, 215, 55, 0, 55, 55, 0, 0],
                        [55, 55, 55, 55, 55, 55, 55, 0, 0, 0, 0, 0]],
                       dtype=np.uint8)

        # expected area closing with area 2
        expected_2 = np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [15, 55, 55, 15, 15, 15, 55, 55, 15,
                                15, 55, 15],
                               [15, 55, 55, 15, 15, 15, 15, 15, 15,
                                15, 55, 15],
                               [15, 15, 15, 15, 15, 15, 155, 155, 15,
                                15, 55, 15],
                               [15, 15, 15, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [55, 55, 55, 55, 55, 55, 55, 15, 55,
                                55, 0, 0],
                               [55, 0, 55, 55, 55, 0, 55, 15, 0,
                                0, 0, 0],
                               [55, 55, 55, 155, 55, 55, 55, 15, 0,
                                0, 0, 0],
                               [55, 55, 55, 155, 55, 55, 55, 15, 55,
                                55, 0, 0],
                               [55, 55, 55, 55, 55, 215, 55, 15, 15,
                                55, 0, 0],
                               [55, 55, 0, 0, 0, 215, 55, 0, 55,
                                55, 0, 0],
                               [55, 55, 55, 55, 55, 55, 55, 0, 0,
                                0, 0, 0]],
                              dtype=np.uint8)

        # expected diameter closing with diameter 4
        expected_4 = np.array([[15, 15, 15, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [15, 55, 55, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [15, 55, 55, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [15, 15, 15, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [15, 15, 15, 15, 15, 15, 15, 15, 15,
                                15, 15, 15],
                               [55, 55, 55, 55, 55, 55, 55, 15, 15,
                                15, 0, 0],
                               [55, 0, 55, 55, 55, 0, 55, 15, 0,
                                0, 0, 0],
                               [55, 55, 55, 55, 55, 55, 55, 15, 0,
                                0, 0, 0],
                               [55, 55, 55, 55, 55, 55, 55, 15, 55,
                                55, 0, 0],
                               [55, 55, 55, 55, 55, 55, 55, 15, 15,
                                55, 0, 0],
                               [55, 55, 0, 0, 0, 55, 55, 0, 55,
                                55, 0, 0],
                               [55, 55, 55, 55, 55, 55, 55, 0, 0,
                                0, 0, 0]],
                              dtype=np.uint8)

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, expected_2, area_opening, connectivity=2)
        _full_type_test(img, 4, expected_4, area_opening, connectivity=2)

        P, S = max_tree(img, connectivity=2)
        _full_type_test(img, 4, expected_4, area_opening,
                        parent=P, tree_traverser=S)

    def test_diameter_closing(self):
        "Test for Diameter Opening (2 thresholds, all types)"
        img = np.array([[97, 95, 93, 92, 91, 90, 90, 90, 91, 92, 93, 95],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93],
                        [93, 63, 63, 63, 63, 86, 86, 86, 87, 43, 43, 91],
                        [92, 89, 88, 86, 85, 85, 84, 85, 85, 43, 43, 89],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [90, 88, 86, 84, 83, 83, 82, 83, 83, 84, 86, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [92, 89, 23, 23, 85, 85, 84, 85, 85, 3, 3, 89],
                        [93, 91, 23, 23, 87, 86, 86, 86, 87, 88, 3, 91],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93]],
                       dtype=np.uint8)

        ex2 = np.array([[97, 95, 93, 92, 91, 90, 90, 90, 91, 92, 93, 95],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93],
                        [93, 63, 63, 63, 63, 86, 86, 86, 87, 43, 43, 91],
                        [92, 89, 88, 86, 85, 85, 84, 85, 85, 43, 43, 89],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [90, 88, 86, 84, 83, 83, 83, 83, 83, 84, 86, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [92, 89, 23, 23, 85, 85, 84, 85, 85, 3, 3, 89],
                        [93, 91, 23, 23, 87, 86, 86, 86, 87, 88, 3, 91],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93]],
                       dtype=np.uint8)

        ex4 = np.array([[97, 95, 93, 92, 91, 90, 90, 90, 91, 92, 93, 95],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93],
                        [93, 63, 63, 63, 63, 86, 86, 86, 87, 84, 84, 91],
                        [92, 89, 88, 86, 85, 85, 84, 85, 85, 84, 84, 89],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [90, 88, 86, 84, 83, 83, 83, 83, 83, 84, 86, 88],
                        [90, 88, 86, 85, 84, 83, 83, 83, 84, 85, 86, 88],
                        [91, 88, 87, 85, 84, 84, 83, 84, 84, 85, 87, 88],
                        [92, 89, 84, 84, 85, 85, 84, 85, 85, 84, 84, 89],
                        [93, 91, 84, 84, 87, 86, 86, 86, 87, 88, 84, 91],
                        [95, 93, 91, 89, 88, 88, 88, 88, 88, 89, 91, 93]],
                       dtype=np.uint8)

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, ex2, diameter_closing, connectivity=2)
        _full_type_test(img, 4, ex4, diameter_closing, connectivity=2)

        P, S = max_tree(invert(img), connectivity=2)
        _full_type_test(img, 4, ex4, diameter_opening,
                        parent=P, tree_traverser=S)

    def test_diameter_opening(self):
        "Test for Diameter Opening (2 thresholds, all types)"
        img = np.array([[5, 7, 9, 11, 12, 12, 12, 12, 12, 11, 9, 7],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10],
                        [9, 40, 40, 40, 40, 16, 16, 16, 16, 60, 60, 11],
                        [11, 13, 15, 16, 17, 18, 18, 18, 17, 60, 60, 13],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 15, 16, 18, 19, 19, 20, 19, 19, 18, 16, 15],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [11, 13, 80, 80, 17, 18, 18, 18, 17, 100, 100, 13],
                        [9, 11, 80, 80, 16, 16, 16, 16, 16, 15, 100, 11],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10]])

        ex2 = np.array([[5, 7, 9, 11, 12, 12, 12, 12, 12, 11, 9, 7],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10],
                        [9, 40, 40, 40, 40, 16, 16, 16, 16, 60, 60, 11],
                        [11, 13, 15, 16, 17, 18, 18, 18, 17, 60, 60, 13],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 15, 16, 18, 19, 19, 19, 19, 19, 18, 16, 15],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [11, 13, 80, 80, 17, 18, 18, 18, 17, 100, 100, 13],
                        [9, 11, 80, 80, 16, 16, 16, 16, 16, 15, 100, 11],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10]])

        ex4 = np.array([[5, 7, 9, 11, 12, 12, 12, 12, 12, 11, 9, 7],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10],
                        [9, 40, 40, 40, 40, 16, 16, 16, 16, 18, 18, 11],
                        [11, 13, 15, 16, 17, 18, 18, 18, 17, 18, 18, 13],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 15, 16, 18, 19, 19, 19, 19, 19, 18, 16, 15],
                        [12, 14, 16, 18, 19, 19, 19, 19, 19, 18, 16, 14],
                        [12, 14, 16, 17, 18, 19, 19, 19, 18, 17, 16, 14],
                        [11, 13, 18, 18, 17, 18, 18, 18, 17, 18, 18, 13],
                        [9, 11, 18, 18, 16, 16, 16, 16, 16, 15, 18, 11],
                        [7, 10, 11, 13, 14, 14, 15, 14, 14, 13, 11, 10]])

        # _full_type_test makes a test with many image types.
        _full_type_test(img, 2, ex2, diameter_opening, connectivity=2)
        _full_type_test(img, 4, ex4, diameter_opening, connectivity=2)

        P, S = max_tree(img, connectivity=2)
        _full_type_test(img, 4, ex4, diameter_opening,
                        parent=P, tree_traverser=S)

    def test_local_maxima(self):
        "local maxima for various data types"
        data = np.array([[10, 11, 13, 14, 14, 15, 14, 14, 13, 11],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13],
                         [13, 15, 40, 40, 18, 18, 18, 60, 60, 15],
                         [14, 16, 40, 40, 19, 19, 19, 60, 60, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [15, 16, 18, 19, 19, 20, 19, 19, 18, 16],
                         [14, 16, 18, 19, 19, 19, 19, 19, 18, 16],
                         [14, 16, 80, 80, 19, 19, 19, 100, 100, 16],
                         [13, 15, 80, 80, 18, 18, 18, 100, 100, 15],
                         [11, 13, 15, 16, 16, 16, 16, 16, 15, 13]],
                        dtype=np.uint8)
        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint64)
        for dtype in [np.uint8, np.uint64, np.int8, np.int64]:

            test_data = data.astype(dtype)
            out = max_tree_local_maxima(test_data, connectivity=1)
            out_bin = out > 0
            assert_array_equal(expected_result, out_bin)
            assert out.dtype == expected_result.dtype
            assert np.max(out) == 5

            P, S = max_tree(test_data)
            out = max_tree_local_maxima(test_data,
                                        parent=P,
                                        tree_traverser=S)

            assert_array_equal(expected_result, out_bin)

            assert out.dtype == expected_result.dtype
            assert np.max(out) == 5

    def test_extrema_float(self):
        "specific tests for float type"
        data = np.array([[0.10, 0.11, 0.13, 0.14, 0.14, 0.15, 0.14,
                          0.14, 0.13, 0.11],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13],
                         [0.13, 0.15, 0.40, 0.40, 0.18, 0.18, 0.18,
                          0.60, 0.60, 0.15],
                         [0.14, 0.16, 0.40, 0.40, 0.19, 0.19, 0.19,
                          0.60, 0.60, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.15, 0.182, 0.18, 0.19, 0.204, 0.20, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.18, 0.19, 0.19, 0.19, 0.19,
                          0.19, 0.18, 0.16],
                         [0.14, 0.16, 0.80, 0.80, 0.19, 0.19, 0.19,
                          4.0, 1.0, 0.16],
                         [0.13, 0.15, 0.80, 0.80, 0.18, 0.18, 0.18,
                          1.0, 1.0, 0.15],
                         [0.11, 0.13, 0.15, 0.16, 0.16, 0.16, 0.16,
                          0.16, 0.15, 0.13]],
                        dtype=np.float32)

        expected_result = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                   dtype=np.uint8)

        # test for local maxima
        out = max_tree_local_maxima(data, connectivity=1)
        out_bin = out > 0
        assert_array_equal(expected_result, out_bin)
        assert np.max(out) == 6

    def test_3d(self):
        """tests the detection of maxima in 3D."""
        img = np.zeros((8, 8, 8), dtype=np.uint8)
        local_maxima = np.zeros((8, 8, 8), dtype=np.uint64)

        # first maximum: only one pixel
        img[1, 1:3, 1:3] = 100
        img[2, 2, 2] = 200
        img[3, 1:3, 1:3] = 100
        local_maxima[2, 2, 2] = 1

        # second maximum: three pixels in z-direction
        img[5:8, 1, 1] = 200
        local_maxima[5:8, 1, 1] = 1

        # third: two maxima in 0 and 3.
        img[0, 5:8, 5:8] = 200
        img[1, 6, 6] = 100
        img[2, 5:7, 5:7] = 200
        img[0:3, 5:8, 5:8] += 50
        local_maxima[0, 5:8, 5:8] = 1
        local_maxima[2, 5:7, 5:7] = 1

        # four : one maximum in the corner of the square
        img[6:8, 6:8, 6:8] = 200
        img[7, 7, 7] = 255
        local_maxima[7, 7, 7] = 1

        out = max_tree_local_maxima(img)
        out_bin = out > 0
        assert_array_equal(local_maxima, out_bin)
        assert np.max(out) == 5
