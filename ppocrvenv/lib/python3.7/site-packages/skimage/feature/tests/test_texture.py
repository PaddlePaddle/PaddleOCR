import numpy as np
import pytest

from skimage._shared.testing import expected_warnings, test_parallel
from skimage.feature import (graycomatrix,
                             graycoprops,
                             greycomatrix,
                             greycoprops,
                             local_binary_pattern,
                             multiblock_lbp)
from skimage.transform import integral_image


class TestGLCM():

    def setup(self):
        self.image = np.array([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [0, 2, 2, 2],
                               [2, 2, 3, 3]], dtype=np.uint8)

    @test_parallel()
    def test_output_angles(self):
        result = graycomatrix(
            self.image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 4
        )
        assert result.shape == (4, 4, 1, 4)
        expected1 = np.array([[2, 2, 1, 0],
                             [0, 2, 0, 0],
                             [0, 0, 3, 1],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
        expected2 = np.array([[1, 1, 3, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 1], expected2)
        expected3 = np.array([[3, 0, 2, 0],
                             [0, 2, 2, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 2], expected3)
        expected4 = np.array([[2, 0, 0, 0],
                             [1, 1, 2, 0],
                             [0, 0, 2, 1],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 3], expected4)

    def test_output_symmetric_1(self):
        result = graycomatrix(self.image, [1], [np.pi / 2], 4,
                              symmetric=True)
        assert result.shape == (4, 4, 1, 1)
        expected = np.array([[6, 0, 2, 0],
                             [0, 4, 2, 0],
                             [2, 2, 2, 2],
                             [0, 0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_error_raise_float(self):
        for dtype in [
                float, np.double, np.float16, np.float32, np.float64
                ]:
            with pytest.raises(ValueError):
                graycomatrix(self.image.astype(dtype), [1], [np.pi], 4)

    def test_error_raise_int_types(self):
        for dtype in [
                np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64
                ]:
            with pytest.raises(ValueError):
                graycomatrix(self.image.astype(dtype), [1], [np.pi])

    def test_error_raise_negative(self):
        with pytest.raises(ValueError):
            graycomatrix(self.image.astype(np.int16) - 1, [1], [np.pi], 4)

    def test_error_raise_levels_smaller_max(self):
        with pytest.raises(ValueError):
            graycomatrix(self.image - 1, [1], [np.pi], 3)

    def test_image_data_types(self):
        for dtype in [
            np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64
        ]:
            img = self.image.astype(dtype)
            result = graycomatrix(img, [1], [np.pi / 2], 4,
                                  symmetric=True)
            assert result.shape == (4, 4, 1, 1)
            expected = np.array([[6, 0, 2, 0],
                                 [0, 4, 2, 0],
                                 [2, 2, 2, 2],
                                 [0, 0, 2, 0]], dtype=np.uint32)
            np.testing.assert_array_equal(result[:, :, 0, 0], expected)

        return

    def test_output_distance(self):
        im = np.array([[0, 0, 0, 0],
                       [1, 0, 0, 1],
                       [2, 0, 0, 2],
                       [3, 0, 0, 3]], dtype=np.uint8)
        result = graycomatrix(im, [3], [0], 4, symmetric=False)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_output_combo(self):
        im = np.array([[0],
                       [1],
                       [2],
                       [3]], dtype=np.uint8)
        result = graycomatrix(im, [1, 2], [0, np.pi / 2], 4)
        assert result.shape == (4, 4, 2, 2)

        z = np.zeros((4, 4), dtype=np.uint32)
        e1 = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]], dtype=np.uint32)
        e2 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=np.uint32)

        np.testing.assert_array_equal(result[:, :, 0, 0], z)
        np.testing.assert_array_equal(result[:, :, 1, 0], z)
        np.testing.assert_array_equal(result[:, :, 0, 1], e1)
        np.testing.assert_array_equal(result[:, :, 1, 1], e2)

    def test_output_empty(self):
        result = graycomatrix(self.image, [10], [0], 4)
        np.testing.assert_array_equal(result[:, :, 0, 0],
                                      np.zeros((4, 4), dtype=np.uint32))
        result = graycomatrix(self.image, [10], [0], 4, normed=True)
        np.testing.assert_array_equal(result[:, :, 0, 0],
                                      np.zeros((4, 4), dtype=np.uint32))

    def test_normed_symmetric(self):
        result = graycomatrix(self.image, [1, 2, 3],
                              [0, np.pi / 2, np.pi], 4,
                              normed=True, symmetric=True)
        for d in range(result.shape[2]):
            for a in range(result.shape[3]):
                np.testing.assert_almost_equal(result[:, :, d, a].sum(),
                                               1.0)
                np.testing.assert_array_equal(result[:, :, d, a],
                                              result[:, :, d, a].transpose())

    def test_contrast(self):
        result = graycomatrix(self.image, [1, 2], [0], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        contrast = graycoprops(result, 'contrast')
        np.testing.assert_almost_equal(contrast[0, 0], 0.585, decimal=3)

    def test_dissimilarity(self):
        result = graycomatrix(self.image, [1], [0, np.pi / 2], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = graycoprops(result, 'dissimilarity')
        np.testing.assert_almost_equal(dissimilarity[0, 0], 0.418, decimal=3)

    def test_greycomatrix_and_greycoprops_deprecations(self):
        expected = graycomatrix(self.image, [1], [0, np.pi / 2], 4,
                                normed=True, symmetric=True)
        with expected_warnings(["Function ``greycomatrix``"]):
            result = greycomatrix(self.image, [1], [0, np.pi / 2], 4,
                                  normed=True, symmetric=True)
        np.testing.assert_array_equal(expected, result)

        result = np.round(result, 3)
        dissimilarity_expected = graycoprops(result, 'dissimilarity')
        with expected_warnings(["Function ``greycoprops``"]):
            dissimilarity_result = greycoprops(result, 'dissimilarity')
        np.testing.assert_array_equal(
            dissimilarity_expected, dissimilarity_result
        )

    def test_dissimilarity_2(self):
        result = graycomatrix(self.image, [1, 3], [np.pi / 2], 4,
                              normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = graycoprops(result, 'dissimilarity')[0, 0]
        np.testing.assert_almost_equal(dissimilarity, 0.665, decimal=3)

    def test_non_normalized_glcm(self):
        img = (np.random.random((100, 100)) * 8).astype(np.uint8)
        p = graycomatrix(img, [1, 2, 4, 5], [0, 0.25, 1, 1.5], levels=8)
        np.testing.assert_(np.max(graycoprops(p, 'correlation')) < 1.0)

    def test_invalid_property(self):
        result = graycomatrix(self.image, [1], [0], 4)
        with pytest.raises(ValueError):
            graycoprops(result, 'ABC')

    def test_homogeneity(self):
        result = graycomatrix(self.image, [1], [0, 6], 4, normed=True,
                              symmetric=True)
        homogeneity = graycoprops(result, 'homogeneity')[0, 0]
        np.testing.assert_almost_equal(homogeneity, 0.80833333)

    def test_energy(self):
        result = graycomatrix(self.image, [1], [0, 4], 4, normed=True,
                              symmetric=True)
        energy = graycoprops(result, 'energy')[0, 0]
        np.testing.assert_almost_equal(energy, 0.38188131)

    def test_correlation(self):
        result = graycomatrix(self.image, [1, 2], [0], 4, normed=True,
                              symmetric=True)
        energy = graycoprops(result, 'correlation')
        np.testing.assert_almost_equal(energy[0, 0], 0.71953255)
        np.testing.assert_almost_equal(energy[1, 0], 0.41176470)

    def test_uniform_properties(self):
        im = np.ones((4, 4), dtype=np.uint8)
        result = graycomatrix(im, [1, 2, 8], [0, np.pi / 2], 4, normed=True,
                              symmetric=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity',
                     'energy', 'correlation', 'ASM']:
            graycoprops(result, prop)


class TestLBP():

    def setup(self):
        self.image = np.array([[255,   6, 255,   0,  141,   0],
                               [ 48, 250, 204, 166,  223,  63],
                               [  8,   0, 159,  50,  255,  30],
                               [167, 255,  63,  40,  128, 255],
                               [  0, 255,  30,  34,  255,  24],
                               [146, 241, 255,   0,  189, 126]],
                              dtype='double')

    @test_parallel()
    def test_default(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'default')
        ref = np.array([[  0, 251,   0, 255,  96, 255],
                        [143,   0,  20, 153,  64,  56],
                        [238, 255,  12, 191,   0, 252],
                        [129,  64.,  62, 159, 199,   0],
                        [255,   4, 255, 175,   0, 254],
                        [  3,   5,   0, 255,   4,  24]])
        np.testing.assert_array_equal(lbp, ref)

    def test_ror(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'ror')
        ref = np.array([[  0, 127,   0, 255,   3, 255],
                        [ 31,   0,   5,  51,   1,   7],
                        [119, 255,   3, 127,   0,  63],
                        [  3,   1,  31,  63,  31,   0],
                        [255,   1, 255,  95,   0, 127],
                        [  3,   5,   0, 255,   1,   3]])
        np.testing.assert_array_equal(lbp, ref)

    def test_uniform(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'uniform')
        ref = np.array([[0, 7, 0, 8, 2, 8],
                        [5, 0, 9, 9, 1, 3],
                        [9, 8, 2, 7, 0, 6],
                        [2, 1, 5, 6, 5, 0],
                        [8, 1, 8, 9, 0, 7],
                        [2, 9, 0, 8, 1, 2]])
        np.testing.assert_array_equal(lbp, ref)

    def test_var(self):
        # Test idea: mean of variance is estimate of overall variance.

        # Fix random seed for test stability.
        np.random.seed(13141516)

        # Create random image with known variance.
        image = np.random.rand(500, 500)
        target_std = 0.3
        image = image / image.std() * target_std

        # Use P=4 to avoid interpolation effects
        P, R = 4, 1
        lbp = local_binary_pattern(image, P, R, 'var')

        # Take central part to avoid border effect.
        lbp = lbp[5:-5, 5:-5]

        # The LBP variance is biased (ddof=0), correct for that.
        expected = target_std**2 * (P-1)/P

        np.testing.assert_almost_equal(lbp.mean(), expected, 4)

    def test_nri_uniform(self):
        lbp = local_binary_pattern(self.image, 8, 1, 'nri_uniform')
        ref = np.array([[ 0, 54,  0, 57, 12, 57],
                        [34,  0, 58, 58,  3, 22],
                        [58, 57, 15, 50,  0, 47],
                        [10,  3, 40, 42, 35,  0],
                        [57,  7, 57, 58,  0, 56],
                        [ 9, 58,  0, 57,  7, 14]])
        np.testing.assert_array_almost_equal(lbp, ref)


class TestMBLBP():

    def test_single_mblbp(self):

        # Create dummy matrix where first and fifth rectangles have greater
        # value than the central one. Therefore, the following bits
        # should be 1.
        test_img = np.zeros((9, 9), dtype='uint8')
        test_img[3:6, 3:6] = 1
        test_img[:3, :3] = 255
        test_img[6:, 6:] = 255

        # MB-LBP is filled in reverse order. So the first and fifth bits from
        # the end should be filled.
        correct_answer = 0b10001000

        int_img = integral_image(test_img)

        lbp_code = multiblock_lbp(int_img, 0, 0, 3, 3)

        np.testing.assert_equal(lbp_code, correct_answer)
