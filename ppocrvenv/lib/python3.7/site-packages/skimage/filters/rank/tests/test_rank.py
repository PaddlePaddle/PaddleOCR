import numpy as np
import pytest

from skimage import data, morphology, util
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (assert_allclose, assert_array_almost_equal,
                                     assert_equal, fetch, test_parallel)
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import subtract_mean
from skimage.morphology import ball, disk, gray
from skimage.util import img_as_float, img_as_ubyte


def test_otsu_edge_case():
    # This is an edge case that causes OTSU to appear to misbehave
    # Pixel [1, 1] may take a value of of 41 or 81. Both should be considered
    # valid. The value will change depending on the particular implementation
    # of OTSU.
    # To better understand, see
    # https://mybinder.org/v2/gist/hmaarrfk/4afae1cfded1d78e44c9e4f58285d552/master

    footprint = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)

    img = np.array([[  0,  41,   0],
                    [ 30,  81, 106],
                    [  0, 147,   0]], dtype=np.uint8)

    result = rank.otsu(img, footprint)
    assert result[1, 1] in [41, 81]

    img = np.array([[  0, 214,   0],
                    [229, 104, 141],
                    [  0, 172,   0]], dtype=np.uint8)
    result = rank.otsu(img, footprint)
    assert result[1, 1] in [141, 172]


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_subtract_mean_underflow_correction(dtype):
    # Input: [10, 10, 10]
    footprint = np.ones((1, 3))
    arr = np.array([[10, 10, 10]], dtype=dtype)
    result = subtract_mean(arr, footprint)

    if dtype == np.uint8:
        expected_val = 127
    else:
        expected_val = (arr.max() + 1) // 2 - 1

    assert np.all(result == expected_val)


# Note: Explicitly read all values into a dict. Otherwise, stochastic test
#       failures related to I/O can occur during parallel test cases.
ref_data = dict(np.load(fetch("data/rank_filter_tests.npz")))
ref_data_3d = dict(np.load(fetch('data/rank_filters_tests_3d.npz')))


class TestRank():
    def setup(self):
        np.random.seed(0)
        # This image is used along with @test_parallel
        # to ensure that the same seed is used for each thread.
        self.image = np.random.rand(25, 25)
        np.random.seed(0)
        self.volume = np.random.rand(10, 10, 10)
        # Set again the seed for the other tests.
        np.random.seed(0)
        self.footprint = morphology.disk(1)
        self.footprint_3d = morphology.ball(1)
        self.refs = ref_data
        self.refs_3d = ref_data_3d

    @pytest.mark.parametrize('outdt', [None, np.float32, np.float64])
    @pytest.mark.parametrize('filter', all_rank_filters)
    def test_rank_filter(self, filter, outdt):
        @test_parallel(warnings_matching=['Possible precision loss'])
        def check():
            expected = self.refs[filter]
            if outdt is not None:
                out = np.zeros_like(expected, dtype=outdt)
            else:
                out = None
            result = getattr(rank, filter)(self.image, self.footprint, out=out)
            if filter == "entropy":
                # There may be some arch dependent rounding errors
                # See the discussions in
                # https://github.com/scikit-image/scikit-image/issues/3091
                # https://github.com/scikit-image/scikit-image/issues/2528
                if outdt is not None:
                    # Adjust expected precision
                    expected = expected.astype(outdt)
                assert_allclose(expected, result, atol=0, rtol=1E-15)
            elif filter == "otsu":
                # OTSU May also have some optimization dependent failures
                # See the discussions in
                # https://github.com/scikit-image/scikit-image/issues/3091
                # Pixel 3, 5 was found to be problematic. It can take either
                # a value of 41 or 81 depending on the specific optimizations
                # used.
                assert result[3, 5] in [41, 81]
                result[3, 5] = 81
                # Pixel [19, 18] is also found to be problematic for the same
                # reason.
                assert result[19, 18] in [141, 172]
                result[19, 18] = 172
                assert_array_almost_equal(expected, result)
            else:
                if outdt is not None:
                    # Avoid rounding issues comparing to expected result.
                    # Take modulus first to avoid undefined behavior for
                    # float->uint8 conversions.
                    result = np.mod(result, 256.0).astype(expected.dtype)
                assert_array_almost_equal(expected, result)

        check()

    @pytest.mark.parametrize('filter', all_rank_filters)
    def test_rank_filter_selem_kwarg_deprecation(self, filter):
        with expected_warnings(["`selem` is a deprecated argument name"]):
            getattr(rank, filter)(self.image.astype(np.uint8),
                                  selem=self.footprint)

    @pytest.mark.parametrize('outdt', [None, np.float32, np.float64])
    @pytest.mark.parametrize(
        'filter', ['equalize', 'otsu', 'autolevel', 'gradient',
                   'majority', 'maximum', 'mean', 'geometric_mean',
                   'subtract_mean', 'median', 'minimum', 'modal',
                   'enhance_contrast', 'pop', 'sum', 'threshold',
                   'noise_filter', 'entropy']
    )
    def test_rank_filters_3D(self, filter, outdt):
        @test_parallel(warnings_matching=['Possible precision loss'])
        def check():
            expected = self.refs_3d[filter]
            if outdt is not None:
                out = np.zeros_like(expected, dtype=outdt)
            else:
                out = None
            result = getattr(rank, filter)(self.volume, self.footprint_3d,
                                           out=out)
            if outdt is not None:
                # Avoid rounding issues comparing to expected result
                if filter == 'sum':
                    # sum test data seems to be 8-bit disguised as 16-bit
                    datadt = np.uint8
                else:
                    datadt = expected.dtype
                # Take modulus first to avoid undefined behavior for
                # float->uint8 conversions.
                result = np.mod(result, 256.0).astype(datadt)
            assert_array_almost_equal(expected, result)

        check()


    def test_random_sizes(self):
        # make sure the size is not a problem

        elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        for m, n in np.random.randint(1, 101, size=(10, 2)):
            mask = np.ones((m, n), dtype=np.uint8)

            image8 = np.ones((m, n), dtype=np.uint8)
            out8 = np.empty_like(image8)
            rank.mean(image=image8, footprint=elem, mask=mask, out=out8,
                      shift_x=0, shift_y=0)
            assert_equal(image8.shape, out8.shape)
            rank.mean(image=image8, footprint=elem, mask=mask, out=out8,
                      shift_x=+1, shift_y=+1)
            assert_equal(image8.shape, out8.shape)

            rank.geometric_mean(image=image8, footprint=elem, mask=mask,
                                out=out8, shift_x=0, shift_y=0)
            assert_equal(image8.shape, out8.shape)
            rank.geometric_mean(image=image8, footprint=elem, mask=mask,
                                out=out8, shift_x=+1, shift_y=+1)
            assert_equal(image8.shape, out8.shape)

            image16 = np.ones((m, n), dtype=np.uint16)
            out16 = np.empty_like(image8, dtype=np.uint16)
            rank.mean(image=image16, footprint=elem, mask=mask, out=out16,
                      shift_x=0, shift_y=0)
            assert_equal(image16.shape, out16.shape)
            rank.mean(image=image16, footprint=elem, mask=mask, out=out16,
                      shift_x=+1, shift_y=+1)
            assert_equal(image16.shape, out16.shape)

            rank.geometric_mean(image=image16, footprint=elem, mask=mask,
                                out=out16, shift_x=0, shift_y=0)
            assert_equal(image16.shape, out16.shape)
            rank.geometric_mean(image=image16, footprint=elem, mask=mask,
                                out=out16, shift_x=+1, shift_y=+1)
            assert_equal(image16.shape, out16.shape)

            rank.mean_percentile(image=image16, mask=mask, out=out16,
                                 footprint=elem, shift_x=0, shift_y=0, p0=.1,
                                 p1=.9)
            assert_equal(image16.shape, out16.shape)
            rank.mean_percentile(image=image16, mask=mask, out=out16,
                                 footprint=elem, shift_x=+1, shift_y=+1, p0=.1,
                                 p1=.9)
            assert_equal(image16.shape, out16.shape)

    def test_compare_with_gray_dilation(self):
        # compare the result of maximum filter with dilate

        image = (np.random.rand(100, 100) * 256).astype(np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)

        for r in range(3, 20, 2):
            elem = np.ones((r, r), dtype=np.uint8)
            rank.maximum(image=image, footprint=elem, out=out, mask=mask)
            cm = gray.dilation(image, elem)
            assert_equal(out, cm)

    def test_compare_with_gray_erosion(self):
        # compare the result of maximum filter with erode

        image = (np.random.rand(100, 100) * 256).astype(np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)

        for r in range(3, 20, 2):
            elem = np.ones((r, r), dtype=np.uint8)
            rank.minimum(image=image, footprint=elem, out=out, mask=mask)
            cm = gray.erosion(image, elem)
            assert_equal(out, cm)

    def test_bitdepth(self):
        # test the different bit depth for rank16

        elem = np.ones((3, 3), dtype=np.uint8)
        out = np.empty((100, 100), dtype=np.uint16)
        mask = np.ones((100, 100), dtype=np.uint8)

        for i in range(8, 13):
            max_val = 2 ** i - 1
            image = np.full((100, 100), max_val, dtype=np.uint16)
            if i > 10:
                expected = ["Bad rank filter performance"]
            else:
                expected = []
            with expected_warnings(expected):
                rank.mean_percentile(image=image, footprint=elem, mask=mask,
                                     out=out, shift_x=0, shift_y=0, p0=.1,
                                     p1=.9)

    def test_population(self):
        # check the number of valid pixels in the neighborhood

        image = np.zeros((5, 5), dtype=np.uint8)
        elem = np.ones((3, 3), dtype=np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)

        rank.pop(image=image, footprint=elem, out=out, mask=mask)
        r = np.array([[4, 6, 6, 6, 4],
                      [6, 9, 9, 9, 6],
                      [6, 9, 9, 9, 6],
                      [6, 9, 9, 9, 6],
                      [4, 6, 6, 6, 4]])
        assert_equal(r, out)

    def test_structuring_element8(self):
        # check the output for a custom footprint

        r = np.array([[0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 255, 0, 0, 0],
                      [0, 0, 255, 255, 255, 0],
                      [0, 0, 0, 255, 255, 0],
                      [0, 0, 0, 0, 0, 0]])

        # 8-bit
        image = np.zeros((6, 6), dtype=np.uint8)
        image[2, 2] = 255
        elem = np.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)

        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=1, shift_y=1)
        assert_equal(r, out)

        # 16-bit
        image = np.zeros((6, 6), dtype=np.uint16)
        image[2, 2] = 255
        out = np.empty_like(image)

        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=1, shift_y=1)
        assert_equal(r, out)

    def test_pass_on_bitdepth(self):
        # should pass because data bitdepth is not too high for the function

        image = np.full((100, 100), 2 ** 11, dtype=np.uint16)
        elem = np.ones((3, 3), dtype=np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)
        with expected_warnings(["Bad rank filter performance"]):
            rank.maximum(image=image, footprint=elem, out=out, mask=mask)

    def test_inplace_output(self):
        # rank filters are not supposed to filter inplace

        footprint = disk(20)
        image = (np.random.rand(500, 500) * 256).astype(np.uint8)
        out = image
        with pytest.raises(NotImplementedError):
            rank.mean(image, footprint, out=out)

    def test_compare_autolevels(self):
        # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
        # should returns the same arrays

        image = util.img_as_ubyte(data.camera())

        footprint = disk(20)
        loc_autolevel = rank.autolevel(image, footprint=footprint)
        loc_perc_autolevel = rank.autolevel_percentile(image,
                                                       footprint=footprint,
                                                       p0=.0, p1=1.)

        assert_equal(loc_autolevel, loc_perc_autolevel)

    def test_compare_autolevels_16bit(self):
        # compare autolevel(16-bit) and percentile autolevel(16-bit) with
        # p0=0.0 and p1=1.0 should returns the same arrays

        image = data.camera().astype(np.uint16) * 4

        footprint = disk(20)
        loc_autolevel = rank.autolevel(image, footprint=footprint)
        loc_perc_autolevel = rank.autolevel_percentile(image,
                                                       footprint=footprint,
                                                       p0=.0, p1=1.)

        assert_equal(loc_autolevel, loc_perc_autolevel)

    def test_compare_ubyte_vs_float(self):

        # Create signed int8 image that and convert it to uint8
        image_uint = img_as_ubyte(data.camera()[:50, :50])
        image_float = img_as_float(image_uint)

        methods = ['autolevel', 'equalize', 'gradient', 'threshold',
                   'subtract_mean', 'enhance_contrast', 'pop']

        for method in methods:
            func = getattr(rank, method)
            out_u = func(image_uint, disk(3))
            with expected_warnings(["Possible precision loss"]):
                out_f = func(image_float, disk(3))
            assert_equal(out_u, out_f)

    def test_compare_ubyte_vs_float_3d(self):

        # Create signed int8 volume that and convert it to uint8
        np.random.seed(0)
        volume_uint = np.random.randint(0, high=256,
                                        size=(10, 20, 30), dtype=np.uint8)
        volume_float = img_as_float(volume_uint)

        methods_3d = ['equalize', 'otsu', 'autolevel', 'gradient',
                     'majority', 'maximum', 'mean', 'geometric_mean',
                     'subtract_mean', 'median', 'minimum', 'modal',
                     'enhance_contrast', 'pop', 'sum', 'threshold',
                     'noise_filter', 'entropy']

        for method in methods_3d:
            func = getattr(rank, method)
            out_u = func(volume_uint, ball(3))
            with expected_warnings(["Possible precision loss"]):
                out_f = func(volume_float, ball(3))
            assert_equal(out_u, out_f)

    def test_compare_8bit_unsigned_vs_signed(self):
        # filters applied on 8-bit image ore 16-bit image (having only real 8-bit
        # of dynamic) should be identical

        # Create signed int8 image that and convert it to uint8
        image = img_as_ubyte(data.camera())[::2, ::2]
        image[image > 127] = 0
        image_s = image.astype(np.int8)
        image_u = img_as_ubyte(image_s)
        assert_equal(image_u, img_as_ubyte(image_s))

        methods = ['autolevel', 'equalize', 'gradient', 'maximum',
                   'mean', 'geometric_mean', 'subtract_mean', 'median', 'minimum',
                   'modal', 'enhance_contrast', 'pop', 'threshold']

        for method in methods:
            func = getattr(rank, method)
            out_u = func(image_u, disk(3))
            with expected_warnings(["Possible precision loss"]):
                out_s = func(image_s, disk(3))
            assert_equal(out_u, out_s)

    def test_compare_8bit_unsigned_vs_signed_3d(self):
        # filters applied on 8-bit volume ore 16-bit volume (having only real 8-bit
        # of dynamic) should be identical

        # Create signed int8 volume that and convert it to uint8
        np.random.seed(0)
        volume_s = np.random.randint(0, high=127,
                                     size=(10, 20, 30), dtype=np.int8)
        volume_u = img_as_ubyte(volume_s)
        assert_equal(volume_u, img_as_ubyte(volume_s))

        methods_3d = ['equalize', 'otsu', 'autolevel', 'gradient',
                     'majority', 'maximum', 'mean', 'geometric_mean',
                     'subtract_mean', 'median', 'minimum', 'modal',
                     'enhance_contrast', 'pop', 'sum', 'threshold',
                     'noise_filter', 'entropy']

        for method in methods_3d:
            func = getattr(rank, method)
            out_u = func(volume_u, ball(3))
            with expected_warnings(["Possible precision loss"]):
                out_s = func(volume_s, ball(3))
            assert_equal(out_u, out_s)

    @pytest.mark.parametrize(
        'method', ['autolevel', 'equalize', 'gradient', 'maximum',
                   'mean', 'subtract_mean', 'median', 'minimum', 'modal',
                   'enhance_contrast', 'pop', 'threshold']
    )
    def test_compare_8bit_vs_16bit(self, method):
        # filters applied on 8-bit image ore 16-bit image (having only real 8-bit
        # of dynamic) should be identical
        image8 = util.img_as_ubyte(data.camera())[::2, ::2]
        image16 = image8.astype(np.uint16)
        assert_equal(image8, image16)

        np.random.seed(0)
        volume8 = np.random.randint(128, high=256,
                                    size=(10, 10, 10), dtype=np.uint8)
        volume16 = volume8.astype(np.uint16)

        methods_3d = ['equalize', 'otsu', 'autolevel', 'gradient',
                     'majority', 'maximum', 'mean', 'geometric_mean',
                     'subtract_mean', 'median', 'minimum', 'modal',
                     'enhance_contrast', 'pop', 'sum', 'threshold',
                     'noise_filter', 'entropy']

        func = getattr(rank, method)
        f8 = func(image8, disk(3))
        f16 = func(image16, disk(3))
        assert_equal(f8, f16)

        if (method in methods_3d):
            f8 = func(volume8, ball(3))
            f16 = func(volume16, ball(3))

            assert_equal(f8, f16)

    def test_trivial_footprint8(self):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = np.zeros((5, 5), dtype=np.uint8)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        rank.mean(image=image, footprint=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.geometric_mean(image=image, footprint=elem, out=out, mask=mask,
                            shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)

    def test_trivial_footprint16(self):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = np.zeros((5, 5), dtype=np.uint16)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        rank.mean(image=image, footprint=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.geometric_mean(image=image, footprint=elem, out=out, mask=mask,
                            shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)

    def test_smallest_footprint8(self):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = np.zeros((5, 5), dtype=np.uint8)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = np.array([[1]], dtype=np.uint8)
        rank.mean(image=image, footprint=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)

    def test_smallest_footprint16(self):
        # check that min, max and mean returns identity if footprint
        # contains only central pixel

        image = np.zeros((5, 5), dtype=np.uint16)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = np.array([[1]], dtype=np.uint8)
        rank.mean(image=image, footprint=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.geometric_mean(image=image, footprint=elem, out=out, mask=mask,
                            shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(image, out)

    def test_empty_footprint(self):
        # check that min, max and mean returns zeros if footprint is empty

        image = np.zeros((5, 5), dtype=np.uint16)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        res = np.zeros_like(image)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        elem = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8)

        rank.mean(image=image, footprint=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_equal(res, out)
        rank.geometric_mean(image=image, footprint=elem, out=out, mask=mask,
                            shift_x=0, shift_y=0)
        assert_equal(res, out)
        rank.minimum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(res, out)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask,
                     shift_x=0, shift_y=0)
        assert_equal(res, out)

    def test_otsu(self):
        # test the local Otsu segmentation on a synthetic image
        # (left to right ramp * sinus)

        test = np.tile([128, 145, 103, 127, 165, 83, 127, 185, 63, 127, 205, 43,
                        127, 225, 23, 127],
                       (16, 1))
        test = test.astype(np.uint8)
        res = np.tile([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], (16, 1))
        footprint = np.ones((6, 6), dtype=np.uint8)
        th = 1 * (test >= rank.otsu(test, footprint))
        assert_equal(th, res)

    def test_entropy(self):
        #  verify that entropy is coherent with bitdepth of the input data

        footprint = np.ones((16, 16), dtype=np.uint8)
        # 1 bit per pixel
        data = np.tile(np.asarray([0, 1]), (100, 100)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 1)

        # 2 bit per pixel
        data = np.tile(np.asarray([[0, 1], [2, 3]]), (10, 10)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 2)

        # 3 bit per pixel
        data = np.tile(
            np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]), (10, 10)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 3)

        # 4 bit per pixel
        data = np.tile(
            np.reshape(np.arange(16), (4, 4)), (10, 10)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 4)

        # 6 bit per pixel
        data = np.tile(
            np.reshape(np.arange(64), (8, 8)), (10, 10)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 6)

        # 8-bit per pixel
        data = np.tile(
            np.reshape(np.arange(256), (16, 16)), (10, 10)).astype(np.uint8)
        assert(np.max(rank.entropy(data, footprint)) == 8)

        # 12 bit per pixel
        footprint = np.ones((64, 64), dtype=np.uint8)
        data = np.zeros((65, 65), dtype=np.uint16)
        data[:64, :64] = np.reshape(np.arange(4096), (64, 64))
        with expected_warnings(['Bad rank filter performance']):
            assert(np.max(rank.entropy(data, footprint)) == 12)

        # make sure output is of dtype double
        with expected_warnings(['Bad rank filter performance']):
            out = rank.entropy(data, np.ones((16, 16), dtype=np.uint8))
        assert out.dtype == np.double

    def test_footprint_dtypes(self):

        image = np.zeros((5, 5), dtype=np.uint8)
        out = np.zeros_like(image)
        mask = np.ones_like(image, dtype=np.uint8)
        image[2, 2] = 255
        image[2, 3] = 128
        image[1, 2] = 16

        for dtype in (bool, np.uint8, np.uint16, np.int32, np.int64,
                      np.float32, np.float64):
            elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=dtype)
            rank.mean(image=image, footprint=elem, out=out, mask=mask,
                      shift_x=0, shift_y=0)
            assert_equal(image, out)
            rank.geometric_mean(image=image, footprint=elem, out=out,
                                mask=mask, shift_x=0, shift_y=0)
            assert_equal(image, out)
            rank.mean_percentile(image=image, footprint=elem, out=out,
                                 mask=mask, shift_x=0, shift_y=0)
            assert_equal(image, out)

    def test_16bit(self):
        image = np.zeros((21, 21), dtype=np.uint16)
        footprint = np.ones((3, 3), dtype=np.uint8)

        for bitdepth in range(17):
            value = 2 ** bitdepth - 1
            image[10, 10] = value
            if bitdepth >= 11:
                expected = ['Bad rank filter performance']
            else:
                expected = []
            with expected_warnings(expected):
                assert rank.minimum(image, footprint)[10, 10] == 0
                assert rank.maximum(image, footprint)[10, 10] == value
                mean_val = rank.mean(image, footprint)[10, 10]
                assert mean_val == int(value / footprint.size)

    def test_bilateral(self):
        image = np.zeros((21, 21), dtype=np.uint16)
        footprint = np.ones((3, 3), dtype=np.uint8)

        image[10, 10] = 1000
        image[10, 11] = 1010
        image[10, 9] = 900

        kwargs = dict(s0=1, s1=1)
        assert rank.mean_bilateral(image, footprint, **kwargs)[10, 10] == 1000
        assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10] == 1
        kwargs = dict(s0=11, s1=11)
        assert rank.mean_bilateral(image, footprint, **kwargs)[10, 10] == 1005
        assert rank.pop_bilateral(image, footprint, **kwargs)[10, 10] == 2

    def test_percentile_min(self):
        # check that percentile p0 = 0 is identical to local min
        img = data.camera()
        img16 = img.astype(np.uint16)
        footprint = disk(15)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=0)
        img_min = rank.minimum(img, footprint=footprint)
        assert_equal(img_p0, img_min)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=0)
        img_min = rank.minimum(img16, footprint=footprint)
        assert_equal(img_p0, img_min)

    def test_percentile_max(self):
        # check that percentile p0 = 1 is identical to local max
        img = data.camera()
        img16 = img.astype(np.uint16)
        footprint = disk(15)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=1.)
        img_max = rank.maximum(img, footprint=footprint)
        assert_equal(img_p0, img_max)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=1.)
        img_max = rank.maximum(img16, footprint=footprint)
        assert_equal(img_p0, img_max)

    def test_percentile_median(self):
        # check that percentile p0 = 0.5 is identical to local median
        img = data.camera()
        img16 = img.astype(np.uint16)
        footprint = disk(15)
        # check for 8bit
        img_p0 = rank.percentile(img, footprint=footprint, p0=.5)
        img_max = rank.median(img, footprint=footprint)
        assert_equal(img_p0, img_max)
        # check for 16bit
        img_p0 = rank.percentile(img16, footprint=footprint, p0=.5)
        img_max = rank.median(img16, footprint=footprint)
        assert_equal(img_p0, img_max)

    def test_sum(self):
        # check the number of valid pixels in the neighborhood

        image8 = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
        image16 = 400 * np.array([[0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0]], dtype=np.uint16)
        elem = np.ones((3, 3), dtype=np.uint8)
        out8 = np.empty_like(image8)
        out16 = np.empty_like(image16)
        mask = np.ones(image8.shape, dtype=np.uint8)

        r = np.array([[1, 2, 3, 2, 1],
                      [2, 4, 6, 4, 2],
                      [3, 6, 9, 6, 3],
                      [2, 4, 6, 4, 2],
                      [1, 2, 3, 2, 1]], dtype=np.uint8)
        rank.sum(image=image8, footprint=elem, out=out8, mask=mask)
        assert_equal(r, out8)
        rank.sum_percentile(
            image=image8, footprint=elem, out=out8, mask=mask, p0=.0, p1=1.)
        assert_equal(r, out8)
        rank.sum_bilateral(
            image=image8, footprint=elem, out=out8, mask=mask, s0=255, s1=255)
        assert_equal(r, out8)

        r = 400 * np.array([[1, 2, 3, 2, 1],
                            [2, 4, 6, 4, 2],
                            [3, 6, 9, 6, 3],
                            [2, 4, 6, 4, 2],
                            [1, 2, 3, 2, 1]], dtype=np.uint16)
        rank.sum(image=image16, footprint=elem, out=out16, mask=mask)
        assert_equal(r, out16)
        rank.sum_percentile(
            image=image16, footprint=elem, out=out16, mask=mask, p0=.0, p1=1.)
        assert_equal(r, out16)
        rank.sum_bilateral(
            image=image16, footprint=elem, out=out16, mask=mask, s0=1000,
            s1=1000
        )
        assert_equal(r, out16)

    def test_windowed_histogram(self):
        # check the number of valid pixels in the neighborhood

        image8 = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]], dtype=np.uint8)
        elem = np.ones((3, 3), dtype=np.uint8)
        outf = np.empty(image8.shape + (2,), dtype=float)
        mask = np.ones(image8.shape, dtype=np.uint8)

        # Population so we can normalize the expected output while maintaining
        # code readability
        pop = np.array([[4, 6, 6, 6, 4],
                        [6, 9, 9, 9, 6],
                        [6, 9, 9, 9, 6],
                        [6, 9, 9, 9, 6],
                        [4, 6, 6, 6, 4]], dtype=float)

        r0 = np.array([[3, 4, 3, 4, 3],
                       [4, 5, 3, 5, 4],
                       [3, 3, 0, 3, 3],
                       [4, 5, 3, 5, 4],
                       [3, 4, 3, 4, 3]], dtype=float) / pop
        r1 = np.array([[1, 2, 3, 2, 1],
                       [2, 4, 6, 4, 2],
                       [3, 6, 9, 6, 3],
                       [2, 4, 6, 4, 2],
                       [1, 2, 3, 2, 1]], dtype=float) / pop
        rank.windowed_histogram(
            image=image8, footprint=elem, out=outf, mask=mask
        )
        assert_equal(r0, outf[:, :, 0])
        assert_equal(r1, outf[:, :, 1])

        # Test n_bins parameter
        larger_output = rank.windowed_histogram(
            image=image8, footprint=elem, mask=mask, n_bins=5
        )
        assert larger_output.shape[2] == 5

    def test_median_default_value(self):
        a = np.zeros((3, 3), dtype=np.uint8)
        a[1] = 1
        full_footprint = np.ones((3, 3), dtype=np.uint8)
        assert_equal(rank.median(a), rank.median(a, full_footprint))
        assert rank.median(a)[1, 1] == 0
        assert rank.median(a, disk(1))[1, 1] == 1

    def test_majority(self):
        img = data.camera()
        elem = np.ones((3, 3), dtype=np.uint8)
        expected = rank.windowed_histogram(
            img, elem).argmax(-1).astype(np.uint8)
        assert_equal(expected, rank.majority(img, elem))

    def test_output_same_dtype(self):
        image = (np.random.rand(100, 100) * 256).astype(np.uint8)
        out = np.empty_like(image)
        mask = np.ones(image.shape, dtype=np.uint8)
        elem = np.ones((3, 3), dtype=np.uint8)
        rank.maximum(image=image, footprint=elem, out=out, mask=mask)
        assert_equal(image.dtype, out.dtype)

    def test_input_boolean_dtype(self):
        image = (np.random.rand(100, 100) * 256).astype(bool)
        elem = np.ones((3, 3), dtype=bool)
        with pytest.raises(ValueError):
            rank.maximum(image=image, footprint=elem)
