import numpy as np

from skimage import data, img_as_float
from skimage._shared import testing
from skimage._shared.testing import assert_allclose, expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error, normalized_root_mse
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint


@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
@testing.parametrize('split_into_regions', [False, True])
def test_inpaint_biharmonic_2d(dtype, split_into_regions):
    img = np.tile(np.square(np.linspace(0, 1, 5, dtype=dtype)), (5, 1))
    mask = np.zeros_like(img)
    mask[2, 2:] = 1
    mask[1, 3:] = 1
    mask[0, 4:] = 1
    img[np.where(mask)] = 0
    out = inpaint.inpaint_biharmonic(img, mask,
                                     split_into_regions=split_into_regions)
    assert out.dtype == _supported_float_type(img)

    ref = np.array(
        [[0., 0.0625, 0.25000000, 0.5625000, 0.73925058],
         [0., 0.0625, 0.25000000, 0.5478048, 0.76557821],
         [0., 0.0625, 0.25842878, 0.5623079, 0.85927796],
         [0., 0.0625, 0.25000000, 0.5625000, 1.00000000],
         [0., 0.0625, 0.25000000, 0.5625000, 1.00000000]]
    )
    rtol = 1e-7 if dtype == np.float64 else 1e-6
    assert_allclose(ref, out, rtol=rtol)


@testing.parametrize('channel_axis', [0, 1, -1])
def test_inpaint_biharmonic_2d_color(channel_axis):
    img = img_as_float(data.astronaut()[:64, :64])

    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[8:16, :16] = 1
    img_defect = img * ~mask[..., np.newaxis]
    mse_defect = mean_squared_error(img, img_defect)

    img_defect = np.moveaxis(img_defect, -1, channel_axis)
    img_restored = inpaint.inpaint_biharmonic(img_defect, mask,
                                              channel_axis=channel_axis)
    img_restored = np.moveaxis(img_restored, channel_axis, -1)
    mse_restored = mean_squared_error(img, img_restored)

    assert mse_restored < 0.01 * mse_defect


@testing.parametrize('dtype', [np.float32, np.float64])
def test_inpaint_biharmonic_2d_float_dtypes(dtype):
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    mask = np.zeros_like(img)
    mask[2, 2:] = 1
    mask[1, 3:] = 1
    mask[0, 4:] = 1
    img[np.where(mask)] = 0
    img = img.astype(dtype, copy=False)
    out = inpaint.inpaint_biharmonic(img, mask)
    assert out.dtype == img.dtype
    ref = np.array(
        [[0., 0.0625, 0.25000000, 0.5625000, 0.73925058],
         [0., 0.0625, 0.25000000, 0.5478048, 0.76557821],
         [0., 0.0625, 0.25842878, 0.5623079, 0.85927796],
         [0., 0.0625, 0.25000000, 0.5625000, 1.00000000],
         [0., 0.0625, 0.25000000, 0.5625000, 1.00000000]]
    )
    assert_allclose(ref, out, rtol=1e-5)


def test_inpaint_biharmonic_2d_color_deprecated():
    img = img_as_float(data.astronaut()[:64, :64])

    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[8:16, :16] = 1
    img_defect = img * ~mask[..., np.newaxis]
    mse_defect = mean_squared_error(img, img_defect)

    # providing multichannel argument positionally also warns
    channel_warning = "`multichannel` is a deprecated argument"
    matrix_warning = "the matrix subclass is not the recommended way"
    with expected_warnings([channel_warning + '|' + matrix_warning]):
        img_restored = inpaint.inpaint_biharmonic(img_defect, mask,
                                                  multichannel=True)
    mse_restored = mean_squared_error(img, img_restored)

    assert mse_restored < 0.01 * mse_defect

    # providing multichannel argument positionally also warns
    channel_warning = "Providing the `multichannel` argument"
    with expected_warnings([channel_warning + '|' + matrix_warning]):
        img_restored = inpaint.inpaint_biharmonic(img_defect, mask, True)
    mse_restored = mean_squared_error(img, img_restored)

    assert mse_restored < 0.01 * mse_defect


@testing.parametrize('split_into_regions', [False, True])
def test_inpaint_biharmonic_3d(split_into_regions):
    img = np.tile(np.square(np.linspace(0, 1, 5)), (5, 1))
    img = np.dstack((img, img.T))
    mask = np.zeros_like(img)
    mask[2, 2:, :] = 1
    mask[1, 3:, :] = 1
    mask[0, 4:, :] = 1
    img[np.where(mask)] = 0
    out = inpaint.inpaint_biharmonic(img, mask,
                                     split_into_regions=split_into_regions)
    ref = np.dstack((
        np.array(
            [[0.0000, 0.0625, 0.25000000, 0.56250000, 0.53752796],
             [0.0000, 0.0625, 0.25000000, 0.44443780, 0.53762210],
             [0.0000, 0.0625, 0.23693666, 0.46621112, 0.68615592],
             [0.0000, 0.0625, 0.25000000, 0.56250000, 1.00000000],
             [0.0000, 0.0625, 0.25000000, 0.56250000, 1.00000000]]),
        np.array(
            [[0.0000, 0.0000, 0.00000000, 0.00000000, 0.19621902],
             [0.0625, 0.0625, 0.06250000, 0.17470756, 0.30140091],
             [0.2500, 0.2500, 0.27241289, 0.35155440, 0.43068654],
             [0.5625, 0.5625, 0.56250000, 0.56250000, 0.56250000],
             [1.0000, 1.0000, 1.00000000, 1.00000000, 1.00000000]])
    ))
    assert_allclose(ref, out)


def test_invalid_input():
    img, mask = np.zeros([]), np.zeros([])
    with testing.raises(ValueError):
        inpaint.inpaint_biharmonic(img, mask)

    img, mask = np.zeros((2, 2)), np.zeros((4, 1))
    with testing.raises(ValueError):
        inpaint.inpaint_biharmonic(img, mask)

    img = np.ma.array(np.zeros((2, 2)), mask=[[0, 0], [0, 0]])
    mask = np.zeros((2, 2))
    with testing.raises(TypeError):
        inpaint.inpaint_biharmonic(img, mask)


@testing.parametrize('dtype', [np.uint8, np.float32, np.float64])
@testing.parametrize('channel_axis', [None, -1])
@testing.parametrize('split_into_regions', [False, True])
def test_inpaint_nrmse(dtype, channel_axis, split_into_regions):
    image_orig = data.astronaut()[:, :200]
    float_dtype = np.float32 if dtype == np.float32 else np.float64
    image_orig = image_orig.astype(float_dtype, copy=False)

    # Create mask with six block defect regions
    mask = np.zeros(image_orig.shape[:-1], dtype=bool)
    mask[20:50, 3:20] = 1
    mask[165:180, 90:155] = 1
    mask[40:60, 170:195] = 1
    mask[-60:-40, 170:195] = 1
    mask[-180:-165, 90:155] = 1
    mask[-50:-20, :20] = 1

    # add a few long, narrow defects
    mask[200:205, -200:] = 1
    mask[150:255, 20:22] = 1
    mask[365:368, 60:130] = 1

    # add randomly positioned small point-like defects
    rstate = np.random.default_rng(0)
    for radius in [0, 2, 4]:
        # larger defects are less common
        thresh = 3.25 + 0.25 * radius  # larger defects less commmon
        tmp_mask = rstate.standard_normal(image_orig.shape[:-1]) > thresh
        if radius > 0:
            tmp_mask = binary_dilation(tmp_mask, disk(radius, dtype=bool))
        mask[tmp_mask] = 1

    # Defect image over the same region in each color channel
    image_defect = image_orig.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 0

    if channel_axis is None:
        image_orig = rgb2gray(image_orig)
        image_defect = rgb2gray(image_defect)

    image_orig = image_orig.astype(dtype, copy=False)
    image_defect = image_defect.astype(dtype, copy=False)

    image_result = inpaint.inpaint_biharmonic(
        image_defect, mask, channel_axis=channel_axis,
        split_into_regions=split_into_regions
    )
    assert image_result.dtype == float_dtype

    nrmse_defect = normalized_root_mse(image_orig, image_defect)
    nrmse_result = normalized_root_mse(img_as_float(image_orig), image_result)
    assert nrmse_result < 0.2 * nrmse_defect
