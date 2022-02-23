
import numpy as np
from skimage.restoration import unwrap_phase
import sys

from skimage._shared import testing
from skimage._shared.testing import (assert_array_almost_equal_nulp,
                                     assert_almost_equal, assert_array_equal,
                                     assert_, skipif)
from skimage._shared._warnings import expected_warnings


def assert_phase_almost_equal(a, b, *args, **kwargs):
    """An assert_almost_equal insensitive to phase shifts of n*2*pi."""
    shift = 2 * np.pi * np.round((b.mean() - a.mean()) / (2 * np.pi))
    with expected_warnings([r'invalid value encountered|\A\Z',
                            r'divide by zero encountered|\A\Z']):
        print('assert_phase_allclose, abs', np.max(np.abs(a - (b - shift))))
        print('assert_phase_allclose, rel',
              np.max(np.abs((a - (b - shift)) / a)))
    if np.ma.isMaskedArray(a):
        assert_(np.ma.isMaskedArray(b))
        assert_array_equal(a.mask, b.mask)
        assert_(a.fill_value == b.fill_value)
        au = np.asarray(a)
        bu = np.asarray(b)
        with expected_warnings([r'invalid value encountered|\A\Z',
                                r'divide by zero encountered|\A\Z']):
            print('assert_phase_allclose, no mask, abs',
                  np.max(np.abs(au - (bu - shift))))
            print('assert_phase_allclose, no mask, rel',
                  np.max(np.abs((au - (bu - shift)) / au)))
    assert_array_almost_equal_nulp(a + shift, b, *args, **kwargs)


def check_unwrap(image, mask=None):
    image_wrapped = np.angle(np.exp(1j * image))
    if mask is not None:
        print('Testing a masked image')
        image = np.ma.array(image, mask=mask, fill_value=0.5)
        image_wrapped = np.ma.array(image_wrapped, mask=mask, fill_value=0.5)
    image_unwrapped = unwrap_phase(image_wrapped, seed=0)
    assert_phase_almost_equal(image_unwrapped, image)


def test_unwrap_1d():
    image = np.linspace(0, 10 * np.pi, 100)
    check_unwrap(image)
    # Masked arrays are not allowed in 1D
    with testing.raises(ValueError):
        check_unwrap(image, True)
    # wrap_around is not allowed in 1D
    with testing.raises(ValueError):
        unwrap_phase(image, True, seed=0)


@testing.parametrize("check_with_mask", (False, True))
def test_unwrap_2d(check_with_mask):
    mask = None
    x, y = np.ogrid[:8, :16]
    image = 2 * np.pi * (x * 0.2 + y * 0.1)
    if check_with_mask:
        mask = np.zeros(image.shape, dtype=bool)
        mask[4:6, 4:8] = True
    check_unwrap(image, mask)


@testing.parametrize("check_with_mask", (False, True))
def test_unwrap_3d(check_with_mask):
    mask = None
    x, y, z = np.ogrid[:8, :12, :16]
    image = 2 * np.pi * (x * 0.2 + y * 0.1 + z * 0.05)
    if check_with_mask:
        mask = np.zeros(image.shape, dtype=bool)
        mask[4:6, 4:6, 1:3] = True
    check_unwrap(image, mask)


def check_wrap_around(ndim, axis):
    # create a ramp, but with the last pixel along axis equalling the first
    elements = 100
    ramp = np.linspace(0, 12 * np.pi, elements)
    ramp[-1] = ramp[0]
    image = ramp.reshape(tuple([elements if n == axis else 1
                                for n in range(ndim)]))
    image_wrapped = np.angle(np.exp(1j * image))

    index_first = tuple([0] * ndim)
    index_last = tuple([-1 if n == axis else 0 for n in range(ndim)])
    # unwrap the image without wrap around
    # We do not want warnings about length 1 dimensions
    with expected_warnings([r'Image has a length 1 dimension|\A\Z']):
        image_unwrap_no_wrap_around = unwrap_phase(image_wrapped, seed=0)
    print('endpoints without wrap_around:',
          image_unwrap_no_wrap_around[index_first],
          image_unwrap_no_wrap_around[index_last])
    # without wrap around, the endpoints of the image should differ
    assert_(abs(image_unwrap_no_wrap_around[index_first] -
                image_unwrap_no_wrap_around[index_last]) > np.pi)
    # unwrap the image with wrap around
    wrap_around = [n == axis for n in range(ndim)]
    # We do not want warnings about length 1 dimensions
    with expected_warnings([r'Image has a length 1 dimension.|\A\Z']):
        image_unwrap_wrap_around = unwrap_phase(image_wrapped, wrap_around,
                                                seed=0)
    print('endpoints with wrap_around:',
          image_unwrap_wrap_around[index_first],
          image_unwrap_wrap_around[index_last])
    # with wrap around, the endpoints of the image should be equal
    assert_almost_equal(image_unwrap_wrap_around[index_first],
                        image_unwrap_wrap_around[index_last])


dim_axis = [(ndim, axis) for ndim in (2, 3) for axis in range(ndim)]


@skipif(sys.version_info[:2] == (3, 4),
        reason="Doesn't work with python 3.4. See issue #3079")
@testing.parametrize("ndim, axis", dim_axis)
def test_wrap_around(ndim, axis):
    check_wrap_around(ndim, axis)


def test_mask():
    length = 100
    ramps = [np.linspace(0, 4 * np.pi, length),
             np.linspace(0, 8 * np.pi, length),
             np.linspace(0, 6 * np.pi, length)]
    image = np.vstack(ramps)
    mask_1d = np.ones((length,), dtype=bool)
    mask_1d[0] = mask_1d[-1] = False
    for i in range(len(ramps)):
        # mask all ramps but the i'th one
        mask = np.zeros(image.shape, dtype=bool)
        mask |= mask_1d.reshape(1, -1)
        mask[i, :] = False   # unmask i'th ramp
        image_wrapped = np.ma.array(np.angle(np.exp(1j * image)), mask=mask)
        image_unwrapped = unwrap_phase(image_wrapped)
        image_unwrapped -= image_unwrapped[0, 0]    # remove phase shift
        # The end of the unwrapped array should have value equal to the
        # endpoint of the unmasked ramp
        assert_array_almost_equal_nulp(image_unwrapped[:, -1], image[i, -1])
        assert_(np.ma.isMaskedArray(image_unwrapped))

        # Same tests, but forcing use of the 3D unwrapper by reshaping
        with expected_warnings(['length 1 dimension']):
            shape = (1,) + image_wrapped.shape
            image_wrapped_3d = image_wrapped.reshape(shape)
            image_unwrapped_3d = unwrap_phase(image_wrapped_3d)
            # remove phase shift
            image_unwrapped_3d -= image_unwrapped_3d[0, 0, 0]
        assert_array_almost_equal_nulp(image_unwrapped_3d[:, :, -1],
                                       image[i, -1])


def test_invalid_input():
    with testing.raises(ValueError):
        unwrap_phase(np.zeros([]))
    with testing.raises(ValueError):
        unwrap_phase(np.zeros((1, 1, 1, 1)))
    with testing.raises(ValueError):
        unwrap_phase(np.zeros((1, 1)), 3 * [False])
    with testing.raises(ValueError):
        unwrap_phase(np.zeros((1, 1)), 'False')


def test_unwrap_3d_middle_wrap_around():
    # Segmentation fault in 3D unwrap phase with middle dimension connected
    # GitHub issue #1171
    image = np.zeros((20, 30, 40), dtype=np.float32)
    unwrap = unwrap_phase(image, wrap_around=[False, True, False])
    assert_(np.all(unwrap == 0))


def test_unwrap_2d_compressed_mask():
    # ValueError when image is masked array with a compressed mask (no masked
    # elements).  GitHub issue #1346
    image = np.ma.zeros((10, 10))
    unwrap = unwrap_phase(image)
    assert_(np.all(unwrap == 0))


def test_unwrap_2d_all_masked():
    # Segmentation fault when image is masked array with a all elements masked
    # GitHub issue #1347
    # all elements masked
    image = np.ma.zeros((10, 10))
    image[:] = np.ma.masked
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.all(unwrap.mask))

    # 1 unmasked element, still zero edges
    image = np.ma.zeros((10, 10))
    image[:] = np.ma.masked
    image[0, 0] = 0
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.sum(unwrap.mask) == 99)    # all but one masked
    assert_(unwrap[0, 0] == 0)


def test_unwrap_3d_all_masked():
    # all elements masked
    image = np.ma.zeros((10, 10, 10))
    image[:] = np.ma.masked
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.all(unwrap.mask))

    # 1 unmasked element, still zero edges
    image = np.ma.zeros((10, 10, 10))
    image[:] = np.ma.masked
    image[0, 0, 0] = 0
    unwrap = unwrap_phase(image)
    assert_(np.ma.isMaskedArray(unwrap))
    assert_(np.sum(unwrap.mask) == 999)   # all but one masked
    assert_(unwrap[0, 0, 0] == 0)
