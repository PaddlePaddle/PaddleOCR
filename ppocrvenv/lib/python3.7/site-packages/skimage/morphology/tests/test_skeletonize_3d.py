import numpy as np
import scipy.ndimage as ndi

from skimage import io, draw
from skimage.data import binary_blobs
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize, skeletonize_3d

from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch

# basic behavior tests (mostly copied over from 2D skeletonize)

def test_skeletonize_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with testing.raises(ValueError):
        skeletonize(im, method='lee')


def test_skeletonize_1D_old_api():
    # a corner case of an image of a shape(1, N)
    im = np.ones((5, 1), dtype=np.uint8)
    res = skeletonize_3d(im)
    assert_equal(res, im)


def test_skeletonize_1D():
    # a corner case of an image of a shape(1, N)
    im = np.ones((5, 1), dtype=np.uint8)
    res = skeletonize(im, method='lee')
    assert_equal(res, im)


def test_skeletonize_no_foreground():
    im = np.zeros((5, 5), dtype=np.uint8)
    result = skeletonize(im, method='lee')
    assert_equal(result, im)


def test_skeletonize_all_foreground():
    im = np.ones((3, 4), dtype=np.uint8)
    assert_equal(skeletonize(im, method='lee'),
                 np.array([[0, 0, 0, 0],
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]], dtype=np.uint8))


def test_skeletonize_single_point():
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 3] = 1
    result = skeletonize(im, method='lee')
    assert_equal(result, im)


def test_skeletonize_already_thinned():
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 1:-1] = 1
    im[2, -1] = 1
    im[4, 0] = 1
    result = skeletonize(im, method='lee')
    assert_equal(result, im)


def test_dtype_conv():
    # check that the operation does the right thing with floats etc
    # also check non-contiguous input
    img = np.random.random((16, 16))[::2, ::2]
    img[img < 0.5] = 0

    orig = img.copy()
    res = skeletonize(img, method='lee')
    img_max = img_as_ubyte(img).max()

    assert_equal(res.dtype, np.uint8)
    assert_equal(img, orig)  # operation does not clobber the original
    assert_equal(res.max(), img_max)    # the intensity range is preserved


@parametrize("img", [
    np.ones((8, 8), dtype=float), np.ones((4, 8, 8), dtype=float)
])
def test_input_with_warning(img):
    # check that the input is not clobbered
    # for 2D and 3D images of varying dtypes
    check_input(img)


@parametrize("img", [
    np.ones((8, 8), dtype=np.uint8), np.ones((4, 8, 8), dtype=np.uint8),
    np.ones((8, 8), dtype=bool), np.ones((4, 8, 8), dtype=bool)
])
def test_input_without_warning(img):
    # check that the input is not clobbered
    # for 2D and 3D images of varying dtypes
    check_input(img)


def check_input(img):
    orig = img.copy()
    skeletonize(img, method='lee')
    assert_equal(img, orig)


def test_skeletonize_num_neighbours():
    # an empty image
    image = np.zeros((300, 300))

    # foreground object 1
    image[10:-10, 10:100] = 1
    image[-100:-10, 10:-10] = 1
    image[10:-10, -100:-10] = 1

    # foreground object 2
    rs, cs = draw.line(250, 150, 10, 280)
    for i in range(10):
        image[rs + i, cs] = 1
    rs, cs = draw.line(10, 150, 250, 280)
    for i in range(20):
        image[rs + i, cs] = 1

    # foreground object 3
    ir, ic = np.indices(image.shape)
    circle1 = (ic - 135)**2 + (ir - 150)**2 < 30**2
    circle2 = (ic - 135)**2 + (ir - 150)**2 < 20**2
    image[circle1] = 1
    image[circle2] = 0
    result = skeletonize(image, method='lee')

    # there should never be a 2x2 block of foreground pixels in a skeleton
    mask = np.array([[1,  1],
                     [1,  1]], np.uint8)
    blocks = ndi.correlate(result, mask, mode='constant')
    assert_(not np.any(blocks == 4))


def test_two_hole_image():
    # test a simple 2D image against FIJI
    img_o = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.uint8)
    img_f = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                     dtype=np.uint8)
    res = skeletonize(img_o, method='lee')
    assert_equal(res, img_f)


def test_3d_vs_fiji():
    # generate an image with blobs and compate its skeleton to
    # the skeleton generated by FIJI (Plugins>Skeleton->Skeletonize)
    img = binary_blobs(32, 0.05, n_dim=3, seed=1234)
    img = img[:-2, ...]
    img = img.astype(np.uint8)*255

    img_s = skeletonize(img)
    img_f = io.imread(fetch("data/_blobs_3d_fiji_skeleton.tif"))
    assert_equal(img_s, img_f)
