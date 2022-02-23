import numpy as np
import scipy
from packaging import version

from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import xfail, arch32
from skimage.segmentation import random_walker
from skimage.transform import resize

# older versions of scipy raise a warning with new NumPy because they use
# numpy.rank() instead of arr.ndim or numpy.linalg.matrix_rank.
SCIPY_RANK_WARNING = r'numpy.linalg.matrix_rank|\A\Z'
PYAMG_MISSING_WARNING = r'pyamg|\A\Z'
PYAMG_OR_SCIPY_WARNING = SCIPY_RANK_WARNING + '|' + PYAMG_MISSING_WARNING

if version.parse(scipy.__version__) < version.parse('1.3'):
    NUMPY_MATRIX_WARNING = 'matrix subclass'
else:
    NUMPY_MATRIX_WARNING = None


def make_2d_syntheticdata(lx, ly=None):
    if ly is None:
        ly = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly)) + 0.1 * np.random.randn(lx, ly)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l,
         ly // 2 - small_l:ly // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1,
         ly // 2 - small_l + 1:ly // 2 + small_l - 1] = (
            0.1 * np.random.randn(2 * small_l - 2, 2 * small_l - 2))
    data[lx // 2 - small_l, ly // 2 - small_l // 8:ly // 2 + small_l // 8] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5] = 1
    seeds[lx // 2 + small_l // 4, ly // 2 - small_l // 4] = 2
    return data, seeds


def make_3d_syntheticdata(lx, ly=None, lz=None):
    if ly is None:
        ly = lx
    if lz is None:
        lz = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly, lz)) + 0.1 * np.random.randn(lx, ly, lz)
    small_l = int(lx // 5)
    data[lx // 2 - small_l:lx // 2 + small_l,
         ly // 2 - small_l:ly // 2 + small_l,
         lz // 2 - small_l:lz // 2 + small_l] = 1
    data[lx // 2 - small_l + 1:lx // 2 + small_l - 1,
         ly // 2 - small_l + 1:ly // 2 + small_l - 1,
         lz // 2 - small_l + 1:lz // 2 + small_l - 1] = 0
    # make a hole
    hole_size = np.max([1, small_l // 8])
    data[lx // 2 - small_l,
         ly // 2 - hole_size:ly // 2 + hole_size,
         lz // 2 - hole_size:lz // 2 + hole_size] = 0
    seeds = np.zeros_like(data)
    seeds[lx // 5, ly // 5, lz // 5] = 1
    seeds[lx // 2 + small_l // 4,
          ly // 2 - small_l // 4,
          lz // 2 - small_l // 4] = 2
    return data, seeds


@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_bf(dtype):
    lx = 70
    ly = 100

    # have to use a smaller beta to avoid warning with lower precision input
    beta = 90 if dtype == np.float64 else 25

    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        labels_bf = random_walker(data, labels, beta=beta, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        full_prob_bf = random_walker(data, labels, beta=beta, mode='bf',
                                     return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    # Now test with more than two labels
    labels[55, 80] = 3
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        full_prob_bf = random_walker(data, labels, beta=beta, mode='bf',
                                     return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert len(full_prob_bf) == 3
    assert data.shape == labels.shape


@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_cg(dtype):
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        labels_cg = random_walker(data, labels, beta=90, mode='cg')
    assert (labels_cg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        full_prob = random_walker(data, labels, beta=90, mode='cg',
                                  return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg


@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_cg_mg(dtype):
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    anticipated_warnings = [
        'scipy.sparse.sparsetools|%s' % PYAMG_OR_SCIPY_WARNING,
        NUMPY_MATRIX_WARNING]
    with expected_warnings(anticipated_warnings):
        labels_cg_mg = random_walker(data, labels, beta=90, mode='cg_mg')
    assert (labels_cg_mg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    with expected_warnings(anticipated_warnings):
        full_prob = random_walker(data, labels, beta=90, mode='cg_mg',
                                  return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg_mg


@testing.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_2d_cg_j(dtype):
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        labels_cg = random_walker(data, labels, beta=90, mode='cg_j')
    assert (labels_cg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        full_prob = random_walker(data, labels, beta=90, mode='cg_j',
                                  return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60]
            >= full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape


def test_types():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = 255 * (data - data.min()) // (data.max() - data.min())
    data = data.astype(np.uint8)
    with expected_warnings([PYAMG_OR_SCIPY_WARNING, NUMPY_MATRIX_WARNING]):
        labels_cg_mg = random_walker(data, labels, beta=90, mode='cg_mg')
    assert (labels_cg_mg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels_cg_mg


def test_reorder_labels():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels[labels == 2] = 4
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        labels_bf = random_walker(data, labels, beta=90, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels_bf


def test_2d_inactive():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels[10:20, 10:20] = -1
    labels[46:50, 33:38] = -2
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        labels = random_walker(data, labels, beta=90)
    assert (labels.reshape((lx, ly))[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels


def test_2d_laplacian_size():
    # test case from: https://github.com/scikit-image/scikit-image/issues/5034
    # The markers here were modified from the ones in the original issue to
    # avoid a singular matrix, but still reproduce the issue.
    data = np.asarray([[12823, 12787, 12710],
                       [12883, 13425, 12067],
                       [11934, 11929, 12309]])
    markers = np.asarray([[0, -1, 2],
                          [0, -1, 0],
                          [1, 0, -1]])
    expected_labels = np.asarray([[1, -1, 2],
                                  [1, -1, 2],
                                  [1, 1, -1]])
    labels = random_walker(data, markers, beta=10)
    np.testing.assert_array_equal(labels, expected_labels)
    return data, labels


@testing.parametrize('dtype', [np.float32, np.float64])
def test_3d(dtype):
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    data = data.astype(dtype, copy=False)
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        labels = random_walker(data, labels, mode='cg')
    assert (labels.reshape(data.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data.shape == labels.shape
    return data, labels


def test_3d_inactive():
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    old_labels = np.copy(labels)
    labels[5:25, 26:29, 26:29] = -1
    after_labels = np.copy(labels)
    with expected_warnings(['"cg" mode|CObject type' + '|'
                            + SCIPY_RANK_WARNING, NUMPY_MATRIX_WARNING]):
        labels = random_walker(data, labels, mode='cg')
    assert (labels.reshape(data.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data.shape == labels.shape
    return data, labels, old_labels, after_labels


@testing.parametrize('channel_axis', [0, 1, -1])
@testing.parametrize('dtype', [np.float32, np.float64])
def test_multispectral_2d(dtype, channel_axis):
    lx, ly = 70, 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data.astype(dtype, copy=False)
    data = data[..., np.newaxis].repeat(2, axis=-1)  # Expect identical output

    data = np.moveaxis(data, -1, channel_axis)
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        multi_labels = random_walker(data, labels, mode='cg',
                                     channel_axis=channel_axis)
    data = np.moveaxis(data, channel_axis, -1)

    assert data[..., 0].shape == labels.shape
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[25:45, 40:60] == 2).all()
    assert data[..., 0].shape == labels.shape
    return data, multi_labels, single_labels, labels


def test_multispectral_2d_deprecated():
    lx, ly = 70, 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data[..., np.newaxis].repeat(2, axis=-1)  # Expect identical output

    # checking for multichannel kwarg warning
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                           "`multichannel` is a deprecated argument",
                            NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        multi_labels = random_walker(data, labels, mode='cg',
                                     multichannel=True)
    assert data[..., 0].shape == labels.shape

    # checking for positional multichannel warning
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                           "Providing the `multichannel` argument",
                            NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        multi_labels = random_walker(data, labels, 130, 'cg', 1.e-3, True,
                                     True)
    assert data[..., 0].shape == labels.shape

    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[25:45, 40:60] == 2).all()
    assert data[..., 0].shape == labels.shape
    return data, multi_labels, single_labels, labels


@testing.parametrize('dtype', [np.float32, np.float64])
def test_multispectral_3d(dtype):
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    data = data.astype(dtype, copy=False)
    data = data[..., np.newaxis].repeat(2, axis=-1)  # Expect identical output
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        multi_labels = random_walker(data, labels, mode='cg', channel_axis=-1)
    assert data[..., 0].shape == labels.shape
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert (single_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data[..., 0].shape == labels.shape
    return data, multi_labels, single_labels, labels


def test_spacing_0():
    n = 30
    lx, ly, lz = n, n, n
    data, _ = make_3d_syntheticdata(lx, ly, lz)

    # Rescale `data` along Z axis
    data_aniso = np.zeros((n, n, n // 2))
    for i, yz in enumerate(data):
        data_aniso[i, :, :] = resize(yz, (n, n // 2),
                                     mode='constant',
                                     anti_aliasing=False)

    # Generate new labels
    small_l = int(lx // 5)
    labels_aniso = np.zeros_like(data_aniso)
    labels_aniso[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso[lx // 2 + small_l // 4,
                 ly // 2 - small_l // 4,
                 lz // 4 - small_l // 8] = 2

    # Test with `spacing` kwarg
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        labels_aniso = random_walker(data_aniso, labels_aniso, mode='cg',
                                     spacing=(1., 1., 0.5))

    assert (labels_aniso[13:17, 13:17, 7:9] == 2).all()


@xfail(condition=arch32,
       reason=('Known test failure on 32-bit platforms. See links for '
               'details: '
               'https://github.com/scikit-image/scikit-image/issues/3091 '
               'https://github.com/scikit-image/scikit-image/issues/3092'))
def test_spacing_1():
    n = 30
    lx, ly, lz = n, n, n
    data, _ = make_3d_syntheticdata(lx, ly, lz)

    # Rescale `data` along Y axis
    # `resize` is not yet 3D capable, so this must be done by looping in 2D.
    data_aniso = np.zeros((n, n * 2, n))
    for i, yz in enumerate(data):
        data_aniso[i, :, :] = resize(yz, (n * 2, n),
                                     mode='constant',
                                     anti_aliasing=False)

    # Generate new labels
    small_l = int(lx // 5)
    labels_aniso = np.zeros_like(data_aniso)
    labels_aniso[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso[lx // 2 + small_l // 4,
                 ly - small_l // 2,
                 lz // 2 - small_l // 4] = 2

    # Test with `spacing` kwarg
    # First, anisotropic along Y
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        labels_aniso = random_walker(data_aniso, labels_aniso, mode='cg',
                                     spacing=(1., 2., 1.))
    assert (labels_aniso[13:17, 26:34, 13:17] == 2).all()

    # Rescale `data` along X axis
    # `resize` is not yet 3D capable, so this must be done by looping in 2D.
    data_aniso = np.zeros((n, n * 2, n))
    for i in range(data.shape[1]):
        data_aniso[i, :, :] = resize(data[:, 1, :], (n * 2, n),
                                     mode='constant',
                                     anti_aliasing=False)

    # Generate new labels
    small_l = int(lx // 5)
    labels_aniso2 = np.zeros_like(data_aniso)
    labels_aniso2[lx // 5, ly // 5, lz // 5] = 1
    labels_aniso2[lx - small_l // 2,
                  ly // 2 + small_l // 4,
                  lz // 2 - small_l // 4] = 2

    # Anisotropic along X
    with expected_warnings(['"cg" mode' + '|' + SCIPY_RANK_WARNING,
                            NUMPY_MATRIX_WARNING]):
        labels_aniso2 = random_walker(data_aniso,
                                      labels_aniso2,
                                      mode='cg', spacing=(2., 1., 1.))
    assert (labels_aniso2[26:34, 13:17, 13:17] == 2).all()


def test_trivial_cases():
    # When all voxels are labeled
    img = np.ones((10, 10))
    labels = np.ones((10, 10))

    with expected_warnings(["Returning provided labels"]):
        pass_through = random_walker(img, labels)
    np.testing.assert_array_equal(pass_through, labels)

    # When all voxels are labeled AND return_full_prob is True
    labels[:, :5] = 3
    expected = np.concatenate(((labels == 1)[..., np.newaxis],
                               (labels == 3)[..., np.newaxis]), axis=2)
    with expected_warnings(["Returning provided labels"]):
        test = random_walker(img, labels, return_full_prob=True)
    np.testing.assert_array_equal(test, expected)

    # Unlabeled voxels not connected to seed, so nothing can be done
    img = np.full((10, 10), False)
    object_A = np.array([(6,7), (6,8), (7,7), (7,8)])
    object_B = np.array([(3,1), (4,1), (2,2), (3,2), (4,2), (2,3), (3,3)])
    for x, y in np.vstack((object_A, object_B)):
            img[y][x] = True

    markers = np.zeros((10, 10), dtype=np.int8)
    for x, y in object_B:
            markers[y][x] = 1

    markers[img == 0] = -1
    with expected_warnings(["All unlabeled pixels are isolated"]):
            output_labels = random_walker(img, markers)
    assert np.all(output_labels[markers == 1] == 1)
    # Here 0-labeled pixels could not be determined (no connexion to seed)
    assert np.all(output_labels[markers == 0] == -1)
    with expected_warnings(["All unlabeled pixels are isolated"]):
        test = random_walker(img, markers, return_full_prob=True)


def test_length2_spacing():
    # If this passes without raising an exception (warnings OK), the new
    #   spacing code is working properly.
    np.random.seed(42)
    img = np.ones((10, 10)) + 0.2 * np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[2, 4] = 1
    labels[6, 8] = 4
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        random_walker(img, labels, spacing=(1., 2.))


def test_bad_inputs():
    # Too few dimensions
    img = np.ones(10)
    labels = np.arange(10)
    with testing.raises(ValueError):
        random_walker(img, labels)
    with testing.raises(ValueError):
        random_walker(img, labels, channel_axis=-1)

    # Too many dimensions
    np.random.seed(42)
    img = np.random.normal(size=(3, 3, 3, 3, 3))
    labels = np.arange(3 ** 5).reshape(img.shape)
    with testing.raises(ValueError):
        random_walker(img, labels)
    with testing.raises(ValueError):
        random_walker(img, labels, channel_axis=-1)

    # Spacing incorrect length
    img = np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10))
    labels[2, 4] = 2
    labels[6, 8] = 5
    with testing.raises(ValueError):
        random_walker(img, labels, spacing=(1,))

    # Invalid mode
    img = np.random.normal(size=(10, 10))
    labels = np.zeros((10, 10))
    with testing.raises(ValueError):
        random_walker(img, labels, mode='bad')


def test_isolated_seeds():
    np.random.seed(0)
    a = np.random.random((7, 7))
    mask = - np.ones(a.shape)
    # This pixel is an isolated seed
    mask[1, 1] = 1
    # Unlabeled pixels
    mask[3:, 3:] = 0
    # Seeds connected to unlabeled pixels
    mask[4, 4] = 2
    mask[6, 6] = 1

    # Test that no error is raised, and that labels of isolated seeds are OK
    with expected_warnings([NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        res = random_walker(a, mask)
    assert res[1, 1] == 1
    with expected_warnings([NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        res = random_walker(a, mask, return_full_prob=True)
    assert res[0, 1, 1] == 1
    assert res[1, 1, 1] == 0


def test_isolated_area():
    np.random.seed(0)
    a = np.random.random((7, 7))
    mask = - np.ones(a.shape)
    # This pixel is an isolated seed
    mask[1, 1] = 0
    # Unlabeled pixels
    mask[3:, 3:] = 0
    # Seeds connected to unlabeled pixels
    mask[4, 4] = 2
    mask[6, 6] = 1

    # Test that no error is raised, and that labels of isolated seeds are OK
    with expected_warnings([NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        res = random_walker(a, mask)
    assert res[1, 1] == 0
    with expected_warnings([NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        res = random_walker(a, mask, return_full_prob=True)
    assert res[0, 1, 1] == 0
    assert res[1, 1, 1] == 0


def test_prob_tol():
    np.random.seed(0)
    a = np.random.random((7, 7))
    mask = - np.ones(a.shape)
    # This pixel is an isolated seed
    mask[1, 1] = 1
    # Unlabeled pixels
    mask[3:, 3:] = 0
    # Seeds connected to unlabeled pixels
    mask[4, 4] = 2
    mask[6, 6] = 1

    with expected_warnings([NUMPY_MATRIX_WARNING,
                            'The probability range is outside']):
        res = random_walker(a, mask, return_full_prob=True)

    # Lower beta, no warning is expected.
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        res = random_walker(a, mask, return_full_prob=True, beta=10)
    assert res[0, 1, 1] == 1
    assert res[1, 1, 1] == 0

    # Being more prob_tol tolerant, no warning is expected.
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        res = random_walker(a, mask, return_full_prob=True, prob_tol=1e-1)
    assert res[0, 1, 1] == 1
    assert res[1, 1, 1] == 0

    # Reduced tol, no warning is expected.
    with expected_warnings([NUMPY_MATRIX_WARNING]):
        res = random_walker(a, mask, return_full_prob=True, tol=1e-9)
    assert res[0, 1, 1] == 1
    assert res[1, 1, 1] == 0


def test_umfpack_import():
    from skimage.segmentation import random_walker_segmentation
    UmfpackContext = random_walker_segmentation.UmfpackContext
    try:
        # when scikit-umfpack is installed UmfpackContext should not be None
        import scikits.umfpack
        assert UmfpackContext is not None
    except ImportError:
        assert UmfpackContext is None
    return
