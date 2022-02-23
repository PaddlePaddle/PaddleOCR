import re
import textwrap

import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal)

from skimage.transform._geometric import (_affine_matrix_from_vector,
                                          _center_and_normalize_points,
                                          _euler_rotation_matrix,
                                          GeometricTransform)
from skimage.transform import (estimate_transform, matrix_transform,
                               EuclideanTransform, SimilarityTransform,
                               AffineTransform, FundamentalMatrixTransform,
                               EssentialMatrixTransform, ProjectiveTransform,
                               PolynomialTransform, PiecewiseAffineTransform)


SRC = np.array([
    [-12.3705, -10.5075],
    [-10.7865, 15.4305],
    [8.6985, 10.8675],
    [11.4975, -9.5715],
    [7.8435, 7.4835],
    [-5.3325, 6.5025],
    [6.7905, -6.3765],
    [-6.1695, -0.8235],
])
DST = np.array([
    [0, 0],
    [0, 5800],
    [4900, 5800],
    [4900, 0],
    [4479, 4580],
    [1176, 3660],
    [3754, 790],
    [1024, 1931],
])


def test_estimate_transform():
    for tform in ('euclidean', 'similarity', 'affine', 'projective',
                  'polynomial'):
        estimate_transform(tform, SRC[:2, :], DST[:2, :])
    with pytest.raises(ValueError):
        estimate_transform('foobar', SRC[:2, :], DST[:2, :])


def test_matrix_transform():
    tform = AffineTransform(scale=(0.1, 0.5), rotation=2)
    assert_equal(tform(SRC), matrix_transform(SRC, tform.params))


def test_euclidean_estimation():
    # exact solution
    tform = estimate_transform('euclidean', SRC[:2, :], SRC[:2, :] + 10)
    assert_almost_equal(tform(SRC[:2, :]), SRC[:2, :] + 10)
    assert_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_almost_equal(tform.params[0, 1], - tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('euclidean', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_almost_equal(tform2.params[0, 1], - tform2.params[1, 0])

    # via estimate method
    tform3 = EuclideanTransform()
    tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_euclidean_init():
    # init with implicit parameters
    rotation = 1
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = EuclideanTransform(tform.params)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    rotation = 0
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    rotation = np.pi / 2
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)


def test_similarity_estimation():
    # exact solution
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_almost_equal(tform.params[0, 1], - tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('similarity', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_almost_equal(tform2.params[0, 1], - tform2.params[1, 0])

    # via estimate method
    tform3 = SimilarityTransform()
    tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_similarity_init():
    # init with implicit parameters
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = SimilarityTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    scale = 0.1
    rotation = 0
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    scale = 0.1
    rotation = np.pi / 2
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale where the rotation isn't exactly 90deg,
    # but very close
    scale = 1.0
    rotation = np.pi / 2
    translation = (0, 0)
    params = np.array([[0, -1, 1.33226763e-15],
                       [1, 2.22044605e-16, -1.33226763e-15],
                       [0, 0, 1]])
    tform = SimilarityTransform(params)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)


def test_affine_estimation():
    # exact solution
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_almost_equal(tform(SRC[:3, :]), DST[:3, :])

    # over-determined
    tform2 = estimate_transform('affine', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = AffineTransform()
    tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_affine_init():
    # init with implicit parameters
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear,
                            translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.shear, shear)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = AffineTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.shear, shear)
    assert_almost_equal(tform2.translation, translation)

    # scalar vs. tuple scale arguments
    assert_almost_equal(AffineTransform(scale=0.5).scale, AffineTransform(scale=(0.5, 0.5)).scale)


def test_piecewise_affine():
    tform = PiecewiseAffineTransform()
    tform.estimate(SRC, DST)
    # make sure each single affine transform is exactly estimated
    assert_almost_equal(tform(SRC), DST)
    assert_almost_equal(tform.inverse(DST), SRC)


def test_fundamental_matrix_estimation():
    src = np.array([1.839035, 1.924743, 0.543582,  0.375221,
                    0.473240, 0.142522, 0.964910,  0.598376,
                    0.102388, 0.140092, 15.994343, 9.622164,
                    0.285901, 0.430055, 0.091150,  0.254594]).reshape(-1, 2)
    dst = np.array([1.002114, 1.129644, 1.521742, 1.846002,
                    1.084332, 0.275134, 0.293328, 0.588992,
                    0.839509, 0.087290, 1.779735, 1.116857,
                    0.878616, 0.602447, 0.642616, 1.028681]).reshape(-1, 2)

    tform = estimate_transform('fundamental', src, dst)

    # Reference values obtained using COLMAP SfM library.
    tform_ref = np.array([[-0.217859, 0.419282, -0.0343075],
                          [-0.0717941, 0.0451643, 0.0216073],
                          [0.248062, -0.429478, 0.0221019]])
    assert_almost_equal(tform.params, tform_ref, 6)


def test_fundamental_matrix_residuals():
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0]))
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 0], [0, 0]])
    dst = np.array([[2, 0], [2, 1], [2, 2]])
    assert_almost_equal(tform.residuals(src, dst)**2, [0, 0.5, 2])


def test_fundamental_matrix_forward():
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0]))
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform(src), [[0, -1, 0], [0, -1, 1], [0, -1, 1]])


def test_fundamental_matrix_inverse():
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0]))
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform.inverse(src),
                        [[0, 1, 0], [0, 1, -1], [0, 1, -1]])


def test_essential_matrix_init():
    tform = EssentialMatrixTransform(rotation=np.eye(3),
                                     translation=np.array([0, 0, 1]))
    assert_equal(tform.params,
                 np.array([0, -1, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3))


def test_essential_matrix_estimation():
    src = np.array([1.839035, 1.924743, 0.543582,  0.375221,
                    0.473240, 0.142522, 0.964910,  0.598376,
                    0.102388, 0.140092, 15.994343, 9.622164,
                    0.285901, 0.430055, 0.091150,  0.254594]).reshape(-1, 2)
    dst = np.array([1.002114, 1.129644, 1.521742, 1.846002,
                    1.084332, 0.275134, 0.293328, 0.588992,
                    0.839509, 0.087290, 1.779735, 1.116857,
                    0.878616, 0.602447, 0.642616, 1.028681]).reshape(-1, 2)

    tform = estimate_transform('essential', src, dst)

    # Reference values obtained using COLMAP SfM library.
    tform_ref = np.array([[-0.0811666, 0.255449, -0.0478999],
                          [-0.192392, -0.0531675, 0.119547],
                          [0.177784, -0.22008, -0.015203]])
    assert_almost_equal(tform.params, tform_ref, 6)


def test_essential_matrix_forward():
    tform = EssentialMatrixTransform(rotation=np.eye(3),
                                     translation=np.array([1, 0, 0]))
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform(src), [[0, -1, 0], [0, -1, 1], [0, -1, 1]])


def test_essential_matrix_inverse():
    tform = EssentialMatrixTransform(rotation=np.eye(3),
                                     translation=np.array([1, 0, 0]))
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform.inverse(src),
                        [[0, 1, 0], [0, 1, -1], [0, 1, -1]])


def test_essential_matrix_residuals():
    tform = EssentialMatrixTransform(rotation=np.eye(3),
                                     translation=np.array([1, 0, 0]))
    src = np.array([[0, 0], [0, 0], [0, 0]])
    dst = np.array([[2, 0], [2, 1], [2, 2]])
    assert_almost_equal(tform.residuals(src, dst)**2, [0, 0.5, 2])


def test_projective_estimation():
    # exact solution
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    assert_almost_equal(tform(SRC[:4, :]), DST[:4, :])

    # over-determined
    tform2 = estimate_transform('projective', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = ProjectiveTransform()
    tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_projective_weighted_estimation():

    # Exact solution with same points, and unity weights
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    tform_w = estimate_transform('projective',
                                 SRC[:4, :], DST[:4, :], np.ones(4))
    assert_almost_equal(tform.params, tform_w.params)

    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('projective', SRC, DST)
    tform_w = estimate_transform('projective',
                                 SRC, DST, np.ones(SRC.shape[0]))
    assert_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('projective', SRC, DST)
    tform2 = estimate_transform('projective',
                                SRC[np.arange(-1, SRC.shape[0]), :],
                                DST[np.arange(-1, SRC.shape[0]), :],
                                point_weights)
    assert_almost_equal(tform1.params, tform2.params, decimal=3)


def test_projective_init():
    tform = estimate_transform('projective', SRC, DST)
    # init with transformation matrix
    tform2 = ProjectiveTransform(tform.params)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_estimation():
    # over-determined
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_almost_equal(tform(SRC), DST, 6)

    # via estimate method
    tform2 = PolynomialTransform()
    tform2.estimate(SRC, DST, order=10)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_weighted_estimation():
    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    tform_w = estimate_transform('polynomial',
                                 SRC,
                                 DST,
                                 order=10,
                                 weights=np.ones(SRC.shape[0]))
    assert_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('polynomial', SRC, DST, order=10)
    tform2 = estimate_transform('polynomial',
                                SRC[np.arange(-1, SRC.shape[0]), :],
                                DST[np.arange(-1, SRC.shape[0]), :],
                                order=10,
                                weights=point_weights)
    assert_almost_equal(tform1.params, tform2.params, decimal=4)


def test_polynomial_init():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    # init with transformation parameters
    tform2 = PolynomialTransform(tform.params)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_default_order():
    tform = estimate_transform('polynomial', SRC, DST)
    tform2 = estimate_transform('polynomial', SRC, DST, order=2)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_inverse():
    with pytest.raises(Exception):
        PolynomialTransform().inverse(0)


def test_union():
    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_almost_equal(tform.params, tform3.params)

    tform1 = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_almost_equal(tform.params, tform3.params)
    assert tform.__class__ == ProjectiveTransform

    tform = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    assert_almost_equal((tform + tform.inverse).params, np.eye(3))

    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 * 1/0.1, rotation=0.3 - 0.9)
    tform = tform1 + tform2.inverse
    assert_almost_equal(tform.params, tform3.params)


def test_union_differing_types():
    tform1 = SimilarityTransform()
    tform2 = PolynomialTransform()
    with pytest.raises(TypeError):
        tform1.__add__(tform2)


def test_geometric_tform():
    tform = GeometricTransform()
    with pytest.raises(NotImplementedError):
        tform(0)
    with pytest.raises(NotImplementedError):
        tform.inverse(0)
    with pytest.raises(NotImplementedError):
        tform.__add__(0)

    # See gh-3926 for discussion details
    for i in range(20):
        # Generate random Homography
        H = np.random.rand(3, 3) * 100
        H[2, H[2] == 0] += np.finfo(float).eps
        H /= H[2, 2]

        # Craft some src coords
        src = np.array([
            [(H[2, 1] + 1) / -H[2, 0], 1],
            [1, (H[2, 0] + 1) / -H[2, 1]],
            [1, 1],
        ])
        # Prior to gh-3926, under the above circumstances,
        # destination coordinates could be returned with nan/inf values.
        tform = ProjectiveTransform(H)  # Construct the transform
        dst = tform(src)  # Obtain the dst coords
        # Ensure dst coords are finite numeric values
        assert(np.isfinite(dst).all())

def test_invalid_input():
    with pytest.raises(ValueError):
        ProjectiveTransform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        AffineTransform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        SimilarityTransform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        EuclideanTransform(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        AffineTransform(matrix=np.zeros((2, 3)), scale=1)
    with pytest.raises(ValueError):
        SimilarityTransform(matrix=np.zeros((2, 3)), scale=1)
    with pytest.raises(ValueError):
        EuclideanTransform(
            matrix=np.zeros((2, 3)), translation=(0, 0))
    with pytest.raises(ValueError):
        PolynomialTransform(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        FundamentalMatrixTransform(matrix=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(matrix=np.zeros((3, 2)))

    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(
            rotation=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(
            rotation=np.eye(3))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.eye(3),
                                 translation=np.zeros((2,)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.eye(3),
                                 translation=np.zeros((2,)))
    with pytest.raises(ValueError):
        EssentialMatrixTransform(
            rotation=np.eye(3), translation=np.zeros((3,)))


def test_degenerate():
    src = dst = np.zeros((10, 2))

    tform = SimilarityTransform()
    tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    tform = AffineTransform()
    tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    # See gh-3926 for discussion details
    tform = ProjectiveTransform()
    for i in range(20):
        # Some random coordinates
        src = np.random.rand(4, 2) * 100
        dst = np.random.rand(4, 2) * 100

        # Degenerate the case by arranging points on a single line
        src[:, 1] = np.random.rand()
        # Prior to gh-3926, under the above circumstances,
        # a transform could be returned with nan values.
        assert(not tform.estimate(src, dst) or np.isfinite(tform.params).all())


def test_normalize_degenerate_points():
    """Return nan matrix *of appropriate size* when point is repeated."""
    pts = np.array([[73.42834308, 94.2977623 ],] * 3)
    mat, pts_tf = _center_and_normalize_points(pts)
    assert np.all(np.isnan(mat))
    assert np.all(np.isnan(pts_tf))
    assert mat.shape == (3, 3)
    assert pts_tf.shape == pts.shape



def test_projective_repr():
    tform = ProjectiveTransform()
    want = re.escape(textwrap.dedent(
        '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]) at
        ''').strip()) + ' 0x[a-f0-9]+' + re.escape('>')
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    assert re.match(want, repr(tform))


def test_projective_str():
    tform = ProjectiveTransform()
    want = re.escape(textwrap.dedent(
        '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])>
        ''').strip())
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    print(want)
    assert re.match(want, str(tform))


def _assert_least_squares(tf, src, dst):
    baseline = np.sum((tf(src) - dst) ** 2)
    for i in range(tf.params.size):
        for update in [0.001, -0.001]:
            params = np.copy(tf.params)
            params.flat[i] += update
            new_tf = tf.__class__(matrix=params)
            new_ssq = np.sum((new_tf(src) - dst) ** 2)
            assert new_ssq > baseline


def test_estimate_affine_3d():
    ndim = 3
    src = np.random.random((25, ndim)) * 2 ** np.arange(7, 7 + ndim)
    matrix = np.array(
        [[4.8, 0.1, 0.2, 25],
         [0.0, 1.0, 0.1, 30],
         [0.0, 0.0, 1.0, -2],
         [0.0, 0.0, 0.0, 1.]]
    )
    tf = AffineTransform(matrix=matrix)
    dst = tf(src)
    dst_noisy = dst + np.random.random((25, ndim))
    tf2 = AffineTransform(dimensionality=ndim)
    tf2.estimate(src, dst_noisy)
    # we check rot/scale/etc more tightly than translation because translation
    # estimation is on the 1 pixel scale
    assert_almost_equal(tf2.params[:, :-1], matrix[:, :-1], decimal=2)
    assert_almost_equal(tf2.params[:, -1], matrix[:, -1], decimal=0)
    _assert_least_squares(tf2, src, dst_noisy)


def test_fundamental_3d_not_implemented():
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(np.eye(4))


def test_array_protocol():
    mat = np.eye(4)
    tf = ProjectiveTransform(mat)
    assert_equal(np.array(tf), mat)
    assert_equal(np.array(tf, dtype=int), mat.astype(int))


def test_affine_transform_from_linearized_parameters():
    mat = np.concatenate(
        (np.random.random((3, 4)), np.eye(4)[-1:]), axis=0
    )
    v = mat[:-1].ravel()
    mat_from_v = _affine_matrix_from_vector(v)
    tf = AffineTransform(matrix=mat_from_v)
    assert_equal(np.array(tf), mat)
    # incorrect number of parameters
    with pytest.raises(ValueError):
        _ = _affine_matrix_from_vector(v[:-1])
    with pytest.raises(ValueError):
        _ = AffineTransform(matrix=v[:-1])


def test_affine_params_nD_error():
    with pytest.raises(ValueError):
        _ = AffineTransform(scale=5, dimensionality=3)


def test_euler_rotation():
    v = [0, 10, 0]
    angles = np.radians([90, 45, 45])
    expected = [-5, -5, 7.1]
    R = _euler_rotation_matrix(angles)
    assert_almost_equal(R @ v, expected, decimal=1)


def test_euclidean_param_defaults():
    # 2D rotation is 0 when only translation is given
    tf = EuclideanTransform(translation=(5, 5))
    assert np.array(tf)[0, 1] == 0
    # off diagonals are 0 when only translation is given
    tf = EuclideanTransform(translation=(4, 5, 9), dimensionality=3)
    assert_equal(np.array(tf)[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], 0)
    with pytest.raises(ValueError):
        # specifying parameters for D>3 is not supported
        _ = EuclideanTransform(translation=(5, 6, 7, 8), dimensionality=4)
    with pytest.raises(ValueError):
        # incorrect number of angles for given dimensionality
        _ = EuclideanTransform(rotation=(4, 8), dimensionality=3)
    # translation is 0 when rotation is given
    tf = EuclideanTransform(rotation=np.pi * np.arange(3), dimensionality=3)
    assert_equal(np.array(tf)[:-1, 3], 0)


def test_similarity_transform_params():
    with pytest.raises(ValueError):
        _ = SimilarityTransform(translation=(4, 5, 6, 7), dimensionality=4)
    tf = SimilarityTransform(scale=4, dimensionality=3)
    assert_equal(tf([[1, 1, 1]]), [[4, 4, 4]])


def test_euler_angle_consistency():
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    euclid = EuclideanTransform(rotation=angles, dimensionality=3)
    similar = SimilarityTransform(rotation=angles, dimensionality=3)
    assert_array_almost_equal(euclid, similar)


def test_2D_only_implementations():
    with pytest.raises(NotImplementedError):
        _ = PolynomialTransform(dimensionality=3)
    tf = AffineTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = tf.rotation
    with pytest.raises(NotImplementedError):
        _ = tf.shear
