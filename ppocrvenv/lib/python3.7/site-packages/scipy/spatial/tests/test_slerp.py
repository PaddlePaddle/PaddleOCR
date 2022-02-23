from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import assert_allclose

import pytest
from scipy.spatial import geometric_slerp


def _generate_spherical_points(ndim=3, n_pts=2):
    # generate uniform points on sphere
    # see: https://stackoverflow.com/a/23785326
    # tentatively extended to arbitrary dims
    # for 0-sphere it will always produce antipodes
    np.random.seed(123)
    points = np.random.normal(size=(n_pts, ndim))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points[0], points[1]


class TestGeometricSlerp:
    # Test various properties of the geometric slerp code

    @pytest.mark.parametrize("n_dims", [2, 3, 5, 7, 9])
    @pytest.mark.parametrize("n_pts", [0, 3, 17])
    def test_shape_property(self, n_dims, n_pts):
        # geometric_slerp output shape should match
        # input dimensionality & requested number
        # of interpolation points
        start, end = _generate_spherical_points(n_dims, 2)

        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, n_pts))

        assert actual.shape == (n_pts, n_dims)

    @pytest.mark.parametrize("n_dims", [2, 3, 5, 7, 9])
    @pytest.mark.parametrize("n_pts", [3, 17])
    def test_include_ends(self, n_dims, n_pts):
        # geometric_slerp should return a data structure
        # that includes the start and end coordinates
        # when t includes 0 and 1 ends
        # this is convenient for plotting surfaces represented
        # by interpolations for example

        # the generator doesn't work so well for the unit
        # sphere (it always produces antipodes), so use
        # custom values there
        start, end = _generate_spherical_points(n_dims, 2)

        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, n_pts))

        assert_allclose(actual[0], start)
        assert_allclose(actual[-1], end)

    @pytest.mark.parametrize("start, end", [
        # both arrays are not flat
        (np.zeros((1, 3)), np.ones((1, 3))),
        # only start array is not flat
        (np.zeros((1, 3)), np.ones(3)),
        # only end array is not flat
        (np.zeros(1), np.ones((3, 1))),
        ])
    def test_input_shape_flat(self, start, end):
        # geometric_slerp should handle input arrays that are
        # not flat appropriately
        with pytest.raises(ValueError, match='one-dimensional'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end", [
        # 7-D and 3-D ends
        (np.zeros(7), np.ones(3)),
        # 2-D and 1-D ends
        (np.zeros(2), np.ones(1)),
        # empty, "3D" will also get caught this way
        (np.array([]), np.ones(3)),
        ])
    def test_input_dim_mismatch(self, start, end):
        # geometric_slerp must appropriately handle cases where
        # an interpolation is attempted across two different
        # dimensionalities
        with pytest.raises(ValueError, match='dimensions'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end", [
        # both empty
        (np.array([]), np.array([])),
        ])
    def test_input_at_least1d(self, start, end):
        # empty inputs to geometric_slerp must
        # be handled appropriately when not detected
        # by mismatch
        with pytest.raises(ValueError, match='at least two-dim'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize("start, end, expected", [
        # North and South Poles are definitely antipodes
        # but should be handled gracefully now
        (np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), "warning"),
        # this case will issue a warning & be handled
        # gracefully as well;
        # North Pole was rotated very slightly
        # using r = R.from_euler('x', 0.035, degrees=True)
        # to achieve Euclidean distance offset from diameter by
        # 9.328908379124812e-08, within the default tol
        (np.array([0.00000000e+00,
                  -6.10865200e-04,
                  9.99999813e-01]), np.array([0, 0, -1.0]), "warning"),
        # this case should succeed without warning because a
        # sufficiently large
        # rotation was applied to North Pole point to shift it
        # to a Euclidean distance of 2.3036691931821451e-07
        # from South Pole, which is larger than tol
        (np.array([0.00000000e+00,
                  -9.59930941e-04,
                  9.99999539e-01]), np.array([0, 0, -1.0]), "success"),
        ])
    def test_handle_antipodes(self, start, end, expected):
        # antipodal points must be handled appropriately;
        # there are an infinite number of possible geodesic
        # interpolations between them in higher dims
        if expected == "warning":
            with pytest.warns(UserWarning, match='antipodes'):
                res = geometric_slerp(start=start,
                                      end=end,
                                      t=np.linspace(0, 1, 10))
        else:
            res = geometric_slerp(start=start,
                                  end=end,
                                  t=np.linspace(0, 1, 10))

        # antipodes or near-antipodes should still produce
        # slerp paths on the surface of the sphere (but they
        # may be ambiguous):
        assert_allclose(np.linalg.norm(res, axis=1), 1.0)

    @pytest.mark.parametrize("start, end, expected", [
        # 2-D with n_pts=4 (two new interpolation points)
        # this is an actual circle
        (np.array([1, 0]),
         np.array([0, 1]),
         np.array([[1, 0],
                   [np.sqrt(3) / 2, 0.5],  # 30 deg on unit circle
                   [0.5, np.sqrt(3) / 2],  # 60 deg on unit circle
                   [0, 1]])),
        # likewise for 3-D (add z = 0 plane)
        # this is an ordinary sphere
        (np.array([1, 0, 0]),
         np.array([0, 1, 0]),
         np.array([[1, 0, 0],
                   [np.sqrt(3) / 2, 0.5, 0],
                   [0.5, np.sqrt(3) / 2, 0],
                   [0, 1, 0]])),
        # for 5-D, pad more columns with constants
        # zeros are easiest--non-zero values on unit
        # circle are more difficult to reason about
        # at higher dims
        (np.array([1, 0, 0, 0, 0]),
         np.array([0, 1, 0, 0, 0]),
         np.array([[1, 0, 0, 0, 0],
                   [np.sqrt(3) / 2, 0.5, 0, 0, 0],
                   [0.5, np.sqrt(3) / 2, 0, 0, 0],
                   [0, 1, 0, 0, 0]])),

    ])
    def test_straightforward_examples(self, start, end, expected):
        # some straightforward interpolation tests, sufficiently
        # simple to use the unit circle to deduce expected values;
        # for larger dimensions, pad with constants so that the
        # data is N-D but simpler to reason about
        actual = geometric_slerp(start=start,
                                 end=end,
                                 t=np.linspace(0, 1, 4))
        assert_allclose(actual, expected, atol=1e-16)

    @pytest.mark.parametrize("t", [
        # both interval ends clearly violate limits
        np.linspace(-20, 20, 300),
        # only one interval end violating limit slightly
        np.linspace(-0.0001, 0.0001, 17),
        ])
    def test_t_values_limits(self, t):
        # geometric_slerp() should appropriately handle
        # interpolation parameters < 0 and > 1
        with pytest.raises(ValueError, match='interpolation parameter'):
            _ = geometric_slerp(start=np.array([1, 0]),
                                end=np.array([0, 1]),
                                t=t)

    @pytest.mark.parametrize("start, end", [
        (np.array([1]),
         np.array([0])),
        (np.array([0]),
         np.array([1])),
        (np.array([-17.7]),
         np.array([165.9])),
     ])
    def test_0_sphere_handling(self, start, end):
        # it does not make sense to interpolate the set of
        # two points that is the 0-sphere
        with pytest.raises(ValueError, match='at least two-dim'):
            _ = geometric_slerp(start=start,
                                end=end,
                                t=np.linspace(0, 1, 4))

    @pytest.mark.parametrize("tol", [
        # an integer currently raises
        5,
        # string raises
        "7",
        # list and arrays also raise
        [5, 6, 7], np.array(9.0),
        ])
    def test_tol_type(self, tol):
        # geometric_slerp() should raise if tol is not
        # a suitable float type
        with pytest.raises(ValueError, match='must be a float'):
            _ = geometric_slerp(start=np.array([1, 0]),
                                end=np.array([0, 1]),
                                t=np.linspace(0, 1, 5),
                                tol=tol)

    @pytest.mark.parametrize("tol", [
        -5e-6,
        -7e-10,
        ])
    def test_tol_sign(self, tol):
        # geometric_slerp() currently handles negative
        # tol values, as long as they are floats
        _ = geometric_slerp(start=np.array([1, 0]),
                            end=np.array([0, 1]),
                            t=np.linspace(0, 1, 5),
                            tol=tol)

    @pytest.mark.parametrize("start, end", [
        # 1-sphere (circle) with one point at origin
        # and the other on the circle
        (np.array([1, 0]), np.array([0, 0])),
        # 2-sphere (normal sphere) with both points
        # just slightly off sphere by the same amount
        # in different directions
        (np.array([1 + 1e-6, 0, 0]),
         np.array([0, 1 - 1e-6, 0])),
        # same thing in 4-D
        (np.array([1 + 1e-6, 0, 0, 0]),
         np.array([0, 1 - 1e-6, 0, 0])),
        ])
    def test_unit_sphere_enforcement(self, start, end):
        # geometric_slerp() should raise on input that clearly
        # cannot be on an n-sphere of radius 1
        with pytest.raises(ValueError, match='unit n-sphere'):
            geometric_slerp(start=start,
                            end=end,
                            t=np.linspace(0, 1, 5))

    @pytest.mark.parametrize("start, end", [
        # 1-sphere 45 degree case
        (np.array([1, 0]),
         np.array([np.sqrt(2) / 2.,
                   np.sqrt(2) / 2.])),
        # 2-sphere 135 degree case
        (np.array([1, 0]),
         np.array([-np.sqrt(2) / 2.,
                   np.sqrt(2) / 2.])),
        ])
    @pytest.mark.parametrize("t_func", [
        np.linspace, np.logspace])
    def test_order_handling(self, start, end, t_func):
        # geometric_slerp() should handle scenarios with
        # ascending and descending t value arrays gracefully;
        # results should simply be reversed

        # for scrambled / unsorted parameters, the same values
        # should be returned, just in scrambled order

        num_t_vals = 20
        np.random.seed(789)
        forward_t_vals = t_func(0, 10, num_t_vals)
        # normalize to max of 1
        forward_t_vals /= forward_t_vals.max()
        reverse_t_vals = np.flipud(forward_t_vals)
        shuffled_indices = np.arange(num_t_vals)
        np.random.shuffle(shuffled_indices)
        scramble_t_vals = forward_t_vals.copy()[shuffled_indices]

        forward_results = geometric_slerp(start=start,
                                          end=end,
                                          t=forward_t_vals)
        reverse_results = geometric_slerp(start=start,
                                          end=end,
                                          t=reverse_t_vals)
        scrambled_results = geometric_slerp(start=start,
                                            end=end,
                                            t=scramble_t_vals)

        # check fidelity to input order
        assert_allclose(forward_results, np.flipud(reverse_results))
        assert_allclose(forward_results[shuffled_indices],
                        scrambled_results)

    @pytest.mark.parametrize("t", [
        # string:
        "15, 5, 7",
        # complex numbers currently produce a warning
        # but not sure we need to worry about it too much:
        # [3 + 1j, 5 + 2j],
        ])
    def test_t_values_conversion(self, t):
        with pytest.raises(ValueError):
            _ = geometric_slerp(start=np.array([1]),
                                end=np.array([0]),
                                t=t)

    def test_accept_arraylike(self):
        # array-like support requested by reviewer
        # in gh-10380
        actual = geometric_slerp([1, 0], [0, 1], [0, 1/3, 0.5, 2/3, 1])

        # expected values are based on visual inspection
        # of the unit circle for the progressions along
        # the circumference provided in t
        expected = np.array([[1, 0],
                             [np.sqrt(3) / 2, 0.5],
                             [np.sqrt(2) / 2,
                              np.sqrt(2) / 2],
                             [0.5, np.sqrt(3) / 2],
                             [0, 1]], dtype=np.float64)
        # Tyler's original Cython implementation of geometric_slerp
        # can pass at atol=0 here, but on balance we will accept
        # 1e-16 for an implementation that avoids Cython and
        # makes up accuracy ground elsewhere
        assert_allclose(actual, expected, atol=1e-16)

    def test_scalar_t(self):
        # when t is a scalar, return value is a single
        # interpolated point of the appropriate dimensionality
        # requested by reviewer in gh-10380
        actual = geometric_slerp([1, 0], [0, 1], 0.5)
        expected = np.array([np.sqrt(2) / 2,
                             np.sqrt(2) / 2], dtype=np.float64)
        assert actual.shape == (2,)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('start', [
        np.array([1, 0, 0]),
        np.array([0, 1]),
    ])
    @pytest.mark.parametrize('t', [
        np.array(1),
        np.array([1]),
        np.array([[1]]),
        np.array([[[1]]]),
        np.array([]),
        np.linspace(0, 1, 5),
    ])
    def test_degenerate_input(self, start, t):
        if np.asarray(t).ndim > 1:
            with pytest.raises(ValueError):
                geometric_slerp(start=start, end=start, t=t)
        else:

            shape = (t.size,) + start.shape
            expected = np.full(shape, start)

            actual = geometric_slerp(start=start, end=start, t=t)
            assert_allclose(actual, expected)

            # Check that degenerate and non-degenerate
            # inputs yield the same size
            non_degenerate = geometric_slerp(start=start, end=start[::-1], t=t)
            assert actual.size == non_degenerate.size

    @pytest.mark.parametrize('k', np.logspace(-10, -1, 10))
    def test_numerical_stability_pi(self, k):
        # geometric_slerp should have excellent numerical
        # stability for angles approaching pi between
        # the start and end points
        angle = np.pi - k
        ts = np.linspace(0, 1, 100)
        P = np.array([1, 0, 0, 0])
        Q = np.array([np.cos(angle), np.sin(angle), 0, 0])
        # the test should only be enforced for cases where
        # geometric_slerp determines that the input is actually
        # on the unit sphere
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            result = geometric_slerp(P, Q, ts, 1e-18)
            norms = np.linalg.norm(result, axis=1)
            error = np.max(np.abs(norms - 1))
            assert error < 4e-15

    @pytest.mark.parametrize('t', [
     [[0, 0.5]],
     [[[[[[[[[0, 0.5]]]]]]]]],
    ])
    def test_interpolation_param_ndim(self, t):
        # regression test for gh-14465
        arr1 = np.array([0, 1])
        arr2 = np.array([1, 0])

        with pytest.raises(ValueError):
            geometric_slerp(start=arr1,
                            end=arr2,
                            t=t)

        with pytest.raises(ValueError):
            geometric_slerp(start=arr1,
                            end=arr1,
                            t=t)
