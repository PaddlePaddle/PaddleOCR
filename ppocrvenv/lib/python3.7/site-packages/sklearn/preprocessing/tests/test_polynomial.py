import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import random as sparse_random
from sklearn.utils._testing import assert_array_almost_equal

from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import BSpline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)
from sklearn.utils.fixes import linspace, sp_version, parse_version


@pytest.mark.parametrize("est", (PolynomialFeatures, SplineTransformer))
def test_polynomial_and_spline_array_order(est):
    """Test that output array has the given order."""
    X = np.arange(10).reshape(5, 2)

    def is_c_contiguous(a):
        return np.isfortran(a.T)

    assert is_c_contiguous(est().fit_transform(X))
    assert is_c_contiguous(est(order="C").fit_transform(X))
    assert np.isfortran(est(order="F").fit_transform(X))


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"degree": -1}, "degree must be a non-negative integer"),
        ({"degree": 2.5}, "degree must be a non-negative integer"),
        ({"degree": "string"}, "degree must be a non-negative integer"),
        ({"n_knots": 1}, "n_knots must be a positive integer >= 2."),
        ({"n_knots": 1}, "n_knots must be a positive integer >= 2."),
        ({"n_knots": 2.5}, "n_knots must be a positive integer >= 2."),
        ({"n_knots": "string"}, "n_knots must be a positive integer >= 2."),
        ({"knots": 1}, "Expected 2D array, got scalar array instead:"),
        ({"knots": [1, 2]}, "Expected 2D array, got 1D array instead:"),
        (
            {"knots": [[1]]},
            r"Number of knots, knots.shape\[0\], must be >= 2.",
        ),
        (
            {"knots": [[1, 5], [2, 6]]},
            r"knots.shape\[1\] == n_features is violated.",
        ),
        (
            {"knots": [[1], [1], [2]]},
            "knots must be sorted without duplicates.",
        ),
        ({"knots": [[2], [1]]}, "knots must be sorted without duplicates."),
        (
            {"extrapolation": None},
            "extrapolation must be one of 'error', 'constant', 'linear', "
            "'continue' or 'periodic'.",
        ),
        (
            {"extrapolation": 1},
            "extrapolation must be one of 'error', 'constant', 'linear', "
            "'continue' or 'periodic'.",
        ),
        (
            {"extrapolation": "string"},
            "extrapolation must be one of 'error', 'constant', 'linear', "
            "'continue' or 'periodic'.",
        ),
        ({"include_bias": None}, "include_bias must be bool."),
        ({"include_bias": 1}, "include_bias must be bool."),
        ({"include_bias": "string"}, "include_bias must be bool."),
        (
            {"extrapolation": "periodic", "n_knots": 3, "degree": 3},
            "Periodic splines require degree < n_knots. Got n_knots=3 and degree=3.",
        ),
        (
            {"extrapolation": "periodic", "knots": [[0], [1]], "degree": 2},
            "Periodic splines require degree < n_knots. Got n_knots=2 and degree=2.",
        ),
    ],
)
def test_spline_transformer_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in SplineTransformer."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        SplineTransformer(**params).fit(X)


def test_spline_transformer_manual_knot_input():
    """
    Test that array-like knot positions in SplineTransformer are accepted.
    """
    X = np.arange(20).reshape(10, 2)
    knots = [[0.5, 1], [1.5, 2], [5, 10]]
    st1 = SplineTransformer(degree=3, knots=knots, n_knots=None).fit(X)
    knots = np.asarray(knots)
    st2 = SplineTransformer(degree=3, knots=knots, n_knots=None).fit(X)
    for i in range(X.shape[1]):
        assert_allclose(st1.bsplines_[i].t, st2.bsplines_[i].t)


@pytest.mark.parametrize("extrapolation", ["continue", "periodic"])
def test_spline_transformer_integer_knots(extrapolation):
    """Test that SplineTransformer accepts integer value knot positions."""
    X = np.arange(20).reshape(10, 2)
    knots = [[0, 1], [1, 2], [5, 5], [11, 10], [12, 11]]
    _ = SplineTransformer(
        degree=3, knots=knots, extrapolation=extrapolation
    ).fit_transform(X)


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
@pytest.mark.parametrize("get_names", ["get_feature_names", "get_feature_names_out"])
def test_spline_transformer_feature_names(get_names):
    """Test that SplineTransformer generates correct features name."""
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(n_knots=3, degree=3, include_bias=True).fit(X)
    feature_names = getattr(splt, get_names)()
    assert_array_equal(
        feature_names,
        [
            "x0_sp_0",
            "x0_sp_1",
            "x0_sp_2",
            "x0_sp_3",
            "x0_sp_4",
            "x1_sp_0",
            "x1_sp_1",
            "x1_sp_2",
            "x1_sp_3",
            "x1_sp_4",
        ],
    )

    splt = SplineTransformer(n_knots=3, degree=3, include_bias=False).fit(X)
    feature_names = getattr(splt, get_names)(["a", "b"])
    assert_array_equal(
        feature_names,
        [
            "a_sp_0",
            "a_sp_1",
            "a_sp_2",
            "a_sp_3",
            "b_sp_0",
            "b_sp_1",
            "b_sp_2",
            "b_sp_3",
        ],
    )


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("n_knots", range(3, 5))
@pytest.mark.parametrize("knots", ["uniform", "quantile"])
@pytest.mark.parametrize("extrapolation", ["constant", "periodic"])
def test_spline_transformer_unity_decomposition(degree, n_knots, knots, extrapolation):
    """Test that B-splines are indeed a decomposition of unity.

    Splines basis functions must sum up to 1 per row, if we stay in between
    boundaries.
    """
    X = np.linspace(0, 1, 100)[:, None]
    # make the boundaries 0 and 1 part of X_train, for sure.
    X_train = np.r_[[[0]], X[::2, :], [[1]]]
    X_test = X[1::2, :]

    if extrapolation == "periodic":
        n_knots = n_knots + degree  # periodic splines require degree < n_knots

    splt = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        knots=knots,
        include_bias=True,
        extrapolation=extrapolation,
    )
    splt.fit(X_train)
    for X in [X_train, X_test]:
        assert_allclose(np.sum(splt.transform(X), axis=1), 1)


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_linear_regression(bias, intercept):
    """Test that B-splines fit a sinusodial curve pretty well."""
    X = np.linspace(0, 10, 100)[:, None]
    y = np.sin(X[:, 0]) + 2  # +2 to avoid the value 0 in assert_allclose
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=15,
                    degree=3,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict(X), y, rtol=1e-3)


@pytest.mark.parametrize(
    ["knots", "n_knots", "sample_weight", "expected_knots"],
    [
        ("uniform", 3, None, np.array([[0, 2], [3, 8], [6, 14]])),
        (
            "uniform",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [4, 8], [6, 14]]),
        ),
        ("uniform", 4, None, np.array([[0, 2], [2, 6], [4, 10], [6, 14]])),
        ("quantile", 3, None, np.array([[0, 2], [3, 3], [6, 14]])),
        (
            "quantile",
            3,
            np.array([0, 0, 1, 1, 0, 3, 1]),
            np.array([[2, 2], [5, 8], [6, 14]]),
        ),
    ],
)
def test_spline_transformer_get_base_knot_positions(
    knots, n_knots, sample_weight, expected_knots
):
    # Check the behaviour to find the positions of the knots with and without
    # `sample_weight`
    X = np.array([[0, 2], [0, 2], [2, 2], [3, 3], [4, 6], [5, 8], [6, 14]])
    base_knots = SplineTransformer._get_base_knot_positions(
        X=X, knots=knots, n_knots=n_knots, sample_weight=sample_weight
    )
    assert_allclose(base_knots, expected_knots)


@pytest.mark.parametrize(
    "knots, n_knots, degree",
    [
        ("uniform", 5, 3),
        ("uniform", 12, 8),
        (
            [[-1.0, 0.0], [0, 1.0], [0.1, 2.0], [0.2, 3.0], [0.3, 4.0], [1, 5.0]],
            None,
            3,
        ),
    ],
)
def test_spline_transformer_periodicity_of_extrapolation(knots, n_knots, degree):
    """Test that the SplineTransformer is periodic for multiple features."""
    X_1 = linspace((-1, 0), (1, 5), 10)
    X_2 = linspace((1, 5), (3, 10), 10)

    splt = SplineTransformer(
        knots=knots, n_knots=n_knots, degree=degree, extrapolation="periodic"
    )
    splt.fit(X_1)

    assert_allclose(splt.transform(X_1), splt.transform(X_2))


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
def test_spline_transformer_periodic_linear_regression(bias, intercept):
    """Test that B-splines fit a periodic curve pretty well."""
    # "+ 3" to avoid the value 0 in assert_allclose
    def f(x):
        return np.sin(2 * np.pi * x) - np.sin(8 * np.pi * x) + 3

    X = np.linspace(0, 1, 101)[:, None]
    pipe = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=20,
                    degree=3,
                    include_bias=bias,
                    extrapolation="periodic",
                ),
            ),
            ("ols", LinearRegression(fit_intercept=intercept)),
        ]
    )
    pipe.fit(X, f(X[:, 0]))

    # Generate larger array to check periodic extrapolation
    X_ = np.linspace(-1, 2, 301)[:, None]
    predictions = pipe.predict(X_)
    assert_allclose(predictions, f(X_[:, 0]), atol=0.01, rtol=0.01)
    assert_allclose(predictions[0:100], predictions[100:200], rtol=1e-3)


@pytest.mark.skipif(
    sp_version < parse_version("1.0.0"),
    reason="Periodic extrapolation not yet implemented for BSpline.",
)
def test_spline_transformer_periodic_spline_backport():
    """Test that the backport of extrapolate="periodic" works correctly"""
    X = np.linspace(-2, 3.5, 10)[:, None]
    degree = 2

    # Use periodic extrapolation backport in SplineTransformer
    transformer = SplineTransformer(
        degree=degree, extrapolation="periodic", knots=[[-1.0], [0.0], [1.0]]
    )
    Xt = transformer.fit_transform(X)

    # Use periodic extrapolation in BSpline
    coef = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    spl = BSpline(np.arange(-3, 4), coef, degree, "periodic")
    Xspl = spl(X[:, 0])
    assert_allclose(Xt, Xspl)


def test_spline_transformer_periodic_splines_periodicity():
    """
    Test if shifted knots result in the same transformation up to permutation.
    """
    X = np.linspace(0, 10, 101)[:, None]

    transformer_1 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )

    transformer_2 = SplineTransformer(
        degree=3,
        extrapolation="periodic",
        knots=[[1.0], [3.0], [4.0], [5.0], [8.0], [9.0]],
    )

    Xt_1 = transformer_1.fit_transform(X)
    Xt_2 = transformer_2.fit_transform(X)

    assert_allclose(Xt_1, Xt_2[:, [4, 0, 1, 2, 3]])


@pytest.mark.parametrize("degree", [3, 5])
def test_spline_transformer_periodic_splines_smoothness(degree):
    """Test that spline transformation is smooth at first / last knot."""
    X = np.linspace(-2, 10, 10_000)[:, None]

    transformer = SplineTransformer(
        degree=degree,
        extrapolation="periodic",
        knots=[[0.0], [1.0], [3.0], [4.0], [5.0], [8.0]],
    )
    Xt = transformer.fit_transform(X)

    delta = (X.max() - X.min()) / len(X)
    tol = 10 * delta

    dXt = Xt
    # We expect splines of degree `degree` to be (`degree`-1) times
    # continuously differentiable. I.e. for d = 0, ..., `degree` - 1 the d-th
    # derivative should be continuous. This is the case if the (d+1)-th
    # numerical derivative is reasonably small (smaller than `tol` in absolute
    # value). We thus compute d-th numeric derivatives for d = 1, ..., `degree`
    # and compare them to `tol`.
    #
    # Note that the 0-th derivative is the function itself, such that we are
    # also checking its continuity.
    for d in range(1, degree + 1):
        # Check continuity of the (d-1)-th derivative
        diff = np.diff(dXt, axis=0)
        assert np.abs(diff).max() < tol
        # Compute d-th numeric derivative
        dXt = diff / delta

    # As degree `degree` splines are not `degree` times continuously
    # differentiable at the knots, the `degree + 1`-th numeric derivative
    # should have spikes at the knots.
    diff = np.diff(dXt, axis=0)
    assert np.abs(diff).max() > 1


@pytest.mark.parametrize(["bias", "intercept"], [(True, False), (False, True)])
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
def test_spline_transformer_extrapolation(bias, intercept, degree):
    """Test that B-spline extrapolation works correctly."""
    # we use a straight line for that
    X = np.linspace(-1, 1, 100)[:, None]
    y = X.squeeze()

    # 'constant'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="constant",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-1, 1])

    # 'linear'
    pipe = Pipeline(
        [
            [
                "spline",
                SplineTransformer(
                    n_knots=4,
                    degree=degree,
                    include_bias=bias,
                    extrapolation="linear",
                ),
            ],
            ["ols", LinearRegression(fit_intercept=intercept)],
        ]
    )
    pipe.fit(X, y)
    assert_allclose(pipe.predict([[-10], [5]]), [-10, 5])

    # 'error'
    splt = SplineTransformer(
        n_knots=4, degree=degree, include_bias=bias, extrapolation="error"
    )
    splt.fit(X)
    with pytest.raises(ValueError):
        splt.transform([[-10]])
    with pytest.raises(ValueError):
        splt.transform([[5]])


def test_spline_transformer_kbindiscretizer():
    """Test that a B-spline of degree=0 is equivalent to KBinsDiscretizer."""
    rng = np.random.RandomState(97531)
    X = rng.randn(200).reshape(200, 1)
    n_bins = 5
    n_knots = n_bins + 1

    splt = SplineTransformer(
        n_knots=n_knots, degree=0, knots="quantile", include_bias=True
    )
    splines = splt.fit_transform(X)

    kbd = KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")
    kbins = kbd.fit_transform(X)

    # Though they should be exactly equal, we test approximately with high
    # accuracy.
    assert_allclose(splines, kbins, rtol=1e-13)


@pytest.mark.parametrize("n_knots", [5, 10])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("degree", [3, 5])
def test_spline_transformer_n_features_out(n_knots, include_bias, degree):
    """Test that transform results in n_features_out_ features."""
    splt = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=include_bias)
    X = np.linspace(0, 1, 10)[:, None]
    splt.fit(X)

    assert splt.transform(X).shape[1] == splt.n_features_out_


@pytest.mark.parametrize(
    "params, err_msg",
    [
        ({"degree": -1}, "degree must be a non-negative integer"),
        ({"degree": 2.5}, "degree must be a non-negative int or tuple"),
        ({"degree": "12"}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": "string"}, "degree must be a non-negative int or tuple"),
        ({"degree": (-1, 2)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (0, 1.5)}, r"degree=\(min_degree, max_degree\) must"),
        ({"degree": (3, 2)}, r"degree=\(min_degree, max_degree\) must"),
    ],
)
def test_polynomial_features_input_validation(params, err_msg):
    """Test that we raise errors for invalid input in PolynomialFeatures."""
    X = [[1], [2]]

    with pytest.raises(ValueError, match=err_msg):
        PolynomialFeatures(**params).fit(X)


@pytest.fixture()
def single_feature_degree3():
    X = np.arange(6)[:, np.newaxis]
    P = np.hstack([np.ones_like(X), X, X ** 2, X ** 3])
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1]),
        (3, False, True, [1]),
        ((2, 3), True, False, [0, 2, 3]),
        ((2, 3), False, False, [2, 3]),
        ((2, 3), True, True, [0]),
        ((2, 3), False, True, []),
    ],
)
@pytest.mark.parametrize(
    "sparse_X",
    [False, sparse.csr_matrix, sparse.csc_matrix],
)
def test_polynomial_features_one_feature(
    single_feature_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    sparse_X,
):
    """Test PolynomialFeatures on single feature up to degree 3."""
    X, P = single_feature_degree3
    if sparse_X:
        X = sparse_X(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if sparse_X:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


@pytest.fixture()
def two_features_degree3():
    X = np.arange(6).reshape((3, 2))
    x1 = X[:, :1]
    x2 = X[:, 1:]
    P = np.hstack(
        [
            x1 ** 0 * x2 ** 0,  # 0
            x1 ** 1 * x2 ** 0,  # 1
            x1 ** 0 * x2 ** 1,  # 2
            x1 ** 2 * x2 ** 0,  # 3
            x1 ** 1 * x2 ** 1,  # 4
            x1 ** 0 * x2 ** 2,  # 5
            x1 ** 3 * x2 ** 0,  # 6
            x1 ** 2 * x2 ** 1,  # 7
            x1 ** 1 * x2 ** 2,  # 8
            x1 ** 0 * x2 ** 3,  # 9
        ]
    )
    return X, P


@pytest.mark.parametrize(
    "degree, include_bias, interaction_only, indices",
    [
        (2, True, False, slice(0, 6)),
        (2, False, False, slice(1, 6)),
        (2, True, True, [0, 1, 2, 4]),
        (2, False, True, [1, 2, 4]),
        ((2, 2), True, False, [0, 3, 4, 5]),
        ((2, 2), False, False, [3, 4, 5]),
        ((2, 2), True, True, [0, 4]),
        ((2, 2), False, True, [4]),
        (3, True, False, slice(None, None)),
        (3, False, False, slice(1, None)),
        (3, True, True, [0, 1, 2, 4]),
        (3, False, True, [1, 2, 4]),
        ((2, 3), True, False, [0, 3, 4, 5, 6, 7, 8, 9]),
        ((2, 3), False, False, slice(3, None)),
        ((2, 3), True, True, [0, 4]),
        ((2, 3), False, True, [4]),
        ((3, 3), True, False, [0, 6, 7, 8, 9]),
        ((3, 3), False, False, [6, 7, 8, 9]),
        ((3, 3), True, True, [0]),
        ((3, 3), False, True, []),  # would need 3 input features
    ],
)
@pytest.mark.parametrize(
    "sparse_X",
    [False, sparse.csr_matrix, sparse.csc_matrix],
)
def test_polynomial_features_two_features(
    two_features_degree3,
    degree,
    include_bias,
    interaction_only,
    indices,
    sparse_X,
):
    """Test PolynomialFeatures on 2 features up to degree 3."""
    X, P = two_features_degree3
    if sparse_X:
        X = sparse_X(X)
    tf = PolynomialFeatures(
        degree=degree, include_bias=include_bias, interaction_only=interaction_only
    ).fit(X)
    out = tf.transform(X)
    if sparse_X:
        out = out.toarray()
    assert_allclose(out, P[:, indices])
    if tf.n_output_features_ > 0:
        assert tf.powers_.shape == (tf.n_output_features_, tf.n_features_in_)


# TODO: Remove in 1.2 when get_feature_names is removed.
@pytest.mark.filterwarnings("ignore::FutureWarning:sklearn")
@pytest.mark.parametrize("get_names", ["get_feature_names", "get_feature_names_out"])
def test_polynomial_feature_names(get_names):
    X = np.arange(30).reshape(10, 3)
    poly = PolynomialFeatures(degree=2, include_bias=True).fit(X)
    feature_names = poly.get_feature_names()
    assert_array_equal(
        ["1", "x0", "x1", "x2", "x0^2", "x0 x1", "x0 x2", "x1^2", "x1 x2", "x2^2"],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=3, include_bias=False).fit(X)
    feature_names = getattr(poly, get_names)(["a", "b", "c"])
    assert_array_equal(
        [
            "a",
            "b",
            "c",
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(degree=(2, 3), include_bias=False).fit(X)
    feature_names = getattr(poly, get_names)(["a", "b", "c"])
    assert_array_equal(
        [
            "a^2",
            "a b",
            "a c",
            "b^2",
            "b c",
            "c^2",
            "a^3",
            "a^2 b",
            "a^2 c",
            "a b^2",
            "a b c",
            "a c^2",
            "b^3",
            "b^2 c",
            "b c^2",
            "c^3",
        ],
        feature_names,
    )
    assert len(feature_names) == poly.transform(X).shape[1]

    poly = PolynomialFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(X)
    feature_names = getattr(poly, get_names)(["a", "b", "c"])
    assert_array_equal(["1", "a b c"], feature_names)
    assert len(feature_names) == poly.transform(X).shape[1]

    # test some unicode
    poly = PolynomialFeatures(degree=1, include_bias=True).fit(X)
    feature_names = poly.get_feature_names(["\u0001F40D", "\u262E", "\u05D0"])
    assert_array_equal(["1", "\u0001F40D", "\u262E", "\u05D0"], feature_names)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
        (4, False, False, np.float64),
        (4, False, True, np.float64),
    ],
)
def test_polynomial_features_csc_X(deg, include_bias, interaction_only, dtype):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csc = sparse.csc_matrix(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csc = est.fit_transform(X_csc.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert isinstance(Xt_csc, sparse.csc_matrix)
    assert Xt_csc.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csc.A, Xt_dense)


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (1, True, False, int),
        (2, True, False, int),
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
def test_polynomial_features_csr_X(deg, include_bias, interaction_only, dtype):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, (100, 2))
    X_csr = sparse.csr_matrix(X)

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype, copy=False))

    assert isinstance(Xt_csr, sparse.csr_matrix)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize("n_features", [1, 4, 5])
@pytest.mark.parametrize(
    "min_degree, max_degree", [(0, 1), (0, 2), (1, 3), (0, 4), (3, 4)]
)
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_num_combinations(
    n_features,
    min_degree,
    max_degree,
    interaction_only,
    include_bias,
):
    """
    Test that n_output_features_ is calculated correctly.
    """
    x = sparse.csr_matrix(([1], ([0], [n_features - 1])))
    est = PolynomialFeatures(
        degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    est.fit(x)
    num_combos = est.n_output_features_

    combos = PolynomialFeatures._combinations(
        n_features=n_features,
        min_degree=0,
        max_degree=max_degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    assert num_combos == sum([1 for _ in combos])


@pytest.mark.parametrize(
    ["deg", "include_bias", "interaction_only", "dtype"],
    [
        (2, True, False, np.float32),
        (2, True, False, np.float64),
        (3, False, False, np.float64),
        (3, False, True, np.float64),
    ],
)
def test_polynomial_features_csr_X_floats(deg, include_bias, interaction_only, dtype):
    X_csr = sparse_random(1000, 10, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(
        deg, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr.astype(dtype))
    Xt_dense = est.fit_transform(X.astype(dtype))

    assert isinstance(Xt_csr, sparse.csr_matrix)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize(
    ["zero_row_index", "deg", "interaction_only"],
    [
        (0, 2, True),
        (1, 2, True),
        (2, 2, True),
        (0, 3, True),
        (1, 3, True),
        (2, 3, True),
        (0, 2, False),
        (1, 2, False),
        (2, 2, False),
        (0, 3, False),
        (1, 3, False),
        (2, 3, False),
    ],
)
def test_polynomial_features_csr_X_zero_row(zero_row_index, deg, interaction_only):
    X_csr = sparse_random(3, 10, 1.0, random_state=0).tocsr()
    X_csr[zero_row_index, :] = 0.0
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, include_bias=False, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert isinstance(Xt_csr, sparse.csr_matrix)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


# This degree should always be one more than the highest degree supported by
# _csr_expansion.
@pytest.mark.parametrize(
    ["include_bias", "interaction_only"],
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_polynomial_features_csr_X_degree_4(include_bias, interaction_only):
    X_csr = sparse_random(1000, 10, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(
        4, include_bias=include_bias, interaction_only=interaction_only
    )
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert isinstance(Xt_csr, sparse.csr_matrix)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


@pytest.mark.parametrize(
    ["deg", "dim", "interaction_only"],
    [
        (2, 1, True),
        (2, 2, True),
        (3, 1, True),
        (3, 2, True),
        (3, 3, True),
        (2, 1, False),
        (2, 2, False),
        (3, 1, False),
        (3, 2, False),
        (3, 3, False),
    ],
)
def test_polynomial_features_csr_X_dim_edges(deg, dim, interaction_only):
    X_csr = sparse_random(1000, dim, 0.5, random_state=0).tocsr()
    X = X_csr.toarray()

    est = PolynomialFeatures(deg, interaction_only=interaction_only)
    Xt_csr = est.fit_transform(X_csr)
    Xt_dense = est.fit_transform(X)

    assert isinstance(Xt_csr, sparse.csr_matrix)
    assert Xt_csr.dtype == Xt_dense.dtype
    assert_array_almost_equal(Xt_csr.A, Xt_dense)


def test_polynomial_features_deprecated_n_input_features():
    # check that we raise a deprecation warning when accessing
    # `n_input_features_`. FIXME: remove in 1.2
    depr_msg = (
        "The attribute `n_input_features_` was deprecated in version "
        "1.0 and will be removed in 1.2."
    )
    X = np.arange(10).reshape(5, 2)

    with pytest.warns(FutureWarning, match=depr_msg):
        PolynomialFeatures().fit(X).n_input_features_


# TODO: Remove in 1.2 when get_feature_names is removed
@pytest.mark.parametrize("Transformer", [SplineTransformer, PolynomialFeatures])
def test_get_feature_names_deprecated(Transformer):
    X = np.arange(30).reshape(10, 3)
    poly = Transformer().fit(X)
    msg = "get_feature_names is deprecated in 1.0"
    with pytest.warns(FutureWarning, match=msg):
        poly.get_feature_names()
