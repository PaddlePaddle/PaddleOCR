import numpy as np

from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import (arch32, assert_almost_equal,
                                     assert_array_less, assert_equal, xfail)
from skimage.measure import LineModelND, CircleModel, EllipseModel, ransac
from skimage.measure.fit import _dynamic_max_trials
from skimage.transform import AffineTransform


def test_line_model_invalid_input():
    with testing.raises(ValueError):
        LineModelND().estimate(np.empty((1, 3)))


def test_line_model_predict():
    model = LineModelND()
    model.params = ((0, 0), (1, 1))
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))


def test_line_model_nd_invalid_input():
    with testing.raises(ValueError):
        LineModelND().predict_x(np.zeros(1))

    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1))

    with testing.raises(ValueError):
        LineModelND().predict_x(np.zeros(1), np.zeros(1))

    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1))

    with testing.raises(ValueError):
        LineModelND().predict_y(np.zeros(1), np.zeros(1))

    with testing.raises(ValueError):
        LineModelND().estimate(np.empty((1, 3)))

    with testing.raises(ValueError):
        LineModelND().residuals(np.empty((1, 3)))

    data = np.empty((1, 2))
    with testing.raises(ValueError):
        LineModelND().estimate(data)


def test_line_model_nd_predict():
    model = LineModelND()
    model.params = (np.array([0, 0]), np.array([0.2, 0.8]))
    x = np.arange(-10, 10)
    y = model.predict_y(x)
    assert_almost_equal(x, model.predict_x(y))


def test_line_model_nd_estimate():
    # generate original data without noise
    model0 = LineModelND()
    model0.params = (np.array([0, 0, 0], dtype='float'),
                     np.array([1, 1, 1], dtype='float')/np.sqrt(3))
    # we scale the unit vector with a factor 10 when generating points on the
    # line in order to compensate for the scale of the random noise
    data0 = (model0.params[0] +
             10 * np.arange(-100, 100)[..., np.newaxis] * model0.params[1])

    # add gaussian noise to data
    random_state = np.random.default_rng(1234)
    data = data0 + random_state.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = LineModelND()
    model_est.estimate(data)
    # assert_almost_equal(model_est.residuals(data0), np.zeros(len(data)), 1)

    # test whether estimated parameters are correct
    # we use the following geometric property: two aligned vectors have
    # a cross-product equal to zero
    # test if direction vectors are aligned
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1],
                                                model_est.params[1])), 0, 1)
    # test if origins are aligned with the direction
    a = model_est.params[0] - model0.params[0]
    if np.linalg.norm(a) > 0:
        a /= np.linalg.norm(a)
    assert_almost_equal(np.linalg.norm(np.cross(model0.params[1], a)), 0, 1)


def test_line_model_nd_residuals():
    model = LineModelND()
    model.params = (np.array([0, 0, 0]), np.array([0, 0, 1]))
    assert_equal(abs(model.residuals(np.array([[0, 0, 0]]))), 0)
    assert_equal(abs(model.residuals(np.array([[0, 0, 1]]))), 0)
    assert_equal(abs(model.residuals(np.array([[10, 0, 0]]))), 10)
    # test params argument in model.rediduals
    data = np.array([[10, 0, 0]])
    params = (np.array([0, 0, 0]), np.array([2, 0, 0]))
    assert_equal(abs(model.residuals(data, params=params)), 30)


def test_line_modelND_under_determined():
    data = np.empty((1, 3))
    with testing.raises(ValueError):
        LineModelND().estimate(data)


def test_circle_model_invalid_input():
    with testing.raises(ValueError):
        CircleModel().estimate(np.empty((5, 3)))


def test_circle_model_predict():
    model = CircleModel()
    r = 5
    model.params = (0, 0, r)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 5), (-5, 0), (0, -5)))
    assert_almost_equal(xy, model.predict_xy(t))


def test_circle_model_estimate():
    # generate original data without noise
    model0 = CircleModel()
    model0.params = (10, 12, 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add gaussian noise to data
    random_state = np.random.default_rng(1234)
    data = data0 + random_state.normal(size=data0.shape)

    # estimate parameters of noisy data
    model_est = CircleModel()
    model_est.estimate(data)

    # test whether estimated parameters almost equal original parameters
    assert_almost_equal(model0.params, model_est.params, 0)


def test_circle_model_int_overflow():
    xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)
    xy += 500

    model = CircleModel()
    model.estimate(xy)

    assert_almost_equal(model.params, [500, 500, 1])


def test_circle_model_residuals():
    model = CircleModel()
    model.params = (0, 0, 5)
    assert_almost_equal(abs(model.residuals(np.array([[5, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[6, 6]]))),
                        np.sqrt(2 * 6**2) - 5)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 5)


def test_circle_model_insufficient_data():
    model = CircleModel()

    with expected_warnings(["Input data does not contain enough significant"]):
        model.estimate(np.array([[1, 2], [3, 4]]))

    with expected_warnings(["Input data does not contain enough significant"]):
        model.estimate(np.ones((6, 2)))

    with expected_warnings(["Input data does not contain enough significant"]):
        model.estimate(np.array([[0, 0], [1, 1], [2, 2]]))


def test_ellipse_model_invalid_input():
    with testing.raises(ValueError):
        EllipseModel().estimate(np.empty((5, 3)))


def test_ellipse_model_predict():
    model = EllipseModel()
    model.params = (0, 0, 5, 10, 0)
    t = np.arange(0, 2 * np.pi, np.pi / 2)

    xy = np.array(((5, 0), (0, 10), (-5, 0), (0, -10)))
    assert_almost_equal(xy, model.predict_xy(t))


def test_ellipse_model_estimate():
    for angle in range(0, 180, 15):
        rad = np.deg2rad(angle)
        # generate original data without noise
        model0 = EllipseModel()
        model0.params = (10, 20, 15, 25, rad)
        t = np.linspace(0, 2 * np.pi, 100)
        data0 = model0.predict_xy(t)

        # add gaussian noise to data
        random_state = np.random.default_rng(1234)
        data = data0 + random_state.normal(size=data0.shape)

        # estimate parameters of noisy data
        model_est = EllipseModel()
        model_est.estimate(data)

        # test whether estimated parameters almost equal original parameters
        assert_almost_equal(model0.params[:2], model_est.params[:2], 0)
        res = model_est.residuals(data0)
        assert_array_less(res, np.ones(res.shape))


def test_ellipse_model_estimate_from_data():
    data = np.array([
        [264, 854], [265, 875], [268, 863], [270, 857], [275, 905], [285, 915],
        [305, 925], [324, 934], [335, 764], [336, 915], [345, 925], [345, 945],
        [354, 933], [355, 745], [364, 936], [365, 754], [375, 745], [375, 735],
        [385, 736], [395, 735], [394, 935], [405, 727], [415, 736], [415, 727],
        [425, 727], [426, 929], [435, 735], [444, 933], [445, 735], [455, 724],
        [465, 934], [465, 735], [475, 908], [475, 726], [485, 753], [485, 728],
        [492, 762], [495, 745], [491, 910], [493, 909], [499, 904], [505, 905],
        [504, 747], [515, 743], [516, 752], [524, 855], [525, 844], [525, 885],
        [533, 845], [533, 873], [535, 883], [545, 874], [543, 864], [553, 865],
        [553, 845], [554, 825], [554, 835], [563, 845], [565, 826], [563, 855],
        [563, 795], [565, 735], [573, 778], [572, 815], [574, 804], [575, 665],
        [575, 685], [574, 705], [574, 745], [575, 875], [572, 732], [582, 795],
        [579, 709], [583, 805], [583, 854], [586, 755], [584, 824], [585, 655],
        [581, 718], [586, 844], [585, 915], [587, 905], [594, 824], [593, 855],
        [590, 891], [594, 776], [596, 767], [593, 763], [603, 785], [604, 775],
        [603, 885], [605, 753], [605, 655], [606, 935], [603, 761], [613, 802],
        [613, 945], [613, 965], [615, 693], [617, 665], [623, 962], [624, 972],
        [625, 995], [633, 673], [633, 965], [633, 683], [633, 692], [633, 954],
        [634, 1016], [635, 664], [641, 804], [637, 999], [641, 956], [643, 946],
        [643, 926], [644, 975], [643, 655], [646, 705], [651, 664], [651, 984],
        [647, 665], [651, 715], [651, 725], [651, 734], [647, 809], [651, 825],
        [651, 873], [647, 900], [652, 917], [651, 944], [652, 742], [648, 811],
        [651, 994], [652, 783], [650, 911], [654, 879]], dtype=np.int32)

    # estimate parameters of real data
    model = EllipseModel()
    model.estimate(data)

    # test whether estimated parameters are smaller then 1000, so means stable
    assert_array_less(model.params[:4], np.full(4, 1000))

    # test whether all parameters are more than 0. Negative values were the
    # result of an integer overflow
    assert_array_less(np.zeros(4), np.abs(model.params[:4]))


@xfail(condition=arch32,
       reason=('Known test failure on 32-bit platforms. See links for '
               'details: '
               'https://github.com/scikit-image/scikit-image/issues/3091 '
               'https://github.com/scikit-image/scikit-image/issues/2670'))
def test_ellipse_model_estimate_failers():
    # estimate parameters of real data
    model = EllipseModel()
    assert not model.estimate(np.ones((5, 2)))
    assert not model.estimate(np.array([[50, 80], [51, 81], [52, 80]]))


def test_ellipse_model_residuals():
    model = EllipseModel()
    # vertical line through origin
    model.params = (0, 0, 10, 5, 0)
    assert_almost_equal(abs(model.residuals(np.array([[10, 0]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 5]]))), 0)
    assert_almost_equal(abs(model.residuals(np.array([[0, 10]]))), 5)


def test_ransac_shape():
    # generate original data without noise
    model0 = CircleModel()
    model0.params = (10, 12, 3)
    t = np.linspace(0, 2 * np.pi, 1000)
    data0 = model0.predict_xy(t)

    # add some faulty data
    outliers = (10, 30, 200)
    data0[outliers[0], :] = (1000, 1000)
    data0[outliers[1], :] = (-50, 50)
    data0[outliers[2], :] = (-100, -10)

    # estimate parameters of corrupted data
    model_est, inliers = ransac(data0, CircleModel, 3, 5, random_state=1)

    # test whether estimated parameters equal original parameters
    assert_almost_equal(model0.params, model_est.params)
    for outlier in outliers:
        assert outlier not in inliers


def test_ransac_geometric():
    random_state = np.random.default_rng(12373240)

    # generate original data without noise
    src = 100 * random_state.random((50, 2))
    model0 = AffineTransform(scale=(0.5, 0.3), rotation=1,
                             translation=(10, 20))
    dst = model0(src)

    # add some faulty data
    outliers = (0, 5, 20)
    dst[outliers[0]] = (10000, 10000)
    dst[outliers[1]] = (-100, 100)
    dst[outliers[2]] = (50, 50)

    # estimate parameters of corrupted data
    model_est, inliers = ransac((src, dst), AffineTransform, 2, 20,
                                random_state=random_state)

    # test whether estimated parameters equal original parameters
    assert_almost_equal(model0.params, model_est.params)
    assert np.all(np.nonzero(inliers == False)[0] == outliers)


def test_ransac_is_data_valid():
    def is_data_valid(data):
        return data.shape[0] > 2
    with expected_warnings(["No inliers found"]):
        model, inliers = ransac(np.empty((10, 2)), LineModelND, 2, np.inf,
                                is_data_valid=is_data_valid, random_state=1)
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_is_model_valid():
    def is_model_valid(model, data):
        return False
    with expected_warnings(["No inliers found"]):
        model, inliers = ransac(np.empty((10, 2)), LineModelND, 2, np.inf,
                                is_model_valid=is_model_valid, random_state=1)
    assert_equal(model, None)
    assert_equal(inliers, None)


def test_ransac_dynamic_max_trials():
    # Numbers hand-calculated and confirmed on page 119 (Table 4.3) in
    #   Hartley, R.~I. and Zisserman, A., 2004,
    #   Multiple View Geometry in Computer Vision, Second Edition,
    #   Cambridge University Press, ISBN: 0521540518

    # e = 0%, min_samples = X
    assert_equal(_dynamic_max_trials(100, 100, 2, 0.99), 1)

    # e = 5%, min_samples = 2
    assert_equal(_dynamic_max_trials(95, 100, 2, 0.99), 2)
    # e = 10%, min_samples = 2
    assert_equal(_dynamic_max_trials(90, 100, 2, 0.99), 3)
    # e = 30%, min_samples = 2
    assert_equal(_dynamic_max_trials(70, 100, 2, 0.99), 7)
    # e = 50%, min_samples = 2
    assert_equal(_dynamic_max_trials(50, 100, 2, 0.99), 17)

    # e = 5%, min_samples = 8
    assert_equal(_dynamic_max_trials(95, 100, 8, 0.99), 5)
    # e = 10%, min_samples = 8
    assert_equal(_dynamic_max_trials(90, 100, 8, 0.99), 9)
    # e = 30%, min_samples = 8
    assert_equal(_dynamic_max_trials(70, 100, 8, 0.99), 78)
    # e = 50%, min_samples = 8
    assert_equal(_dynamic_max_trials(50, 100, 8, 0.99), 1177)

    # e = 0%, min_samples = 5
    assert_equal(_dynamic_max_trials(1, 100, 5, 0), 0)
    assert_equal(_dynamic_max_trials(1, 100, 5, 1), np.inf)


def test_ransac_invalid_input():
    # `residual_threshold` must be greater than zero
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2,
               residual_threshold=-0.5)
    # "`max_trials` must be greater than zero"
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2,
               residual_threshold=0, max_trials=-1)
    # `stop_probability` must be in range (0, 1)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2,
               residual_threshold=0, stop_probability=-1)
    # `stop_probability` must be in range (0, 1)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=2,
               residual_threshold=0, stop_probability=1.01)
    # `min_samples` as ratio must be in range (0, nb)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=0,
               residual_threshold=0)
    # `min_samples` as ratio must be in range (0, nb)
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=10,
               residual_threshold=0)
    # `min_samples` must be greater than zero
    with testing.raises(ValueError):
        ransac(np.zeros((10, 2)), None, min_samples=-1,
               residual_threshold=0)


def test_ransac_sample_duplicates():
    class DummyModel(object):

        """Dummy model to check for duplicates."""

        def estimate(self, data):
            # Assert that all data points are unique.
            assert_equal(np.unique(data).size, data.size)
            return True

        def residuals(self, data):
            return np.ones(len(data), dtype=np.double)

    # Create dataset with four unique points. Force 10 iterations
    # and check that there are no duplicated data points.
    data = np.arange(4)
    with expected_warnings(["No inliers found"]):
        ransac(data, DummyModel, min_samples=3, residual_threshold=0.0,
               max_trials=10)


def test_ransac_with_no_final_inliers():
    data = np.random.rand(5, 2)
    with expected_warnings(['No inliers found. Model not fitted']):
        model, inliers = ransac(data, model_class=LineModelND, min_samples=3,
                                residual_threshold=0, random_state=1523427)
    assert inliers is None
    assert model is None


def test_ransac_non_valid_best_model():
    """Example from GH issue #5572"""
    def is_model_valid(model, *random_data) -> bool:
        """Allow models with a maximum of 10 degree tilt from the vertical

        """
        tilt = abs(np.arccos(np.dot(model.params[1], [0, 0, 1])))
        return tilt <= (10 / 180 * np.pi)

    rnd = np.random.RandomState(1)
    data = np.linspace([0, 0, 0], [0.3, 0, 1], 1000) + rnd.rand(1000, 3) - 0.5
    with expected_warnings(["Estimated model is not valid"]):
        ransac(data, LineModelND, min_samples=2,
               residual_threshold=0.3, max_trials=50, random_state=0,
               is_model_valid=is_model_valid)
