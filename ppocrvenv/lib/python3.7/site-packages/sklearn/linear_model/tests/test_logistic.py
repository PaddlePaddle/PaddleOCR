import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
import scipy.sparse as sp
from scipy import linalg, optimize, sparse

import pytest

from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import log_loss
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import compute_class_weight, _IS_32BIT
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import scale
from sklearn.utils._testing import skip_if_no_parallel

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._logistic import (
    LogisticRegression,
    _logistic_regression_path,
    LogisticRegressionCV,
    _logistic_loss_and_grad,
    _logistic_grad_hess,
    _multinomial_grad_hess,
    _logistic_loss,
    _log_reg_scoring_path,
)

X = [[-1, 0], [0, 1], [1, 1]]
X_sp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
iris = load_iris()


def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]

    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)

    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_samples, n_classes)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)


def test_predict_2_classes():
    # Simple sanity check on a 2 classes dataset
    # Make sure it predicts the correct result on simple datasets.
    check_predictions(LogisticRegression(random_state=0), X, Y1)
    check_predictions(LogisticRegression(random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(C=100, random_state=0), X, Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(fit_intercept=False, random_state=0), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False, random_state=0), X_sp, Y1)


def test_error():
    # Test for appropriate exception on errors
    msg = "Penalty term must be positive"

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(C=-1).fit(X, Y1)

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(C="test").fit(X, Y1)

    msg = "is not a valid scoring value"
    with pytest.raises(ValueError, match=msg):
        LogisticRegressionCV(scoring="bad-scorer", cv=2).fit(X, Y1)

    for LR in [LogisticRegression, LogisticRegressionCV]:
        msg = "Tolerance for stopping criteria must be positive"

        with pytest.raises(ValueError, match=msg):
            LR(tol=-1).fit(X, Y1)

        with pytest.raises(ValueError, match=msg):
            LR(tol="test").fit(X, Y1)

        msg = "Maximum number of iteration must be positive"

        with pytest.raises(ValueError, match=msg):
            LR(max_iter=-1).fit(X, Y1)

        with pytest.raises(ValueError, match=msg):
            LR(max_iter="test").fit(X, Y1)


def test_logistic_cv_mock_scorer():
    class MockScorer:
        def __init__(self):
            self.calls = 0
            self.scores = [0.1, 0.4, 0.8, 0.5]

        def __call__(self, model, X, y, sample_weight=None):
            score = self.scores[self.calls % len(self.scores)]
            self.calls += 1
            return score

    mock_scorer = MockScorer()
    Cs = [1, 2, 3, 4]
    cv = 2

    lr = LogisticRegressionCV(Cs=Cs, scoring=mock_scorer, cv=cv)
    lr.fit(X, Y1)

    # Cs[2] has the highest score (0.8) from MockScorer
    assert lr.C_[0] == Cs[2]

    # scorer called 8 times (cv*len(Cs))
    assert mock_scorer.calls == cv * len(Cs)

    # reset mock_scorer
    mock_scorer.calls = 0
    custom_score = lr.score(X, lr.predict(X))

    assert custom_score == mock_scorer.scores[0]
    assert mock_scorer.calls == 1


def test_logistic_cv_score_does_not_warn_by_default():
    lr = LogisticRegressionCV(cv=2)
    lr.fit(X, Y1)

    with pytest.warns(None) as record:
        lr.score(X, lr.predict(X))
    assert len(record) == 0


@skip_if_no_parallel
def test_lr_liblinear_warning():
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]

    lr = LogisticRegression(solver="liblinear", n_jobs=2)
    warning_message = (
        "'n_jobs' > 1 does not have any effect when"
        " 'solver' is set to 'liblinear'. Got 'n_jobs'"
        " = 2."
    )
    with pytest.warns(UserWarning, match=warning_message):
        lr.fit(iris.data, target)


def test_predict_3_classes():
    check_predictions(LogisticRegression(C=10), X, Y2)
    check_predictions(LogisticRegression(C=10), X_sp, Y2)


def test_predict_iris():
    # Test logistic regression with the iris dataset
    n_samples, n_features = iris.data.shape

    target = iris.target_names[iris.target]

    # Test that both multinomial and OvR solvers handle
    # multiclass data correctly and give good accuracy
    # score (>0.95) for the training data.
    for clf in [
        LogisticRegression(C=len(iris.data), solver="liblinear", multi_class="ovr"),
        LogisticRegression(C=len(iris.data), solver="lbfgs", multi_class="multinomial"),
        LogisticRegression(
            C=len(iris.data), solver="newton-cg", multi_class="multinomial"
        ),
        LogisticRegression(
            C=len(iris.data), solver="sag", tol=1e-2, multi_class="ovr", random_state=42
        ),
        LogisticRegression(
            C=len(iris.data),
            solver="saga",
            tol=1e-2,
            multi_class="ovr",
            random_state=42,
        ),
    ]:
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        assert np.mean(pred == target) > 0.95

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))

        pred = iris.target_names[probabilities.argmax(axis=1)]
        assert np.mean(pred == target) > 0.95


@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg", "sag", "saga"])
def test_multinomial_validation(solver):
    lr = LogisticRegression(C=-1, solver=solver, multi_class="multinomial")

    with pytest.raises(ValueError):
        lr.fit([[0, 1], [1, 0]], [0, 1])


@pytest.mark.parametrize("LR", [LogisticRegression, LogisticRegressionCV])
def test_check_solver_option(LR):
    X, y = iris.data, iris.target

    msg = (
        r"Logistic Regression supports only solvers in \['liblinear', "
        r"'newton-cg', 'lbfgs', 'sag', 'saga'\], got wrong_name."
    )
    lr = LR(solver="wrong_name", multi_class="ovr")
    with pytest.raises(ValueError, match=msg):
        lr.fit(X, y)

    msg = "multi_class should be 'multinomial', 'ovr' or 'auto'. Got wrong_name"
    lr = LR(solver="newton-cg", multi_class="wrong_name")
    with pytest.raises(ValueError, match=msg):
        lr.fit(X, y)

    # only 'liblinear' solver
    msg = "Solver liblinear does not support a multinomial backend."
    lr = LR(solver="liblinear", multi_class="multinomial")
    with pytest.raises(ValueError, match=msg):
        lr.fit(X, y)

    # all solvers except 'liblinear' and 'saga'
    for solver in ["newton-cg", "lbfgs", "sag"]:
        msg = "Solver %s supports only 'l2' or 'none' penalties," % solver
        lr = LR(solver=solver, penalty="l1", multi_class="ovr")
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)
    for solver in ["newton-cg", "lbfgs", "sag", "saga"]:
        msg = "Solver %s supports only dual=False, got dual=True" % solver
        lr = LR(solver=solver, dual=True, multi_class="ovr")
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # only saga supports elasticnet. We only test for liblinear because the
    # error is raised before for the other solvers (solver %s supports only l2
    # penalties)
    for solver in ["liblinear"]:
        msg = "Only 'saga' solver supports elasticnet penalty, got solver={}.".format(
            solver
        )
        lr = LR(solver=solver, penalty="elasticnet")
        with pytest.raises(ValueError, match=msg):
            lr.fit(X, y)

    # liblinear does not support penalty='none'
    msg = "penalty='none' is not supported for the liblinear solver"
    lr = LR(penalty="none", solver="liblinear")
    with pytest.raises(ValueError, match=msg):
        lr.fit(X, y)


@pytest.mark.parametrize("solver", ["lbfgs", "newton-cg", "sag", "saga"])
def test_multinomial_binary(solver):
    # Test multinomial LR on a binary problem.
    target = (iris.target > 0).astype(np.intp)
    target = np.array(["setosa", "not-setosa"])[target]

    clf = LogisticRegression(
        solver=solver, multi_class="multinomial", random_state=42, max_iter=2000
    )
    clf.fit(iris.data, target)

    assert clf.coef_.shape == (1, iris.data.shape[1])
    assert clf.intercept_.shape == (1,)
    assert_array_equal(clf.predict(iris.data), target)

    mlr = LogisticRegression(
        solver=solver, multi_class="multinomial", random_state=42, fit_intercept=False
    )
    mlr.fit(iris.data, target)
    pred = clf.classes_[np.argmax(clf.predict_log_proba(iris.data), axis=1)]
    assert np.mean(pred == target) > 0.9


def test_multinomial_binary_probabilities():
    # Test multinomial LR gives expected probabilities based on the
    # decision function, for a binary problem.
    X, y = make_classification()
    clf = LogisticRegression(multi_class="multinomial", solver="saga")
    clf.fit(X, y)

    decision = clf.decision_function(X)
    proba = clf.predict_proba(X)

    expected_proba_class_1 = np.exp(decision) / (np.exp(decision) + np.exp(-decision))
    expected_proba = np.c_[1 - expected_proba_class_1, expected_proba_class_1]

    assert_almost_equal(proba, expected_proba)


def test_sparsify():
    # Test sparsify and densify members.
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    clf = LogisticRegression(random_state=0).fit(iris.data, target)

    pred_d_d = clf.decision_function(iris.data)

    clf.sparsify()
    assert sp.issparse(clf.coef_)
    pred_s_d = clf.decision_function(iris.data)

    sp_data = sp.coo_matrix(iris.data)
    pred_s_s = clf.decision_function(sp_data)

    clf.densify()
    pred_d_s = clf.decision_function(sp_data)

    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)


def test_inconsistent_input():
    # Test that an exception is raised on inconsistent input
    rng = np.random.RandomState(0)
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0

    clf = LogisticRegression(random_state=0)

    # Wrong dimensions for training data
    y_wrong = y_[:-1]

    with pytest.raises(ValueError):
        clf.fit(X, y_wrong)

    # Wrong dimensions for test data
    with pytest.raises(ValueError):
        clf.fit(X_, y_).predict(rng.random_sample((3, 12)))


def test_write_parameters():
    # Test that we can write to coef_ and intercept_
    clf = LogisticRegression(random_state=0)
    clf.fit(X, Y1)
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    assert_array_almost_equal(clf.decision_function(X), 0)


def test_nan():
    # Test proper NaN handling.
    # Regression test for Issue #252: fit used to go into an infinite loop.
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    logistic = LogisticRegression(random_state=0)

    with pytest.raises(ValueError):
        logistic.fit(Xnan, Y1)


def test_consistency_path():
    # Test that the path algorithm is consistent
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = np.logspace(0, 4, 10)

    f = ignore_warnings
    # can't test with fit_intercept=True since LIBLINEAR
    # penalizes the intercept
    for solver in ["sag", "saga"]:
        coefs, Cs, _ = f(_logistic_regression_path)(
            X,
            y,
            Cs=Cs,
            fit_intercept=False,
            tol=1e-5,
            solver=solver,
            max_iter=1000,
            multi_class="ovr",
            random_state=0,
        )
        for i, C in enumerate(Cs):
            lr = LogisticRegression(
                C=C,
                fit_intercept=False,
                tol=1e-5,
                solver=solver,
                multi_class="ovr",
                random_state=0,
                max_iter=1000,
            )
            lr.fit(X, y)
            lr_coef = lr.coef_.ravel()
            assert_array_almost_equal(
                lr_coef, coefs[i], decimal=4, err_msg="with solver = %s" % solver
            )

    # test for fit_intercept=True
    for solver in ("lbfgs", "newton-cg", "liblinear", "sag", "saga"):
        Cs = [1e3]
        coefs, Cs, _ = f(_logistic_regression_path)(
            X,
            y,
            Cs=Cs,
            tol=1e-6,
            solver=solver,
            intercept_scaling=10000.0,
            random_state=0,
            multi_class="ovr",
        )
        lr = LogisticRegression(
            C=Cs[0],
            tol=1e-4,
            intercept_scaling=10000.0,
            random_state=0,
            multi_class="ovr",
            solver=solver,
        )
        lr.fit(X, y)
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        assert_array_almost_equal(
            lr_coef, coefs[0], decimal=4, err_msg="with solver = %s" % solver
        )


def test_logistic_regression_path_convergence_fail():
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = [1e3]

    # Check that the convergence message points to both a model agnostic
    # advice (scaling the data) and to the logistic regression specific
    # documentation that includes hints on the solver configuration.
    with pytest.warns(ConvergenceWarning) as record:
        with warnings.catch_warnings():
            # scipy 1.3.0 uses tostring which is deprecated in numpy
            warnings.filterwarnings("ignore", "tostring", DeprecationWarning)
            _logistic_regression_path(
                X, y, Cs=Cs, tol=0.0, max_iter=1, random_state=0, verbose=0
            )

    assert len(record) == 1
    warn_msg = record[0].message.args[0]
    assert "lbfgs failed to converge" in warn_msg
    assert "Increase the number of iterations" in warn_msg
    assert "scale the data" in warn_msg
    assert "linear_model.html#logistic-regression" in warn_msg


def test_liblinear_dual_random_state():
    # random_state is relevant for liblinear solver only if dual=True
    X, y = make_classification(n_samples=20, random_state=0)
    lr1 = LogisticRegression(
        random_state=0,
        dual=True,
        max_iter=1,
        tol=1e-15,
        solver="liblinear",
        multi_class="ovr",
    )
    lr1.fit(X, y)
    lr2 = LogisticRegression(
        random_state=0,
        dual=True,
        max_iter=1,
        tol=1e-15,
        solver="liblinear",
        multi_class="ovr",
    )
    lr2.fit(X, y)
    lr3 = LogisticRegression(
        random_state=8,
        dual=True,
        max_iter=1,
        tol=1e-15,
        solver="liblinear",
        multi_class="ovr",
    )
    lr3.fit(X, y)

    # same result for same random state
    assert_array_almost_equal(lr1.coef_, lr2.coef_)
    # different results for different random states
    msg = "Arrays are not almost equal to 6 decimals"
    with pytest.raises(AssertionError, match=msg):
        assert_array_almost_equal(lr1.coef_, lr3.coef_)


def test_logistic_loss_and_grad():
    X_ref, y = make_classification(n_samples=20, random_state=0)
    n_features = X_ref.shape[1]

    X_sp = X_ref.copy()
    X_sp[X_sp < 0.1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.zeros(n_features)

        # First check that our derivation of the grad is correct
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.0)
        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.0)[0], 1e-3
        )
        assert_array_almost_equal(grad, approx_grad, decimal=2)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(w, X, y, alpha=1.0)
        assert_array_almost_equal(loss, loss_interp)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.0)[0], 1e-3
        )
        assert_array_almost_equal(grad_interp, approx_grad, decimal=2)


def test_logistic_grad_hess():
    rng = np.random.RandomState(0)
    n_samples, n_features = 50, 5
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    X_sp = X_ref.copy()
    X_sp[X_sp < 0.1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.full(n_features, 0.1)

        # First check that _logistic_grad_hess is consistent
        # with _logistic_loss_and_grad
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.0)
        grad_2, hess = _logistic_grad_hess(w, X, y, alpha=1.0)
        assert_array_almost_equal(grad, grad_2)

        # Now check our hessian along the second direction of the grad
        vector = np.zeros_like(grad)
        vector[1] = 1
        hess_col = hess(vector)

        # Computation of the Hessian is particularly fragile to numerical
        # errors when doing simple finite differences. Here we compute the
        # grad along a path in the direction of the vector and then use a
        # least-square regression to estimate the slope
        e = 1e-3
        d_x = np.linspace(-e, e, 30)
        d_grad = np.array(
            [_logistic_loss_and_grad(w + t * vector, X, y, alpha=1.0)[1] for t in d_x]
        )

        d_grad -= d_grad.mean(axis=0)
        approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()

        assert_array_almost_equal(approx_hess_col, hess_col, decimal=3)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(w, X, y, alpha=1.0)
        loss_interp_2 = _logistic_loss(w, X, y, alpha=1.0)
        grad_interp_2, hess = _logistic_grad_hess(w, X, y, alpha=1.0)
        assert_array_almost_equal(loss_interp, loss_interp_2)
        assert_array_almost_equal(grad_interp, grad_interp_2)


def test_logistic_cv():
    # test for LogisticRegressionCV object
    n_samples, n_features = 50, 5
    rng = np.random.RandomState(0)
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    lr_cv = LogisticRegressionCV(
        Cs=[1.0], fit_intercept=False, solver="liblinear", multi_class="ovr", cv=3
    )
    lr_cv.fit(X_ref, y)
    lr = LogisticRegression(
        C=1.0, fit_intercept=False, solver="liblinear", multi_class="ovr"
    )
    lr.fit(X_ref, y)
    assert_array_almost_equal(lr.coef_, lr_cv.coef_)

    assert_array_equal(lr_cv.coef_.shape, (1, n_features))
    assert_array_equal(lr_cv.classes_, [-1, 1])
    assert len(lr_cv.classes_) == 2

    coefs_paths = np.asarray(list(lr_cv.coefs_paths_.values()))
    assert_array_equal(coefs_paths.shape, (1, 3, 1, n_features))
    assert_array_equal(lr_cv.Cs_.shape, (1,))
    scores = np.asarray(list(lr_cv.scores_.values()))
    assert_array_equal(scores.shape, (1, 3, 1))


@pytest.mark.parametrize(
    "scoring, multiclass_agg_list",
    [
        ("accuracy", [""]),
        ("precision", ["_macro", "_weighted"]),
        # no need to test for micro averaging because it
        # is the same as accuracy for f1, precision,
        # and recall (see https://github.com/
        # scikit-learn/scikit-learn/pull/
        # 11578#discussion_r203250062)
        ("f1", ["_macro", "_weighted"]),
        ("neg_log_loss", [""]),
        ("recall", ["_macro", "_weighted"]),
    ],
)
def test_logistic_cv_multinomial_score(scoring, multiclass_agg_list):
    # test that LogisticRegressionCV uses the right score to compute its
    # cross-validation scores when using a multinomial scoring
    # see https://github.com/scikit-learn/scikit-learn/issues/8720
    X, y = make_classification(
        n_samples=100, random_state=0, n_classes=3, n_informative=6
    )
    train, test = np.arange(80), np.arange(80, 100)
    lr = LogisticRegression(C=1.0, multi_class="multinomial")
    # we use lbfgs to support multinomial
    params = lr.get_params()
    # we store the params to set them further in _log_reg_scoring_path
    for key in ["C", "n_jobs", "warm_start"]:
        del params[key]
    lr.fit(X[train], y[train])
    for averaging in multiclass_agg_list:
        scorer = get_scorer(scoring + averaging)
        assert_array_almost_equal(
            _log_reg_scoring_path(
                X, y, train, test, Cs=[1.0], scoring=scorer, **params
            )[2][0],
            scorer(lr, X[test], y[test]),
        )


def test_multinomial_logistic_regression_string_inputs():
    # Test with string labels for LogisticRegression(CV)
    n_samples, n_features, n_classes = 50, 5, 3
    X_ref, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=3,
        random_state=0,
    )
    y_str = LabelEncoder().fit(["bar", "baz", "foo"]).inverse_transform(y)
    # For numerical labels, let y values be taken from set (-1, 0, 1)
    y = np.array(y) - 1
    # Test for string labels
    lr = LogisticRegression(multi_class="multinomial")
    lr_cv = LogisticRegressionCV(multi_class="multinomial", Cs=3)
    lr_str = LogisticRegression(multi_class="multinomial")
    lr_cv_str = LogisticRegressionCV(multi_class="multinomial", Cs=3)

    lr.fit(X_ref, y)
    lr_cv.fit(X_ref, y)
    lr_str.fit(X_ref, y_str)
    lr_cv_str.fit(X_ref, y_str)

    assert_array_almost_equal(lr.coef_, lr_str.coef_)
    assert sorted(lr_str.classes_) == ["bar", "baz", "foo"]
    assert_array_almost_equal(lr_cv.coef_, lr_cv_str.coef_)
    assert sorted(lr_str.classes_) == ["bar", "baz", "foo"]
    assert sorted(lr_cv_str.classes_) == ["bar", "baz", "foo"]

    # The predictions should be in original labels
    assert sorted(np.unique(lr_str.predict(X_ref))) == ["bar", "baz", "foo"]
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ["bar", "baz", "foo"]

    # Make sure class weights can be given with string labels
    lr_cv_str = LogisticRegression(
        class_weight={"bar": 1, "baz": 2, "foo": 0}, multi_class="multinomial"
    ).fit(X_ref, y_str)
    assert sorted(np.unique(lr_cv_str.predict(X_ref))) == ["bar", "baz"]


def test_logistic_cv_sparse():
    X, y = make_classification(n_samples=50, n_features=5, random_state=0)
    X[X < 1.0] = 0.0
    csr = sp.csr_matrix(X)

    clf = LogisticRegressionCV()
    clf.fit(X, y)
    clfs = LogisticRegressionCV()
    clfs.fit(csr, y)
    assert_array_almost_equal(clfs.coef_, clf.coef_)
    assert_array_almost_equal(clfs.intercept_, clf.intercept_)
    assert clfs.C_ == clf.C_


def test_intercept_logistic_helper():
    n_samples, n_features = 10, 5
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=0
    )

    # Fit intercept case.
    alpha = 1.0
    w = np.ones(n_features + 1)
    grad_interp, hess_interp = _logistic_grad_hess(w, X, y, alpha)
    loss_interp = _logistic_loss(w, X, y, alpha)

    # Do not fit intercept. This can be considered equivalent to adding
    # a feature vector of ones, i.e column of one vectors.
    X_ = np.hstack((X, np.ones(10)[:, np.newaxis]))
    grad, hess = _logistic_grad_hess(w, X_, y, alpha)
    loss = _logistic_loss(w, X_, y, alpha)

    # In the fit_intercept=False case, the feature vector of ones is
    # penalized. This should be taken care of.
    assert_almost_equal(loss_interp + 0.5 * (w[-1] ** 2), loss)

    # Check gradient.
    assert_array_almost_equal(grad_interp[:n_features], grad[:n_features])
    assert_almost_equal(grad_interp[-1] + alpha * w[-1], grad[-1])

    rng = np.random.RandomState(0)
    grad = rng.rand(n_features + 1)
    hess_interp = hess_interp(grad)
    hess = hess(grad)
    assert_array_almost_equal(hess_interp[:n_features], hess[:n_features])
    assert_almost_equal(hess_interp[-1] + alpha * grad[-1], hess[-1])


def test_ovr_multinomial_iris():
    # Test that OvR and multinomial are correct using the iris dataset.
    train, target = iris.data, iris.target
    n_samples, n_features = train.shape

    # The cv indices from stratified kfold (where stratification is done based
    # on the fine-grained iris classes, i.e, before the classes 0 and 1 are
    # conflated) is used for both clf and clf1
    n_cv = 2
    cv = StratifiedKFold(n_cv)
    precomputed_folds = list(cv.split(train, target))

    # Train clf on the original dataset where classes 0 and 1 are separated
    clf = LogisticRegressionCV(cv=precomputed_folds, multi_class="ovr")
    clf.fit(train, target)

    # Conflate classes 0 and 1 and train clf1 on this modified dataset
    clf1 = LogisticRegressionCV(cv=precomputed_folds, multi_class="ovr")
    target_copy = target.copy()
    target_copy[target_copy == 0] = 1
    clf1.fit(train, target_copy)

    # Ensure that what OvR learns for class2 is same regardless of whether
    # classes 0 and 1 are separated or not
    assert_allclose(clf.scores_[2], clf1.scores_[2])
    assert_allclose(clf.intercept_[2:], clf1.intercept_)
    assert_allclose(clf.coef_[2][np.newaxis, :], clf1.coef_)

    # Test the shape of various attributes.
    assert clf.coef_.shape == (3, n_features)
    assert_array_equal(clf.classes_, [0, 1, 2])
    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
    assert clf.Cs_.shape == (10,)
    scores = np.asarray(list(clf.scores_.values()))
    assert scores.shape == (3, n_cv, 10)

    # Test that for the iris data multinomial gives a better accuracy than OvR
    for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
        max_iter = 500 if solver in ["sag", "saga"] else 15
        clf_multi = LogisticRegressionCV(
            solver=solver,
            multi_class="multinomial",
            max_iter=max_iter,
            random_state=42,
            tol=1e-3 if solver in ["sag", "saga"] else 1e-2,
            cv=2,
        )
        clf_multi.fit(train, target)
        multi_score = clf_multi.score(train, target)
        ovr_score = clf.score(train, target)
        assert multi_score > ovr_score

        # Test attributes of LogisticRegressionCV
        assert clf.coef_.shape == clf_multi.coef_.shape
        assert_array_equal(clf_multi.classes_, [0, 1, 2])
        coefs_paths = np.asarray(list(clf_multi.coefs_paths_.values()))
        assert coefs_paths.shape == (3, n_cv, 10, n_features + 1)
        assert clf_multi.Cs_.shape == (10,)
        scores = np.asarray(list(clf_multi.scores_.values()))
        assert scores.shape == (3, n_cv, 10)


def test_logistic_regression_solvers():
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)

    params = dict(fit_intercept=False, random_state=42, multi_class="ovr")
    ncg = LogisticRegression(solver="newton-cg", **params)
    lbf = LogisticRegression(solver="lbfgs", **params)
    lib = LogisticRegression(solver="liblinear", **params)
    sag = LogisticRegression(solver="sag", **params)
    saga = LogisticRegression(solver="saga", **params)
    ncg.fit(X, y)
    lbf.fit(X, y)
    sag.fit(X, y)
    saga.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=3)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, lib.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, ncg.coef_, decimal=3)
    assert_array_almost_equal(sag.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(saga.coef_, sag.coef_, decimal=3)
    assert_array_almost_equal(saga.coef_, lbf.coef_, decimal=3)
    assert_array_almost_equal(saga.coef_, ncg.coef_, decimal=3)
    assert_array_almost_equal(saga.coef_, lib.coef_, decimal=3)


def test_logistic_regression_solvers_multiclass():
    X, y = make_classification(
        n_samples=20, n_features=20, n_informative=10, n_classes=3, random_state=0
    )
    tol = 1e-7
    params = dict(fit_intercept=False, tol=tol, random_state=42, multi_class="ovr")
    ncg = LogisticRegression(solver="newton-cg", **params)
    lbf = LogisticRegression(solver="lbfgs", **params)
    lib = LogisticRegression(solver="liblinear", **params)
    sag = LogisticRegression(solver="sag", max_iter=1000, **params)
    saga = LogisticRegression(solver="saga", max_iter=10000, **params)
    ncg.fit(X, y)
    lbf.fit(X, y)
    sag.fit(X, y)
    saga.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=4)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, lib.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, ncg.coef_, decimal=4)
    assert_array_almost_equal(sag.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(saga.coef_, sag.coef_, decimal=4)
    assert_array_almost_equal(saga.coef_, lbf.coef_, decimal=4)
    assert_array_almost_equal(saga.coef_, ncg.coef_, decimal=4)
    assert_array_almost_equal(saga.coef_, lib.coef_, decimal=4)


def test_logistic_regressioncv_class_weights():
    for weight in [{0: 0.1, 1: 0.2}, {0: 0.1, 1: 0.2, 2: 0.5}]:
        n_classes = len(weight)
        for class_weight in (weight, "balanced"):
            X, y = make_classification(
                n_samples=30,
                n_features=3,
                n_repeated=0,
                n_informative=3,
                n_redundant=0,
                n_classes=n_classes,
                random_state=0,
            )

            clf_lbf = LogisticRegressionCV(
                solver="lbfgs",
                Cs=1,
                fit_intercept=False,
                multi_class="ovr",
                class_weight=class_weight,
            )
            clf_ncg = LogisticRegressionCV(
                solver="newton-cg",
                Cs=1,
                fit_intercept=False,
                multi_class="ovr",
                class_weight=class_weight,
            )
            clf_lib = LogisticRegressionCV(
                solver="liblinear",
                Cs=1,
                fit_intercept=False,
                multi_class="ovr",
                class_weight=class_weight,
            )
            clf_sag = LogisticRegressionCV(
                solver="sag",
                Cs=1,
                fit_intercept=False,
                multi_class="ovr",
                class_weight=class_weight,
                tol=1e-5,
                max_iter=10000,
                random_state=0,
            )
            clf_saga = LogisticRegressionCV(
                solver="saga",
                Cs=1,
                fit_intercept=False,
                multi_class="ovr",
                class_weight=class_weight,
                tol=1e-5,
                max_iter=10000,
                random_state=0,
            )
            clf_lbf.fit(X, y)
            clf_ncg.fit(X, y)
            clf_lib.fit(X, y)
            clf_sag.fit(X, y)
            clf_saga.fit(X, y)
            assert_array_almost_equal(clf_lib.coef_, clf_lbf.coef_, decimal=4)
            assert_array_almost_equal(clf_ncg.coef_, clf_lbf.coef_, decimal=4)
            assert_array_almost_equal(clf_sag.coef_, clf_lbf.coef_, decimal=4)
            assert_array_almost_equal(clf_saga.coef_, clf_lbf.coef_, decimal=4)


def test_logistic_regression_sample_weights():
    X, y = make_classification(
        n_samples=20, n_features=5, n_informative=3, n_classes=2, random_state=0
    )
    sample_weight = y + 1

    for LR in [LogisticRegression, LogisticRegressionCV]:

        kw = {"random_state": 42, "fit_intercept": False, "multi_class": "ovr"}
        if LR is LogisticRegressionCV:
            kw.update({"Cs": 3, "cv": 3})

        # Test that passing sample_weight as ones is the same as
        # not passing them at all (default None)
        for solver in ["lbfgs", "liblinear"]:
            clf_sw_none = LR(solver=solver, **kw)
            clf_sw_ones = LR(solver=solver, **kw)
            clf_sw_none.fit(X, y)
            clf_sw_ones.fit(X, y, sample_weight=np.ones(y.shape[0]))
            assert_array_almost_equal(clf_sw_none.coef_, clf_sw_ones.coef_, decimal=4)

        # Test that sample weights work the same with the lbfgs,
        # newton-cg, and 'sag' solvers
        clf_sw_lbfgs = LR(**kw)
        clf_sw_lbfgs.fit(X, y, sample_weight=sample_weight)
        clf_sw_n = LR(solver="newton-cg", **kw)
        clf_sw_n.fit(X, y, sample_weight=sample_weight)
        clf_sw_sag = LR(solver="sag", tol=1e-10, **kw)
        # ignore convergence warning due to small dataset
        with ignore_warnings():
            clf_sw_sag.fit(X, y, sample_weight=sample_weight)
        clf_sw_liblinear = LR(solver="liblinear", **kw)
        clf_sw_liblinear.fit(X, y, sample_weight=sample_weight)
        assert_array_almost_equal(clf_sw_lbfgs.coef_, clf_sw_n.coef_, decimal=4)
        assert_array_almost_equal(clf_sw_lbfgs.coef_, clf_sw_sag.coef_, decimal=4)
        assert_array_almost_equal(clf_sw_lbfgs.coef_, clf_sw_liblinear.coef_, decimal=4)

        # Test that passing class_weight as [1,2] is the same as
        # passing class weight = [1,1] but adjusting sample weights
        # to be 2 for all instances of class 2
        for solver in ["lbfgs", "liblinear"]:
            clf_cw_12 = LR(solver=solver, class_weight={0: 1, 1: 2}, **kw)
            clf_cw_12.fit(X, y)
            clf_sw_12 = LR(solver=solver, **kw)
            clf_sw_12.fit(X, y, sample_weight=sample_weight)
            assert_array_almost_equal(clf_cw_12.coef_, clf_sw_12.coef_, decimal=4)

    # Test the above for l1 penalty and l2 penalty with dual=True.
    # since the patched liblinear code is different.
    clf_cw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        class_weight={0: 1, 1: 2},
        penalty="l1",
        tol=1e-5,
        random_state=42,
        multi_class="ovr",
    )
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        penalty="l1",
        tol=1e-5,
        random_state=42,
        multi_class="ovr",
    )
    clf_sw.fit(X, y, sample_weight)
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)

    clf_cw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        class_weight={0: 1, 1: 2},
        penalty="l2",
        dual=True,
        random_state=42,
        multi_class="ovr",
    )
    clf_cw.fit(X, y)
    clf_sw = LogisticRegression(
        solver="liblinear",
        fit_intercept=False,
        penalty="l2",
        dual=True,
        random_state=42,
        multi_class="ovr",
    )
    clf_sw.fit(X, y, sample_weight)
    assert_array_almost_equal(clf_cw.coef_, clf_sw.coef_, decimal=4)


def _compute_class_weight_dictionary(y):
    # helper for returning a dictionary instead of an array
    classes = np.unique(y)
    class_weight = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weight))
    return class_weight_dict


def test_logistic_regression_class_weights():
    # Multinomial case: remove 90% of class 0
    X = iris.data[45:, :]
    y = iris.target[45:]
    solvers = ("lbfgs", "newton-cg")
    class_weight_dict = _compute_class_weight_dictionary(y)

    for solver in solvers:
        clf1 = LogisticRegression(
            solver=solver, multi_class="multinomial", class_weight="balanced"
        )
        clf2 = LogisticRegression(
            solver=solver, multi_class="multinomial", class_weight=class_weight_dict
        )
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=4)

    # Binary case: remove 90% of class 0 and 100% of class 2
    X = iris.data[45:100, :]
    y = iris.target[45:100]
    solvers = ("lbfgs", "newton-cg", "liblinear")
    class_weight_dict = _compute_class_weight_dictionary(y)

    for solver in solvers:
        clf1 = LogisticRegression(
            solver=solver, multi_class="ovr", class_weight="balanced"
        )
        clf2 = LogisticRegression(
            solver=solver, multi_class="ovr", class_weight=class_weight_dict
        )
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=6)


def test_logistic_regression_multinomial():
    # Tests for the multinomial option in logistic regression

    # Some basic attributes of Logistic Regression
    n_samples, n_features, n_classes = 50, 20, 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=0,
    )

    X = StandardScaler(with_mean=False).fit_transform(X)

    # 'lbfgs' is used as a referenced
    solver = "lbfgs"
    ref_i = LogisticRegression(solver=solver, multi_class="multinomial")
    ref_w = LogisticRegression(
        solver=solver, multi_class="multinomial", fit_intercept=False
    )
    ref_i.fit(X, y)
    ref_w.fit(X, y)
    assert ref_i.coef_.shape == (n_classes, n_features)
    assert ref_w.coef_.shape == (n_classes, n_features)
    for solver in ["sag", "saga", "newton-cg"]:
        clf_i = LogisticRegression(
            solver=solver,
            multi_class="multinomial",
            random_state=42,
            max_iter=2000,
            tol=1e-7,
        )
        clf_w = LogisticRegression(
            solver=solver,
            multi_class="multinomial",
            random_state=42,
            max_iter=2000,
            tol=1e-7,
            fit_intercept=False,
        )
        clf_i.fit(X, y)
        clf_w.fit(X, y)
        assert clf_i.coef_.shape == (n_classes, n_features)
        assert clf_w.coef_.shape == (n_classes, n_features)

        # Compare solutions between lbfgs and the other solvers
        assert_allclose(ref_i.coef_, clf_i.coef_, rtol=1e-2)
        assert_allclose(ref_w.coef_, clf_w.coef_, rtol=1e-2)
        assert_allclose(ref_i.intercept_, clf_i.intercept_, rtol=1e-2)

    # Test that the path give almost the same results. However since in this
    # case we take the average of the coefs after fitting across all the
    # folds, it need not be exactly the same.
    for solver in ["lbfgs", "newton-cg", "sag", "saga"]:
        clf_path = LogisticRegressionCV(
            solver=solver, max_iter=2000, tol=1e-6, multi_class="multinomial", Cs=[1.0]
        )
        clf_path.fit(X, y)
        assert_allclose(clf_path.coef_, ref_i.coef_, rtol=2e-2)
        assert_allclose(clf_path.intercept_, ref_i.intercept_, rtol=2e-2)


def test_multinomial_grad_hess():
    rng = np.random.RandomState(0)
    n_samples, n_features, n_classes = 100, 5, 3
    X = rng.randn(n_samples, n_features)
    w = rng.rand(n_classes, n_features)
    Y = np.zeros((n_samples, n_classes))
    ind = np.argmax(np.dot(X, w.T), axis=1)
    Y[range(0, n_samples), ind] = 1
    w = w.ravel()
    sample_weights = np.ones(X.shape[0])
    grad, hessp = _multinomial_grad_hess(
        w, X, Y, alpha=1.0, sample_weight=sample_weights
    )
    # extract first column of hessian matrix
    vec = np.zeros(n_features * n_classes)
    vec[0] = 1
    hess_col = hessp(vec)

    # Estimate hessian using least squares as done in
    # test_logistic_grad_hess
    e = 1e-3
    d_x = np.linspace(-e, e, 30)
    d_grad = np.array(
        [
            _multinomial_grad_hess(
                w + t * vec, X, Y, alpha=1.0, sample_weight=sample_weights
            )[0]
            for t in d_x
        ]
    )
    d_grad -= d_grad.mean(axis=0)
    approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()
    assert_array_almost_equal(hess_col, approx_hess_col)


def test_liblinear_decision_function_zero():
    # Test negative prediction when decision_function values are zero.
    # Liblinear predicts the positive class when decision_function values
    # are zero. This is a test to verify that we do not do the same.
    # See Issue: https://github.com/scikit-learn/scikit-learn/issues/3600
    # and the PR https://github.com/scikit-learn/scikit-learn/pull/3623
    X, y = make_classification(n_samples=5, n_features=5, random_state=0)
    clf = LogisticRegression(fit_intercept=False, solver="liblinear", multi_class="ovr")
    clf.fit(X, y)

    # Dummy data such that the decision function becomes zero.
    X = np.zeros((5, 5))
    assert_array_equal(clf.predict(X), np.zeros(5))


def test_liblinear_logregcv_sparse():
    # Test LogRegCV with solver='liblinear' works for sparse matrices

    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    clf = LogisticRegressionCV(solver="liblinear", multi_class="ovr")
    clf.fit(sparse.csr_matrix(X), y)


def test_saga_sparse():
    # Test LogRegCV with solver='liblinear' works for sparse matrices

    X, y = make_classification(n_samples=10, n_features=5, random_state=0)
    clf = LogisticRegressionCV(solver="saga")
    clf.fit(sparse.csr_matrix(X), y)


def test_logreg_intercept_scaling():
    # Test that the right error message is thrown when intercept_scaling <= 0

    for i in [-1, 0]:
        clf = LogisticRegression(
            intercept_scaling=i, solver="liblinear", multi_class="ovr"
        )
        msg = (
            "Intercept scaling is %r but needs to be greater than 0."
            " To disable fitting an intercept,"
            " set fit_intercept=False."
            % clf.intercept_scaling
        )
        with pytest.raises(ValueError, match=msg):
            clf.fit(X, Y1)


def test_logreg_intercept_scaling_zero():
    # Test that intercept_scaling is ignored when fit_intercept is False

    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X, Y1)
    assert clf.intercept_ == 0.0


def test_logreg_l1():
    # Because liblinear penalizes the intercept and saga does not, we do not
    # fit the intercept to make it possible to compare the coefficients of
    # the two models at convergence.
    rng = np.random.RandomState(42)
    n_samples = 50
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(size=(n_samples, 3))
    X_constant = np.ones(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    lr_liblinear = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        multi_class="ovr",
        tol=1e-10,
    )
    lr_liblinear.fit(X, y)

    lr_saga = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)

    # Noise and constant features should be regularized to zero by the l1
    # penalty
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))


def test_logreg_l1_sparse_data():
    # Because liblinear penalizes the intercept and saga does not, we do not
    # fit the intercept to make it possible to compare the coefficients of
    # the two models at convergence.
    rng = np.random.RandomState(42)
    n_samples = 50
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(scale=0.1, size=(n_samples, 3))
    X_constant = np.zeros(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    X[X < 1] = 0
    X = sparse.csr_matrix(X)

    lr_liblinear = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        multi_class="ovr",
        tol=1e-10,
    )
    lr_liblinear.fit(X, y)

    lr_saga = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)
    # Noise and constant features should be regularized to zero by the l1
    # penalty
    assert_array_almost_equal(lr_liblinear.coef_[0, -5:], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[0, -5:], np.zeros(5))

    # Check that solving on the sparse and dense data yield the same results
    lr_saga_dense = LogisticRegression(
        penalty="l1",
        C=1.0,
        solver="saga",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga_dense.fit(X.toarray(), y)
    assert_array_almost_equal(lr_saga.coef_, lr_saga_dense.coef_)


@pytest.mark.parametrize("random_seed", [42])
@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_logistic_regression_cv_refit(random_seed, penalty):
    # Test that when refit=True, logistic regression cv with the saga solver
    # converges to the same solution as logistic regression with a fixed
    # regularization parameter.
    # Internally the LogisticRegressionCV model uses a warm start to refit on
    # the full data model with the optimal C found by CV. As the penalized
    # logistic regression loss is convex, we should still recover exactly
    # the same solution as long as the stopping criterion is strict enough (and
    # that there are no exactly duplicated features when penalty='l1').
    X, y = make_classification(n_samples=100, n_features=20, random_state=random_seed)
    common_params = dict(
        solver="saga",
        penalty=penalty,
        random_state=random_seed,
        max_iter=1000,
        tol=1e-12,
    )
    lr_cv = LogisticRegressionCV(Cs=[1.0], refit=True, **common_params)
    lr_cv.fit(X, y)
    lr = LogisticRegression(C=1.0, **common_params)
    lr.fit(X, y)
    assert_array_almost_equal(lr_cv.coef_, lr.coef_)


def test_logreg_predict_proba_multinomial():
    X, y = make_classification(
        n_samples=10, n_features=20, random_state=0, n_classes=3, n_informative=10
    )

    # Predicted probabilities using the true-entropy loss should give a
    # smaller loss than those using the ovr method.
    clf_multi = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf_multi.fit(X, y)
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_ovr = LogisticRegression(multi_class="ovr", solver="lbfgs")
    clf_ovr.fit(X, y)
    clf_ovr_loss = log_loss(y, clf_ovr.predict_proba(X))
    assert clf_ovr_loss > clf_multi_loss

    # Predicted probabilities using the soft-max function should give a
    # smaller loss than those using the logistic function.
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_wrong_loss = log_loss(y, clf_multi._predict_proba_lr(X))
    assert clf_wrong_loss > clf_multi_loss


@pytest.mark.parametrize("max_iter", np.arange(1, 5))
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize(
    "solver, message",
    [
        (
            "newton-cg",
            "newton-cg failed to converge. Increase the number of iterations.",
        ),
        (
            "liblinear",
            "Liblinear failed to converge, increase the number of iterations.",
        ),
        ("sag", "The max_iter was reached which means the coef_ did not converge"),
        ("saga", "The max_iter was reached which means the coef_ did not converge"),
        ("lbfgs", "lbfgs failed to converge"),
    ],
)
def test_max_iter(max_iter, multi_class, solver, message):
    # Test that the maximum number of iteration is reached
    X, y_bin = iris.data, iris.target.copy()
    y_bin[y_bin == 2] = 0

    if solver == "liblinear" and multi_class == "multinomial":
        pytest.skip("'multinomial' is unavailable when solver='liblinear'")

    lr = LogisticRegression(
        max_iter=max_iter,
        tol=1e-15,
        multi_class=multi_class,
        random_state=0,
        solver=solver,
    )
    with pytest.warns(ConvergenceWarning, match=message):
        lr.fit(X, y_bin)

    assert lr.n_iter_[0] == max_iter


@pytest.mark.parametrize("solver", ["newton-cg", "liblinear", "sag", "saga", "lbfgs"])
def test_n_iter(solver):
    # Test that self.n_iter_ has the correct format.
    X, y = iris.data, iris.target

    y_bin = y.copy()
    y_bin[y_bin == 2] = 0

    n_Cs = 4
    n_cv_fold = 2

    # OvR case
    n_classes = 1 if solver == "liblinear" else np.unique(y).shape[0]
    clf = LogisticRegression(
        tol=1e-2, multi_class="ovr", solver=solver, C=1.0, random_state=42
    )
    clf.fit(X, y)
    assert clf.n_iter_.shape == (n_classes,)

    n_classes = np.unique(y).shape[0]
    clf = LogisticRegressionCV(
        tol=1e-2,
        multi_class="ovr",
        solver=solver,
        Cs=n_Cs,
        cv=n_cv_fold,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.n_iter_.shape == (n_classes, n_cv_fold, n_Cs)
    clf.fit(X, y_bin)
    assert clf.n_iter_.shape == (1, n_cv_fold, n_Cs)

    # multinomial case
    n_classes = 1
    if solver in ("liblinear", "sag", "saga"):
        return

    clf = LogisticRegression(
        tol=1e-2, multi_class="multinomial", solver=solver, C=1.0, random_state=42
    )
    clf.fit(X, y)
    assert clf.n_iter_.shape == (n_classes,)

    clf = LogisticRegressionCV(
        tol=1e-2,
        multi_class="multinomial",
        solver=solver,
        Cs=n_Cs,
        cv=n_cv_fold,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.n_iter_.shape == (n_classes, n_cv_fold, n_Cs)
    clf.fit(X, y_bin)
    assert clf.n_iter_.shape == (1, n_cv_fold, n_Cs)


@pytest.mark.parametrize("solver", ("newton-cg", "sag", "saga", "lbfgs"))
@pytest.mark.parametrize("warm_start", (True, False))
@pytest.mark.parametrize("fit_intercept", (True, False))
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
def test_warm_start(solver, warm_start, fit_intercept, multi_class):
    # A 1-iteration second fit on same data should give almost same result
    # with warm starting, and quite different result without warm starting.
    # Warm starting does not work with liblinear solver.
    X, y = iris.data, iris.target

    clf = LogisticRegression(
        tol=1e-4,
        multi_class=multi_class,
        warm_start=warm_start,
        solver=solver,
        random_state=42,
        fit_intercept=fit_intercept,
    )
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
        coef_1 = clf.coef_

        clf.max_iter = 1
        clf.fit(X, y)
    cum_diff = np.sum(np.abs(coef_1 - clf.coef_))
    msg = (
        "Warm starting issue with %s solver in %s mode "
        "with fit_intercept=%s and warm_start=%s"
        % (solver, multi_class, str(fit_intercept), str(warm_start))
    )
    if warm_start:
        assert 2.0 > cum_diff, msg
    else:
        assert cum_diff > 2.0, msg


def test_saga_vs_liblinear():
    iris = load_iris()
    X, y = iris.data, iris.target
    X = np.concatenate([X] * 3)
    y = np.concatenate([y] * 3)

    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1

    X_sparse, y_sparse = make_classification(
        n_samples=50, n_features=20, random_state=0
    )
    X_sparse = sparse.csr_matrix(X_sparse)

    for (X, y) in ((X_bin, y_bin), (X_sparse, y_sparse)):
        for penalty in ["l1", "l2"]:
            n_samples = X.shape[0]
            # alpha=1e-3 is time consuming
            for alpha in np.logspace(-1, 1, 3):
                saga = LogisticRegression(
                    C=1.0 / (n_samples * alpha),
                    solver="saga",
                    multi_class="ovr",
                    max_iter=200,
                    fit_intercept=False,
                    penalty=penalty,
                    random_state=0,
                    tol=1e-24,
                )

                liblinear = LogisticRegression(
                    C=1.0 / (n_samples * alpha),
                    solver="liblinear",
                    multi_class="ovr",
                    max_iter=200,
                    fit_intercept=False,
                    penalty=penalty,
                    random_state=0,
                    tol=1e-24,
                )

                saga.fit(X, y)
                liblinear.fit(X, y)
                # Convergence for alpha=1e-3 is very slow
                assert_array_almost_equal(saga.coef_, liblinear.coef_, 3)


@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
@pytest.mark.parametrize("solver", ["newton-cg", "liblinear", "saga"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_dtype_match(solver, multi_class, fit_intercept):
    # Test that np.float32 input data is not cast to np.float64 when possible
    # and that the output is approximately the same no matter the input format.

    if solver == "liblinear" and multi_class == "multinomial":
        pytest.skip("liblinear does not support multinomial logistic")

    out32_type = np.float64 if solver == "liblinear" else np.float32

    X_32 = np.array(X).astype(np.float32)
    y_32 = np.array(Y1).astype(np.float32)
    X_64 = np.array(X).astype(np.float64)
    y_64 = np.array(Y1).astype(np.float64)
    X_sparse_32 = sp.csr_matrix(X, dtype=np.float32)
    X_sparse_64 = sp.csr_matrix(X, dtype=np.float64)
    solver_tol = 5e-4

    lr_templ = LogisticRegression(
        solver=solver,
        multi_class=multi_class,
        random_state=42,
        tol=solver_tol,
        fit_intercept=fit_intercept,
    )

    # Check 32-bit type consistency
    lr_32 = clone(lr_templ)
    lr_32.fit(X_32, y_32)
    assert lr_32.coef_.dtype == out32_type

    # Check 32-bit type consistency with sparsity
    lr_32_sparse = clone(lr_templ)
    lr_32_sparse.fit(X_sparse_32, y_32)
    assert lr_32_sparse.coef_.dtype == out32_type

    # Check 64-bit type consistency
    lr_64 = clone(lr_templ)
    lr_64.fit(X_64, y_64)
    assert lr_64.coef_.dtype == np.float64

    # Check 64-bit type consistency with sparsity
    lr_64_sparse = clone(lr_templ)
    lr_64_sparse.fit(X_sparse_64, y_64)
    assert lr_64_sparse.coef_.dtype == np.float64

    # solver_tol bounds the norm of the loss gradient
    # dw ~= inv(H)*grad ==> |dw| ~= |inv(H)| * solver_tol, where H - hessian
    #
    # See https://github.com/scikit-learn/scikit-learn/pull/13645
    #
    # with  Z = np.hstack((np.ones((3,1)), np.array(X)))
    # In [8]: np.linalg.norm(np.diag([0,2,2]) + np.linalg.inv((Z.T @ Z)/4))
    # Out[8]: 1.7193336918135917

    # factor of 2 to get the ball diameter
    atol = 2 * 1.72 * solver_tol
    if os.name == "nt" and _IS_32BIT:
        # FIXME
        atol = 1e-2

    # Check accuracy consistency
    assert_allclose(lr_32.coef_, lr_64.coef_.astype(np.float32), atol=atol)

    if solver == "saga" and fit_intercept:
        # FIXME: SAGA on sparse data fits the intercept inaccurately with the
        # default tol and max_iter parameters.
        atol = 1e-1

    assert_allclose(lr_32.coef_, lr_32_sparse.coef_, atol=atol)
    assert_allclose(lr_64.coef_, lr_64_sparse.coef_, atol=atol)


def test_warm_start_converge_LR():
    # Test to see that the logistic regression converges on warm start,
    # with multi_class='multinomial'. Non-regressive test for #10836

    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = np.array([1] * 100 + [-1] * 100)
    lr_no_ws = LogisticRegression(
        multi_class="multinomial", solver="sag", warm_start=False, random_state=0
    )
    lr_ws = LogisticRegression(
        multi_class="multinomial", solver="sag", warm_start=True, random_state=0
    )

    lr_no_ws_loss = log_loss(y, lr_no_ws.fit(X, y).predict_proba(X))
    for i in range(5):
        lr_ws.fit(X, y)
    lr_ws_loss = log_loss(y, lr_ws.predict_proba(X))
    assert_allclose(lr_no_ws_loss, lr_ws_loss, rtol=1e-5)


def test_elastic_net_coeffs():
    # make sure elasticnet penalty gives different coefficients from l1 and l2
    # with saga solver (l1_ratio different from 0 or 1)
    X, y = make_classification(random_state=0)

    C = 2.0
    l1_ratio = 0.5
    coeffs = list()
    for penalty in ("elasticnet", "l1", "l2"):
        lr = LogisticRegression(
            penalty=penalty, C=C, solver="saga", random_state=0, l1_ratio=l1_ratio
        )
        lr.fit(X, y)
        coeffs.append(lr.coef_)

    elastic_net_coeffs, l1_coeffs, l2_coeffs = coeffs
    # make sure coeffs differ by at least .1
    assert not np.allclose(elastic_net_coeffs, l1_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(elastic_net_coeffs, l2_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(l2_coeffs, l1_coeffs, rtol=0, atol=0.1)


@pytest.mark.parametrize("C", [0.001, 0.1, 1, 10, 100, 1000, 1e6])
@pytest.mark.parametrize("penalty, l1_ratio", [("l1", 1), ("l2", 0)])
def test_elastic_net_l1_l2_equivalence(C, penalty, l1_ratio):
    # Make sure elasticnet is equivalent to l1 when l1_ratio=1 and to l2 when
    # l1_ratio=0.
    X, y = make_classification(random_state=0)

    lr_enet = LogisticRegression(
        penalty="elasticnet", C=C, l1_ratio=l1_ratio, solver="saga", random_state=0
    )
    lr_expected = LogisticRegression(
        penalty=penalty, C=C, solver="saga", random_state=0
    )
    lr_enet.fit(X, y)
    lr_expected.fit(X, y)

    assert_array_almost_equal(lr_enet.coef_, lr_expected.coef_)


@pytest.mark.parametrize("C", [0.001, 1, 100, 1e6])
def test_elastic_net_vs_l1_l2(C):
    # Make sure that elasticnet with grid search on l1_ratio gives same or
    # better results than just l1 or just l2.

    X, y = make_classification(500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    param_grid = {"l1_ratio": np.linspace(0, 1, 5)}

    enet_clf = LogisticRegression(
        penalty="elasticnet", C=C, solver="saga", random_state=0
    )
    gs = GridSearchCV(enet_clf, param_grid, refit=True)

    l1_clf = LogisticRegression(penalty="l1", C=C, solver="saga", random_state=0)
    l2_clf = LogisticRegression(penalty="l2", C=C, solver="saga", random_state=0)

    for clf in (gs, l1_clf, l2_clf):
        clf.fit(X_train, y_train)

    assert gs.score(X_test, y_test) >= l1_clf.score(X_test, y_test)
    assert gs.score(X_test, y_test) >= l2_clf.score(X_test, y_test)


@pytest.mark.parametrize("C", np.logspace(-3, 2, 4))
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
def test_LogisticRegression_elastic_net_objective(C, l1_ratio):
    # Check that training with a penalty matching the objective leads
    # to a lower objective.
    # Here we train a logistic regression with l2 (a) and elasticnet (b)
    # penalties, and compute the elasticnet objective. That of a should be
    # greater than that of b (both objectives are convex).
    X, y = make_classification(
        n_samples=1000,
        n_classes=2,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = scale(X)

    lr_enet = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        random_state=0,
        C=C,
        l1_ratio=l1_ratio,
        fit_intercept=False,
    )
    lr_l2 = LogisticRegression(
        penalty="l2", solver="saga", random_state=0, C=C, fit_intercept=False
    )
    lr_enet.fit(X, y)
    lr_l2.fit(X, y)

    def enet_objective(lr):
        coef = lr.coef_.ravel()
        obj = C * log_loss(y, lr.predict_proba(X))
        obj += l1_ratio * np.sum(np.abs(coef))
        obj += (1.0 - l1_ratio) * 0.5 * np.dot(coef, coef)
        return obj

    assert enet_objective(lr_enet) < enet_objective(lr_l2)


@pytest.mark.parametrize("multi_class", ("ovr", "multinomial"))
def test_LogisticRegressionCV_GridSearchCV_elastic_net(multi_class):
    # make sure LogisticRegressionCV gives same best params (l1 and C) as
    # GridSearchCV when penalty is elasticnet

    if multi_class == "ovr":
        # This is actually binary classification, ovr multiclass is treated in
        # test_LogisticRegressionCV_GridSearchCV_elastic_net_ovr
        X, y = make_classification(random_state=0)
    else:
        X, y = make_classification(
            n_samples=100, n_classes=3, n_informative=3, random_state=0
        )

    cv = StratifiedKFold(5)

    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)

    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=cv,
        l1_ratios=l1_ratios,
        random_state=0,
        multi_class=multi_class,
    )
    lrcv.fit(X, y)

    param_grid = {"C": Cs, "l1_ratio": l1_ratios}
    lr = LogisticRegression(
        penalty="elasticnet", solver="saga", random_state=0, multi_class=multi_class
    )
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X, y)

    assert gs.best_params_["l1_ratio"] == lrcv.l1_ratio_[0]
    assert gs.best_params_["C"] == lrcv.C_[0]


def test_LogisticRegressionCV_GridSearchCV_elastic_net_ovr():
    # make sure LogisticRegressionCV gives same best params (l1 and C) as
    # GridSearchCV when penalty is elasticnet and multiclass is ovr. We can't
    # compare best_params like in the previous test because
    # LogisticRegressionCV with multi_class='ovr' will have one C and one
    # l1_param for each class, while LogisticRegression will share the
    # parameters over the *n_classes* classifiers.

    X, y = make_classification(
        n_samples=100, n_classes=3, n_informative=3, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    cv = StratifiedKFold(5)

    l1_ratios = np.linspace(0, 1, 3)
    Cs = np.logspace(-4, 4, 3)

    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=cv,
        l1_ratios=l1_ratios,
        random_state=0,
        multi_class="ovr",
    )
    lrcv.fit(X_train, y_train)

    param_grid = {"C": Cs, "l1_ratio": l1_ratios}
    lr = LogisticRegression(
        penalty="elasticnet", solver="saga", random_state=0, multi_class="ovr"
    )
    gs = GridSearchCV(lr, param_grid, cv=cv)
    gs.fit(X_train, y_train)

    # Check that predictions are 80% the same
    assert (lrcv.predict(X_train) == gs.predict(X_train)).mean() >= 0.8
    assert (lrcv.predict(X_test) == gs.predict(X_test)).mean() >= 0.8


@pytest.mark.parametrize("penalty", ("l2", "elasticnet"))
@pytest.mark.parametrize("multi_class", ("ovr", "multinomial", "auto"))
def test_LogisticRegressionCV_no_refit(penalty, multi_class):
    # Test LogisticRegressionCV attribute shapes when refit is False

    n_classes = 3
    n_features = 20
    X, y = make_classification(
        n_samples=200,
        n_classes=n_classes,
        n_informative=n_classes,
        n_features=n_features,
        random_state=0,
    )

    Cs = np.logspace(-4, 4, 3)
    if penalty == "elasticnet":
        l1_ratios = np.linspace(0, 1, 2)
    else:
        l1_ratios = None

    lrcv = LogisticRegressionCV(
        penalty=penalty,
        Cs=Cs,
        solver="saga",
        l1_ratios=l1_ratios,
        random_state=0,
        multi_class=multi_class,
        refit=False,
    )
    lrcv.fit(X, y)
    assert lrcv.C_.shape == (n_classes,)
    assert lrcv.l1_ratio_.shape == (n_classes,)
    assert lrcv.coef_.shape == (n_classes, n_features)


def test_LogisticRegressionCV_elasticnet_attribute_shapes():
    # Make sure the shapes of scores_ and coefs_paths_ attributes are correct
    # when using elasticnet (added one dimension for l1_ratios)

    n_classes = 3
    n_features = 20
    X, y = make_classification(
        n_samples=200,
        n_classes=n_classes,
        n_informative=n_classes,
        n_features=n_features,
        random_state=0,
    )

    Cs = np.logspace(-4, 4, 3)
    l1_ratios = np.linspace(0, 1, 2)

    n_folds = 2
    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        Cs=Cs,
        solver="saga",
        cv=n_folds,
        l1_ratios=l1_ratios,
        multi_class="ovr",
        random_state=0,
    )
    lrcv.fit(X, y)
    coefs_paths = np.asarray(list(lrcv.coefs_paths_.values()))
    assert coefs_paths.shape == (
        n_classes,
        n_folds,
        Cs.size,
        l1_ratios.size,
        n_features + 1,
    )
    scores = np.asarray(list(lrcv.scores_.values()))
    assert scores.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)

    assert lrcv.n_iter_.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)


@pytest.mark.parametrize("l1_ratio", (-1, 2, None, "something_wrong"))
def test_l1_ratio_param(l1_ratio):

    msg = r"l1_ratio must be between 0 and 1; got \(l1_ratio=%r\)" % l1_ratio
    with pytest.raises(ValueError, match=msg):
        LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1_ratio).fit(
            X, Y1
        )

    if l1_ratio is not None:
        msg = (
            r"l1_ratio parameter is only used when penalty is"
            r" 'elasticnet'\. Got \(penalty=l1\)"
        )
        with pytest.warns(UserWarning, match=msg):
            LogisticRegression(penalty="l1", solver="saga", l1_ratio=l1_ratio).fit(
                X, Y1
            )


@pytest.mark.parametrize("l1_ratios", ([], [0.5, 2], None, "something_wrong"))
def test_l1_ratios_param(l1_ratios):

    msg = (
        "l1_ratios must be a list of numbers between 0 and 1; got (l1_ratios=%r)"
        % l1_ratios
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        LogisticRegressionCV(
            penalty="elasticnet", solver="saga", l1_ratios=l1_ratios, cv=2
        ).fit(X, Y1)

    if l1_ratios is not None:
        msg = (
            r"l1_ratios parameter is only used when penalty"
            r" is 'elasticnet'. Got \(penalty=l1\)"
        )
        function = LogisticRegressionCV(
            penalty="l1", solver="saga", l1_ratios=l1_ratios, cv=2
        ).fit
        with pytest.warns(UserWarning, match=msg):
            function(X, Y1)


@pytest.mark.parametrize("C", np.logspace(-3, 2, 4))
@pytest.mark.parametrize("l1_ratio", [0.1, 0.5, 0.9])
def test_elastic_net_versus_sgd(C, l1_ratio):
    # Compare elasticnet penalty in LogisticRegression() and SGD(loss='log')
    n_samples = 500
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=1,
    )
    X = scale(X)

    sgd = SGDClassifier(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=-np.inf,
        max_iter=2000,
        l1_ratio=l1_ratio,
        alpha=1.0 / C / n_samples,
        loss="log",
    )
    log = LogisticRegression(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=1e-5,
        max_iter=1000,
        l1_ratio=l1_ratio,
        C=C,
        solver="saga",
    )

    sgd.fit(X, y)
    log.fit(X, y)
    assert_array_almost_equal(sgd.coef_, log.coef_, decimal=1)


def test_logistic_regression_path_coefs_multinomial():
    # Make sure that the returned coefs by logistic_regression_path when
    # multi_class='multinomial' don't override each other (used to be a
    # bug).
    X, y = make_classification(
        n_samples=200,
        n_classes=3,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
        n_features=2,
    )
    Cs = [0.00001, 1, 10000]
    coefs, _, _ = _logistic_regression_path(
        X,
        y,
        penalty="l1",
        Cs=Cs,
        solver="saga",
        random_state=0,
        multi_class="multinomial",
    )

    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[1], decimal=1)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[0], coefs[2], decimal=1)
    with pytest.raises(AssertionError):
        assert_array_almost_equal(coefs[1], coefs[2], decimal=1)


@pytest.mark.parametrize(
    "est",
    [
        LogisticRegression(random_state=0, max_iter=500),
        LogisticRegressionCV(random_state=0, cv=3, Cs=3, tol=1e-3, max_iter=500),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
def test_logistic_regression_multi_class_auto(est, solver):
    # check multi_class='auto' => multi_class='ovr' iff binary y or liblinear

    def fit(X, y, **kw):
        return clone(est).set_params(**kw).fit(X, y)

    scaled_data = scale(iris.data)
    X = scaled_data[::10]
    X2 = scaled_data[1::10]
    y_multi = iris.target[::10]
    y_bin = y_multi == 0
    est_auto_bin = fit(X, y_bin, multi_class="auto", solver=solver)
    est_ovr_bin = fit(X, y_bin, multi_class="ovr", solver=solver)
    assert_allclose(est_auto_bin.coef_, est_ovr_bin.coef_)
    assert_allclose(est_auto_bin.predict_proba(X2), est_ovr_bin.predict_proba(X2))

    est_auto_multi = fit(X, y_multi, multi_class="auto", solver=solver)
    if solver == "liblinear":
        est_ovr_multi = fit(X, y_multi, multi_class="ovr", solver=solver)
        assert_allclose(est_auto_multi.coef_, est_ovr_multi.coef_)
        assert_allclose(
            est_auto_multi.predict_proba(X2), est_ovr_multi.predict_proba(X2)
        )
    else:
        est_multi_multi = fit(X, y_multi, multi_class="multinomial", solver=solver)
        assert_allclose(est_auto_multi.coef_, est_multi_multi.coef_)
        assert_allclose(
            est_auto_multi.predict_proba(X2), est_multi_multi.predict_proba(X2)
        )

        # Make sure multi_class='ovr' is distinct from ='multinomial'
        assert not np.allclose(
            est_auto_bin.coef_,
            fit(X, y_bin, multi_class="multinomial", solver=solver).coef_,
        )
        assert not np.allclose(
            est_auto_bin.coef_,
            fit(X, y_multi, multi_class="multinomial", solver=solver).coef_,
        )


@pytest.mark.parametrize("solver", ("lbfgs", "newton-cg", "sag", "saga"))
def test_penalty_none(solver):
    # - Make sure warning is raised if penalty='none' and C is set to a
    #   non-default value.
    # - Make sure setting penalty='none' is equivalent to setting C=np.inf with
    #   l2 penalty.
    X, y = make_classification(n_samples=1000, random_state=0)

    msg = "Setting penalty='none' will ignore the C"
    lr = LogisticRegression(penalty="none", solver=solver, C=4)
    with pytest.warns(UserWarning, match=msg):
        lr.fit(X, y)

    lr_none = LogisticRegression(penalty="none", solver=solver, random_state=0)
    lr_l2_C_inf = LogisticRegression(
        penalty="l2", C=np.inf, solver=solver, random_state=0
    )
    pred_none = lr_none.fit(X, y).predict(X)
    pred_l2_C_inf = lr_l2_C_inf.fit(X, y).predict(X)
    assert_array_equal(pred_none, pred_l2_C_inf)

    lr = LogisticRegressionCV(penalty="none")
    err_msg = "penalty='none' is not useful and not supported by LogisticRegressionCV"
    with pytest.raises(ValueError, match=err_msg):
        lr.fit(X, y)


@pytest.mark.parametrize(
    "params",
    [
        {"penalty": "l1", "dual": False, "tol": 1e-12, "max_iter": 1000},
        {"penalty": "l2", "dual": True, "tol": 1e-12, "max_iter": 1000},
        {"penalty": "l2", "dual": False, "tol": 1e-12, "max_iter": 1000},
    ],
)
def test_logisticregression_liblinear_sample_weight(params):
    # check that we support sample_weight with liblinear in all possible cases:
    # l1-primal, l2-primal, l2-dual
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.dtype("float"),
    )
    y = np.array(
        [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.dtype("int")
    )

    X2 = np.vstack([X, X])
    y2 = np.hstack([y, 3 - y])
    sample_weight = np.ones(shape=len(y) * 2)
    sample_weight[len(y) :] = 0
    X2, y2, sample_weight = shuffle(X2, y2, sample_weight, random_state=0)

    base_clf = LogisticRegression(solver="liblinear", random_state=42)
    base_clf.set_params(**params)
    clf_no_weight = clone(base_clf).fit(X, y)
    clf_with_weight = clone(base_clf).fit(X2, y2, sample_weight=sample_weight)

    for method in ("predict", "predict_proba", "decision_function"):
        X_clf_no_weight = getattr(clf_no_weight, method)(X)
        X_clf_with_weight = getattr(clf_with_weight, method)(X)
        assert_allclose(X_clf_no_weight, X_clf_with_weight)


def test_scores_attribute_layout_elasticnet():
    # Non regression test for issue #14955.
    # when penalty is elastic net the scores_ attribute has shape
    # (n_classes, n_Cs, n_l1_ratios)
    # We here make sure that the second dimension indeed corresponds to Cs and
    # the third dimension corresponds to l1_ratios.

    X, y = make_classification(n_samples=1000, random_state=0)
    cv = StratifiedKFold(n_splits=5)

    l1_ratios = [0.1, 0.9]
    Cs = [0.1, 1, 10]

    lrcv = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=l1_ratios,
        Cs=Cs,
        cv=cv,
        random_state=0,
    )
    lrcv.fit(X, y)

    avg_scores_lrcv = lrcv.scores_[1].mean(axis=0)  # average over folds

    for i, C in enumerate(Cs):
        for j, l1_ratio in enumerate(l1_ratios):

            lr = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=C,
                l1_ratio=l1_ratio,
                random_state=0,
            )

            avg_score_lr = cross_val_score(lr, X, y, cv=cv).mean()
            assert avg_scores_lrcv[i, j] == pytest.approx(avg_score_lr)


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_multinomial_identifiability_on_iris(fit_intercept):
    """Test that the multinomial classification is identifiable.

    A multinomial with c classes can be modeled with
    probability_k = exp(X@coef_k) / sum(exp(X@coef_l), l=1..c) for k=1..c.
    This is not identifiable, unless one chooses a further constraint.
    According to [1], the maximum of the L2 penalized likelihood automatically
    satisfies the symmetric constraint:
    sum(coef_k, k=1..c) = 0

    Further details can be found in the appendix of [2].

    Reference
    ---------
    .. [1] Zhu, Ji and Trevor J. Hastie. "Classification of gene microarrays by
    penalized logistic regression". Biostatistics 5 3 (2004): 427-43.
    https://doi.org/10.1093/biostatistics%2Fkxg046

    .. [2] Powers, Scott, Trevor J. Hastie and Robert Tibshirani. "Nuclear
    penalized multinomial regression with an application to predicting at bat
    outcomes in baseball." Statistical modelling 18 5-6 (2017): 388-410 .
    https://arxiv.org/pdf/1706.10272.pdf
    """
    # Test logistic regression with the iris dataset
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]

    clf = LogisticRegression(
        C=len(iris.data),
        solver="lbfgs",
        max_iter=300,
        multi_class="multinomial",
        fit_intercept=fit_intercept,
    )
    clf.fit(iris.data, target)

    # axis=0 is sum over classes
    assert_allclose(clf.coef_.sum(axis=0), 0, atol=1e-10)
    if fit_intercept:
        clf.intercept_.sum(axis=0) == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("multi_class", ["ovr", "multinomial", "auto"])
@pytest.mark.parametrize("class_weight", [{0: 1.0, 1: 10.0, 2: 1.0}, "balanced"])
def test_sample_weight_not_modified(multi_class, class_weight):
    X, y = load_iris(return_X_y=True)
    n_features = len(X)
    W = np.ones(n_features)
    W[: n_features // 2] = 2

    expected = W.copy()

    clf = LogisticRegression(
        random_state=0, class_weight=class_weight, max_iter=200, multi_class=multi_class
    )
    clf.fit(X, y, sample_weight=W)
    assert_allclose(expected, W)


@pytest.mark.parametrize("solver", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
def test_large_sparse_matrix(solver):
    # Solvers either accept large sparse matrices, or raise helpful error.
    # Non-regression test for pull-request #21093.

    # generate sparse matrix with int64 indices
    X = sp.rand(20, 10, format="csr")
    for attr in ["indices", "indptr"]:
        setattr(X, attr, getattr(X, attr).astype("int64"))
    y = np.random.randint(2, size=X.shape[0])

    if solver in ["liblinear", "sag", "saga"]:
        msg = "Only sparse matrices with 32-bit integer indices"
        with pytest.raises(ValueError, match=msg):
            LogisticRegression(solver=solver).fit(X, y)
    else:
        LogisticRegression(solver=solver).fit(X, y)
