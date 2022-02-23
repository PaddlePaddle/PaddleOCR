"""
Testing for Support Vector Machine module (sklearn.svm)

TODO: remove hard coded numerical results when possible
"""
import numpy as np
import itertools
import pytest
import re

from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn import svm, linear_model, datasets, metrics, base
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.validation import _num_samples
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning
from sklearn.multiclass import OneVsRestClassifier

# mypy error: Module 'sklearn.svm' has no attribute '_libsvm'
from sklearn.svm import _libsvm  # type: ignore

# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
Y = [1, 1, 1, 2, 2, 2]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [1, 2, 2]

# also load the iris dataset
iris = datasets.load_iris()
rng = check_random_state(42)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_libsvm_parameters():
    # Test parameters on classes that make use of libsvm.
    clf = svm.SVC(kernel="linear").fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.0])
    assert_array_equal(clf.predict(X), Y)


def test_libsvm_iris():
    # Check consistency on dataset iris.

    # shuffle the dataset so that labels are not ordered
    for k in ("linear", "rbf"):
        clf = svm.SVC(kernel=k).fit(iris.data, iris.target)
        assert np.mean(clf.predict(iris.data) == iris.target) > 0.9
        assert hasattr(clf, "coef_") == (k == "linear")

    assert_array_equal(clf.classes_, np.sort(clf.classes_))

    # check also the low-level API
    model = _libsvm.fit(iris.data, iris.target.astype(np.float64))
    pred = _libsvm.predict(iris.data, *model)
    assert np.mean(pred == iris.target) > 0.95

    model = _libsvm.fit(iris.data, iris.target.astype(np.float64), kernel="linear")
    pred = _libsvm.predict(iris.data, *model, kernel="linear")
    assert np.mean(pred == iris.target) > 0.95

    pred = _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )
    assert np.mean(pred == iris.target) > 0.95

    # If random_seed >= 0, the libsvm rng is seeded (by calling `srand`), hence
    # we should get deterministic results (assuming that there is no other
    # thread calling this wrapper calling `srand` concurrently).
    pred2 = _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )
    assert_array_equal(pred, pred2)


def test_precomputed():
    # SVC with a precomputed kernel.
    # We test it with a toy dataset and with iris.
    clf = svm.SVC(kernel="precomputed")
    # Gram matrix for train data (square matrix)
    # (we use just a linear kernel)
    K = np.dot(X, np.array(X).T)
    clf.fit(K, Y)
    # Gram matrix for test data (rectangular matrix)
    KT = np.dot(T, np.array(X).T)
    pred = clf.predict(KT)
    with pytest.raises(ValueError):
        clf.predict(KT.T)

    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)

    # Gram matrix for test data but compute KT[i,j]
    # for support vectors j only.
    KT = np.zeros_like(KT)
    for i in range(len(T)):
        for j in clf.support_:
            KT[i, j] = np.dot(T[i], X[j])

    pred = clf.predict(KT)
    assert_array_equal(pred, true_result)

    # same as before, but using a callable function instead of the kernel
    # matrix. kernel is just a linear kernel

    def kfunc(x, y):
        return np.dot(x, y.T)

    clf = svm.SVC(kernel=kfunc)
    clf.fit(np.array(X), Y)
    pred = clf.predict(T)

    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.support_, [1, 3])
    assert_array_equal(pred, true_result)

    # test a precomputed kernel with the iris dataset
    # and check parameters against a linear SVC
    clf = svm.SVC(kernel="precomputed")
    clf2 = svm.SVC(kernel="linear")
    K = np.dot(iris.data, iris.data.T)
    clf.fit(K, iris.target)
    clf2.fit(iris.data, iris.target)
    pred = clf.predict(K)
    assert_array_almost_equal(clf.support_, clf2.support_)
    assert_array_almost_equal(clf.dual_coef_, clf2.dual_coef_)
    assert_array_almost_equal(clf.intercept_, clf2.intercept_)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)

    # Gram matrix for test data but compute KT[i,j]
    # for support vectors j only.
    K = np.zeros_like(K)
    for i in range(len(iris.data)):
        for j in clf.support_:
            K[i, j] = np.dot(iris.data[i], iris.data[j])

    pred = clf.predict(K)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)

    clf = svm.SVC(kernel=kfunc)
    clf.fit(iris.data, iris.target)
    assert_almost_equal(np.mean(pred == iris.target), 0.99, decimal=2)


def test_svr():
    # Test Support Vector Regression

    diabetes = datasets.load_diabetes()
    for clf in (
        svm.NuSVR(kernel="linear", nu=0.4, C=1.0),
        svm.NuSVR(kernel="linear", nu=0.4, C=10.0),
        svm.SVR(kernel="linear", C=10.0),
        svm.LinearSVR(C=10.0),
        svm.LinearSVR(C=10.0),
    ):
        clf.fit(diabetes.data, diabetes.target)
        assert clf.score(diabetes.data, diabetes.target) > 0.02

    # non-regression test; previously, BaseLibSVM would check that
    # len(np.unique(y)) < 2, which must only be done for SVC
    svm.SVR().fit(diabetes.data, np.ones(len(diabetes.data)))
    svm.LinearSVR().fit(diabetes.data, np.ones(len(diabetes.data)))


def test_linearsvr():
    # check that SVR(kernel='linear') and LinearSVC() give
    # comparable results
    diabetes = datasets.load_diabetes()
    lsvr = svm.LinearSVR(C=1e3).fit(diabetes.data, diabetes.target)
    score1 = lsvr.score(diabetes.data, diabetes.target)

    svr = svm.SVR(kernel="linear", C=1e3).fit(diabetes.data, diabetes.target)
    score2 = svr.score(diabetes.data, diabetes.target)

    assert_allclose(np.linalg.norm(lsvr.coef_), np.linalg.norm(svr.coef_), 1, 0.0001)
    assert_almost_equal(score1, score2, 2)


def test_linearsvr_fit_sampleweight():
    # check correct result when sample_weight is 1
    # check that SVR(kernel='linear') and LinearSVC() give
    # comparable results
    diabetes = datasets.load_diabetes()
    n_samples = len(diabetes.target)
    unit_weight = np.ones(n_samples)
    lsvr = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target, sample_weight=unit_weight
    )
    score1 = lsvr.score(diabetes.data, diabetes.target)

    lsvr_no_weight = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target
    )
    score2 = lsvr_no_weight.score(diabetes.data, diabetes.target)

    assert_allclose(
        np.linalg.norm(lsvr.coef_), np.linalg.norm(lsvr_no_weight.coef_), 1, 0.0001
    )
    assert_almost_equal(score1, score2, 2)

    # check that fit(X)  = fit([X1, X2, X3],sample_weight = [n1, n2, n3]) where
    # X = X1 repeated n1 times, X2 repeated n2 times and so forth
    random_state = check_random_state(0)
    random_weight = random_state.randint(0, 10, n_samples)
    lsvr_unflat = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(
        diabetes.data, diabetes.target, sample_weight=random_weight
    )
    score3 = lsvr_unflat.score(
        diabetes.data, diabetes.target, sample_weight=random_weight
    )

    X_flat = np.repeat(diabetes.data, random_weight, axis=0)
    y_flat = np.repeat(diabetes.target, random_weight, axis=0)
    lsvr_flat = svm.LinearSVR(C=1e3, tol=1e-12, max_iter=10000).fit(X_flat, y_flat)
    score4 = lsvr_flat.score(X_flat, y_flat)

    assert_almost_equal(score3, score4, 2)


def test_svr_errors():
    X = [[0.0], [1.0]]
    y = [0.0, 0.5]

    # Bad kernel
    clf = svm.SVR(kernel=lambda x, y: np.array([[1.0]]))
    clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(X)


def test_oneclass():
    # Test OneClassSVM
    clf = svm.OneClassSVM()
    clf.fit(X)
    pred = clf.predict(T)

    assert_array_equal(pred, [1, -1, -1])
    assert pred.dtype == np.dtype("intp")
    assert_array_almost_equal(clf.intercept_, [-1.218], decimal=3)
    assert_array_almost_equal(clf.dual_coef_, [[0.750, 0.750, 0.750, 0.750]], decimal=3)
    with pytest.raises(AttributeError):
        (lambda: clf.coef_)()


def test_oneclass_decision_function():
    # Test OneClassSVM decision function
    clf = svm.OneClassSVM()
    rnd = check_random_state(2)

    # Generate train data
    X = 0.3 * rnd.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]

    # Generate some regular novel observations
    X = 0.3 * rnd.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rnd.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    # predict things
    y_pred_test = clf.predict(X_test)
    assert np.mean(y_pred_test == 1) > 0.9
    y_pred_outliers = clf.predict(X_outliers)
    assert np.mean(y_pred_outliers == -1) > 0.9
    dec_func_test = clf.decision_function(X_test)
    assert_array_equal((dec_func_test > 0).ravel(), y_pred_test == 1)
    dec_func_outliers = clf.decision_function(X_outliers)
    assert_array_equal((dec_func_outliers > 0).ravel(), y_pred_outliers == 1)


def test_oneclass_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf = svm.OneClassSVM(gamma=1).fit(X_train)
    assert_array_equal(
        clf.score_samples([[2.0, 2.0]]),
        clf.decision_function([[2.0, 2.0]]) + clf.offset_,
    )


# TODO: Remove in v1.2
def test_oneclass_fit_params_is_deprecated():
    clf = svm.OneClassSVM()
    params = {
        "unused_param": "",
        "extra_param": None,
    }
    msg = (
        "Passing additional keyword parameters has no effect and is deprecated "
        "in 1.0. An error will be raised from 1.2 and beyond. The ignored "
        f"keyword parameter(s) are: {params.keys()}."
    )
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        clf.fit(X, **params)


def test_tweak_params():
    # Make sure some tweaking of parameters works.
    # We change clf.dual_coef_ at run time and expect .predict() to change
    # accordingly. Notice that this is not trivial since it involves a lot
    # of C/Python copying in the libsvm bindings.
    # The success of this test ensures that the mapping between libsvm and
    # the python classifier is complete.
    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.predict([[-0.1, -0.1]]), [1])
    clf._dual_coef_ = np.array([[0.0, 1.0]])
    assert_array_equal(clf.predict([[-0.1, -0.1]]), [2])


def test_probability():
    # Predict probabilities using SVC
    # This uses cross validation, so we use a slightly bigger testing set.

    for clf in (
        svm.SVC(probability=True, random_state=0, C=1.0),
        svm.NuSVC(probability=True, random_state=0),
    ):
        clf.fit(iris.data, iris.target)

        prob_predict = clf.predict_proba(iris.data)
        assert_array_almost_equal(np.sum(prob_predict, 1), np.ones(iris.data.shape[0]))
        assert np.mean(np.argmax(prob_predict, 1) == clf.predict(iris.data)) > 0.9

        assert_almost_equal(
            clf.predict_proba(iris.data), np.exp(clf.predict_log_proba(iris.data)), 8
        )


def test_decision_function():
    # Test decision_function
    # Sanity check, test that decision_function implemented in python
    # returns the same as the one in libsvm
    # multi class:
    clf = svm.SVC(kernel="linear", C=0.1, decision_function_shape="ovo").fit(
        iris.data, iris.target
    )

    dec = np.dot(iris.data, clf.coef_.T) + clf.intercept_

    assert_array_almost_equal(dec, clf.decision_function(iris.data))

    # binary:
    clf.fit(X, Y)
    dec = np.dot(X, clf.coef_.T) + clf.intercept_
    prediction = clf.predict(X)
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))
    assert_array_almost_equal(
        prediction, clf.classes_[(clf.decision_function(X) > 0).astype(int)]
    )
    expected = np.array([-1.0, -0.66, -1.0, 0.66, 1.0, 1.0])
    assert_array_almost_equal(clf.decision_function(X), expected, 2)

    # kernel binary:
    clf = svm.SVC(kernel="rbf", gamma=1, decision_function_shape="ovo")
    clf.fit(X, Y)

    rbfs = rbf_kernel(X, clf.support_vectors_, gamma=clf.gamma)
    dec = np.dot(rbfs, clf.dual_coef_.T) + clf.intercept_
    assert_array_almost_equal(dec.ravel(), clf.decision_function(X))


@pytest.mark.parametrize("SVM", (svm.SVC, svm.NuSVC))
def test_decision_function_shape(SVM):
    # check that decision_function_shape='ovr' or 'ovo' gives
    # correct shape and is consistent with predict

    clf = SVM(kernel="linear", decision_function_shape="ovr").fit(
        iris.data, iris.target
    )
    dec = clf.decision_function(iris.data)
    assert dec.shape == (len(iris.data), 3)
    assert_array_equal(clf.predict(iris.data), np.argmax(dec, axis=1))

    # with five classes:
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = SVM(kernel="linear", decision_function_shape="ovr").fit(X_train, y_train)
    dec = clf.decision_function(X_test)
    assert dec.shape == (len(X_test), 5)
    assert_array_equal(clf.predict(X_test), np.argmax(dec, axis=1))

    # check shape of ovo_decition_function=True
    clf = SVM(kernel="linear", decision_function_shape="ovo").fit(X_train, y_train)
    dec = clf.decision_function(X_train)
    assert dec.shape == (len(X_train), 10)

    with pytest.raises(ValueError, match="must be either 'ovr' or 'ovo'"):
        SVM(decision_function_shape="bad").fit(X_train, y_train)


def test_svr_predict():
    # Test SVR's decision_function
    # Sanity check, test that predict implemented in python
    # returns the same as the one in libsvm

    X = iris.data
    y = iris.target

    # linear kernel
    reg = svm.SVR(kernel="linear", C=0.1).fit(X, y)

    dec = np.dot(X, reg.coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())

    # rbf kernel
    reg = svm.SVR(kernel="rbf", gamma=1).fit(X, y)

    rbfs = rbf_kernel(X, reg.support_vectors_, gamma=reg.gamma)
    dec = np.dot(rbfs, reg.dual_coef_.T) + reg.intercept_
    assert_array_almost_equal(dec.ravel(), reg.predict(X).ravel())


def test_weight():
    # Test class weights
    clf = svm.SVC(class_weight={1: 0.1})
    # we give a small weights to class 1
    clf.fit(X, Y)
    # so all predicted values belong to class 2
    assert_array_almost_equal(clf.predict(X), [2] * 6)

    X_, y_ = make_classification(
        n_samples=200, n_features=10, weights=[0.833, 0.167], random_state=2
    )

    for clf in (
        linear_model.LogisticRegression(),
        svm.LinearSVC(random_state=0),
        svm.SVC(),
    ):
        clf.set_params(class_weight={0: 0.1, 1: 10})
        clf.fit(X_[:100], y_[:100])
        y_pred = clf.predict(X_[100:])
        assert f1_score(y_[100:], y_pred) > 0.3


@pytest.mark.parametrize("estimator", [svm.SVC(C=1e-2), svm.NuSVC()])
def test_svm_classifier_sided_sample_weight(estimator):
    # fit a linear SVM and check that giving more weight to opposed samples
    # in the space will flip the decision toward these samples.
    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    estimator.set_params(kernel="linear")

    # check that with unit weights, a sample is supposed to be predicted on
    # the boundary
    sample_weight = [1] * 6
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred == pytest.approx(0)

    # give more weights to opposed samples
    sample_weight = [10.0, 0.1, 0.1, 0.1, 0.1, 10]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred < 0

    sample_weight = [1.0, 0.1, 10.0, 10.0, 0.1, 0.1]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.decision_function([[-1.0, 1.0]])
    assert y_pred > 0


@pytest.mark.parametrize("estimator", [svm.SVR(C=1e-2), svm.NuSVR(C=1e-2)])
def test_svm_regressor_sided_sample_weight(estimator):
    # similar test to test_svm_classifier_sided_sample_weight but for
    # SVM regressors
    X = [[-2, 0], [-1, -1], [0, -2], [0, 2], [1, 1], [2, 0]]
    estimator.set_params(kernel="linear")

    # check that with unit weights, a sample is supposed to be predicted on
    # the boundary
    sample_weight = [1] * 6
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.predict([[-1.0, 1.0]])
    assert y_pred == pytest.approx(1.5)

    # give more weights to opposed samples
    sample_weight = [10.0, 0.1, 0.1, 0.1, 0.1, 10]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.predict([[-1.0, 1.0]])
    assert y_pred < 1.5

    sample_weight = [1.0, 0.1, 10.0, 10.0, 0.1, 0.1]
    estimator.fit(X, Y, sample_weight=sample_weight)
    y_pred = estimator.predict([[-1.0, 1.0]])
    assert y_pred > 1.5


def test_svm_equivalence_sample_weight_C():
    # test that rescaling all samples is the same as changing C
    clf = svm.SVC()
    clf.fit(X, Y)
    dual_coef_no_weight = clf.dual_coef_
    clf.set_params(C=100)
    clf.fit(X, Y, sample_weight=np.repeat(0.01, len(X)))
    assert_allclose(dual_coef_no_weight, clf.dual_coef_)


@pytest.mark.parametrize(
    "Estimator, err_msg",
    [
        (svm.SVC, "Invalid input - all samples have zero or negative weights."),
        (svm.NuSVC, "(negative dimensions are not allowed|nu is infeasible)"),
        (svm.SVR, "Invalid input - all samples have zero or negative weights."),
        (svm.NuSVR, "Invalid input - all samples have zero or negative weights."),
        (svm.OneClassSVM, "Invalid input - all samples have zero or negative weights."),
    ],
    ids=["SVC", "NuSVC", "SVR", "NuSVR", "OneClassSVM"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [[0] * len(Y), [-0.3] * len(Y)],
    ids=["weights-are-zero", "weights-are-negative"],
)
def test_negative_sample_weights_mask_all_samples(Estimator, err_msg, sample_weight):
    est = Estimator(kernel="linear")
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, Y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "Classifier, err_msg",
    [
        (
            svm.SVC,
            "Invalid input - all samples with positive weights have the same label",
        ),
        (svm.NuSVC, "specified nu is infeasible"),
    ],
    ids=["SVC", "NuSVC"],
)
@pytest.mark.parametrize(
    "sample_weight",
    [[0, -0.5, 0, 1, 1, 1], [1, 1, 1, 0, -0.1, -0.3]],
    ids=["mask-label-1", "mask-label-2"],
)
def test_negative_weights_svc_leave_just_one_label(Classifier, err_msg, sample_weight):
    clf = Classifier(kernel="linear")
    with pytest.raises(ValueError, match=err_msg):
        clf.fit(X, Y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "Classifier, model",
    [
        (svm.SVC, {"when-left": [0.3998, 0.4], "when-right": [0.4, 0.3999]}),
        (svm.NuSVC, {"when-left": [0.3333, 0.3333], "when-right": [0.3333, 0.3333]}),
    ],
    ids=["SVC", "NuSVC"],
)
@pytest.mark.parametrize(
    "sample_weight, mask_side",
    [([1, -0.5, 1, 1, 1, 1], "when-left"), ([1, 1, 1, 0, 1, 1], "when-right")],
    ids=["partial-mask-label-1", "partial-mask-label-2"],
)
def test_negative_weights_svc_leave_two_labels(
    Classifier, model, sample_weight, mask_side
):
    clf = Classifier(kernel="linear")
    clf.fit(X, Y, sample_weight=sample_weight)
    assert_allclose(clf.coef_, [model[mask_side]], rtol=1e-3)


@pytest.mark.parametrize(
    "Estimator", [svm.SVC, svm.NuSVC, svm.NuSVR], ids=["SVC", "NuSVC", "NuSVR"]
)
@pytest.mark.parametrize(
    "sample_weight",
    [[1, -0.5, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]],
    ids=["partial-mask-label-1", "partial-mask-label-2"],
)
def test_negative_weight_equal_coeffs(Estimator, sample_weight):
    # model generates equal coefficients
    est = Estimator(kernel="linear")
    est.fit(X, Y, sample_weight=sample_weight)
    coef = np.abs(est.coef_).ravel()
    assert coef[0] == pytest.approx(coef[1], rel=1e-3)


@ignore_warnings(category=UndefinedMetricWarning)
def test_auto_weight():
    # Test class weights for imbalanced data
    from sklearn.linear_model import LogisticRegression

    # We take as dataset the two-dimensional projection of iris so
    # that it is not separable and remove half of predictors from
    # class 1.
    # We add one to the targets as a non-regression test:
    # class_weight="balanced"
    # used to work only when the labels where a range [0..K).
    from sklearn.utils import compute_class_weight

    X, y = iris.data[:, :2], iris.target + 1
    unbalanced = np.delete(np.arange(y.size), np.where(y > 2)[0][::2])

    classes = np.unique(y[unbalanced])
    class_weights = compute_class_weight("balanced", classes=classes, y=y[unbalanced])
    assert np.argmax(class_weights) == 2

    for clf in (
        svm.SVC(kernel="linear"),
        svm.LinearSVC(random_state=0),
        LogisticRegression(),
    ):
        # check that score is better when class='balanced' is set.
        y_pred = clf.fit(X[unbalanced], y[unbalanced]).predict(X)
        clf.set_params(class_weight="balanced")
        y_pred_balanced = clf.fit(
            X[unbalanced],
            y[unbalanced],
        ).predict(X)
        assert metrics.f1_score(y, y_pred, average="macro") <= metrics.f1_score(
            y, y_pred_balanced, average="macro"
        )


def test_bad_input():
    # Test that it gives proper exception on deficient input
    # impossible value of C
    with pytest.raises(ValueError):
        svm.SVC(C=-1).fit(X, Y)

    # impossible value of nu
    clf = svm.NuSVC(nu=0.0)
    with pytest.raises(ValueError):
        clf.fit(X, Y)

    Y2 = Y[:-1]  # wrong dimensions for labels
    with pytest.raises(ValueError):
        clf.fit(X, Y2)

    # Test with arrays that are non-contiguous.
    for clf in (svm.SVC(), svm.LinearSVC(random_state=0)):
        Xf = np.asfortranarray(X)
        assert not Xf.flags["C_CONTIGUOUS"]
        yf = np.ascontiguousarray(np.tile(Y, (2, 1)).T)
        yf = yf[:, -1]
        assert not yf.flags["F_CONTIGUOUS"]
        assert not yf.flags["C_CONTIGUOUS"]
        clf.fit(Xf, yf)
        assert_array_equal(clf.predict(T), true_result)

    # error for precomputed kernelsx
    clf = svm.SVC(kernel="precomputed")
    with pytest.raises(ValueError):
        clf.fit(X, Y)

    # predict with sparse input when trained with dense
    clf = svm.SVC().fit(X, Y)
    with pytest.raises(ValueError):
        clf.predict(sparse.lil_matrix(X))

    Xt = np.array(X).T
    clf.fit(np.dot(X, Xt), Y)
    with pytest.raises(ValueError):
        clf.predict(X)

    clf = svm.SVC()
    clf.fit(X, Y)
    with pytest.raises(ValueError):
        clf.predict(Xt)


@pytest.mark.parametrize(
    "Estimator, data",
    [
        (svm.SVC, datasets.load_iris(return_X_y=True)),
        (svm.NuSVC, datasets.load_iris(return_X_y=True)),
        (svm.SVR, datasets.load_diabetes(return_X_y=True)),
        (svm.NuSVR, datasets.load_diabetes(return_X_y=True)),
        (svm.OneClassSVM, datasets.load_iris(return_X_y=True)),
    ],
)
def test_svm_gamma_error(Estimator, data):
    X, y = data
    est = Estimator(gamma="auto_deprecated")
    err_msg = "When 'gamma' is a string, it should be either 'scale' or 'auto'"
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)


def test_unicode_kernel():
    # Test that a unicode kernel name does not cause a TypeError
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(X, Y)
    clf.predict_proba(T)
    _libsvm.cross_validation(
        iris.data, iris.target.astype(np.float64), 5, kernel="linear", random_seed=0
    )


def test_sparse_precomputed():
    clf = svm.SVC(kernel="precomputed")
    sparse_gram = sparse.csr_matrix([[1, 0], [0, 1]])
    with pytest.raises(TypeError, match="Sparse precomputed"):
        clf.fit(sparse_gram, [0, 1])


def test_sparse_fit_support_vectors_empty():
    # Regression test for #14893
    X_train = sparse.csr_matrix(
        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    y_train = np.array([0.04, 0.04, 0.10, 0.16])
    model = svm.SVR(kernel="linear")
    model.fit(X_train, y_train)
    assert not model.support_vectors_.data.size
    assert not model.dual_coef_.data.size


def test_linearsvc_parameters():
    # Test possible parameter combinations in LinearSVC
    # Generate list of possible parameter combinations
    losses = ["hinge", "squared_hinge", "logistic_regression", "foo"]
    penalties, duals = ["l1", "l2", "bar"], [True, False]

    X, y = make_classification(n_samples=5, n_features=5)

    for loss, penalty, dual in itertools.product(losses, penalties, duals):
        clf = svm.LinearSVC(penalty=penalty, loss=loss, dual=dual)
        if (
            (loss, penalty) == ("hinge", "l1")
            or (loss, penalty, dual) == ("hinge", "l2", False)
            or (penalty, dual) == ("l1", True)
            or loss == "foo"
            or penalty == "bar"
        ):

            with pytest.raises(
                ValueError,
                match="Unsupported set of arguments.*penalty='%s.*loss='%s.*dual=%s"
                % (penalty, loss, dual),
            ):
                clf.fit(X, y)
        else:
            clf.fit(X, y)

    # Incorrect loss value - test if explicit error message is raised
    with pytest.raises(ValueError, match=".*loss='l3' is not supported.*"):
        svm.LinearSVC(loss="l3").fit(X, y)


def test_linear_svx_uppercase_loss_penality_raises_error():
    # Check if Upper case notation raises error at _fit_liblinear
    # which is called by fit

    X, y = [[0.0], [1.0]], [0, 1]

    msg = "loss='SQuared_hinge' is not supported"
    with pytest.raises(ValueError, match=msg):
        svm.LinearSVC(loss="SQuared_hinge").fit(X, y)

    msg = "The combination of penalty='L2' and loss='squared_hinge' is not supported"
    with pytest.raises(ValueError, match=msg):
        svm.LinearSVC(penalty="L2").fit(X, y)


def test_linearsvc():
    # Test basic routines using LinearSVC
    clf = svm.LinearSVC(random_state=0).fit(X, Y)

    # by default should have intercept
    assert clf.fit_intercept

    assert_array_equal(clf.predict(T), true_result)
    assert_array_almost_equal(clf.intercept_, [0], decimal=3)

    # the same with l1 penalty
    clf = svm.LinearSVC(
        penalty="l1", loss="squared_hinge", dual=False, random_state=0
    ).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # l2 penalty with dual formulation
    clf = svm.LinearSVC(penalty="l2", dual=True, random_state=0).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # l2 penalty, l1 loss
    clf = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, random_state=0)
    clf.fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)

    # test also decision function
    dec = clf.decision_function(T)
    res = (dec > 0).astype(int) + 1
    assert_array_equal(res, true_result)


def test_linearsvc_crammer_singer():
    # Test LinearSVC with crammer_singer multi-class svm
    ovr_clf = svm.LinearSVC(random_state=0).fit(iris.data, iris.target)
    cs_clf = svm.LinearSVC(multi_class="crammer_singer", random_state=0)
    cs_clf.fit(iris.data, iris.target)

    # similar prediction for ovr and crammer-singer:
    assert (ovr_clf.predict(iris.data) == cs_clf.predict(iris.data)).mean() > 0.9

    # classifiers shouldn't be the same
    assert (ovr_clf.coef_ != cs_clf.coef_).all()

    # test decision function
    assert_array_equal(
        cs_clf.predict(iris.data),
        np.argmax(cs_clf.decision_function(iris.data), axis=1),
    )
    dec_func = np.dot(iris.data, cs_clf.coef_.T) + cs_clf.intercept_
    assert_array_almost_equal(dec_func, cs_clf.decision_function(iris.data))


def test_linearsvc_fit_sampleweight():
    # check correct result when sample_weight is 1
    n_samples = len(X)
    unit_weight = np.ones(n_samples)
    clf = svm.LinearSVC(random_state=0).fit(X, Y)
    clf_unitweight = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X, Y, sample_weight=unit_weight
    )

    # check if same as sample_weight=None
    assert_array_equal(clf_unitweight.predict(T), clf.predict(T))
    assert_allclose(clf.coef_, clf_unitweight.coef_, 1, 0.0001)

    # check that fit(X)  = fit([X1, X2, X3],sample_weight = [n1, n2, n3]) where
    # X = X1 repeated n1 times, X2 repeated n2 times and so forth

    random_state = check_random_state(0)
    random_weight = random_state.randint(0, 10, n_samples)
    lsvc_unflat = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X, Y, sample_weight=random_weight
    )
    pred1 = lsvc_unflat.predict(T)

    X_flat = np.repeat(X, random_weight, axis=0)
    y_flat = np.repeat(Y, random_weight, axis=0)
    lsvc_flat = svm.LinearSVC(random_state=0, tol=1e-12, max_iter=1000).fit(
        X_flat, y_flat
    )
    pred2 = lsvc_flat.predict(T)

    assert_array_equal(pred1, pred2)
    assert_allclose(lsvc_unflat.coef_, lsvc_flat.coef_, 1, 0.0001)


def test_crammer_singer_binary():
    # Test Crammer-Singer formulation in the binary case
    X, y = make_classification(n_classes=2, random_state=0)

    for fit_intercept in (True, False):
        acc = (
            svm.LinearSVC(
                fit_intercept=fit_intercept,
                multi_class="crammer_singer",
                random_state=0,
            )
            .fit(X, y)
            .score(X, y)
        )
        assert acc > 0.9


def test_linearsvc_iris():
    # Test that LinearSVC gives plausible predictions on the iris dataset
    # Also, test symbolic class names (classes_).
    target = iris.target_names[iris.target]
    clf = svm.LinearSVC(random_state=0).fit(iris.data, target)
    assert set(clf.classes_) == set(iris.target_names)
    assert np.mean(clf.predict(iris.data) == target) > 0.8

    dec = clf.decision_function(iris.data)
    pred = iris.target_names[np.argmax(dec, 1)]
    assert_array_equal(pred, clf.predict(iris.data))


def test_dense_liblinear_intercept_handling(classifier=svm.LinearSVC):
    # Test that dense liblinear honours intercept_scaling param
    X = [[2, 1], [3, 1], [1, 3], [2, 3]]
    y = [0, 0, 1, 1]
    clf = classifier(
        fit_intercept=True,
        penalty="l1",
        loss="squared_hinge",
        dual=False,
        C=4,
        tol=1e-7,
        random_state=0,
    )
    assert clf.intercept_scaling == 1, clf.intercept_scaling
    assert clf.fit_intercept

    # when intercept_scaling is low the intercept value is highly "penalized"
    # by regularization
    clf.intercept_scaling = 1
    clf.fit(X, y)
    assert_almost_equal(clf.intercept_, 0, decimal=5)

    # when intercept_scaling is sufficiently high, the intercept value
    # is not affected by regularization
    clf.intercept_scaling = 100
    clf.fit(X, y)
    intercept1 = clf.intercept_
    assert intercept1 < -1

    # when intercept_scaling is sufficiently high, the intercept value
    # doesn't depend on intercept_scaling value
    clf.intercept_scaling = 1000
    clf.fit(X, y)
    intercept2 = clf.intercept_
    assert_array_almost_equal(intercept1, intercept2, decimal=2)


def test_liblinear_set_coef():
    # multi-class case
    clf = svm.LinearSVC().fit(iris.data, iris.target)
    values = clf.decision_function(iris.data)
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    values2 = clf.decision_function(iris.data)
    assert_array_almost_equal(values, values2)

    # binary-class case
    X = [[2, 1], [3, 1], [1, 3], [2, 3]]
    y = [0, 0, 1, 1]

    clf = svm.LinearSVC().fit(X, y)
    values = clf.decision_function(X)
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    values2 = clf.decision_function(X)
    assert_array_equal(values, values2)


def test_immutable_coef_property():
    # Check that primal coef modification are not silently ignored
    svms = [
        svm.SVC(kernel="linear").fit(iris.data, iris.target),
        svm.NuSVC(kernel="linear").fit(iris.data, iris.target),
        svm.SVR(kernel="linear").fit(iris.data, iris.target),
        svm.NuSVR(kernel="linear").fit(iris.data, iris.target),
        svm.OneClassSVM(kernel="linear").fit(iris.data),
    ]
    for clf in svms:
        with pytest.raises(AttributeError):
            clf.__setattr__("coef_", np.arange(3))
        with pytest.raises((RuntimeError, ValueError)):
            clf.coef_.__setitem__((0, 0), 0)


def test_linearsvc_verbose():
    # stdout: redirect
    import os

    stdout = os.dup(1)  # save original stdout
    os.dup2(os.pipe()[1], 1)  # replace it

    # actual call
    clf = svm.LinearSVC(verbose=1)
    clf.fit(X, Y)

    # stdout: restore
    os.dup2(stdout, 1)  # restore original stdout


def test_svc_clone_with_callable_kernel():
    # create SVM with callable linear kernel, check that results are the same
    # as with built-in linear kernel
    svm_callable = svm.SVC(
        kernel=lambda x, y: np.dot(x, y.T),
        probability=True,
        random_state=0,
        decision_function_shape="ovr",
    )
    # clone for checking clonability with lambda functions..
    svm_cloned = base.clone(svm_callable)
    svm_cloned.fit(iris.data, iris.target)

    svm_builtin = svm.SVC(
        kernel="linear", probability=True, random_state=0, decision_function_shape="ovr"
    )
    svm_builtin.fit(iris.data, iris.target)

    assert_array_almost_equal(svm_cloned.dual_coef_, svm_builtin.dual_coef_)
    assert_array_almost_equal(svm_cloned.intercept_, svm_builtin.intercept_)
    assert_array_equal(svm_cloned.predict(iris.data), svm_builtin.predict(iris.data))

    assert_array_almost_equal(
        svm_cloned.predict_proba(iris.data),
        svm_builtin.predict_proba(iris.data),
        decimal=4,
    )
    assert_array_almost_equal(
        svm_cloned.decision_function(iris.data),
        svm_builtin.decision_function(iris.data),
    )


def test_svc_bad_kernel():
    svc = svm.SVC(kernel=lambda x, y: x)
    with pytest.raises(ValueError):
        svc.fit(X, Y)


def test_timeout():
    a = svm.SVC(
        kernel=lambda x, y: np.dot(x, y.T), probability=True, random_state=0, max_iter=1
    )
    warning_msg = (
        r"Solver terminated early \(max_iter=1\).  Consider pre-processing "
        r"your data with StandardScaler or MinMaxScaler."
    )
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        a.fit(np.array(X), Y)


def test_unfitted():
    X = "foo!"  # input validation not required when SVM not fitted

    clf = svm.SVC()
    with pytest.raises(Exception, match=r".*\bSVC\b.*\bnot\b.*\bfitted\b"):
        clf.predict(X)

    clf = svm.NuSVR()
    with pytest.raises(Exception, match=r".*\bNuSVR\b.*\bnot\b.*\bfitted\b"):
        clf.predict(X)


# ignore convergence warnings from max_iter=1
@ignore_warnings
def test_consistent_proba():
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    proba_1 = a.fit(X, Y).predict_proba(X)
    a = svm.SVC(probability=True, max_iter=1, random_state=0)
    proba_2 = a.fit(X, Y).predict_proba(X)
    assert_array_almost_equal(proba_1, proba_2)


def test_linear_svm_convergence_warnings():
    # Test that warnings are raised if model does not converge

    lsvc = svm.LinearSVC(random_state=0, max_iter=2)
    warning_msg = "Liblinear failed to converge, increase the number of iterations."
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvc.fit(X, Y)
    assert lsvc.n_iter_ == 2

    lsvr = svm.LinearSVR(random_state=0, max_iter=2)
    with pytest.warns(ConvergenceWarning, match=warning_msg):
        lsvr.fit(iris.data, iris.target)
    assert lsvr.n_iter_ == 2


def test_svr_coef_sign():
    # Test that SVR(kernel="linear") has coef_ with the right sign.
    # Non-regression test for #2933.
    X = np.random.RandomState(21).randn(10, 3)
    y = np.random.RandomState(12).randn(10)

    for svr in [svm.SVR(kernel="linear"), svm.NuSVR(kernel="linear"), svm.LinearSVR()]:
        svr.fit(X, y)
        assert_array_almost_equal(
            svr.predict(X), np.dot(X, svr.coef_.ravel()) + svr.intercept_
        )


def test_linear_svc_intercept_scaling():
    # Test that the right error message is thrown when intercept_scaling <= 0

    for i in [-1, 0]:
        lsvc = svm.LinearSVC(intercept_scaling=i)

        msg = (
            "Intercept scaling is %r but needs to be greater than 0."
            " To disable fitting an intercept,"
            " set fit_intercept=False."
            % lsvc.intercept_scaling
        )
        with pytest.raises(ValueError, match=msg):
            lsvc.fit(X, Y)


def test_lsvc_intercept_scaling_zero():
    # Test that intercept_scaling is ignored when fit_intercept is False

    lsvc = svm.LinearSVC(fit_intercept=False)
    lsvc.fit(X, Y)
    assert lsvc.intercept_ == 0.0


def test_hasattr_predict_proba():
    # Method must be (un)available before or after fit, switched by
    # `probability` param

    G = svm.SVC(probability=True)
    assert hasattr(G, "predict_proba")
    G.fit(iris.data, iris.target)
    assert hasattr(G, "predict_proba")

    G = svm.SVC(probability=False)
    assert not hasattr(G, "predict_proba")
    G.fit(iris.data, iris.target)
    assert not hasattr(G, "predict_proba")

    # Switching to `probability=True` after fitting should make
    # predict_proba available, but calling it must not work:
    G.probability = True
    assert hasattr(G, "predict_proba")
    msg = "predict_proba is not available when fitted with probability=False"

    with pytest.raises(NotFittedError, match=msg):
        G.predict_proba(iris.data)


def test_decision_function_shape_two_class():
    for n_classes in [2, 3]:
        X, y = make_blobs(centers=n_classes, random_state=0)
        for estimator in [svm.SVC, svm.NuSVC]:
            clf = OneVsRestClassifier(estimator(decision_function_shape="ovr")).fit(
                X, y
            )
            assert len(clf.predict(X)) == len(y)


def test_ovr_decision_function():
    # One point from each quadrant represents one class
    X_train = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    y_train = [0, 1, 2, 3]

    # First point is closer to the decision boundaries than the second point
    base_points = np.array([[5, 5], [10, 10]])

    # For all the quadrants (classes)
    X_test = np.vstack(
        (
            base_points * [1, 1],  # Q1
            base_points * [-1, 1],  # Q2
            base_points * [-1, -1],  # Q3
            base_points * [1, -1],  # Q4
        )
    )

    y_test = [0] * 2 + [1] * 2 + [2] * 2 + [3] * 2

    clf = svm.SVC(kernel="linear", decision_function_shape="ovr")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Test if the prediction is the same as y
    assert_array_equal(y_pred, y_test)

    deci_val = clf.decision_function(X_test)

    # Assert that the predicted class has the maximum value
    assert_array_equal(np.argmax(deci_val, axis=1), y_pred)

    # Get decision value at test points for the predicted class
    pred_class_deci_val = deci_val[range(8), y_pred].reshape((4, 2))

    # Assert pred_class_deci_val > 0 here
    assert np.min(pred_class_deci_val) > 0.0

    # Test if the first point has lower decision value on every quadrant
    # compared to the second point
    assert np.all(pred_class_deci_val[:, 0] < pred_class_deci_val[:, 1])


@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
def test_svc_invalid_break_ties_param(SVCClass):
    X, y = make_blobs(random_state=42)

    svm = SVCClass(
        kernel="linear", decision_function_shape="ovo", break_ties=True, random_state=42
    ).fit(X, y)

    with pytest.raises(ValueError, match="break_ties must be False"):
        svm.predict(y)


@pytest.mark.parametrize("SVCClass", [svm.SVC, svm.NuSVC])
def test_svc_ovr_tie_breaking(SVCClass):
    """Test if predict breaks ties in OVR mode.
    Related issue: https://github.com/scikit-learn/scikit-learn/issues/8277
    """
    X, y = make_blobs(random_state=0, n_samples=20, n_features=2)

    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xs, ys)

    common_params = dict(
        kernel="rbf", gamma=1e6, random_state=42, decision_function_shape="ovr"
    )
    svm = SVCClass(
        break_ties=False,
        **common_params,
    ).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert not np.all(pred == np.argmax(dv, axis=1))

    svm = SVCClass(
        break_ties=True,
        **common_params,
    ).fit(X, y)
    pred = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    dv = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    assert np.all(pred == np.argmax(dv, axis=1))


def test_gamma_auto():
    X, y = [[0.0, 1.2], [1.0, 1.3]], [0, 1]

    with pytest.warns(None) as record:
        svm.SVC(kernel="linear").fit(X, y)
    assert not len(record)

    with pytest.warns(None) as record:
        svm.SVC(kernel="precomputed").fit(X, y)
    assert not len(record)


def test_gamma_scale():
    X, y = [[0.0], [1.0]], [0, 1]

    clf = svm.SVC()
    with pytest.warns(None) as record:
        clf.fit(X, y)
    assert not len(record)
    assert_almost_equal(clf._gamma, 4)

    # X_var ~= 1 shouldn't raise warning, for when
    # gamma is not explicitly set.
    X, y = [[1, 2], [3, 2 * np.sqrt(6) / 3 + 2]], [0, 1]
    with pytest.warns(None) as record:
        clf.fit(X, y)
    assert not len(record)


@pytest.mark.parametrize(
    "SVM, params",
    [
        (LinearSVC, {"penalty": "l1", "loss": "squared_hinge", "dual": False}),
        (LinearSVC, {"penalty": "l2", "loss": "squared_hinge", "dual": True}),
        (LinearSVC, {"penalty": "l2", "loss": "squared_hinge", "dual": False}),
        (LinearSVC, {"penalty": "l2", "loss": "hinge", "dual": True}),
        (LinearSVR, {"loss": "epsilon_insensitive", "dual": True}),
        (LinearSVR, {"loss": "squared_epsilon_insensitive", "dual": True}),
        (LinearSVR, {"loss": "squared_epsilon_insensitive", "dual": True}),
    ],
)
def test_linearsvm_liblinear_sample_weight(SVM, params):
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

    base_estimator = SVM(random_state=42)
    base_estimator.set_params(**params)
    base_estimator.set_params(tol=1e-12, max_iter=1000)
    est_no_weight = base.clone(base_estimator).fit(X, y)
    est_with_weight = base.clone(base_estimator).fit(
        X2, y2, sample_weight=sample_weight
    )

    for method in ("predict", "decision_function"):
        if hasattr(base_estimator, method):
            X_est_no_weight = getattr(est_no_weight, method)(X)
            X_est_with_weight = getattr(est_with_weight, method)(X)
            assert_allclose(X_est_no_weight, X_est_with_weight)


def test_n_support_oneclass_svr():
    # Make n_support is correct for oneclass and SVR (used to be
    # non-initialized)
    # this is a non regression test for issue #14774
    X = np.array([[0], [0.44], [0.45], [0.46], [1]])
    clf = svm.OneClassSVM()
    assert not hasattr(clf, "n_support_")
    clf.fit(X)
    assert clf.n_support_ == clf.support_vectors_.shape[0]
    assert clf.n_support_.size == 1
    assert clf.n_support_ == 3

    y = np.arange(X.shape[0])
    reg = svm.SVR().fit(X, y)
    assert reg.n_support_ == reg.support_vectors_.shape[0]
    assert reg.n_support_.size == 1
    assert reg.n_support_ == 4


@pytest.mark.parametrize("Estimator", [svm.SVC, svm.SVR])
def test_custom_kernel_not_array_input(Estimator):
    """Test using a custom kernel that is not fed with array-like for floats"""
    data = ["A A", "A", "B", "B B", "A B"]
    X = np.array([[2, 0], [1, 0], [0, 1], [0, 2], [1, 1]])  # count encoding
    y = np.array([1, 1, 2, 2, 1])

    def string_kernel(X1, X2):
        assert isinstance(X1[0], str)
        n_samples1 = _num_samples(X1)
        n_samples2 = _num_samples(X2)
        K = np.zeros((n_samples1, n_samples2))
        for ii in range(n_samples1):
            for jj in range(ii, n_samples2):
                K[ii, jj] = X1[ii].count("A") * X2[jj].count("A")
                K[ii, jj] += X1[ii].count("B") * X2[jj].count("B")
                K[jj, ii] = K[ii, jj]
        return K

    K = string_kernel(data, data)
    assert_array_equal(np.dot(X, X.T), K)

    svc1 = Estimator(kernel=string_kernel).fit(data, y)
    svc2 = Estimator(kernel="linear").fit(X, y)
    svc3 = Estimator(kernel="precomputed").fit(K, y)

    assert svc1.score(data, y) == svc3.score(K, y)
    assert svc1.score(data, y) == svc2.score(X, y)
    if hasattr(svc1, "decision_function"):  # classifier
        assert_allclose(svc1.decision_function(data), svc2.decision_function(X))
        assert_allclose(svc1.decision_function(data), svc3.decision_function(K))
        assert_array_equal(svc1.predict(data), svc2.predict(X))
        assert_array_equal(svc1.predict(data), svc3.predict(K))
    else:  # regressor
        assert_allclose(svc1.predict(data), svc2.predict(X))
        assert_allclose(svc1.predict(data), svc3.predict(K))


def test_svc_raises_error_internal_representation():
    """Check that SVC raises error when internal representation is altered.

    Non-regression test for #18891 and https://nvd.nist.gov/vuln/detail/CVE-2020-28975
    """
    clf = svm.SVC(kernel="linear").fit(X, Y)
    clf._n_support[0] = 1000000

    msg = "The internal representation of SVC was altered"
    with pytest.raises(ValueError, match=msg):
        clf.predict(X)
