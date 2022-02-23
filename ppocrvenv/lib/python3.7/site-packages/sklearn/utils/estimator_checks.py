import types
import warnings
import pickle
import re
from copy import deepcopy
from functools import partial, wraps
from inspect import signature

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import joblib

from . import IS_PYPY
from .. import config_context
from ._testing import _get_args
from ._testing import assert_raise_message
from ._testing import assert_array_equal
from ._testing import assert_array_almost_equal
from ._testing import assert_allclose
from ._testing import assert_allclose_dense_sparse
from ._testing import assert_array_less
from ._testing import set_random_state
from ._testing import SkipTest
from ._testing import ignore_warnings
from ._testing import create_memmap_backed_data
from ._testing import raises
from . import is_scalar_nan

from ..linear_model import LinearRegression
from ..linear_model import LogisticRegression
from ..linear_model import RANSACRegressor
from ..linear_model import Ridge

from ..base import (
    clone,
    ClusterMixin,
    is_classifier,
    is_regressor,
    is_outlier_detector,
    RegressorMixin,
    _is_pairwise,
)

from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..random_projection import BaseRandomProjection
from ..feature_selection import SelectKBest
from ..pipeline import make_pipeline
from ..exceptions import DataConversionWarning
from ..exceptions import NotFittedError
from ..exceptions import SkipTestWarning
from ..model_selection import train_test_split
from ..model_selection import ShuffleSplit
from ..model_selection._validation import _safe_split
from ..metrics.pairwise import rbf_kernel, linear_kernel, pairwise_distances
from ..utils.fixes import threadpool_info
from ..utils.validation import check_is_fitted

from . import shuffle
from ._tags import (
    _DEFAULT_TAGS,
    _safe_tags,
)
from .validation import has_fit_parameter, _num_samples
from ..preprocessing import StandardScaler
from ..preprocessing import scale
from ..datasets import (
    load_iris,
    make_blobs,
    make_multilabel_classification,
    make_regression,
)

REGRESSION_DATASET = None
CROSS_DECOMPOSITION = ["PLSCanonical", "PLSRegression", "CCA", "PLSSVD"]


def _yield_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)
    pairwise = _is_pairwise(estimator)

    yield check_no_attributes_set_in_init
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    if has_fit_parameter(estimator, "sample_weight"):
        yield check_sample_weights_pandas_series
        yield check_sample_weights_not_an_array
        yield check_sample_weights_list
        if not pairwise:
            # We skip pairwise because the data is not pairwise
            yield check_sample_weights_shape
            yield check_sample_weights_not_overwritten
            yield partial(check_sample_weights_invariance, kind="ones")
            yield partial(check_sample_weights_invariance, kind="zeros")
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    if not tags["no_validation"]:
        yield check_complex_data
        yield check_dtype_object
        yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION:
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if not tags["allow_nan"] and not tags["no_validation"]:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    if pairwise:
        # Check that pairwise estimator throws error on non-square input
        yield check_nonsquare_error

    yield check_estimators_overwrite_params
    if hasattr(estimator, "sparsify"):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle

    yield check_estimator_get_tags_default_keys


def _yield_classifier_checks(classifier):
    tags = _safe_tags(classifier)

    # test classifiers can handle non-array data and pandas objects
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    if tags["multioutput"]:
        yield check_classifier_multioutput
    # basic consistency testing
    yield check_classifiers_train
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield partial(check_classifiers_train, readonly_memmap=True, X_dtype="float32")
    yield check_classifiers_regression_target
    if tags["multilabel"]:
        yield check_classifiers_multilabel_representation_invariance
        yield check_classifiers_multilabel_output_format_predict
        yield check_classifiers_multilabel_output_format_predict_proba
        yield check_classifiers_multilabel_output_format_decision_function
    if not tags["no_validation"]:
        yield check_supervised_y_no_nan
        if not tags["multioutput_only"]:
            yield check_supervised_y_2d
    if tags["requires_fit"]:
        yield check_estimators_unfitted
    if "class_weight" in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency


@ignore_warnings(category=FutureWarning)
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    estimator = clone(estimator_orig)
    rng = np.random.RandomState(888)
    X = rng.randn(10, 5)
    y = np.full(10, np.inf)
    y = _enforce_estimator_tags_y(estimator, y)

    match = (
        "Input contains NaN, infinity or a value too large for " r"dtype\('float64'\)."
    )
    err_msg = (
        f"Estimator {name} should have raised error on fitting array y with NaN value."
    )
    with raises(ValueError, match=match, err_msg=err_msg):
        estimator.fit(X, y)


def _yield_regressor_checks(regressor):
    tags = _safe_tags(regressor)
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield partial(check_regressors_train, readonly_memmap=True)
    yield partial(check_regressors_train, readonly_memmap=True, X_dtype="float32")
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    if tags["multioutput"]:
        yield check_regressor_multioutput
    yield check_regressors_no_decision_function
    if not tags["no_validation"] and not tags["multioutput_only"]:
        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    name = regressor.__class__.__name__
    if name != "CCA":
        # check that the regressor handles int input
        yield check_regressors_int
    if tags["requires_fit"]:
        yield check_estimators_unfitted
    yield check_non_transformer_estimators_n_iter


def _yield_transformer_checks(transformer):
    tags = _safe_tags(transformer)
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if not tags["no_validation"]:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    yield check_transformer_general
    if tags["preserves_dtype"]:
        yield check_transformer_preserve_dtypes
    yield partial(check_transformer_general, readonly_memmap=True)
    if not _safe_tags(transformer, key="stateless"):
        yield check_transformers_unfitted
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = [
        "Isomap",
        "KernelPCA",
        "LocallyLinearEmbedding",
        "RandomizedLasso",
        "LogisticRegressionCV",
    ]

    name = transformer.__class__.__name__
    if name not in external_solver:
        yield check_transformer_n_iter


def _yield_clustering_checks(clusterer):
    yield check_clusterer_compute_labels_predict
    name = clusterer.__class__.__name__
    if name not in ("WardAgglomeration", "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield partial(check_clustering, readonly_memmap=True)
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter


def _yield_outliers_checks(estimator):

    # checks for outlier detectors that have a fit_predict method
    if hasattr(estimator, "fit_predict"):
        yield check_outliers_fit_predict

    # checks for estimators that can be used on a test set
    if hasattr(estimator, "predict"):
        yield check_outliers_train
        yield partial(check_outliers_train, readonly_memmap=True)
        # test outlier detectors can handle non-array data
        yield check_classifier_data_not_an_array
        # test if NotFittedError is raised
        if _safe_tags(estimator, key="requires_fit"):
            yield check_estimators_unfitted


def _yield_all_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)
    if "2darray" not in tags["X_types"]:
        warnings.warn(
            "Can't test estimator {} which requires input  of type {}".format(
                name, tags["X_types"]
            ),
            SkipTestWarning,
        )
        return
    if tags["_skip_test"]:
        warnings.warn(
            "Explicit SKIP via _skip_test tag for estimator {}.".format(name),
            SkipTestWarning,
        )
        return

    for check in _yield_checks(estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(estimator):
            yield check
    if hasattr(estimator, "transform"):
        for check in _yield_transformer_checks(estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(estimator):
            yield check
    if is_outlier_detector(estimator):
        for check in _yield_outliers_checks(estimator):
            yield check
    yield check_parameters_default_constructible
    yield check_methods_sample_order_invariance
    yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent
    yield check_fit_check_is_fitted
    if not tags["no_validation"]:
        yield check_n_features_in
        yield check_fit1d
        yield check_fit2d_predict1d
        if tags["requires_y"]:
            yield check_requires_y_none
    if tags["requires_positive_X"]:
        yield check_fit_non_negative


def _get_check_estimator_ids(obj):
    """Create pytest ids for checks.

    When `obj` is an estimator, this returns the pprint version of the
    estimator (with `print_changed_only=True`). When `obj` is a function, the
    name of the function is returned with its keyword arguments.

    `_get_check_estimator_ids` is designed to be used as the `id` in
    `pytest.mark.parametrize` where `check_estimator(..., generate_only=True)`
    is yielding estimators and checks.

    Parameters
    ----------
    obj : estimator or function
        Items generated by `check_estimator`.

    Returns
    -------
    id : str or None

    See Also
    --------
    check_estimator
    """
    if callable(obj):
        if not isinstance(obj, partial):
            return obj.__name__

        if not obj.keywords:
            return obj.func.__name__

        kwstring = ",".join(["{}={}".format(k, v) for k, v in obj.keywords.items()])
        return "{}({})".format(obj.func.__name__, kwstring)
    if hasattr(obj, "get_params"):
        with config_context(print_changed_only=True):
            return re.sub(r"\s", "", str(obj))


def _construct_instance(Estimator):
    """Construct Estimator instance if possible."""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters):
        if required_parameters in (["estimator"], ["base_estimator"]):
            # `RANSACRegressor` will raise an error with any model other
            # than `LinearRegression` if we don't fix `min_samples` parameter.
            # For common test, we can enforce using `LinearRegression` that
            # is the default estimator in `RANSACRegressor` instead of `Ridge`.
            if issubclass(Estimator, RANSACRegressor):
                estimator = Estimator(LinearRegression())
            elif issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            else:
                estimator = Estimator(LogisticRegression(C=1))
        elif required_parameters in (["estimators"],):
            # Heterogeneous ensemble classes (i.e. stacking, voting)
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(
                    estimators=[("est1", Ridge(alpha=0.1)), ("est2", Ridge(alpha=1))]
                )
            else:
                estimator = Estimator(
                    estimators=[
                        ("est1", LogisticRegression(C=0.1)),
                        ("est2", LogisticRegression(C=1)),
                    ]
                )
        else:
            msg = (
                f"Can't instantiate estimator {Estimator.__name__} "
                f"parameters {required_parameters}"
            )
            # raise additional warning to be shown by pytest
            warnings.warn(msg, SkipTestWarning)
            raise SkipTest(msg)
    else:
        estimator = Estimator()
    return estimator


def _maybe_mark_xfail(estimator, check, pytest):
    # Mark (estimator, check) pairs as XFAIL if needed (see conditions in
    # _should_be_skipped_or_marked())
    # This is similar to _maybe_skip(), but this one is used by
    # @parametrize_with_checks() instead of check_estimator()

    should_be_marked, reason = _should_be_skipped_or_marked(estimator, check)
    if not should_be_marked:
        return estimator, check
    else:
        return pytest.param(estimator, check, marks=pytest.mark.xfail(reason=reason))


def _maybe_skip(estimator, check):
    # Wrap a check so that it's skipped if needed (see conditions in
    # _should_be_skipped_or_marked())
    # This is similar to _maybe_mark_xfail(), but this one is used by
    # check_estimator() instead of @parametrize_with_checks which requires
    # pytest
    should_be_skipped, reason = _should_be_skipped_or_marked(estimator, check)
    if not should_be_skipped:
        return check

    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    @wraps(check)
    def wrapped(*args, **kwargs):
        raise SkipTest(
            f"Skipping {check_name} for {estimator.__class__.__name__}: {reason}"
        )

    return wrapped


def _should_be_skipped_or_marked(estimator, check):
    # Return whether a check should be skipped (when using check_estimator())
    # or marked as XFAIL (when using @parametrize_with_checks()), along with a
    # reason.
    # Currently, a check should be skipped or marked if
    # the check is in the _xfail_checks tag of the estimator

    check_name = check.func.__name__ if isinstance(check, partial) else check.__name__

    xfail_checks = _safe_tags(estimator, key="_xfail_checks") or {}
    if check_name in xfail_checks:
        return True, xfail_checks[check_name]

    return False, "placeholder reason that will never be used"


def parametrize_with_checks(estimators):
    """Pytest specific decorator for parametrizing estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

        .. versionchanged:: 0.24
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24. Pass an instance instead.

        .. versionadded:: 0.24

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    check_estimator : Check if estimator adheres to scikit-learn conventions.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)

    """
    import pytest

    if any(isinstance(est, type) for est in estimators):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    def checks_generator():
        for estimator in estimators:
            name = type(estimator).__name__
            for check in _yield_all_checks(estimator):
                check = partial(check, name)
                yield _maybe_mark_xfail(estimator, check, pytest)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )


def check_estimator(Estimator, generate_only=False):
    """Check if estimator adheres to scikit-learn conventions.

    This estimator will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    scikit-learn provides a pytest specific decorator,
    :func:`~sklearn.utils.parametrize_with_checks`, making it easier to test
    multiple estimators.

    Parameters
    ----------
    Estimator : estimator object
        Estimator instance to check.

        .. versionchanged:: 0.24
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24.

    generate_only : bool, default=False
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

        .. versionadded:: 0.22

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.
    """
    if isinstance(Estimator, type):
        msg = (
            "Passing a class was deprecated in version 0.23 "
            "and isn't supported anymore from 0.24."
            "Please pass an instance instead."
        )
        raise TypeError(msg)

    estimator = Estimator
    name = type(estimator).__name__

    def checks_generator():
        for check in _yield_all_checks(estimator):
            check = _maybe_skip(estimator, check)
            yield estimator, partial(check, name)

    if generate_only:
        return checks_generator()

    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            # SkipTest is thrown when pandas can't be imported, or by checks
            # that are in the xfail_checks tag
            warnings.warn(str(exception), SkipTestWarning)


def _regression_dataset():
    global REGRESSION_DATASET
    if REGRESSION_DATASET is None:
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=1,
            bias=5.0,
            noise=20,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X)
        REGRESSION_DATASET = X, y
    return REGRESSION_DATASET


def _set_checking_parameters(estimator):
    # set parameters to speed up some estimators and
    # avoid deprecated behaviour
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if "n_iter" in params and name != "TSNE":
        estimator.set_params(n_iter=5)
    if "max_iter" in params:
        if estimator.max_iter is not None:
            estimator.set_params(max_iter=min(5, estimator.max_iter))
        # LinearSVR, LinearSVC
        if estimator.__class__.__name__ in ["LinearSVR", "LinearSVC"]:
            estimator.set_params(max_iter=20)
        # NMF
        if estimator.__class__.__name__ == "NMF":
            # FIXME : init should be removed in 1.1
            estimator.set_params(max_iter=500, init="nndsvda")
        # MLP
        if estimator.__class__.__name__ in ["MLPClassifier", "MLPRegressor"]:
            estimator.set_params(max_iter=100)
    if "n_resampling" in params:
        # randomized lasso
        estimator.set_params(n_resampling=5)
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if "max_trials" in params:
        # RANSAC
        estimator.set_params(max_trials=10)
    if "n_init" in params:
        # K-Means
        estimator.set_params(n_init=2)
    if name == "MeanShift":
        # In the case of check_fit2d_1sample, bandwidth is set to None and
        # is thus estimated. De facto it is 0.0 as a single sample is provided
        # and this makes the test fails. Hence we give it a placeholder value.
        estimator.set_params(bandwidth=1.0)

    if name == "TruncatedSVD":
        # TruncatedSVD doesn't run with n_components = n_features
        # This is ugly :-/
        estimator.n_components = 1

    if name == "LassoLarsIC":
        # Noise variance estimation does not work when `n_samples < n_features`.
        # We need to provide the noise variance explicitly.
        estimator.set_params(noise_variance=1.0)

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = min(estimator.n_clusters, 2)

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    if name == "SelectFdr":
        # be tolerant of noisy datasets (not actually speed)
        estimator.set_params(alpha=0.5)

    if name == "TheilSenRegressor":
        estimator.max_subpopulation = 100

    if isinstance(estimator, BaseRandomProjection):
        # Due to the jl lemma and often very few samples, the number
        # of components of the random matrix projection will be probably
        # greater than the number of features.
        # So we impose a smaller number (avoid "auto" mode)
        estimator.set_params(n_components=2)

    if isinstance(estimator, SelectKBest):
        # SelectKBest has a default of k=10
        # which is more feature than we have in most case.
        estimator.set_params(k=1)

    if name in ("HistGradientBoostingClassifier", "HistGradientBoostingRegressor"):
        # The default min_samples_leaf (20) isn't appropriate for small
        # datasets (only very shallow trees are built) that the checks use.
        estimator.set_params(min_samples_leaf=5)

    if name == "DummyClassifier":
        # the default strategy prior would output constant predictions and fail
        # for check_classifiers_predictions
        estimator.set_params(strategy="stratified")

    # Speed-up by reducing the number of CV or splits for CV estimators
    loo_cv = ["RidgeCV", "RidgeClassifierCV"]
    if name not in loo_cv and hasattr(estimator, "cv"):
        estimator.set_params(cv=3)
    if hasattr(estimator, "n_splits"):
        estimator.set_params(n_splits=3)

    if name == "OneHotEncoder":
        estimator.set_params(handle_unknown="ignore")

    if name in CROSS_DECOMPOSITION:
        estimator.set_params(n_components=1)


class _NotAnArray:
    """An object that is convertible to an array.

    Parameters
    ----------
    data : array-like
        The data.
    """

    def __init__(self, data):
        self.data = np.asarray(data)

    def __array__(self, dtype=None):
        return self.data

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "may_share_memory":
            return True
        raise TypeError("Don't want to call array_function {}!".format(func.__name__))


def _is_pairwise_metric(estimator):
    """Returns True if estimator accepts pairwise metric.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    metric = getattr(estimator, "metric", None)

    return bool(metric == "precomputed")


def _pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):

    if _is_pairwise_metric(estimator):
        return pairwise_distances(X, metric="euclidean")
    if _is_pairwise(estimator):
        return kernel(X, X)

    return X


def _generate_sparse_matrix(X_csr):
    """Generate sparse matrices with {32,64}bit indices of diverse format.

    Parameters
    ----------
    X_csr: CSR Matrix
        Input matrix in CSR format.

    Returns
    -------
    out: iter(Matrices)
        In format['dok', 'lil', 'dia', 'bsr', 'csr', 'csc', 'coo',
        'coo_64', 'csc_64', 'csr_64']
    """

    assert X_csr.format == "csr"
    yield "csr", X_csr.copy()
    for sparse_format in ["dok", "lil", "dia", "bsr", "csc", "coo"]:
        yield sparse_format, X_csr.asformat(sparse_format)

    # Generate large indices matrix only if its supported by scipy
    X_coo = X_csr.asformat("coo")
    X_coo.row = X_coo.row.astype("int64")
    X_coo.col = X_coo.col.astype("int64")
    yield "coo_64", X_coo

    for sparse_format in ["csc", "csr"]:
        X = X_csr.asformat(sparse_format)
        X.indices = X.indices.astype("int64")
        X.indptr = X.indptr.astype("int64")
        yield sparse_format + "_64", X


def check_estimator_sparse_data(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    X[X < 0.8] = 0
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.rand(40)).astype(int)
    # catch deprecation warnings
    with ignore_warnings(category=FutureWarning):
        estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    tags = _safe_tags(estimator_orig)
    for matrix_format, X in _generate_sparse_matrix(X_csr):
        # catch deprecation warnings
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            if name in ["Scaler", "StandardScaler"]:
                estimator.set_params(with_mean=False)
        # fit and predict
        if "64" in matrix_format:
            err_msg = (
                f"Estimator {name} doesn't seem to support {matrix_format} "
                "matrix, and is not failing gracefully, e.g. by using "
                "check_array(X, accept_large_sparse=False)"
            )
        else:
            err_msg = (
                f"Estimator {name} doesn't seem to fail gracefully on sparse "
                "data: error message should state explicitly that sparse "
                "input is not supported if this is not the case."
            )
        with raises(
            (TypeError, ValueError),
            match=["sparse", "Sparse"],
            may_pass=True,
            err_msg=err_msg,
        ):
            with ignore_warnings(category=FutureWarning):
                estimator.fit(X, y)
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                if tags["multioutput_only"]:
                    assert pred.shape == (X.shape[0], 1)
                else:
                    assert pred.shape == (X.shape[0],)
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X)
                if tags["binary_only"]:
                    expected_probs_shape = (X.shape[0], 2)
                else:
                    expected_probs_shape = (X.shape[0], 4)
                assert probs.shape == expected_probs_shape


@ignore_warnings(category=FutureWarning)
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    estimator = clone(estimator_orig)
    try:
        import pandas as pd

        X = np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]
        )
        X = pd.DataFrame(_pairwise_estimator_convert_X(X, estimator_orig))
        y = pd.Series([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
        weights = pd.Series([1] * 12)
        if _safe_tags(estimator, key="multioutput_only"):
            y = pd.DataFrame(y)
        try:
            estimator.fit(X, y, sample_weight=weights)
        except ValueError:
            raise ValueError(
                "Estimator {0} raises error if "
                "'sample_weight' parameter is of "
                "type pandas.Series".format(name)
            )
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not testing for "
            "input of type pandas.Series to class weight."
        )


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_not_an_array(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type _NotAnArray in the 'fit' function.
    estimator = clone(estimator_orig)
    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    )
    X = _NotAnArray(_pairwise_estimator_convert_X(X, estimator_orig))
    y = _NotAnArray([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
    weights = _NotAnArray([1] * 12)
    if _safe_tags(estimator, key="multioutput_only"):
        y = _NotAnArray(y.data.reshape(-1, 1))
    estimator.fit(X, y, sample_weight=weights)


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _pairwise_estimator_convert_X(rnd.uniform(size=(n_samples, 3)), estimator_orig)
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator, y)
    sample_weight = [3] * n_samples
    # Test that estimators don't raise any exception
    estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_shape(name, estimator_orig):
    # check that estimators raise an error if sample_weight
    # shape mismatches the input
    estimator = clone(estimator_orig)
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
        ]
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y, sample_weight=np.ones(len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones(2 * len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones((len(y), 2)))


@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance(name, estimator_orig, kind="ones"):
    # For kind="ones" check that the estimators yield same results for
    # unit weights and no weights
    # For kind="zeros" check that setting sample_weight to 0 is equivalent
    # to removing corresponding samples.
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    X1 = np.array(
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
        dtype=np.float64,
    )
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    if kind == "ones":
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        err_msg = (
            f"For {name} sample_weight=None is not equivalent to sample_weight=ones"
        )
    elif kind == "zeros":
        # Construct a dataset that is very different to (X, y) if weights
        # are disregarded, but identical to (X, y) given weights.
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1) :] = 0
        X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)

        err_msg = (
            f"For {name}, a zero sample_weight is not equivalent to removing the sample"
        )
    else:  # pragma: no cover
        raise ValueError

    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)

    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)

    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator_orig, method):
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)


def check_sample_weights_not_overwritten(name, estimator_orig):
    # check that estimators don't override the passed sample_weight parameter
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)

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
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    y = _enforce_estimator_tags_y(estimator, y)

    sample_weight_original = np.ones(y.shape[0])
    sample_weight_original[0] = 10.0

    sample_weight_fit = sample_weight_original.copy()

    estimator.fit(X, y, sample_weight=sample_weight_fit)

    err_msg = "{name} overwrote the original `sample_weight` given during fit"
    assert_allclose(sample_weight_fit, sample_weight_original, err_msg=err_msg)


@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = _pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    tags = _safe_tags(estimator_orig)
    y = (X[:, 0] * 4).astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    with raises(Exception, match="Unknown label type", may_pass=True):
        estimator.fit(X, y.astype(object))

    if "string" not in tags["X_types"]:
        X[0, 0] = {"foo": "bar"}
        msg = "argument must be a string.* number"
        with raises(TypeError, match=msg):
            estimator.fit(X, y)
    else:
        # Estimators supporting string will not call np.asarray to convert the
        # data to numeric and therefore, the error will not be raised.
        # Checking for each element dtype in the input array will be costly.
        # Refer to #11401 for full discussion.
        estimator.fit(X, y)


def check_complex_data(name, estimator_orig):
    rng = np.random.RandomState(42)
    # check that estimators raise an exception on providing complex data
    X = rng.uniform(size=10) + 1j * rng.uniform(size=10)
    X = X.reshape(-1, 1)

    # Something both valid for classification and regression
    y = rng.randint(low=0, high=2, size=10) + 1j
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)
    with raises(ValueError, match="Complex data not supported"):
        estimator.fit(X, y)


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    # this estimator raises
    # ValueError: Found array with 0 feature(s) (shape=(23, 0))
    # while a minimum of 1 is required.
    # error
    if name in ["SpectralCoclustering"]:
        return
    rnd = np.random.RandomState(0)
    if name in ["RANSACRegressor"]:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))

    X = _pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert estimator.__dict__ == dict_before, (
                "Estimator changes __dict__ during %s" % method
            )


def _is_public_parameter(attr):
    return not (attr.startswith("_") or attr.endswith("_"))


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but %s added"
        % ", ".join(attrs_added_by_fit)
    )

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        " %s changed"
        % ", ".join(attrs_changed_by_fit)
    )


@ignore_warnings(category=FutureWarning)
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(
                ValueError, "Reshape your data", getattr(estimator, method), X[0]
            )


def _apply_on_subsets(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features)) for batch in X]

    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    if sparse.issparse(result_full):
        result_full = result_full.A
        result_by_batch = [x.A for x in result_by_batch]

    return np.ravel(result_full), np.ravel(result_by_batch)


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini batches or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:

        msg = ("{method} of {name} is not invariant when applied to a subset.").format(
            method=method, name=name
        )

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X
            )
            assert_allclose(result_full, result_by_batch, atol=1e-7, err_msg=msg)


@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on a subset with different sample order
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int64)
    if _safe_tags(estimator_orig, key="binary_only"):
        y[y == 2] = 1
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 2

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    idx = np.random.permutation(X.shape[0])

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        msg = (
            "{method} of {name} is not invariant when applied to a dataset"
            "with different sample order."
        ).format(method=method, name=name)

        if hasattr(estimator, method):
            assert_allclose_dense_sparse(
                getattr(estimator, method)(X)[idx],
                getattr(estimator, method)(X[idx]),
                atol=1e-9,
                err_msg=msg,
            )


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    # min_cluster_size cannot be less than the data size for OPTICS.
    if name == "OPTICS":
        estimator.set_params(min_samples=1)

    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]

    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == "RandomizedLogisticRegression":
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == "RANSACRegressor":
        estimator.residual_threshold = 0.5

    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = [r"1 feature\(s\)", "n_features = 1", "n_features=1"]

    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    with raises(ValueError):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, transformer)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    _check_transformer(name, transformer, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - 0.1
    X = _pairwise_estimator_convert_X(X, transformer)
    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)
    # try the same with some list
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    X, y = _regression_dataset()

    transformer = clone(transformer)
    with raises(
        (AttributeError, ValueError),
        err_msg=(
            "The unfitted "
            f"transformer {name} does not raise an error when "
            "transform is called. Perhaps use "
            "check_is_fitted in transform."
        ),
    ):
        transformer.transform(X)


def _check_transformer(name, transformer_orig, X, y):
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2
        if isinstance(X, _NotAnArray):
            y_ = _NotAnArray(y_)
    else:
        y_ = y

    transformer.fit(X, y_)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y_)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert x_pred.shape[0] == n_samples
    else:
        # check for consistent n_samples
        assert X_pred.shape[0] == n_samples

    if hasattr(transformer, "transform"):
        if name in CROSS_DECOMPOSITION:
            X_pred2 = transformer.transform(X, y_)
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            X_pred2 = transformer.transform(X)
            X_pred3 = transformer.fit_transform(X, y=y_)

        if _safe_tags(transformer_orig, key="non_deterministic"):
            msg = name + " is non deterministic"
            raise SkipTest(msg)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred2,
                    atol=1e-2,
                    err_msg="fit_transform and transform outcomes not consistent in %s"
                    % transformer,
                )
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred3,
                    atol=1e-2,
                    err_msg="consecutive fit_transform outcomes not consistent in %s"
                    % transformer,
                )
        else:
            assert_allclose_dense_sparse(
                X_pred,
                X_pred2,
                err_msg="fit_transform and transform outcomes not consistent in %s"
                % transformer,
                atol=1e-2,
            )
            assert_allclose_dense_sparse(
                X_pred,
                X_pred3,
                atol=1e-2,
                err_msg="consecutive fit_transform outcomes not consistent in %s"
                % transformer,
            )
            assert _num_samples(X_pred2) == n_samples
            assert _num_samples(X_pred3) == n_samples

        # raises error on malformed input for transform
        if (
            hasattr(X, "shape")
            and not _safe_tags(transformer, key="stateless")
            and X.ndim == 2
            and X.shape[1] > 1
        ):

            # If it's not an array, it does not have a 'T' property
            with raises(
                ValueError,
                err_msg=(
                    f"The transformer {name} does not raise an error "
                    "when the number of features in transform is different from "
                    "the number of features in fit."
                ),
            ):
                transformer.transform(X[:, :-1])


@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if _safe_tags(estimator_orig, key="non_deterministic"):
        msg = name + " is non deterministic"
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = rnd.uniform(size=(n_samples, 3))
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(n_samples) % 3
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert args[1] in ["y", "Y"], (
                "Expected y or Y as second argument for method "
                "%s of %s. Got arguments: %r."
                % (func_name, type(estimator).__name__, args)
            )


@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_32 = _pairwise_estimator_convert_X(X_train_32, estimator_orig)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = X_train_int_64[:, 0]
    y = _enforce_estimator_tags_y(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)


def check_transformer_preserve_dtypes(name, transformer_orig):
    # check that dtype are preserved meaning if input X is of some dtype
    # X_transformed should be from the same dtype.
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, transformer_orig)

    for dtype in _safe_tags(transformer_orig, key="preserves_dtype"):
        X_cast = X.astype(dtype)
        transformer = clone(transformer_orig)
        set_random_state(transformer)
        X_trans = transformer.fit_transform(X_cast, y)

        if isinstance(X_trans, tuple):
            # cross-decompostion returns a tuple of (x_scores, y_scores)
            # when given y with fit_transform; only check the first element
            X_trans = X_trans[0]

        # check that the output dtype is preserved
        assert X_trans.dtype == dtype, (
            f"Estimator transform dtype: {X_trans.dtype} - "
            f"original/expected dtype: {dtype.__name__}"
        )


@ignore_warnings(category=FutureWarning)
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)

    X_zero_samples = np.empty(0).reshape(0, 3)
    # The precise message can change depending on whether X or y is
    # validated first. Let us test the type of exception only:
    err_msg = (
        f"The estimator {name} does not raise a ValueError when an "
        "empty data is used to train. Perhaps use check_array in train."
    )
    with raises(ValueError, err_msg=err_msg):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape(12, 0)
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = _enforce_estimator_tags_y(e, np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    msg = r"0 feature\(s\) \(shape=\(\d*, 0\)\) while a minimum of \d* " "is required."
    with raises(ValueError, match=msg):
        e.fit(X_zero_features, y)


@ignore_warnings(category=FutureWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    rnd = np.random.RandomState(0)
    X_train_finite = _pairwise_estimator_convert_X(
        rnd.uniform(size=(10, 3)), estimator_orig
    )
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = _enforce_estimator_tags_y(estimator_orig, y)
    error_string_fit = "Estimator doesn't check for NaN and inf in fit."
    error_string_predict = "Estimator doesn't check for NaN and inf in predict."
    error_string_transform = "Estimator doesn't check for NaN and inf in transform."
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            with raises(ValueError, match=["inf", "NaN"], err_msg=error_string_fit):
                estimator.fit(X_train, y)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_predict,
                ):
                    estimator.predict(X_train)

            # transform
            if hasattr(estimator, "transform"):
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_transform,
                ):
                    estimator.transform(X_train)


@ignore_warnings
def check_nonsquare_error(name, estimator_orig):
    """Test that error is thrown when non-square data provided."""

    X, y = make_blobs(n_samples=20, n_features=10)
    estimator = clone(estimator_orig)

    with raises(
        ValueError,
        err_msg=(
            f"The pairwise estimator {name} does not raise an error on non-square data"
        ),
    ):
        estimator.fit(X, y)


@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators."""
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )

    # some estimators can't do features less than 0
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)

    tags = _safe_tags(estimator_orig)
    # include NaN values when the estimator should deal with them
    if tags["allow_nan"]:
        # set randomly 10 elements to np.nan
        rng = np.random.RandomState(42)
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan

    estimator = clone(estimator_orig)

    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    estimator.fit(X, y)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    module_name = estimator.__module__
    if module_name.startswith("sklearn.") and not (
        "test_" in module_name or module_name.endswith("_testing")
    ):
        # strict check for sklearn estimators that are not implemented in test
        # modules.
        assert b"version" in pickled_estimator
    unpickled_estimator = pickle.loads(pickled_estimator)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


@ignore_warnings(category=FutureWarning)
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # check if number of features changes between calls to partial_fit.
    if not hasattr(estimator_orig, "partial_fit"):
        return
    estimator = clone(estimator_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X -= X.min()
    y = _enforce_estimator_tags_y(estimator_orig, y)

    try:
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return

    with raises(
        ValueError,
        err_msg=(
            f"The estimator {name} does not raise an error when the "
            "number of features changes between calls to partial_fit."
        ),
    ):
        estimator.partial_fit(X[:, :-1], y)


@ignore_warnings(category=FutureWarning)
def check_classifier_multioutput(name, estimator):
    n_samples, n_labels, n_classes = 42, 5, 3
    tags = _safe_tags(estimator)
    estimator = clone(estimator)
    X, y = make_multilabel_classification(
        random_state=42, n_samples=n_samples, n_labels=n_labels, n_classes=n_classes
    )
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    assert y_pred.shape == (n_samples, n_classes), (
        "The shape of the prediction for multioutput data is "
        "incorrect. Expected {}, got {}.".format((n_samples, n_labels), y_pred.shape)
    )
    assert y_pred.dtype.kind == "i"

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        assert isinstance(decision, np.ndarray)
        assert decision.shape == (n_samples, n_classes), (
            "The shape of the decision function output for "
            "multioutput data is incorrect. Expected {}, got {}.".format(
                (n_samples, n_classes), decision.shape
            )
        )

        dec_pred = (decision > 0).astype(int)
        dec_exp = estimator.classes_[dec_pred]
        assert_array_equal(dec_exp, y_pred)

    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X)

        if isinstance(y_prob, list) and not tags["poor_score"]:
            for i in range(n_classes):
                assert y_prob[i].shape == (n_samples, 2), (
                    "The shape of the probability for multioutput data is"
                    " incorrect. Expected {}, got {}.".format(
                        (n_samples, 2), y_prob[i].shape
                    )
                )
                assert_array_equal(
                    np.argmax(y_prob[i], axis=1).astype(int), y_pred[:, i]
                )
        elif not tags["poor_score"]:
            assert y_prob.shape == (n_samples, n_classes), (
                "The shape of the probability for multioutput data is"
                " incorrect. Expected {}, got {}.".format(
                    (n_samples, n_classes), y_prob.shape
                )
            )
            assert_array_equal(y_prob.round().astype(int), y_pred)

    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):
        for i in range(n_classes):
            y_proba = estimator.predict_proba(X)[:, i]
            y_decision = estimator.decision_function(X)
            assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))


@ignore_warnings(category=FutureWarning)
def check_regressor_multioutput(name, estimator):
    estimator = clone(estimator)
    n_samples = n_features = 10

    if not _is_pairwise_metric(estimator):
        n_samples = n_samples + 1

    X, y = make_regression(
        random_state=42, n_targets=5, n_samples=n_samples, n_features=n_features
    )
    X = _pairwise_estimator_convert_X(X, estimator)

    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    assert y_pred.dtype == np.dtype("float64"), (
        "Multioutput predictions by a regressor are expected to be"
        " floating-point precision. Got {} instead".format(y_pred.dtype)
    )
    assert y_pred.shape == y.shape, (
        "The shape of the prediction for multioutput data is incorrect."
        " Expected {}, got {}."
    )


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])

    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    if name == "AffinityPropagation":
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert pred.shape == (n_samples,)
    assert adjusted_rand_score(pred, y) > 0.4
    if _safe_tags(clusterer, key="non_deterministic"):
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype("int32"), np.dtype("int64")]
    assert pred2.dtype in [np.dtype("int32"), np.dtype("int64")]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(
        labels_sorted, np.arange(labels_sorted[0], labels_sorted[-1] + 1)
    )

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, "n_clusters"):
        n_clusters = getattr(clusterer, "n_clusters")
        assert n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=FutureWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels."""
    X, y = make_blobs(n_samples=20, random_state=0)
    clusterer = clone(clusterer_orig)
    set_random_state(clusterer)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label(name, classifier_orig):
    error_string_fit = "Classifier can't train when only one class is present."
    error_string_predict = "Classifier can't predict when only one class is present."
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 3))
    X_test = rnd.uniform(size=(10, 3))
    y = np.ones(10)
    # catch deprecation warnings
    with ignore_warnings(category=FutureWarning):
        classifier = clone(classifier_orig)
        with raises(
            ValueError, match="class", may_pass=True, err_msg=error_string_fit
        ) as cm:
            classifier.fit(X_train, y)

        if cm.raised_and_matched:
            # ValueError was raised with proper error message
            return

        assert_array_equal(classifier.predict(X_test), y, err_msg=error_string_predict)


def _create_memmap_backed_data(numpy_arrays):
    # OpenBLAS is known to segfault with unaligned data on the Prescott architecture
    # See: https://github.com/scipy/scipy/issues/14886
    has_prescott_openblas = any(
        True
        for info in threadpool_info()
        if info["internal_api"] == "openblas"
        # Prudently assume Prescott might be the architecture if it is unknown.
        and info.get("architecture", "prescott").lower() == "prescott"
    )
    return [
        create_memmap_backed_data(array, aligned=has_prescott_openblas)
        for array in numpy_arrays
    ]


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(
    name, classifier_orig, readonly_memmap=False, X_dtype="float64"
):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m = X_m.astype(X_dtype)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if name in ["BernoulliNB", "MultinomialNB", "ComplementNB", "CategoricalNB"]:
        X_m -= X_m.min()
        X_b -= X_b.min()

    if readonly_memmap:
        X_m, y_m, X_b, y_b = _create_memmap_backed_data([X_m, y_m, X_b, y_b])

    problems = [(X_b, y_b)]
    tags = _safe_tags(classifier_orig)
    if not tags["binary_only"]:
        problems.append((X_m, y_m))

    for (X, y) in problems:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        X = _pairwise_estimator_convert_X(X, classifier)
        y = _enforce_estimator_tags_y(classifier, y)

        set_random_state(classifier)
        # raises error on malformed input for fit
        if not tags["no_validation"]:
            with raises(
                ValueError,
                err_msg=(
                    f"The classifier {name} does not raise an error when "
                    "incorrect/malformed input data for fit is passed. The number "
                    "of training examples is not the same as the number of "
                    "labels. Perhaps use check_X_y in fit."
                ),
            ):
                classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_samples,)
        # training set performance
        if not tags["poor_score"]:
            assert accuracy_score(y, y_pred) > 0.83

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)"
        )
        msg = (
            "The classifier {} does not raise an error when the number of "
            "features in {} is different from the number of features in "
            "fit."
        )

        if not tags["no_validation"]:
            if _is_pairwise(classifier):
                with raises(
                    ValueError,
                    err_msg=msg_pairwise.format(name, "predict"),
                ):
                    classifier.predict(X.reshape(-1, 1))
            else:
                with raises(ValueError, err_msg=msg.format(name, "predict")):
                    classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if not tags["multioutput_only"]:
                        assert decision.shape == (n_samples,)
                    else:
                        assert decision.shape == (n_samples, 1)
                    dec_pred = (decision.ravel() > 0).astype(int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert decision.shape == (n_samples, n_classes)
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not tags["no_validation"]:
                    if _is_pairwise(classifier):
                        with raises(
                            ValueError,
                            err_msg=msg_pairwise.format(name, "decision_function"),
                        ):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        with raises(
                            ValueError,
                            err_msg=msg.format(name, "decision_function"),
                        ):
                            classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1), np.ones(n_samples))
            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                if _is_pairwise(classifier_orig):
                    with raises(
                        ValueError,
                        err_msg=msg_pairwise.format(name, "predict_proba"),
                    ):
                        classifier.predict_proba(X.reshape(-1, 1))
                else:
                    with raises(
                        ValueError,
                        err_msg=msg.format(name, "predict_proba"),
                    ):
                        classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))


def check_outlier_corruption(num_outliers, expected_outliers, decision):
    # Check for deviation from the precise given contamination level that may
    # be due to ties in the anomaly scores.
    if num_outliers < expected_outliers:
        start = num_outliers
        end = expected_outliers + 1
    else:
        start = expected_outliers
        end = num_outliers + 1

    # ensure that all values in the 'critical area' are tied,
    # leading to the observed discrepancy between provided
    # and actual contamination levels.
    sorted_decision = np.sort(decision)
    msg = (
        "The number of predicted outliers is not equal to the expected "
        "number of outliers and this difference is not explained by the "
        "number of ties in the decision_function values"
    )
    assert len(np.unique(sorted_decision[start:end])) == 1, msg


def check_outliers_train(name, estimator_orig, readonly_memmap=True):
    n_samples = 300
    X, _ = make_blobs(n_samples=n_samples, random_state=0)
    X = shuffle(X, random_state=7)

    if readonly_memmap:
        X = create_memmap_backed_data(X)

    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    # fit
    estimator.fit(X)
    # with lists
    estimator.fit(X.tolist())

    y_pred = estimator.predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == "i"
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    decision = estimator.decision_function(X)
    scores = estimator.score_samples(X)
    for output in [decision, scores]:
        assert output.dtype == np.dtype("float")
        assert output.shape == (n_samples,)

    # raises error on malformed input for predict
    with raises(ValueError):
        estimator.predict(X.T)

    # decision_function agrees with predict
    dec_pred = (decision >= 0).astype(int)
    dec_pred[dec_pred == 0] = -1
    assert_array_equal(dec_pred, y_pred)

    # raises error on malformed input for decision_function
    with raises(ValueError):
        estimator.decision_function(X.T)

    # decision_function is a translation of score_samples
    y_dec = scores - estimator.offset_
    assert_allclose(y_dec, decision)

    # raises error on malformed input for score_samples
    with raises(ValueError):
        estimator.score_samples(X.T)

    # contamination parameter (not for OneClassSVM which has the nu parameter)
    if hasattr(estimator, "contamination") and not hasattr(estimator, "novelty"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'. This is true for the training set and cannot thus be
        # checked as follows for estimators with a novelty parameter such as
        # LocalOutlierFactor (tested in check_outliers_fit_predict)
        expected_outliers = 30
        contamination = expected_outliers / n_samples
        estimator.set_params(contamination=contamination)
        estimator.fit(X)
        y_pred = estimator.predict(X)

        num_outliers = np.sum(y_pred != 1)
        # num_outliers should be equal to expected_outliers unless
        # there are ties in the decision_function values. this can
        # only be tested for estimators with a decision_function
        # method, i.e. all estimators except LOF which is already
        # excluded from this if branch.
        if num_outliers != expected_outliers:
            decision = estimator.decision_function(X)
            check_outlier_corruption(num_outliers, expected_outliers, decision)

        # raises error when contamination is a scalar and not in [0,1]
        msg = r"contamination must be in \(0, 0.5]"
        for contamination in [-0.5, 2.3]:
            estimator.set_params(contamination=contamination)
            with raises(ValueError, match=msg):
                estimator.fit(X)


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_representation_invariance(name, classifier_orig):
    X, y = make_multilabel_classification(
        n_samples=100,
        n_features=2,
        n_classes=5,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    X = scale(X)

    X_train, y_train = X[:80], y[:80]
    X_test = X[80:]

    y_train_list_of_lists = y_train.tolist()
    y_train_list_of_arrays = list(y_train)

    classifier = clone(classifier_orig)
    set_random_state(classifier)

    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    y_pred_list_of_lists = classifier.fit(X_train, y_train_list_of_lists).predict(
        X_test
    )

    y_pred_list_of_arrays = classifier.fit(X_train, y_train_list_of_arrays).predict(
        X_test
    )

    assert_array_equal(y_pred, y_pred_list_of_arrays)
    assert_array_equal(y_pred, y_pred_list_of_lists)

    assert y_pred.dtype == y_pred_list_of_arrays.dtype
    assert y_pred.dtype == y_pred_list_of_lists.dtype
    assert type(y_pred) == type(y_pred_list_of_arrays)
    assert type(y_pred) == type(y_pred_list_of_lists)


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict(name, classifier_orig):
    """Check the output of the `predict` method for classifiers supporting
    multilabel-indicator targets."""
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 100, 25, 5
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    classifier.fit(X_train, y_train)

    response_method_name = "predict"
    predict_method = getattr(classifier, response_method_name, None)
    if predict_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = predict_method(X_test)

    # y_pred.shape -> y_test.shape with the same dtype
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.predict is expected to output a NumPy array. Got "
        f"{type(y_pred)} instead."
    )
    assert y_pred.shape == y_test.shape, (
        f"{name}.predict outputs a NumPy array of shape {y_pred.shape} "
        f"instead of {y_test.shape}."
    )
    assert y_pred.dtype == y_test.dtype, (
        f"{name}.predict does not output the same dtype than the targets. "
        f"Got {y_pred.dtype} instead of {y_test.dtype}."
    )


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict_proba(name, classifier_orig):
    """Check the output of the `predict_proba` method for classifiers supporting
    multilabel-indicator targets."""
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 100, 25, 5
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    classifier.fit(X_train, y_train)

    response_method_name = "predict_proba"
    predict_proba_method = getattr(classifier, response_method_name, None)
    if predict_proba_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = predict_proba_method(X_test)

    # y_pred.shape -> 2 possibilities:
    # - list of length n_outputs of shape (n_samples, 2);
    # - ndarray of shape (n_samples, n_outputs).
    # dtype should be floating
    if isinstance(y_pred, list):
        assert len(y_pred) == n_outputs, (
            f"When {name}.predict_proba returns a list, the list should "
            "be of length n_outputs and contain NumPy arrays. Got length "
            f"of {len(y_pred)} instead of {n_outputs}."
        )
        for pred in y_pred:
            assert pred.shape == (test_size, 2), (
                f"When {name}.predict_proba returns a list, this list "
                "should contain NumPy arrays of shape (n_samples, 2). Got "
                f"NumPy arrays of shape {pred.shape} instead of "
                f"{(test_size, 2)}."
            )
            assert pred.dtype.kind == "f", (
                f"When {name}.predict_proba returns a list, it should "
                "contain NumPy arrays with floating dtype. Got "
                f"{pred.dtype} instead."
            )
            # check that we have the correct probabilities
            err_msg = (
                f"When {name}.predict_proba returns a list, each NumPy "
                "array should contain probabilities for each class and "
                "thus each row should sum to 1 (or close to 1 due to "
                "numerical errors)."
            )
            assert_allclose(pred.sum(axis=1), 1, err_msg=err_msg)
    elif isinstance(y_pred, np.ndarray):
        assert y_pred.shape == (test_size, n_outputs), (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected shape is (n_samples, n_outputs). Got {y_pred.shape}"
            f" instead of {(test_size, n_outputs)}."
        )
        assert y_pred.dtype.kind == "f", (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected data type is floating. Got {y_pred.dtype} instead."
        )
        err_msg = (
            f"When {name}.predict_proba returns a NumPy array, this array "
            "is expected to provide probabilities of the positive class "
            "and should therefore contain values between 0 and 1."
        )
        assert_array_less(0, y_pred, err_msg=err_msg)
        assert_array_less(y_pred, 1, err_msg=err_msg)
    else:
        raise ValueError(
            f"Unknown returned type {type(y_pred)} by {name}."
            "predict_proba. A list or a Numpy array is expected."
        )


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_decision_function(name, classifier_orig):
    """Check the output of the `decision_function` method for classifiers supporting
    multilabel-indicator targets."""
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 100, 25, 5
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=50,
        allow_unlabeled=True,
        random_state=0,
    )
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    classifier.fit(X_train, y_train)

    response_method_name = "decision_function"
    decision_function_method = getattr(classifier, response_method_name, None)
    if decision_function_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = decision_function_method(X_test)

    # y_pred.shape -> y_test.shape with floating dtype
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.decision_function is expected to output a NumPy array."
        f" Got {type(y_pred)} instead."
    )
    assert y_pred.shape == (test_size, n_outputs), (
        f"{name}.decision_function is expected to provide a NumPy array "
        f"of shape (n_samples, n_outputs). Got {y_pred.shape} instead of "
        f"{(test_size, n_outputs)}."
    )
    assert y_pred.dtype.kind == "f", (
        f"{name}.decision_function is expected to output a floating dtype."
        f" Got {y_pred.dtype} instead."
    )


@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig, readonly_memmap=False):
    """Check if self is returned when calling fit."""
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise a NotFittedError.
    """
    # Common test for Regressors, Classifiers and Outlier detection estimators
    X, y = _regression_dataset()

    estimator = clone(estimator_orig)
    for method in (
        "decision_function",
        "predict",
        "predict_proba",
        "predict_log_proba",
    ):
        if hasattr(estimator, method):
            with raises(NotFittedError):
                getattr(estimator, method)(X)


@ignore_warnings(category=FutureWarning)
def check_supervised_y_2d(name, estimator_orig):
    tags = _safe_tags(estimator_orig)
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _pairwise_estimator_convert_X(rnd.uniform(size=(n_samples, 3)), estimator_orig)
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator_orig, y)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % ", ".join(
        [str(w_x) for w_x in w]
    )
    if not tags["multioutput"]:
        # check that we warned if we don't support multi-output
        assert len(w) > 0, msg
        assert (
            "DataConversionWarning('A column-vector y"
            " was passed when a 1d array was expected"
            in msg
        )
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())


@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)
    if name == "BernoulliNB":
        X = X > X.mean()
    set_random_state(classifier)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        assert isinstance(decision, np.ndarray)
        if len(classes) == 2:
            dec_pred = (decision.ravel() > 0).astype(int)
            dec_exp = classifier.classes_[dec_pred]
            assert_array_equal(
                dec_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (
                    classifier,
                    ", ".join(map(str, dec_exp)),
                    ", ".join(map(str, y_pred)),
                ),
            )
        elif getattr(classifier, "decision_function_shape", "ovr") == "ovr":
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            assert_array_equal(
                y_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (classifier, ", ".join(map(str, y_exp)), ", ".join(map(str, y_pred))),
            )

    # training set performance
    if name != "ComplementNB":
        # This is a pathological data set for ComplementNB.
        # For some specific cases 'ComplementNB' predicts less classes
        # than expected
        assert_array_equal(np.unique(y), np.unique(y_pred))
    assert_array_equal(
        classes,
        classifier.classes_,
        err_msg="Unexpected classes_ attribute for %r: expected '%s', got '%s'"
        % (
            classifier,
            ", ".join(map(str, classes)),
            ", ".join(map(str, classifier.classes_)),
        ),
    )


def _choose_check_classifiers_labels(name, y, y_names):
    # Semisupervised classifiers use -1 as the indicator for an unlabeled
    # sample.
    return (
        y
        if name in ["LabelPropagation", "LabelSpreading", "SelfTrainingClassifier"]
        else y_names
    )


def check_classifiers_classes(name, classifier_orig):
    X_multiclass, y_multiclass = make_blobs(
        n_samples=30, random_state=0, cluster_std=0.1
    )
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass, random_state=7)
    X_multiclass = StandardScaler().fit_transform(X_multiclass)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X_multiclass -= X_multiclass.min() - 0.1

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = _pairwise_estimator_convert_X(X_multiclass, classifier_orig)
    X_binary = _pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    problems = [(X_binary, y_binary, y_names_binary)]
    if not _safe_tags(classifier_orig, key="binary_only"):
        problems.append((X_multiclass, y_multiclass, y_names_multiclass))

    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype("O")]:
            y_ = _choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = _choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


@ignore_warnings(category=FutureWarning)
def check_regressors_int(name, regressor_orig):
    X, _ = _regression_dataset()
    X = _pairwise_estimator_convert_X(X[:50], regressor_orig)
    rnd = np.random.RandomState(0)
    y = rnd.randint(3, size=X.shape[0])
    y = _enforce_estimator_tags_y(regressor_orig, y)
    rnd = np.random.RandomState(0)
    # separate estimators to control random seeds
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    # fit
    regressor_1.fit(X, y_)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y_.astype(float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    X, y = _regression_dataset()
    X = X.astype(X_dtype)
    X = _pairwise_estimator_convert_X(X, regressor_orig)
    y = scale(y)  # X is already scaled
    regressor = clone(regressor_orig)
    y = _enforce_estimator_tags_y(regressor, y)
    if name in CROSS_DECOMPOSITION:
        rnd = np.random.RandomState(0)
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    if readonly_memmap:
        X, y, y_ = _create_memmap_backed_data([X, y, y_])

    if not hasattr(regressor, "alphas") and hasattr(regressor, "alpha"):
        # linear regressors need to set alpha, but not generalized CV ones
        regressor.alpha = 0.01
    if name == "PassiveAggressiveRegressor":
        regressor.C = 0.01

    # raises error on malformed input for fit
    with raises(
        ValueError,
        err_msg=(
            f"The classifier {name} does not raise an error when "
            "incorrect/malformed input data for fit is passed. The number of "
            "training examples is not the same as the number of labels. Perhaps "
            "use check_X_y in fit."
        ),
    ):
        regressor.fit(X, y[:-1])
    # fit
    set_random_state(regressor)
    regressor.fit(X, y_)
    regressor.fit(X.tolist(), y_.tolist())
    y_pred = regressor.predict(X)
    assert y_pred.shape == y_.shape

    # TODO: find out why PLS and CCA fail. RANSAC is random
    # and furthermore assumes the presence of outliers, hence
    # skipped
    if not _safe_tags(regressor, key="poor_score"):
        assert regressor.score(X, y_) > 0.5


@ignore_warnings
def check_regressors_no_decision_function(name, regressor_orig):
    # check that regressors don't have a decision_function, predict_proba, or
    # predict_log_proba method.
    rng = np.random.RandomState(0)
    regressor = clone(regressor_orig)

    X = rng.normal(size=(10, 4))
    X = _pairwise_estimator_convert_X(X, regressor_orig)
    y = _enforce_estimator_tags_y(regressor, X[:, 0])

    regressor.fit(X, y)
    funcs = ["decision_function", "predict_proba", "predict_log_proba"]
    for func_name in funcs:
        assert not hasattr(regressor, func_name)


@ignore_warnings(category=FutureWarning)
def check_class_weight_classifiers(name, classifier_orig):

    if _safe_tags(classifier_orig, key="binary_only"):
        problems = [2]
    else:
        problems = [2, 3]

    for n_centers in problems:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _is_pairwise(classifier_orig):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(class_weight=class_weight)
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)
        if hasattr(classifier, "n_iter_no_change"):
            classifier.set_params(n_iter_no_change=20)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
        #      0.88 (Issue #9111)
        if not _safe_tags(classifier_orig, key="poor_score"):
            assert np.mean(y_pred == 0) > 0.87


@ignore_warnings(category=FutureWarning)
def check_class_weight_balanced_classifiers(
    name, classifier_orig, X_train, y_train, X_test, y_test, weights
):
    classifier = clone(classifier_orig)
    if hasattr(classifier, "n_iter"):
        classifier.set_params(n_iter=100)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)

    set_random_state(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    classifier.set_params(class_weight="balanced")
    classifier.fit(X_train, y_train)
    y_pred_balanced = classifier.predict(X_test)
    assert f1_score(y_test, y_pred_balanced, average="weighted") > f1_score(
        y_test, y_pred, average="weighted"
    )


@ignore_warnings(category=FutureWarning)
def check_class_weight_balanced_linear_classifier(name, Classifier):
    """Test class weights with non-contiguous class labels."""
    # this is run on classes, not instances, though this should be changed
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    classifier = Classifier()

    if hasattr(classifier, "n_iter"):
        # This is a very small dataset, default n_iter are likely to prevent
        # convergence
        classifier.set_params(n_iter=1000)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)
    if hasattr(classifier, "cv"):
        classifier.set_params(cv=3)
    set_random_state(classifier)

    # Let the model compute the class frequencies
    classifier.set_params(class_weight="balanced")
    coef_balanced = classifier.fit(X, y).coef_.copy()

    # Count each label occurrence to reweight manually
    n_samples = len(y)
    n_classes = float(len(np.unique(y)))

    class_weight = {
        1: n_samples / (np.sum(y == 1) * n_classes),
        -1: n_samples / (np.sum(y == -1) * n_classes),
    }
    classifier.set_params(class_weight=class_weight)
    coef_manual = classifier.fit(X, y).coef_.copy()

    assert_allclose(
        coef_balanced,
        coef_manual,
        err_msg="Classifier %s is not computing class_weight=balanced properly." % name,
    )


@ignore_warnings(category=FutureWarning)
def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (name, param_name, original_value, new_value)
        )


@ignore_warnings(category=FutureWarning)
def check_no_attributes_set_in_init(name, estimator_orig):
    """Check setting during init."""
    try:
        # Clone fails if the estimator does not store
        # all parameters as an attribute during init
        estimator = clone(estimator_orig)
    except AttributeError:
        raise AttributeError(
            f"Estimator {name} should store all parameters as an attribute during init."
        )

    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    init_params = _get_args(type(estimator).__init__)
    if IS_PYPY:
        # __init__ signature has additional objects in PyPy
        for key in ["obj"]:
            if key in init_params:
                init_params.remove(key)
    parents_init_params = [
        param
        for params_parent in (_get_args(parent) for parent in type(estimator).__mro__)
        for param in params_parent
    ]

    # Test for no setting apart from parameters during init
    invalid_attr = set(vars(estimator)) - set(init_params) - set(parents_init_params)
    assert not invalid_attr, (
        "Estimator %s should not set any attribute apart"
        " from parameters during init. Found attributes %s."
        % (name, sorted(invalid_attr))
    )


@ignore_warnings(category=FutureWarning)
def check_sparsify_coefficients(name, estimator_orig):
    X = np.array(
        [
            [-2, -1],
            [-1, -1],
            [-1, -2],
            [1, 1],
            [1, 2],
            [2, 1],
            [-1, -2],
            [2, 2],
            [-2, -2],
        ]
    )
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = _enforce_estimator_tags_y(estimator_orig, y)
    est = clone(estimator_orig)

    est.fit(X, y)
    pred_orig = est.predict(X)

    # test sparsify with dense inputs
    est.sparsify()
    assert sparse.issparse(est.coef_)
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)

    # pickle and unpickle with sparse coef_
    est = pickle.loads(pickle.dumps(est))
    assert sparse.issparse(est.coef_)
    pred = est.predict(X)
    assert_array_equal(pred, pred_orig)


@ignore_warnings(category=FutureWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array(
        [
            [3, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [1, 2],
            [2, 1],
            [0, 3],
            [1, 0],
            [2, 0],
            [4, 4],
            [2, 3],
            [3, 2],
        ]
    )
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _regression_dataset()
    X = _pairwise_estimator_convert_X(X, estimator_orig)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type):
    if name in CROSS_DECOMPOSITION:
        raise SkipTest(
            "Skipping check_estimators_data_not_an_array "
            "for cross decomposition module as estimators "
            "are not deterministic."
        )
    # separate estimators to control random seeds
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    if obj_type not in ["NotAnArray", "PandasDataframe"]:
        raise ValueError("Data type {0} not supported".format(obj_type))

    if obj_type == "NotAnArray":
        y_ = _NotAnArray(np.asarray(y))
        X_ = _NotAnArray(np.asarray(X))
    else:
        # Here pandas objects (Series and DataFrame) are tested explicitly
        # because some estimators may handle them (especially their indexing)
        # specially.
        try:
            import pandas as pd

            y_ = np.asarray(y)
            if y_.ndim == 1:
                y_ = pd.Series(y_)
            else:
                y_ = pd.DataFrame(y_)
            X_ = pd.DataFrame(np.asarray(X))

        except ImportError:
            raise SkipTest(
                "pandas is not installed: not checking estimators for pandas objects."
            )

    # fit
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


def check_parameters_default_constructible(name, Estimator):
    # test default-constructibility
    # get rid of deprecation warnings

    Estimator = Estimator.__class__

    with ignore_warnings(category=FutureWarning):
        estimator = _construct_instance(Estimator)
        # test cloning
        clone(estimator)
        # test __repr__
        repr(estimator)
        # test that set_params returns self
        assert estimator.set_params() is estimator

        # test if init does nothing but set parameters
        # this is important for grid_search etc.
        # We get the default parameters from init and then
        # compare these against the actual values of the attributes.

        # this comes from getattr. Gets rid of deprecation decorator.
        init = getattr(estimator.__init__, "deprecated_original", estimator.__init__)

        try:

            def param_filter(p):
                """Identify hyper parameters of an estimator."""
                return (
                    p.name != "self"
                    and p.kind != p.VAR_KEYWORD
                    and p.kind != p.VAR_POSITIONAL
                )

            init_params = [
                p for p in signature(init).parameters.values() if param_filter(p)
            ]

        except (TypeError, ValueError):
            # init is not a python function.
            # true for mixins
            return
        params = estimator.get_params()
        # they can need a non-default argument
        init_params = init_params[len(getattr(estimator, "_required_parameters", [])) :]

        for init_param in init_params:
            assert (
                init_param.default != init_param.empty
            ), "parameter %s for %s has no default value" % (
                init_param.name,
                type(estimator).__name__,
            )
            allowed_types = {
                str,
                int,
                float,
                bool,
                tuple,
                type(None),
                type,
                types.FunctionType,
                joblib.Memory,
            }
            # Any numpy numeric such as np.int32.
            allowed_types.update(np.core.numerictypes.allTypes.values())
            assert type(init_param.default) in allowed_types, (
                f"Parameter '{init_param.name}' of estimator "
                f"'{Estimator.__name__}' is of type "
                f"{type(init_param.default).__name__} which is not "
                "allowed. All init parameters have to be immutable to "
                "make cloning possible. Therefore we restrict the set of "
                "legal types to "
                f"{set(type.__name__ for type in allowed_types)}."
            )
            if init_param.name not in params.keys():
                # deprecated parameter, not in get_params
                assert init_param.default is None, (
                    f"Estimator parameter '{init_param.name}' of estimator "
                    f"'{Estimator.__name__}' is not returned by get_params. "
                    "If it is deprecated, set its default value to None."
                )
                continue

            param_value = params[init_param.name]
            if isinstance(param_value, np.ndarray):
                assert_array_equal(param_value, init_param.default)
            else:
                failure_text = (
                    f"Parameter {init_param.name} was mutated on init. All "
                    "parameters must be stored unchanged."
                )
                if is_scalar_nan(param_value):
                    # Allows to set default parameters to np.nan
                    assert param_value is init_param.default, failure_text
                else:
                    assert param_value == init_param.default, failure_text


def _enforce_estimator_tags_y(estimator, y):
    # Estimators with a `requires_positive_y` tag only accept strictly positive
    # data
    if _safe_tags(estimator, key="requires_positive_y"):
        # Create strictly positive y. The minimal increment above 0 is 1, as
        # y could be of integer dtype.
        y += 1 + abs(y.min())
    # Estimators with a `binary_only` tag only accept up to two unique y values
    if _safe_tags(estimator, key="binary_only") and y.size > 0:
        y = np.where(y == y.flat[0], y, y.flat[0] + 1)
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if _safe_tags(estimator, key="multioutput_only"):
        return np.reshape(y, (-1, 1))
    return y


def _enforce_estimator_tags_x(estimator, X):
    # Pairwise estimators only accept
    # X of shape (`n_samples`, `n_samples`)
    if _is_pairwise(estimator):
        X = X.dot(X.T)
    # Estimators with `1darray` in `X_types` tag only accept
    # X of shape (`n_samples`,)
    if "1darray" in _safe_tags(estimator, key="X_types"):
        X = X[:, 0]
    # Estimators with a `requires_positive_X` tag only accept
    # strictly positive data
    if _safe_tags(estimator, key="requires_positive_X"):
        X -= X.min()
    if "categorical" in _safe_tags(estimator, key="X_types"):
        X = (X - X.min()).astype(np.int32)
    return X


@ignore_warnings(category=FutureWarning)
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    # SelfTrainingClassifier does not perform an iteration if all samples are
    # labeled, hence n_iter_ = 0 is valid.
    not_run_check_n_iter = [
        "Ridge",
        "SVR",
        "NuSVR",
        "NuSVC",
        "RidgeClassifier",
        "SVC",
        "RandomizedLasso",
        "LogisticRegressionCV",
        "LinearSVC",
        "LogisticRegression",
        "SelfTrainingClassifier",
    ]

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == "LassoLars":
        estimator = clone(estimator_orig).set_params(alpha=0.0)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = _enforce_estimator_tags_y(estimator, y_)

        set_random_state(estimator, 0)

        estimator.fit(X, y_)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 5.0, 4.0]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(
                n_samples=30,
                centers=[[0, 0, 0], [1, 1, 1]],
                random_state=0,
                n_features=2,
                cluster_std=0.1,
            )
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert iter_ >= 1
        else:
            assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert all(item in deep_params.items() for item in shallow_params.items())


@ignore_warnings(category=FutureWarning)
def check_set_params(name, estimator_orig):
    # Check that get_params() returns the same thing
    # before and after set_params() with some fuzz
    estimator = clone(estimator_orig)

    orig_params = estimator.get_params(deep=False)
    msg = "get_params result does not match what was passed to set_params"

    estimator.set_params(**orig_params)
    curr_params = estimator.get_params(deep=False)
    assert set(orig_params.keys()) == set(curr_params.keys()), msg
    for k, v in curr_params.items():
        assert orig_params[k] is v, msg

    # some fuzz values
    test_values = [-np.inf, np.inf, None]

    test_params = deepcopy(orig_params)
    for param_name in orig_params.keys():
        default_value = orig_params[param_name]
        for value in test_values:
            test_params[param_name] = value
            try:
                estimator.set_params(**test_params)
            except (TypeError, ValueError) as e:
                e_type = e.__class__.__name__
                # Exception occurred, possibly parameter validation
                warnings.warn(
                    "{0} occurred during set_params of param {1} on "
                    "{2}. It is recommended to delay parameter "
                    "validation until fit.".format(e_type, param_name, name)
                )

                change_warning_msg = (
                    "Estimator's parameters changed after set_params raised {}".format(
                        e_type
                    )
                )
                params_before_exception = curr_params
                curr_params = estimator.get_params(deep=False)
                try:
                    assert set(params_before_exception.keys()) == set(
                        curr_params.keys()
                    )
                    for k, v in curr_params.items():
                        assert params_before_exception[k] is v
                except AssertionError:
                    warnings.warn(change_warning_msg)
            else:
                curr_params = estimator.get_params(deep=False)
                assert set(test_params.keys()) == set(curr_params.keys()), msg
                for k, v in curr_params.items():
                    assert test_params[k] is v, msg
        test_params[param_name] = default_value


@ignore_warnings(category=FutureWarning)
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    X, y = _regression_dataset()

    X = X + 1 + abs(X.min(axis=0))  # be sure that X is non-negative
    e = clone(estimator_orig)
    msg = "Unknown label type: "
    if not _safe_tags(e, key="no_validation"):
        with raises(ValueError, match=msg):
            e.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(
        n_samples=100,
        random_state=0,
        n_features=4,
        centers=centers,
        cluster_std=1.0,
        shuffle=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    estimator = clone(estimator_orig)

    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):

        estimator.fit(X_train, y_train)
        # Since the link function from decision_function() to predict_proba()
        # is sometimes not precise enough (typically expit), we round to the
        # 10th decimal to avoid numerical issues: we compare the rank
        # with deterministic ties rather than get platform specific rank
        # inversions in case of machine level differences.
        a = estimator.predict_proba(X_test)[:, 1].round(decimals=10)
        b = estimator.decision_function(X_test).round(decimals=10)
        assert_array_equal(rankdata(a), rankdata(b))


def check_outliers_fit_predict(name, estimator_orig):
    # Check fit_predict for outlier detectors.

    n_samples = 300
    X, _ = make_blobs(n_samples=n_samples, random_state=0)
    X = shuffle(X, random_state=7)
    n_samples, n_features = X.shape
    estimator = clone(estimator_orig)

    set_random_state(estimator)

    y_pred = estimator.fit_predict(X)
    assert y_pred.shape == (n_samples,)
    assert y_pred.dtype.kind == "i"
    assert_array_equal(np.unique(y_pred), np.array([-1, 1]))

    # check fit_predict = fit.predict when the estimator has both a predict and
    # a fit_predict method. recall that it is already assumed here that the
    # estimator has a fit_predict method
    if hasattr(estimator, "predict"):
        y_pred_2 = estimator.fit(X).predict(X)
        assert_array_equal(y_pred, y_pred_2)

    if hasattr(estimator, "contamination"):
        # proportion of outliers equal to contamination parameter when not
        # set to 'auto'
        expected_outliers = 30
        contamination = float(expected_outliers) / n_samples
        estimator.set_params(contamination=contamination)
        y_pred = estimator.fit_predict(X)

        num_outliers = np.sum(y_pred != 1)
        # num_outliers should be equal to expected_outliers unless
        # there are ties in the decision_function values. this can
        # only be tested for estimators with a decision_function
        # method
        if num_outliers != expected_outliers and hasattr(
            estimator, "decision_function"
        ):
            decision = estimator.decision_function(X)
            check_outlier_corruption(num_outliers, expected_outliers, decision)

        # raises error when contamination is a scalar and not in [0,1]
        msg = r"contamination must be in \(0, 0.5]"
        for contamination in [-0.5, -0.001, 0.5001, 2.3]:
            estimator.set_params(contamination=contamination)
            with raises(ValueError, match=msg):
                estimator.fit_predict(X)


def check_fit_non_negative(name, estimator_orig):
    # Check that proper warning is raised for non-negative X
    # when tag requires_positive_X is present
    X = np.array([[-1.0, 1], [-1.0, 1]])
    y = np.array([1, 2])
    estimator = clone(estimator_orig)
    with raises(ValueError):
        estimator.fit(X, y)


def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    check_methods = ["predict", "transform", "decision_function", "predict_proba"]
    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2 * np.finfo(new_result.dtype).eps
            else:
                tol = 2 * np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method],
                new_result,
                atol=max(tol, 1e-9),
                rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method),
            )


def check_fit_check_is_fitted(name, estimator_orig):
    # Make sure that estimator doesn't pass check_is_fitted before calling fit
    # and that passes check_is_fitted once it's fit.

    rng = np.random.RandomState(42)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    if not _safe_tags(estimator).get("stateless", False):
        # stateless estimators (such as FunctionTransformer) are always "fit"!
        try:
            check_is_fitted(estimator)
            raise AssertionError(
                f"{estimator.__class__.__name__} passes check_is_fitted before being"
                " fit!"
            )
        except NotFittedError:
            pass
    estimator.fit(X, y)
    try:
        check_is_fitted(estimator)
    except NotFittedError as e:
        raise NotFittedError(
            "Estimator fails to pass `check_is_fitted` even though it has been fit."
        ) from e


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ attribute doesn't exist until fit is
    # called, and that its value is correct.

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    if hasattr(estimator, "n_features_in_"):
        assert estimator.n_features_in_ == X.shape[1]
    else:
        warnings.warn(
            "As of scikit-learn 0.23, estimators should expose a "
            "n_features_in_ attribute, unless the 'no_validation' tag is "
            "True. This attribute should be equal to the number of features "
            "passed to the fit method. "
            "An error will be raised from version 1.0 (renaming of 0.25) "
            "when calling check_estimator(). "
            "See SLEP010: "
            "https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html",  # noqa
            FutureWarning,
        )


def check_requires_y_none(name, estimator_orig):
    # Make sure that an estimator with requires_y=True fails gracefully when
    # given y=None

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _pairwise_estimator_convert_X(X, estimator)

    warning_msg = (
        "As of scikit-learn 0.23, estimators should have a "
        "'requires_y' tag set to the appropriate value. "
        "The default value of the tag is False. "
        "An error will be raised from version 1.0 when calling "
        "check_estimator() if the tag isn't properly set."
    )

    expected_err_msgs = (
        "requires y to be passed, but the target y is None",
        "Expected array-like (array or non-string sequence), got None",
        "y should be a 1d array",
    )

    try:
        estimator.fit(X, None)
    except ValueError as ve:
        if not any(msg in str(ve) for msg in expected_err_msgs):
            warnings.warn(warning_msg, FutureWarning)


@ignore_warnings(category=FutureWarning)
def check_n_features_in_after_fitting(name, estimator_orig):
    # Make sure that n_features_in are checked after fitting
    tags = _safe_tags(estimator_orig)

    is_supported_X_types = (
        "2darray" in tags["X_types"] or "categorical" in tags["X_types"]
    )

    if not is_supported_X_types or tags["no_validation"]:
        return

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_samples = 150
    X = rng.normal(size=(n_samples, 8))
    X = _enforce_estimator_tags_x(estimator, X)
    X = _pairwise_estimator_convert_X(X, estimator)

    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    assert estimator.n_features_in_ == X.shape[1]

    # check methods will check n_features_in_
    check_methods = [
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
    ]
    X_bad = X[:, [1]]

    msg = f"X has 1 features, but \\w+ is expecting {X.shape[1]} features as input"
    for method in check_methods:
        if not hasattr(estimator, method):
            continue

        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)

        with raises(ValueError, match=msg):
            callable_method(X_bad)

    # partial_fit will check in the second call
    if not hasattr(estimator, "partial_fit"):
        return

    estimator = clone(estimator_orig)
    if is_classifier(estimator):
        estimator.partial_fit(X, y, classes=np.unique(y))
    else:
        estimator.partial_fit(X, y)
    assert estimator.n_features_in_ == X.shape[1]

    with raises(ValueError, match=msg):
        estimator.partial_fit(X_bad, y)


def check_estimator_get_tags_default_keys(name, estimator_orig):
    # check that if _get_tags is implemented, it contains all keys from
    # _DEFAULT_KEYS
    estimator = clone(estimator_orig)
    if not hasattr(estimator, "_get_tags"):
        return

    tags_keys = set(estimator._get_tags().keys())
    default_tags_keys = set(_DEFAULT_TAGS.keys())
    assert tags_keys.intersection(default_tags_keys) == default_tags_keys, (
        f"{name}._get_tags() is missing entries for the following default tags"
        f": {default_tags_keys - tags_keys.intersection(default_tags_keys)}"
    )


def check_dataframe_column_names_consistency(name, estimator_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    tags = _safe_tags(estimator_orig)
    is_supported_X_types = (
        "2darray" in tags["X_types"] or "categorical" in tags["X_types"]
    )

    if not is_supported_X_types or tags["no_validation"]:
        return

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X_orig = rng.normal(size=(150, 8))

    # Some picky estimators (e.g. SkewedChi2Sampler) only accept skewed positive data.
    X_orig -= X_orig.min() + 0.5
    X_orig = _enforce_estimator_tags_x(estimator, X_orig)
    X_orig = _pairwise_estimator_convert_X(X_orig, estimator)
    n_samples, n_features = X_orig.shape

    names = np.array([f"col_{i}" for i in range(n_features)])
    X = pd.DataFrame(X_orig, columns=names)

    if is_regressor(estimator):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)

    # Check that calling `fit` does not raise any warnings about feature names.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="X does not have valid feature names",
            category=UserWarning,
            module="sklearn",
        )
        estimator.fit(X, y)

    if not hasattr(estimator, "feature_names_in_"):
        raise ValueError(
            "Estimator does not have a feature_names_in_ "
            "attribute after fitting with a dataframe"
        )
    assert isinstance(estimator.feature_names_in_, np.ndarray)
    assert estimator.feature_names_in_.dtype == object
    assert_array_equal(estimator.feature_names_in_, names)

    # Only check sklearn estimators for feature_names_in_ in docstring
    module_name = estimator_orig.__module__
    if (
        module_name.startswith("sklearn.")
        and not ("test_" in module_name or module_name.endswith("_testing"))
        and ("feature_names_in_" not in (estimator_orig.__doc__))
    ):
        raise ValueError(
            f"Estimator {name} does not document its feature_names_in_ attribute"
        )

    check_methods = []
    for method in (
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
        "score_samples",
        "predict_log_proba",
    ):
        if not hasattr(estimator, method):
            continue

        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)
        check_methods.append((method, callable_method))

    for _, method in check_methods:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="X does not have valid feature names",
                category=UserWarning,
                module="sklearn",
            )
            method(X)  # works without UserWarning for valid features

    invalid_names = [
        (names[::-1], "Feature names must be in the same order as they were in fit."),
        (
            [f"another_prefix_{i}" for i in range(n_features)],
            "Feature names unseen at fit time:\n- another_prefix_0\n-"
            " another_prefix_1\n",
        ),
        (
            names[:3],
            f"Feature names seen at fit time, yet now missing:\n- {min(names[3:])}\n",
        ),
    ]
    params = {
        key: value
        for key, value in estimator.get_params().items()
        if "early_stopping" in key
    }
    early_stopping_enabled = any(value is True for value in params.values())

    for invalid_name, additional_message in invalid_names:
        X_bad = pd.DataFrame(X, columns=invalid_name)

        expected_msg = re.escape(
            "The feature names should match those that were passed "
            "during fit. Starting version 1.2, an error will be raised.\n"
            f"{additional_message}"
        )
        for name, method in check_methods:
            # TODO In 1.2, this will be an error.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    category=FutureWarning,
                    module="sklearn",
                )
                with raises(
                    FutureWarning, match=expected_msg, err_msg=f"{name} did not raise"
                ):
                    method(X_bad)

        # partial_fit checks on second call
        # Do not call partial fit if early_stopping is on
        if not hasattr(estimator, "partial_fit") or early_stopping_enabled:
            continue

        estimator = clone(estimator_orig)
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=FutureWarning, module="sklearn")
            with raises(FutureWarning, match=expected_msg):
                estimator.partial_fit(X_bad, y)


def check_transformer_get_feature_names_out(name, transformer_orig):
    tags = transformer_orig._get_tags()
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()

    transformer = clone(transformer_orig)
    X = _enforce_estimator_tags_x(transformer, X)
    X = _pairwise_estimator_convert_X(X, transformer)

    n_features = X.shape[1]
    set_random_state(transformer)

    y_ = y
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2

    X_transform = transformer.fit_transform(X, y=y_)
    input_features = [f"feature{i}" for i in range(n_features)]

    # input_features names is not the same length as n_features_in_
    with raises(ValueError, match="input_features should have length equal"):
        transformer.get_feature_names_out(input_features[::2])

    feature_names_out = transformer.get_feature_names_out(input_features)
    assert feature_names_out is not None
    assert isinstance(feature_names_out, np.ndarray)
    assert all(isinstance(name, str) for name in feature_names_out)

    if isinstance(X_transform, tuple):
        n_features_out = X_transform[0].shape[1]
    else:
        n_features_out = X_transform.shape[1]

    assert (
        len(feature_names_out) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out)}"


def check_transformer_get_feature_names_out_pandas(name, transformer_orig):
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not checking column name consistency for pandas"
        )

    tags = transformer_orig._get_tags()
    if "2darray" not in tags["X_types"] or tags["no_validation"]:
        return

    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X -= X.min()

    transformer = clone(transformer_orig)
    X = _enforce_estimator_tags_x(transformer, X)
    X = _pairwise_estimator_convert_X(X, transformer)

    n_features = X.shape[1]
    set_random_state(transformer)

    y_ = y
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2

    feature_names_in = [f"col{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names_in)
    X_transform = transformer.fit_transform(df, y=y_)

    # error is raised when `input_features` do not match feature_names_in
    invalid_feature_names = [f"bad{i}" for i in range(n_features)]
    with raises(ValueError, match="input_features is not equal to feature_names_in_"):
        transformer.get_feature_names_out(invalid_feature_names)

    feature_names_out_default = transformer.get_feature_names_out()
    feature_names_in_explicit_names = transformer.get_feature_names_out(
        feature_names_in
    )
    assert_array_equal(feature_names_out_default, feature_names_in_explicit_names)

    if isinstance(X_transform, tuple):
        n_features_out = X_transform[0].shape[1]
    else:
        n_features_out = X_transform.shape[1]

    assert (
        len(feature_names_out_default) == n_features_out
    ), f"Expected {n_features_out} feature names, got {len(feature_names_out_default)}"
