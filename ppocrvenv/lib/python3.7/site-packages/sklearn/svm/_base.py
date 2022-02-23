import numpy as np
import scipy.sparse as sp
import warnings
from abc import ABCMeta, abstractmethod

# mypy error: error: Module 'sklearn.svm' has no attribute '_libsvm'
# (and same for other imports)
from . import _libsvm as libsvm  # type: ignore
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelEncoder
from ..utils.multiclass import _ovr_decision_function
from ..utils import check_array, check_random_state
from ..utils import column_or_1d
from ..utils import compute_class_weight
from ..utils.metaestimators import available_if
from ..utils.deprecation import deprecated
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted, _check_large_sparse
from ..utils.validation import _num_samples
from ..utils.validation import _check_sample_weight, check_consistent_length
from ..utils.multiclass import check_classification_targets
from ..exceptions import ConvergenceWarning
from ..exceptions import NotFittedError


LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]


def _one_vs_one_coef(dual_coef, n_support, support_vectors):
    """Generate primal coefficients from dual coefficients
    for the one-vs-one multi class LibSVM in the case
    of a linear kernel."""

    # get 1vs1 weights for all n*(n-1) classifiers.
    # this is somewhat messy.
    # shape of dual_coef_ is nSV * (n_classes -1)
    # see docs for details
    n_class = dual_coef.shape[0] + 1

    # XXX we could do preallocation of coef but
    # would have to take care in the sparse case
    coef = []
    sv_locs = np.cumsum(np.hstack([[0], n_support]))
    for class1 in range(n_class):
        # SVs for class1:
        sv1 = support_vectors[sv_locs[class1] : sv_locs[class1 + 1], :]
        for class2 in range(class1 + 1, n_class):
            # SVs for class1:
            sv2 = support_vectors[sv_locs[class2] : sv_locs[class2 + 1], :]

            # dual coef for class1 SVs:
            alpha1 = dual_coef[class2 - 1, sv_locs[class1] : sv_locs[class1 + 1]]
            # dual coef for class2 SVs:
            alpha2 = dual_coef[class1, sv_locs[class2] : sv_locs[class2 + 1]]
            # build weight for class1 vs class2

            coef.append(safe_sparse_dot(alpha1, sv1) + safe_sparse_dot(alpha2, sv2))
    return coef


class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for estimators that use libsvm as backing library.

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    """

    # The order of these must match the integer values in LibSVM.
    # XXX These are actually the same in the dense case. Need to factor
    # this out.
    _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    @abstractmethod
    def __init__(
        self,
        kernel,
        degree,
        gamma,
        coef0,
        tol,
        C,
        nu,
        epsilon,
        shrinking,
        probability,
        cache_size,
        class_weight,
        verbose,
        max_iter,
        random_state,
    ):

        if self._impl not in LIBSVM_IMPL:
            raise ValueError(
                "impl should be one of %s, %s was given" % (LIBSVM_IMPL, self._impl)
            )

        if gamma == 0:
            msg = (
                "The gamma value of 0.0 is invalid. Use 'auto' to set"
                " gamma to a value of 1 / n_features."
            )
            raise ValueError(msg)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    def _more_tags(self):
        # Used by cross_val_score.
        return {"pairwise": self.kernel == "precomputed"}

    # TODO: Remove in 1.1
    # mypy error: Decorated property not supported
    @deprecated(  # type: ignore
        "Attribute `_pairwise` was deprecated in "
        "version 0.24 and will be removed in 1.1 (renaming of 0.26)."
    )
    @property
    def _pairwise(self):
        # Used by cross_val_score.
        return self.kernel == "precomputed"

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) \
                or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """

        rnd = check_random_state(self.random_state)

        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        if hasattr(self, "decision_function_shape"):
            if self.decision_function_shape not in ("ovr", "ovo"):
                raise ValueError(
                    "decision_function_shape must be either 'ovr' or 'ovo', "
                    f"got {self.decision_function_shape}."
                )

        if callable(self.kernel):
            check_consistent_length(X, y)
        else:
            X, y = self._validate_data(
                X,
                y,
                dtype=np.float64,
                order="C",
                accept_sparse="csr",
                accept_large_sparse=False,
            )

        y = self._validate_targets(y)

        sample_weight = np.asarray(
            [] if sample_weight is None else sample_weight, dtype=np.float64
        )
        solver_type = LIBSVM_IMPL.index(self._impl)

        # input validation
        n_samples = _num_samples(X)
        if solver_type != 2 and n_samples != y.shape[0]:
            raise ValueError(
                "X and y have incompatible shapes.\n"
                + "X has %s samples, but y has %s." % (n_samples, y.shape[0])
            )

        if self.kernel == "precomputed" and n_samples != X.shape[1]:
            raise ValueError(
                "Precomputed matrix must be a square matrix."
                " Input is a {}x{} matrix.".format(X.shape[0], X.shape[1])
            )

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != n_samples:
            raise ValueError(
                "sample_weight and X have incompatible shapes: "
                "%r vs %r\n"
                "Note: Sparse matrices cannot be indexed w/"
                "boolean masks (use `indices=True` in CV)."
                % (sample_weight.shape, X.shape)
            )

        kernel = "precomputed" if callable(self.kernel) else self.kernel

        if kernel == "precomputed":
            # unused but needs to be a float for cython code that ignores
            # it anyway
            self._gamma = 0.0
        elif isinstance(self.gamma, str):
            if self.gamma == "scale":
                # var = E[X^2] - E[X]^2 if sparse
                X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
                self._gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
            elif self.gamma == "auto":
                self._gamma = 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(self.gamma)
                )
        else:
            self._gamma = self.gamma

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:
            print("[LibSVM]", end="")

        seed = rnd.randint(np.iinfo("i").max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        # see comment on the other call to np.iinfo in this file

        self.shape_fit_ = X.shape if hasattr(X, "shape") else (n_samples,)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_
        # internally.
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if self._impl in ["c_svc", "nu_svc"] and len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_

        return self

    def _validate_targets(self, y):
        """Validation of y and class_weight.

        Default implementation for SVR and one-class; overridden in BaseSVC.
        """
        # XXX this is ugly.
        # Regression models should not have a class_weight_ attribute.
        self.class_weight_ = np.empty(0)
        return column_or_1d(y, warn=True).astype(np.float64, copy=False)

    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        if self.fit_status_ == 1:
            warnings.warn(
                "Solver terminated early (max_iter=%i)."
                "  Consider pre-processing your data with"
                " StandardScaler or MinMaxScaler."
                % self.max_iter,
                ConvergenceWarning,
            )

    def _dense_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        if callable(self.kernel):
            # you must store a reference to X to compute the kernel in predict
            # TODO: add keyword copy to copy on demand
            self.__Xfit = X
            X = self._compute_kernel(X)

            if X.shape[0] != X.shape[1]:
                raise ValueError("X.shape[0] should be equal to X.shape[1]")

        libsvm.set_verbosity_wrap(self.verbose)

        # we don't pass **self.get_params() to allow subclasses to
        # add other parameters to __init__
        (
            self.support_,
            self.support_vectors_,
            self._n_support,
            self.dual_coef_,
            self.intercept_,
            self._probA,
            self._probB,
            self.fit_status_,
        ) = libsvm.fit(
            X,
            y,
            svm_type=solver_type,
            sample_weight=sample_weight,
            class_weight=self.class_weight_,
            kernel=kernel,
            C=self.C,
            nu=self.nu,
            probability=self.probability,
            degree=self.degree,
            shrinking=self.shrinking,
            tol=self.tol,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            random_seed=random_seed,
        )

        self._warn_from_fit_status()

    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
        X.data = np.asarray(X.data, dtype=np.float64, order="C")
        X.sort_indices()

        kernel_type = self._sparse_kernels.index(kernel)

        libsvm_sparse.set_verbosity_wrap(self.verbose)

        (
            self.support_,
            self.support_vectors_,
            dual_coef_data,
            self.intercept_,
            self._n_support,
            self._probA,
            self._probB,
            self.fit_status_,
        ) = libsvm_sparse.libsvm_sparse_train(
            X.shape[1],
            X.data,
            X.indices,
            X.indptr,
            y,
            solver_type,
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            self.class_weight_,
            sample_weight,
            self.nu,
            self.cache_size,
            self.epsilon,
            int(self.shrinking),
            int(self.probability),
            self.max_iter,
            random_seed,
        )

        self._warn_from_fit_status()

        if hasattr(self, "classes_"):
            n_class = len(self.classes_) - 1
        else:  # regression
            n_class = 1
        n_SV = self.support_vectors_.shape[0]

        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        if not n_SV:
            self.dual_coef_ = sp.csr_matrix([])
        else:
            dual_coef_indptr = np.arange(
                0, dual_coef_indices.size + 1, dual_coef_indices.size / n_class
            )
            self.dual_coef_ = sp.csr_matrix(
                (dual_coef_data, dual_coef_indices, dual_coef_indptr), (n_class, n_SV)
            )

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):
        X = self._compute_kernel(X)
        if X.ndim == 1:
            X = check_array(X, order="C", accept_large_sparse=False)

        kernel = self.kernel
        if callable(self.kernel):
            kernel = "precomputed"
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError(
                    "X.shape[1] = %d should be equal to %d, "
                    "the number of samples at training time"
                    % (X.shape[1], self.shape_fit_[0])
                )

        svm_type = LIBSVM_IMPL.index(self._impl)

        return libsvm.predict(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=svm_type,
            kernel=kernel,
            degree=self.degree,
            coef0=self.coef0,
            gamma=self._gamma,
            cache_size=self.cache_size,
        )

    def _sparse_predict(self, X):
        # Precondition: X is a csr_matrix of dtype np.float64.
        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        kernel_type = self._sparse_kernels.index(kernel)

        C = 0.0  # C is not useful here

        return libsvm_sparse.libsvm_sparse_predict(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            C,
            self.class_weight_,
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order="C")
        return X

    def _decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        # NOTE: _validate_for_predict contains check for is_fitted
        # hence must be placed before any other attributes are used.
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)

        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function.
        if self._impl in ["c_svc", "nu_svc"] and len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order="C", accept_large_sparse=False)

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        return libsvm.decision_function(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel,
            degree=self.degree,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
        )

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order="C")

        kernel = self.kernel
        if hasattr(kernel, "__call__"):
            kernel = "precomputed"

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_decision_function(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            self.class_weight_,
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )

    def _validate_for_predict(self, X):
        check_is_fitted(self)

        if not callable(self.kernel):
            X = self._validate_data(
                X,
                accept_sparse="csr",
                dtype=np.float64,
                order="C",
                accept_large_sparse=False,
                reset=False,
            )

        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__
            )

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError(
                    "X.shape[1] = %d should be equal to %d, "
                    "the number of samples at training time"
                    % (X.shape[1], self.shape_fit_[0])
                )
        # Fixes https://nvd.nist.gov/vuln/detail/CVE-2020-28975
        # Check that _n_support is consistent with support_vectors
        sv = self.support_vectors_
        if not self._sparse and sv.size > 0 and self.n_support_.sum() != sv.shape[0]:
            raise ValueError(
                f"The internal representation of {self.__class__.__name__} was altered"
            )
        return X

    @property
    def coef_(self):
        """Weights assigned to the features when `kernel="linear"`.

        Returns
        -------
        ndarray of shape (n_features, n_classes)
        """
        if self.kernel != "linear":
            raise AttributeError("coef_ is only available when using a linear kernel")

        coef = self._get_coef()

        # coef_ being a read-only property, it's better to mark the value as
        # immutable to avoid hiding potential bugs for the unsuspecting user.
        if sp.issparse(coef):
            # sparse matrix do not have global flags
            coef.data.flags.writeable = False
        else:
            # regular dense array
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)

    @property
    def n_support_(self):
        """Number of support vectors for each class."""
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError

        svm_type = LIBSVM_IMPL.index(self._impl)
        if svm_type in (0, 1):
            return self._n_support
        else:
            # SVR and OneClass
            # _n_support has size 2, we make it size 1
            return np.array([self._n_support[0]])


class BaseSVC(ClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    """ABC for LibSVM-based classifiers."""

    @abstractmethod
    def __init__(
        self,
        kernel,
        degree,
        gamma,
        coef0,
        tol,
        C,
        nu,
        shrinking,
        probability,
        cache_size,
        class_weight,
        verbose,
        max_iter,
        decision_function_shape,
        random_state,
        break_ties,
    ):
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=nu,
            epsilon=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            random_state=random_state,
        )

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")

    def decision_function(self, X):
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes).

        Notes
        -----
        If decision_function_shape='ovo', the function values are proportional
        to the distance of the samples X to the separating hyperplane. If the
        exact distances are required, divide the function values by the norm of
        the weight vector (``coef_``). See also `this question
        <https://stats.stackexchange.com/questions/14876/
        interpreting-distance-from-hyperplane-in-svm>`_ for further details.
        If decision_function_shape='ovr', the decision function is a monotonic
        transformation of ovo decision function.
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == "ovr" and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec

    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        check_is_fitted(self)
        if self.break_ties and self.decision_function_shape == "ovo":
            raise ValueError(
                "break_ties must be False when decision_function_shape is 'ovo'"
            )

        if (
            self.break_ties
            and self.decision_function_shape == "ovr"
            and len(self.classes_) > 2
        ):
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            y = super().predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp))

    # Hacky way of getting predict_proba to raise an AttributeError when
    # probability=False using properties. Do not use this in new code; when
    # probabilities are not available depending on a setting, introduce two
    # estimators.
    def _check_proba(self):
        if not self.probability:
            raise AttributeError(
                "predict_proba is not available when  probability=False"
            )
        if self._impl not in ("c_svc", "nu_svc"):
            raise AttributeError("predict_proba only implemented for SVC and NuSVC")
        return True

    @available_if(_check_proba)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        X = self._validate_for_predict(X)
        if self.probA_.size == 0 or self.probB_.size == 0:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False"
            )
        pred_proba = (
            self._sparse_predict_proba if self._sparse else self._dense_predict_proba
        )
        return pred_proba(X)

    @available_if(_check_proba)
    def predict_log_proba(self, X):
        """Compute log probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        T : ndarray of shape (n_samples, n_classes)
            Returns the log-probabilities of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        predict. Also, it will produce meaningless results on very small
        datasets.
        """
        return np.log(self.predict_proba(X))

    def _dense_predict_proba(self, X):
        X = self._compute_kernel(X)

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        svm_type = LIBSVM_IMPL.index(self._impl)
        pprob = libsvm.predict_proba(
            X,
            self.support_,
            self.support_vectors_,
            self._n_support,
            self._dual_coef_,
            self._intercept_,
            self._probA,
            self._probB,
            svm_type=svm_type,
            kernel=kernel,
            degree=self.degree,
            cache_size=self.cache_size,
            coef0=self.coef0,
            gamma=self._gamma,
        )

        return pprob

    def _sparse_predict_proba(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order="C")

        kernel = self.kernel
        if callable(kernel):
            kernel = "precomputed"

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_predict_proba(
            X.data,
            X.indices,
            X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data,
            self._intercept_,
            LIBSVM_IMPL.index(self._impl),
            kernel_type,
            self.degree,
            self._gamma,
            self.coef0,
            self.tol,
            self.C,
            self.class_weight_,
            self.nu,
            self.epsilon,
            self.shrinking,
            self.probability,
            self._n_support,
            self._probA,
            self._probB,
        )

    def _get_coef(self):
        if self.dual_coef_.shape[0] == 1:
            # binary classifier
            coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
        else:
            # 1vs1 classifier
            coef = _one_vs_one_coef(
                self.dual_coef_, self._n_support, self.support_vectors_
            )
            if sp.issparse(coef[0]):
                coef = sp.vstack(coef).tocsr()
            else:
                coef = np.vstack(coef)

        return coef

    @property
    def probA_(self):
        """Parameter learned in Platt scaling when `probability=True`.

        Returns
        -------
        ndarray of shape  (n_classes * (n_classes - 1) / 2)
        """
        return self._probA

    @property
    def probB_(self):
        """Parameter learned in Platt scaling when `probability=True`.

        Returns
        -------
        ndarray of shape  (n_classes * (n_classes - 1) / 2)
        """
        return self._probB


def _get_liblinear_solver_type(multi_class, penalty, loss, dual):
    """Find the liblinear magic number for the solver.

    This number depends on the values of the following attributes:
      - multi_class
      - penalty
      - loss
      - dual

    The same number is also internally used by LibLinear to determine
    which solver to use.
    """
    # nested dicts containing level 1: available loss functions,
    # level2: available penalties for the given loss function,
    # level3: whether the dual solver is available for the specified
    # combination of loss function and penalty
    _solver_type_dict = {
        "logistic_regression": {"l1": {False: 6}, "l2": {False: 0, True: 7}},
        "hinge": {"l2": {True: 3}},
        "squared_hinge": {"l1": {False: 5}, "l2": {False: 2, True: 1}},
        "epsilon_insensitive": {"l2": {True: 13}},
        "squared_epsilon_insensitive": {"l2": {False: 11, True: 12}},
        "crammer_singer": 4,
    }

    if multi_class == "crammer_singer":
        return _solver_type_dict[multi_class]
    elif multi_class != "ovr":
        raise ValueError(
            "`multi_class` must be one of `ovr`, `crammer_singer`, got %r" % multi_class
        )

    _solver_pen = _solver_type_dict.get(loss, None)
    if _solver_pen is None:
        error_string = "loss='%s' is not supported" % loss
    else:
        _solver_dual = _solver_pen.get(penalty, None)
        if _solver_dual is None:
            error_string = (
                "The combination of penalty='%s' and loss='%s' is not supported"
                % (penalty, loss)
            )
        else:
            solver_num = _solver_dual.get(dual, None)
            if solver_num is None:
                error_string = (
                    "The combination of penalty='%s' and "
                    "loss='%s' are not supported when dual=%s" % (penalty, loss, dual)
                )
            else:
                return solver_num
    raise ValueError(
        "Unsupported set of arguments: %s, Parameters: penalty=%r, loss=%r, dual=%r"
        % (error_string, penalty, loss, dual)
    )


def _fit_liblinear(
    X,
    y,
    C,
    fit_intercept,
    intercept_scaling,
    class_weight,
    penalty,
    dual,
    verbose,
    max_iter,
    tol,
    random_state=None,
    multi_class="ovr",
    loss="logistic_regression",
    epsilon=0.1,
    sample_weight=None,
):
    """Used by Logistic Regression (and CV) and LinearSVC/LinearSVR.

    Preprocessing is done in this function before supplying it to liblinear.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target vector relative to X

    C : float
        Inverse of cross-validation parameter. Lower the C, the more
        the penalization.

    fit_intercept : bool
        Whether or not to fit the intercept, that is to add a intercept
        term to the decision function.

    intercept_scaling : float
        LibLinear internally penalizes the intercept and this term is subject
        to regularization just like the other terms of the feature vector.
        In order to avoid this, one should increase the intercept_scaling.
        such that the feature vector becomes [x, intercept_scaling].

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    penalty : {'l1', 'l2'}
        The norm of the penalty used in regularization.

    dual : bool
        Dual or primal formulation,

    verbose : int
        Set verbose to any positive number for verbosity.

    max_iter : int
        Number of iterations.

    tol : float
        Stopping condition.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    multi_class : {'ovr', 'crammer_singer'}, default='ovr'
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.

    loss : {'logistic_regression', 'hinge', 'squared_hinge', \
            'epsilon_insensitive', 'squared_epsilon_insensitive}, \
            default='logistic_regression'
        The loss function used to fit the model.

    epsilon : float, default=0.1
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set epsilon=0.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights assigned to each sample.

    Returns
    -------
    coef_ : ndarray of shape (n_features, n_features + 1)
        The coefficient vector got by minimizing the objective function.

    intercept_ : float
        The intercept term added to the vector.

    n_iter_ : int
        Maximum number of iterations run across all classes.
    """
    if loss not in ["epsilon_insensitive", "squared_epsilon_insensitive"]:
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        classes_ = enc.classes_
        if len(classes_) < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )

        class_weight_ = compute_class_weight(class_weight, classes=classes_, y=y)
    else:
        class_weight_ = np.empty(0, dtype=np.float64)
        y_ind = y
    liblinear.set_verbosity_wrap(verbose)
    rnd = check_random_state(random_state)
    if verbose:
        print("[LibLinear]", end="")

    # LinearSVC breaks when intercept_scaling is <= 0
    bias = -1.0
    if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError(
                "Intercept scaling is %r but needs to be greater "
                "than 0. To disable fitting an intercept,"
                " set fit_intercept=False." % intercept_scaling
            )
        else:
            bias = intercept_scaling

    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # Liblinear doesn't support 64bit sparse matrix indices yet
    if sp.issparse(X):
        _check_large_sparse(X)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    y_ind = np.require(y_ind, requirements="W")

    sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)

    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X,
        y_ind,
        sp.isspmatrix(X),
        solver_type,
        tol,
        bias,
        C,
        class_weight_,
        max_iter,
        rnd.randint(np.iinfo("i").max),
        epsilon,
        sample_weight,
    )
    # Regarding rnd.randint(..) in the above signature:
    # seed for srand in range [0..INT_MAX); due to limitations in Numpy
    # on 32-bit platforms, we can't get to the UINT_MAX limit that
    # srand supports
    n_iter_ = max(n_iter_)
    if n_iter_ >= max_iter:
        warnings.warn(
            "Liblinear failed to converge, increase the number of iterations.",
            ConvergenceWarning,
        )

    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.0

    return coef_, intercept_, n_iter_
