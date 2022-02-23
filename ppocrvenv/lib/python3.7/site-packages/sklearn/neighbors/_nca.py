# coding: utf-8
"""
Neighborhood Component Analysis
"""

# Authors: William de Vazelhes <wdevazelhes@gmail.com>
#          John Chiotellis <ioannis.chiotellis@in.tum.de>
# License: BSD 3 clause

from warnings import warn
import numpy as np
import sys
import time
import numbers
from scipy.optimize import minimize
from ..utils.extmath import softmax
from ..metrics import pairwise_distances
from ..base import BaseEstimator, TransformerMixin
from ..preprocessing import LabelEncoder
from ..decomposition import PCA
from ..utils.multiclass import check_classification_targets
from ..utils.random import check_random_state
from ..utils.validation import check_is_fitted, check_array, check_scalar
from ..exceptions import ConvergenceWarning


class NeighborhoodComponentsAnalysis(TransformerMixin, BaseEstimator):
    """Neighborhood Components Analysis.

    Neighborhood Component Analysis (NCA) is a machine learning algorithm for
    metric learning. It learns a linear transformation in a supervised fashion
    to improve the classification accuracy of a stochastic nearest neighbors
    rule in the transformed space.

    Read more in the :ref:`User Guide <nca>`.

    Parameters
    ----------
    n_components : int, default=None
        Preferred dimensionality of the projected space.
        If None it will be set to `n_features`.

    init : {'auto', 'pca', 'lda', 'identity', 'random'} or ndarray of shape \
            (n_features_a, n_features_b), default='auto'
        Initialization of the linear transformation. Possible options are
        `'auto'`, `'pca'`, `'lda'`, `'identity'`, `'random'`, and a numpy
        array of shape `(n_features_a, n_features_b)`.

        - `'auto'`
            Depending on `n_components`, the most reasonable initialization
            will be chosen. If `n_components <= n_classes` we use `'lda'`, as
            it uses labels information. If not, but
            `n_components < min(n_features, n_samples)`, we use `'pca'`, as
            it projects data in meaningful directions (those of higher
            variance). Otherwise, we just use `'identity'`.

        - `'pca'`
            `n_components` principal components of the inputs passed
            to :meth:`fit` will be used to initialize the transformation.
            (See :class:`~sklearn.decomposition.PCA`)

        - `'lda'`
            `min(n_components, n_classes)` most discriminative
            components of the inputs passed to :meth:`fit` will be used to
            initialize the transformation. (If `n_components > n_classes`,
            the rest of the components will be zero.) (See
            :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)

        - `'identity'`
            If `n_components` is strictly smaller than the
            dimensionality of the inputs passed to :meth:`fit`, the identity
            matrix will be truncated to the first `n_components` rows.

        - `'random'`
            The initial transformation will be a random array of shape
            `(n_components, n_features)`. Each value is sampled from the
            standard normal distribution.

        - numpy array
            `n_features_b` must match the dimensionality of the inputs passed
            to :meth:`fit` and n_features_a must be less than or equal to that.
            If `n_components` is not `None`, `n_features_a` must match it.

    warm_start : bool, default=False
        If `True` and :meth:`fit` has been called before, the solution of the
        previous call to :meth:`fit` is used as the initial linear
        transformation (`n_components` and `init` will be ignored).

    max_iter : int, default=50
        Maximum number of iterations in the optimization.

    tol : float, default=1e-5
        Convergence tolerance for the optimization.

    callback : callable, default=None
        If not `None`, this function is called after every iteration of the
        optimizer, taking as arguments the current solution (flattened
        transformation matrix) and the number of iterations. This might be
        useful in case one wants to examine or store the transformation
        found after each iteration.

    verbose : int, default=0
        If 0, no progress messages will be printed.
        If 1, progress messages will be printed to stdout.
        If > 1, progress messages will be printed and the `disp`
        parameter of :func:`scipy.optimize.minimize` will be set to
        `verbose - 2`.

    random_state : int or numpy.RandomState, default=None
        A pseudo random number generator object or a seed for it if int. If
        `init='random'`, `random_state` is used to initialize the random
        transformation. If `init='pca'`, `random_state` is passed as an
        argument to PCA when initializing the transformation. Pass an int
        for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The linear transformation learned during fitting.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    n_iter_ : int
        Counts the number of iterations performed by the optimizer.

    random_state_ : numpy.RandomState
        Pseudo random number generator object used during initialization.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis : Linear
        Discriminant Analysis.
    sklearn.decomposition.PCA : Principal component analysis (PCA).

    References
    ----------
    .. [1] J. Goldberger, G. Hinton, S. Roweis, R. Salakhutdinov.
           "Neighbourhood Components Analysis". Advances in Neural Information
           Processing Systems. 17, 513-520, 2005.
           http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf

    .. [2] Wikipedia entry on Neighborhood Components Analysis
           https://en.wikipedia.org/wiki/Neighbourhood_components_analysis

    Examples
    --------
    >>> from sklearn.neighbors import NeighborhoodComponentsAnalysis
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ... stratify=y, test_size=0.7, random_state=42)
    >>> nca = NeighborhoodComponentsAnalysis(random_state=42)
    >>> nca.fit(X_train, y_train)
    NeighborhoodComponentsAnalysis(...)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(X_test, y_test))
    0.933333...
    >>> knn.fit(nca.transform(X_train), y_train)
    KNeighborsClassifier(...)
    >>> print(knn.score(nca.transform(X_test), y_test))
    0.961904...
    """

    def __init__(
        self,
        n_components=None,
        *,
        init="auto",
        warm_start=False,
        max_iter=50,
        tol=1e-5,
        callback=None,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.

        y : array-like of shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Verify inputs X and y and NCA parameters, and transform a copy if
        # needed
        X, y, init = self._validate_params(X, y)

        # Initialize the random generator
        self.random_state_ = check_random_state(self.random_state)

        # Measure the total training time
        t_train = time.time()

        # Compute a mask that stays fixed during optimization:
        same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]
        # (n_samples, n_samples)

        # Initialize the transformation
        transformation = self._initialize(X, y, init)

        # Create a dictionary of parameters to be passed to the optimizer
        disp = self.verbose - 2 if self.verbose > 1 else -1
        optimizer_params = {
            "method": "L-BFGS-B",
            "fun": self._loss_grad_lbfgs,
            "args": (X, same_class_mask, -1.0),
            "jac": True,
            "x0": transformation,
            "tol": self.tol,
            "options": dict(maxiter=self.max_iter, disp=disp),
            "callback": self._callback,
        }

        # Call the optimizer
        self.n_iter_ = 0
        opt_result = minimize(**optimizer_params)

        # Reshape the solution found by the optimizer
        self.components_ = opt_result.x.reshape(-1, X.shape[1])

        # Stop timer
        t_train = time.time() - t_train
        if self.verbose:
            cls_name = self.__class__.__name__

            # Warn the user if the algorithm did not converge
            if not opt_result.success:
                warn(
                    "[{}] NCA did not converge: {}".format(
                        cls_name, opt_result.message
                    ),
                    ConvergenceWarning,
                )

            print("[{}] Training took {:8.2f}s.".format(cls_name, t_train))

        return self

    def transform(self, X):
        """Apply the learned transformation to the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data samples.

        Returns
        -------
        X_embedded: ndarray of shape (n_samples, n_components)
            The data samples transformed.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called before.
        """

        check_is_fitted(self)
        X = self._validate_data(X, reset=False)

        return np.dot(X, self.components_.T)

    def _validate_params(self, X, y):
        """Validate parameters as soon as :meth:`fit` is called.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.

        y : array-like of shape (n_samples,)
            The corresponding training labels.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The validated training samples.

        y : ndarray of shape (n_samples,)
            The validated training labels, encoded to be integers in
            the `range(0, n_classes)`.

        init : str or ndarray of shape (n_features_a, n_features_b)
            The validated initialization of the linear transformation.

        Raises
        -------
        TypeError
            If a parameter is not an instance of the desired type.

        ValueError
            If a parameter's value violates its legal value range or if the
            combination of two or more given parameters is incompatible.
        """

        # Validate the inputs X and y, and converts y to numerical classes.
        X, y = self._validate_data(X, y, ensure_min_samples=2)
        check_classification_targets(y)
        y = LabelEncoder().fit_transform(y)

        # Check the preferred dimensionality of the projected space
        if self.n_components is not None:
            check_scalar(self.n_components, "n_components", numbers.Integral, min_val=1)

            if self.n_components > X.shape[1]:
                raise ValueError(
                    "The preferred dimensionality of the "
                    "projected space `n_components` ({}) cannot "
                    "be greater than the given data "
                    "dimensionality ({})!".format(self.n_components, X.shape[1])
                )

        # If warm_start is enabled, check that the inputs are consistent
        check_scalar(self.warm_start, "warm_start", bool)
        if self.warm_start and hasattr(self, "components_"):
            if self.components_.shape[1] != X.shape[1]:
                raise ValueError(
                    "The new inputs dimensionality ({}) does not "
                    "match the input dimensionality of the "
                    "previously learned transformation ({}).".format(
                        X.shape[1], self.components_.shape[1]
                    )
                )

        check_scalar(self.max_iter, "max_iter", numbers.Integral, min_val=1)
        check_scalar(self.tol, "tol", numbers.Real, min_val=0.0)
        check_scalar(self.verbose, "verbose", numbers.Integral, min_val=0)

        if self.callback is not None:
            if not callable(self.callback):
                raise ValueError("`callback` is not callable.")

        # Check how the linear transformation should be initialized
        init = self.init

        if isinstance(init, np.ndarray):
            init = check_array(init)

            # Assert that init.shape[1] = X.shape[1]
            if init.shape[1] != X.shape[1]:
                raise ValueError(
                    "The input dimensionality ({}) of the given "
                    "linear transformation `init` must match the "
                    "dimensionality of the given inputs `X` ({}).".format(
                        init.shape[1], X.shape[1]
                    )
                )

            # Assert that init.shape[0] <= init.shape[1]
            if init.shape[0] > init.shape[1]:
                raise ValueError(
                    "The output dimensionality ({}) of the given "
                    "linear transformation `init` cannot be "
                    "greater than its input dimensionality ({}).".format(
                        init.shape[0], init.shape[1]
                    )
                )

            if self.n_components is not None:
                # Assert that self.n_components = init.shape[0]
                if self.n_components != init.shape[0]:
                    raise ValueError(
                        "The preferred dimensionality of the "
                        "projected space `n_components` ({}) does"
                        " not match the output dimensionality of "
                        "the given linear transformation "
                        "`init` ({})!".format(self.n_components, init.shape[0])
                    )
        elif init in ["auto", "pca", "lda", "identity", "random"]:
            pass
        else:
            raise ValueError(
                "`init` must be 'auto', 'pca', 'lda', 'identity', 'random' "
                "or a numpy array of shape (n_components, n_features)."
            )

        return X, y, init

    def _initialize(self, X, y, init):
        """Initialize the transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training samples.

        y : array-like of shape (n_samples,)
            The training labels.

        init : str or ndarray of shape (n_features_a, n_features_b)
            The validated initialization of the linear transformation.

        Returns
        -------
        transformation : ndarray of shape (n_components, n_features)
            The initialized linear transformation.

        """

        transformation = init
        if self.warm_start and hasattr(self, "components_"):
            transformation = self.components_
        elif isinstance(init, np.ndarray):
            pass
        else:
            n_samples, n_features = X.shape
            n_components = self.n_components or n_features
            if init == "auto":
                n_classes = len(np.unique(y))
                if n_components <= min(n_features, n_classes - 1):
                    init = "lda"
                elif n_components < min(n_features, n_samples):
                    init = "pca"
                else:
                    init = "identity"
            if init == "identity":
                transformation = np.eye(n_components, X.shape[1])
            elif init == "random":
                transformation = self.random_state_.randn(n_components, X.shape[1])
            elif init in {"pca", "lda"}:
                init_time = time.time()
                if init == "pca":
                    pca = PCA(
                        n_components=n_components, random_state=self.random_state_
                    )
                    if self.verbose:
                        print("Finding principal components... ", end="")
                        sys.stdout.flush()
                    pca.fit(X)
                    transformation = pca.components_
                elif init == "lda":
                    from ..discriminant_analysis import LinearDiscriminantAnalysis

                    lda = LinearDiscriminantAnalysis(n_components=n_components)
                    if self.verbose:
                        print("Finding most discriminative components... ", end="")
                        sys.stdout.flush()
                    lda.fit(X, y)
                    transformation = lda.scalings_.T[:n_components]
                if self.verbose:
                    print("done in {:5.2f}s".format(time.time() - init_time))
        return transformation

    def _callback(self, transformation):
        """Called after each iteration of the optimizer.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The solution computed by the optimizer in this iteration.
        """
        if self.callback is not None:
            self.callback(transformation, self.n_iter_)

        self.n_iter_ += 1

    def _loss_grad_lbfgs(self, transformation, X, same_class_mask, sign=1.0):
        """Compute the loss and the loss gradient w.r.t. `transformation`.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The raveled linear transformation on which to compute loss and
            evaluate gradient.

        X : ndarray of shape (n_samples, n_features)
            The training samples.

        same_class_mask : ndarray of shape (n_samples, n_samples)
            A mask where `mask[i, j] == 1` if `X[i]` and `X[j]` belong
            to the same class, and `0` otherwise.

        Returns
        -------
        loss : float
            The loss computed for the given transformation.

        gradient : ndarray of shape (n_components * n_features,)
            The new (flattened) gradient of the loss.
        """

        if self.n_iter_ == 0:
            self.n_iter_ += 1
            if self.verbose:
                header_fields = ["Iteration", "Objective Value", "Time(s)"]
                header_fmt = "{:>10} {:>20} {:>10}"
                header = header_fmt.format(*header_fields)
                cls_name = self.__class__.__name__
                print("[{}]".format(cls_name))
                print(
                    "[{}] {}\n[{}] {}".format(
                        cls_name, header, cls_name, "-" * len(header)
                    )
                )

        t_funcall = time.time()

        transformation = transformation.reshape(-1, X.shape[1])
        X_embedded = np.dot(X, transformation.T)  # (n_samples, n_components)

        # Compute softmax distances
        p_ij = pairwise_distances(X_embedded, squared=True)
        np.fill_diagonal(p_ij, np.inf)
        p_ij = softmax(-p_ij)  # (n_samples, n_samples)

        # Compute loss
        masked_p_ij = p_ij * same_class_mask
        p = np.sum(masked_p_ij, axis=1, keepdims=True)  # (n_samples, 1)
        loss = np.sum(p)

        # Compute gradient of loss w.r.t. `transform`
        weighted_p_ij = masked_p_ij - p_ij * p
        weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
        np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))
        gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(X)
        # time complexity of the gradient: O(n_components x n_samples x (
        # n_samples + n_features))

        if self.verbose:
            t_funcall = time.time() - t_funcall
            values_fmt = "[{}] {:>10} {:>20.6e} {:>10.2f}"
            print(
                values_fmt.format(
                    self.__class__.__name__, self.n_iter_, loss, t_funcall
                )
            )
            sys.stdout.flush()

        return sign * loss, sign * gradient.ravel()

    def _more_tags(self):
        return {"requires_y": True}
