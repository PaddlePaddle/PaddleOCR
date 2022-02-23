# Authors: Rob Zinkov, Mathieu Blondel
# License: BSD 3 clause

from ._stochastic_gradient import BaseSGDClassifier
from ._stochastic_gradient import BaseSGDRegressor
from ._stochastic_gradient import DEFAULT_EPSILON


class PassiveAggressiveClassifier(BaseSGDClassifier):
    """Passive Aggressive Classifier.

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------
    C : float, default=1.0
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="hinge"
        The loss function to be used:
        hinge: equivalent to PA-I in the reference paper.
        squared_hinge: equivalent to PA-II in the reference paper.

    n_jobs : int or None, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    class_weight : dict, {class_label: weight} or "balanced" or None, \
            default=None
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        .. versionadded:: 0.17
           parameter *class_weight* to automatically weight samples.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    classes_ : ndarray of shape (n_classes,)
        The unique classes labels.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    loss_function_ : callable
        Loss function used by the algorithm.

    See Also
    --------
    SGDClassifier : Incrementally trained logistic regression.
    Perceptron : Linear perceptron classifier.

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
    ... tol=1e-3)
    >>> clf.fit(X, y)
    PassiveAggressiveClassifier(random_state=0)
    >>> print(clf.coef_)
    [[0.26642044 0.45070924 0.67251877 0.64185414]]
    >>> print(clf.intercept_)
    [1.84127814]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="hinge",
        n_jobs=None,
        random_state=None,
        warm_start=False,
        class_weight=None,
        average=False,
    ):
        super().__init__(
            penalty=None,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            eta0=1.0,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average,
            n_jobs=n_jobs,
        )

        self.C = C
        self.loss = loss

    def partial_fit(self, X, y, classes=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of the training data.

        y : array-like of shape (n_samples,)
            Subset of the target values.

        classes : ndarray of shape (n_classes,)
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params(for_partial_fit=True)
        if self.class_weight == "balanced":
            raise ValueError(
                "class_weight 'balanced' is not supported for "
                "partial_fit. For 'balanced' weights, use "
                "`sklearn.utils.compute_class_weight` with "
                "`class_weight='balanced'`. In place of y you "
                "can use a large enough subset of the full "
                "training set target to properly estimate the "
                "class frequency distributions. Pass the "
                "resulting weights as the class_weight "
                "parameter."
            )
        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._partial_fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="hinge",
            learning_rate=lr,
            max_iter=1,
            classes=classes,
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    def fit(self, X, y, coef_init=None, intercept_init=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        coef_init : ndarray of shape (n_classes, n_features)
            The initial coefficients to warm-start the optimization.

        intercept_init : ndarray of shape (n_classes,)
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="hinge",
            learning_rate=lr,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )


class PassiveAggressiveRegressor(BaseSGDRegressor):
    """Passive Aggressive Regressor.

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float, default=1.0
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float or None, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol).

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    loss : str, default="epsilon_insensitive"
        The loss function to be used:
        epsilon_insensitive: equivalent to PA-I in the reference paper.
        squared_epsilon_insensitive: equivalent to PA-II in the reference
        paper.

    epsilon : float, default=0.1
        If the difference between the current prediction and the correct label
        is below this threshold, the model is not updated.

    random_state : int, RandomState instance, default=None
        Used to shuffle the training data, when ``shuffle`` is set to
        ``True``. Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD.

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,\
            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    See Also
    --------
    SGDRegressor : Linear model fitted by minimizing a regularized
        empirical loss with SGD.

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006).

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveRegressor
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=4, random_state=0)
    >>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
    ... tol=1e-3)
    >>> regr.fit(X, y)
    PassiveAggressiveRegressor(max_iter=100, random_state=0)
    >>> print(regr.coef_)
    [20.48736655 34.18818427 67.59122734 87.94731329]
    >>> print(regr.intercept_)
    [-0.02306214]
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-0.02306214]
    """

    def __init__(
        self,
        *,
        C=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        verbose=0,
        loss="epsilon_insensitive",
        epsilon=DEFAULT_EPSILON,
        random_state=None,
        warm_start=False,
        average=False,
    ):
        super().__init__(
            penalty=None,
            l1_ratio=0,
            epsilon=epsilon,
            eta0=1.0,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            warm_start=warm_start,
            average=average,
        )
        self.C = C
        self.loss = loss

    def partial_fit(self, X, y):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Subset of training data.

        y : numpy array of shape [n_samples]
            Subset of target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params(for_partial_fit=True)
        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        return self._partial_fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="epsilon_insensitive",
            learning_rate=lr,
            max_iter=1,
            sample_weight=None,
            coef_init=None,
            intercept_init=None,
        )

    def fit(self, X, y, coef_init=None, intercept_init=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : numpy array of shape [n_samples]
            Target values.

        coef_init : array, shape = [n_features]
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()
        lr = "pa1" if self.loss == "epsilon_insensitive" else "pa2"
        return self._fit(
            X,
            y,
            alpha=1.0,
            C=self.C,
            loss="epsilon_insensitive",
            learning_rate=lr,
            coef_init=coef_init,
            intercept_init=intercept_init,
        )
