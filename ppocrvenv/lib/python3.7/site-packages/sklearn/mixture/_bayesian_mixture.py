"""Bayesian Gaussian Mixture Model."""
# Author: Wei Xue <xuewei4d@gmail.com>
#         Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import math
import numpy as np
from scipy.special import betaln, digamma, gammaln

from ._base import BaseMixture, _check_shape
from ._gaussian_mixture import _check_precision_matrix
from ._gaussian_mixture import _check_precision_positivity
from ._gaussian_mixture import _compute_log_det_cholesky
from ._gaussian_mixture import _compute_precision_cholesky
from ._gaussian_mixture import _estimate_gaussian_parameters
from ._gaussian_mixture import _estimate_log_gaussian_prob
from ..utils import check_array


def _log_dirichlet_norm(dirichlet_concentration):
    """Compute the log of the Dirichlet distribution normalization term.

    Parameters
    ----------
    dirichlet_concentration : array-like of shape (n_samples,)
        The parameters values of the Dirichlet distribution.

    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return gammaln(np.sum(dirichlet_concentration)) - np.sum(
        gammaln(dirichlet_concentration)
    )


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precision_chol : array-like of shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),
            0,
        )
    )


class BayesianGaussianMixture(BaseMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, default=1e-3
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, default=100
        The number of EM iterations to perform.

    n_init : int, default=1
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'random'}, default='kmeans'
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weight_concentration_prior_type : str, default='dirichlet_process'
        String describing the type of the weight concentration prior.
        Must be one of::

            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).

    weight_concentration_prior : float or None, default=None
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float or None, default=None
        The precision prior on the mean distribution (Gaussian).
        Controls the extent of where means can be placed. Larger
        values concentrate the cluster means around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it is set to 1.

    mean_prior : array-like, shape (n_features,), default=None
        The prior on the mean distribution (Gaussian).
        If it is None, it is set to the mean of X.

    degrees_of_freedom_prior : float or None, default=None
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, default=None
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given to the method chosen to initialize the
        parameters (see `init_params`).
        In addition, it controls the generation of random samples from the
        fitted distribution (see the method `sample`).
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    warm_start : bool, default=False
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default=0
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default=10
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like of shape (n_components,)
        The weights of each mixture components.

    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like of shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior_ : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extent of where means can be placed.
        Larger values concentrate the cluster means around `mean_prior`.
        If mean_precision_prior is set to None, `mean_precision_prior_` is set
        to 1.

    mean_precision_ : array-like of shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    mean_prior_ : array-like of shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like of shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <https://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.mixture import BayesianGaussianMixture
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [12, 4], [10, 7]])
    >>> bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(X)
    >>> bgm.means_
    array([[2.49... , 2.29...],
           [8.45..., 4.52... ]])
    >>> bgm.predict([[0, 0], [9, 3]])
    array([0, 1])
    """

    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.covariance_type = covariance_type
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior

    def _check_parameters(self, X):
        """Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']"
                % self.covariance_type
            )

        if self.weight_concentration_prior_type not in [
            "dirichlet_process",
            "dirichlet_distribution",
        ]:
            raise ValueError(
                "Invalid value for 'weight_concentration_prior_type': %s "
                "'weight_concentration_prior_type' should be in "
                "['dirichlet_process', 'dirichlet_distribution']"
                % self.weight_concentration_prior_type
            )

        self._check_weights_parameters()
        self._check_means_parameters(X)
        self._check_precision_parameters(X)
        self._checkcovariance_prior_parameter(X)

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1.0 / self.n_components
        elif self.weight_concentration_prior > 0.0:
            self.weight_concentration_prior_ = self.weight_concentration_prior
        else:
            raise ValueError(
                "The parameter 'weight_concentration_prior' "
                "should be greater than 0., but got %.3f."
                % self.weight_concentration_prior
            )

    def _check_means_parameters(self, X):
        """Check the parameters of the Gaussian distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.mean_precision_prior is None:
            self.mean_precision_prior_ = 1.0
        elif self.mean_precision_prior > 0.0:
            self.mean_precision_prior_ = self.mean_precision_prior
        else:
            raise ValueError(
                "The parameter 'mean_precision_prior' should be "
                "greater than 0., but got %.3f."
                % self.mean_precision_prior
            )

        if self.mean_prior is None:
            self.mean_prior_ = X.mean(axis=0)
        else:
            self.mean_prior_ = check_array(
                self.mean_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(self.mean_prior_, (n_features,), "means")

    def _check_precision_parameters(self, X):
        """Check the prior parameters of the precision distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = n_features
        elif self.degrees_of_freedom_prior > n_features - 1.0:
            self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior
        else:
            raise ValueError(
                "The parameter 'degrees_of_freedom_prior' "
                "should be greater than %d, but got %.3f."
                % (n_features - 1, self.degrees_of_freedom_prior)
            )

    def _checkcovariance_prior_parameter(self, X):
        """Check the `covariance_prior_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        _, n_features = X.shape

        if self.covariance_prior is None:
            self.covariance_prior_ = {
                "full": np.atleast_2d(np.cov(X.T)),
                "tied": np.atleast_2d(np.cov(X.T)),
                "diag": np.var(X, axis=0, ddof=1),
                "spherical": np.var(X, axis=0, ddof=1).mean(),
            }[self.covariance_type]

        elif self.covariance_type in ["full", "tied"]:
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features, n_features),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_matrix(self.covariance_prior_, self.covariance_type)
        elif self.covariance_type == "diag":
            self.covariance_prior_ = check_array(
                self.covariance_prior, dtype=[np.float64, np.float32], ensure_2d=False
            )
            _check_shape(
                self.covariance_prior_,
                (n_features,),
                "%s covariance_prior" % self.covariance_type,
            )
            _check_precision_positivity(self.covariance_prior_, self.covariance_type)
        # spherical case
        elif self.covariance_prior > 0.0:
            self.covariance_prior_ = self.covariance_prior
        else:
            raise ValueError(
                "The parameter 'spherical covariance_prior' "
                "should be greater than 0., but got %.3f."
                % self.covariance_prior
            )

    def _initialize(self, X, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        nk, xk, sk = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type
        )

        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_weights(self, nk):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)
        """
        if self.weight_concentration_prior_type == "dirichlet_process":
            # For dirichlet process weight_concentration will be a tuple
            # containing the two parameters of the beta distribution
            self.weight_concentration_ = (
                1.0 + nk,
                (
                    self.weight_concentration_prior_
                    + np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
                ),
            )
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            self.weight_concentration_ = self.weight_concentration_prior_ + nk

    def _estimate_means(self, nk, xk):
        """Estimate the parameters of the Gaussian distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)
        """
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = (
            self.mean_precision_prior_ * self.mean_prior_ + nk[:, np.newaxis] * xk
        ) / self.mean_precision_[:, np.newaxis]

    def _estimate_precisions(self, nk, xk, sk):
        """Estimate the precisions parameters of the precision distribution.

        Parameters
        ----------
        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like
            The shape depends of `covariance_type`:
            'full' : (n_components, n_features, n_features)
            'tied' : (n_features, n_features)
            'diag' : (n_components, n_features)
            'spherical' : (n_components,)
        """
        {
            "full": self._estimate_wishart_full,
            "tied": self._estimate_wishart_tied,
            "diag": self._estimate_wishart_diag,
            "spherical": self._estimate_wishart_spherical,
        }[self.covariance_type](nk, xk, sk)

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_wishart_full(self, nk, xk, sk):
        """Estimate the full Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk` is
        # the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        self.covariances_ = np.empty((self.n_components, n_features, n_features))

        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covariances_[k] = (
                self.covariance_prior_
                + nk[k] * sk[k]
                + nk[k]
                * self.mean_precision_prior_
                / self.mean_precision_[k]
                * np.outer(diff, diff)
            )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis, np.newaxis]

    def _estimate_wishart_tied(self, nk, xk, sk):
        """Estimate the tied Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_features, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = (
            self.degrees_of_freedom_prior_ + nk.sum() / self.n_components
        )

        diff = xk - self.mean_prior_
        self.covariances_ = (
            self.covariance_prior_
            + sk * nk.sum() / self.n_components
            + self.mean_precision_prior_
            / self.n_components
            * np.dot((nk / self.mean_precision_) * diff.T, diff)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_

    def _estimate_wishart_diag(self, nk, xk, sk):
        """Estimate the diag Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components, n_features)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = self.covariance_prior_ + nk[:, np.newaxis] * (
            sk
            + (self.mean_precision_prior_ / self.mean_precision_)[:, np.newaxis]
            * np.square(diff)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_[:, np.newaxis]

    def _estimate_wishart_spherical(self, nk, xk, sk):
        """Estimate the spherical Wishart distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        xk : array-like of shape (n_components, n_features)

        sk : array-like of shape (n_components,)
        """
        _, n_features = xk.shape

        # Warning : in some Bishop book, there is a typo on the formula 10.63
        # `degrees_of_freedom_k = degrees_of_freedom_0 + Nk`
        # is the correct formula
        self.degrees_of_freedom_ = self.degrees_of_freedom_prior_ + nk

        diff = xk - self.mean_prior_
        self.covariances_ = self.covariance_prior_ + nk * (
            sk
            + self.mean_precision_prior_
            / self.mean_precision_
            * np.mean(np.square(diff), 1)
        )

        # Contrary to the original bishop book, we normalize the covariances
        self.covariances_ /= self.degrees_of_freedom_

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape

        nk, xk, sk = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            digamma_sum = digamma(
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (
                digamma_a
                - digamma_sum
                + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))
            )
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            return digamma(self.weight_concentration_) - digamma(
                np.sum(self.weight_concentration_)
            )

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        log_lambda = n_features * np.log(2.0) + np.sum(
            digamma(
                0.5
                * (self.degrees_of_freedom_ - np.arange(0, n_features)[:, np.newaxis])
            ),
            0,
        )

        return log_gauss + 0.5 * (log_lambda - n_features / self.mean_precision_)

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.

        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to increase at
        each iteration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.

        log_prob_norm : float
            Logarithm of the probability of each sample in X.

        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.
        (n_features,) = self.mean_prior_.shape

        # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
        # because the precision matrix is normalized.
        log_det_precisions_chol = _compute_log_det_cholesky(
            self.precisions_cholesky_, self.covariance_type, n_features
        ) - 0.5 * n_features * np.log(self.degrees_of_freedom_)

        if self.covariance_type == "tied":
            log_wishart = self.n_components * np.float64(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )
        else:
            log_wishart = np.sum(
                _log_wishart_norm(
                    self.degrees_of_freedom_, log_det_precisions_chol, n_features
                )
            )

        if self.weight_concentration_prior_type == "dirichlet_process":
            log_norm_weight = -np.sum(
                betaln(self.weight_concentration_[0], self.weight_concentration_[1])
            )
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (
            -np.sum(np.exp(log_resp) * log_resp)
            - log_wishart
            - log_norm_weight
            - 0.5 * n_features * np.sum(np.log(self.mean_precision_))
        )

    def _get_parameters(self):
        return (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weight_concentration_,
            self.mean_precision_,
            self.means_,
            self.degrees_of_freedom_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Weights computation
        if self.weight_concentration_prior_type == "dirichlet_process":
            weight_dirichlet_sum = (
                self.weight_concentration_[0] + self.weight_concentration_[1]
            )
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0]
                / weight_dirichlet_sum
                * np.hstack((1, np.cumprod(tmp[:-1])))
            )
            self.weights_ /= np.sum(self.weights_)
        else:
            self.weights_ = self.weight_concentration_ / np.sum(
                self.weight_concentration_
            )

        # Precisions matrices computation
        if self.covariance_type == "full":
            self.precisions_ = np.array(
                [
                    np.dot(prec_chol, prec_chol.T)
                    for prec_chol in self.precisions_cholesky_
                ]
            )

        elif self.covariance_type == "tied":
            self.precisions_ = np.dot(
                self.precisions_cholesky_, self.precisions_cholesky_.T
            )
        else:
            self.precisions_ = self.precisions_cholesky_ ** 2
