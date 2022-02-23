"""

=============================================================
Online Latent Dirichlet Allocation with variational inference
=============================================================

This implementation is modified from Matthew D. Hoffman's onlineldavb code
Link: https://github.com/blei-lab/onlineldavb
"""

# Author: Chyi-Kwei Yau
# Author: Matthew D. Hoffman (original onlineldavb implementation)

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln, logsumexp
from joblib import Parallel, effective_n_jobs

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils.validation import check_non_negative
from ..utils.validation import check_is_fitted
from ..utils.fixes import delayed

from ._online_lda_fast import (
    mean_change,
    _dirichlet_expectation_1d,
    _dirichlet_expectation_2d,
)

EPS = np.finfo(float).eps


def _update_doc_distribution(
    X,
    exp_topic_word_distr,
    doc_topic_prior,
    max_doc_update_iter,
    mean_change_tol,
    cal_sstats,
    random_state,
):
    """E-step: update document-topic distribution.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Document word matrix.

    exp_topic_word_distr : ndarray of shape (n_topics, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    doc_topic_prior : float
        Prior of document topic distribution `theta`.

    max_doc_update_iter : int
        Max number of iterations for updating document topic distribution in
        the E-step.

    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-step.

    cal_sstats : bool
        Parameter that indicate to calculate sufficient statistics or not.
        Set `cal_sstats` to `True` when we need to run M-step.

    random_state : RandomState instance or None
        Parameter that indicate how to initialize document topic distribution.
        Set `random_state` to None will initialize document topic distribution
        to a constant number.

    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.

    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topics = exp_topic_word_distr.shape[0]

    if random_state:
        doc_topic_distr = random_state.gamma(100.0, 0.01, (n_samples, n_topics))
    else:
        doc_topic_distr = np.ones((n_samples, n_topics))

    # In the literature, this is `exp(E[log(theta)])`
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))

    # diff on `component_` (only calculate it when `cal_diff` is True)
    suff_stats = np.zeros(exp_topic_word_distr.shape) if cal_sstats else None

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        doc_topic_d = doc_topic_distr[idx_d, :]
        # The next one is a copy, since the inner loop overwrites it.
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        exp_topic_word_d = exp_topic_word_distr[:, ids]

        # Iterate between `doc_topic_d` and `norm_phi` until convergence
        for _ in range(0, max_doc_update_iter):
            last_d = doc_topic_d

            # The optimal phi_{dwk} is proportional to
            # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS

            doc_topic_d = exp_doc_topic_d * np.dot(cnts / norm_phi, exp_topic_word_d.T)
            # Note: adds doc_topic_prior to doc_topic_d, in-place.
            _dirichlet_expectation_1d(doc_topic_d, doc_topic_prior, exp_doc_topic_d)

            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d

        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)

    return (doc_topic_distr, suff_stats)


class LatentDirichletAllocation(TransformerMixin, BaseEstimator):
    """Latent Dirichlet Allocation with online variational Bayes algorithm.

    The implementation is based on [1]_ and [2]_.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.

        .. versionchanged:: 0.19
            ``n_topics`` was renamed to ``n_components``

    doc_topic_prior : float, default=None
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
        In [1]_, this is called `alpha`.

    topic_word_prior : float, default=None
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
        In [1]_, this is called `eta`.

    learning_method : {'batch', 'online'}, default='batch'
        Method used to update `_component`. Only used in :meth:`fit` method.
        In general, if the data size is large, the online update will be much
        faster than the batch update.

        Valid options::

            'batch': Batch variational Bayes method. Use all training data in
                each EM update.
                Old `components_` will be overwritten in each iteration.
            'online': Online variational Bayes method. In each EM update, use
                mini-batch of training data to update the ``components_``
                variable incrementally. The learning rate is controlled by the
                ``learning_decay`` and the ``learning_offset`` parameters.

        .. versionchanged:: 0.20
            The default learning method is now ``"batch"``.

    learning_decay : float, default=0.7
        It is a parameter that control learning rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence. When the value is 0.0 and batch_size is
        ``n_samples``, the update method is same as batch learning. In the
        literature, this is called kappa.

    learning_offset : float, default=10.0
        A (positive) parameter that downweights early iterations in online
        learning.  It should be greater than 1.0. In the literature, this is
        called tau_0.

    max_iter : int, default=10
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the :meth:`fit` method, and not the
        :meth:`partial_fit` method.

    batch_size : int, default=128
        Number of documents to use in each EM iteration. Only used in online
        learning.

    evaluate_every : int, default=-1
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.

    total_samples : int, default=1e6
        Total number of documents. Only used in the :meth:`partial_fit` method.

    perp_tol : float, default=1e-1
        Perplexity tolerance in batch learning. Only used when
        ``evaluate_every`` is greater than 0.

    mean_change_tol : float, default=1e-3
        Stopping tolerance for updating document topic distribution in E-step.

    max_doc_update_iter : int, default=100
        Max number of iterations for updating document topic distribution in
        the E-step.

    n_jobs : int, default=None
        The number of jobs to use in the E-step.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Variational parameters for topic word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j]`` can be viewed as pseudocount that represents the
        number of times word `j` was assigned to topic `i`.
        It can also be viewed as distribution over the words for each topic
        after normalization:
        ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

    exp_dirichlet_component_ : ndarray of shape (n_components, n_features)
        Exponential value of expectation of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.

    n_batch_iter_ : int
        Number of iterations of the EM step.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of passes over the dataset.

    bound_ : float
        Final perplexity score on training set.

    doc_topic_prior_ : float
        Prior of document topic distribution `theta`. If the value is None,
        it is `1 / n_components`.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generator or by `np.random`.

    topic_word_prior_ : float
        Prior of topic word distribution `beta`. If the value is None, it is
        `1 / n_components`.

    See Also
    --------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis:
        A classifier with a linear decision boundary, generated by fitting
        class conditional densities to the data and using Bayes’ rule.

    References
    ----------
    .. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
           Hoffman, David M. Blei, Francis Bach, 2010
           https://github.com/blei-lab/onlineldavb

    .. [2] "Stochastic Variational Inference", Matthew D. Hoffman,
           David M. Blei, Chong Wang, John Paisley, 2013

    Examples
    --------
    >>> from sklearn.decomposition import LatentDirichletAllocation
    >>> from sklearn.datasets import make_multilabel_classification
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> X, _ = make_multilabel_classification(random_state=0)
    >>> lda = LatentDirichletAllocation(n_components=5,
    ...     random_state=0)
    >>> lda.fit(X)
    LatentDirichletAllocation(...)
    >>> # get topics for some given samples:
    >>> lda.transform(X[-2:])
    array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    """

    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1e6,
        perp_tol=1e-1,
        mean_change_tol=1e-3,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _check_params(self):
        """Check model parameters."""
        if self.n_components <= 0:
            raise ValueError("Invalid 'n_components' parameter: %r" % self.n_components)

        if self.total_samples <= 0:
            raise ValueError(
                "Invalid 'total_samples' parameter: %r" % self.total_samples
            )

        if self.learning_offset < 0:
            raise ValueError(
                "Invalid 'learning_offset' parameter: %r" % self.learning_offset
            )

        if self.learning_method not in ("batch", "online"):
            raise ValueError(
                "Invalid 'learning_method' parameter: %r" % self.learning_method
            )

    def _init_latent_vars(self, n_features):
        """Initialize latent variables."""

        self.random_state_ = check_random_state(self.random_state)
        self.n_batch_iter_ = 1
        self.n_iter_ = 0

        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        init_gamma = 100.0
        init_var = 1.0 / init_gamma
        # In the literature, this is called `lambda`
        self.components_ = self.random_state_.gamma(
            init_gamma, init_var, (self.n_components, n_features)
        )

        # In the literature, this is `exp(E[log(beta)])`
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )

    def _e_step(self, X, cal_sstats, random_init, parallel=None):
        """E-step in EM update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        cal_sstats : bool
            Parameter that indicate whether to calculate sufficient statistics
            or not. Set ``cal_sstats`` to True when we need to run M-step.

        random_init : bool
            Parameter that indicate whether to initialize document topic
            distribution randomly in the E-step. Set it to True in training
            steps.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel.

        Returns
        -------
        (doc_topic_distr, suff_stats) :
            `doc_topic_distr` is unnormalized topic distribution for each
            document. In the literature, this is called `gamma`.
            `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, it will be None.

        """

        # Run e-step in parallel
        random_state = self.random_state_ if random_init else None

        # TODO: make Parallel._effective_n_jobs public instead?
        n_jobs = effective_n_jobs(self.n_jobs)
        if parallel is None:
            parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
        results = parallel(
            delayed(_update_doc_distribution)(
                X[idx_slice, :],
                self.exp_dirichlet_component_,
                self.doc_topic_prior_,
                self.max_doc_update_iter,
                self.mean_change_tol,
                cal_sstats,
                random_state,
            )
            for idx_slice in gen_even_slices(X.shape[0], n_jobs)
        )

        # merge result
        doc_topics, sstats_list = zip(*results)
        doc_topic_distr = np.vstack(doc_topics)

        if cal_sstats:
            # This step finishes computing the sufficient statistics for the
            # M-step.
            suff_stats = np.zeros(self.components_.shape)
            for sstats in sstats_list:
                suff_stats += sstats
            suff_stats *= self.exp_dirichlet_component_
        else:
            suff_stats = None

        return (doc_topic_distr, suff_stats)

    def _em_step(self, X, total_samples, batch_update, parallel=None):
        """EM update for 1 iteration.

        update `_component` by batch VB or online VB.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        total_samples : int
            Total number of documents. It is only used when
            batch_update is `False`.

        batch_update : bool
            Parameter that controls updating method.
            `True` for batch learning, `False` for online learning.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Unnormalized document topic distribution.
        """

        # E-step
        _, suff_stats = self._e_step(
            X, cal_sstats=True, random_init=True, parallel=parallel
        )

        # M-step
        if batch_update:
            self.components_ = self.topic_word_prior_ + suff_stats
        else:
            # online update
            # In the literature, the weight is `rho`
            weight = np.power(
                self.learning_offset + self.n_batch_iter_, -self.learning_decay
            )
            doc_ratio = float(total_samples) / X.shape[0]
            self.components_ *= 1 - weight
            self.components_ += weight * (
                self.topic_word_prior_ + doc_ratio * suff_stats
            )

        # update `component_` related variables
        self.exp_dirichlet_component_ = np.exp(
            _dirichlet_expectation_2d(self.components_)
        )
        self.n_batch_iter_ += 1
        return

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _check_non_neg_array(self, X, reset_n_features, whom):
        """check X format

        check X format and make sure no negative value in X.

        Parameters
        ----------
        X :  array-like or sparse matrix

        """
        X = self._validate_data(X, reset=reset_n_features, accept_sparse="csr")
        check_non_negative(X, whom)
        return X

    def partial_fit(self, X, y=None):
        """Online VB with Mini-Batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Partially fitted estimator.
        """
        self._check_params()
        first_time = not hasattr(self, "components_")
        X = self._check_non_neg_array(
            X, reset_n_features=first_time, whom="LatentDirichletAllocation.partial_fit"
        )
        n_samples, n_features = X.shape
        batch_size = self.batch_size

        # initialize parameters or check
        if first_time:
            self._init_latent_vars(n_features)

        if n_features != self.components_.shape[1]:
            raise ValueError(
                "The provided data has %d dimensions while "
                "the model was trained with feature size %d."
                % (n_features, self.components_.shape[1])
            )

        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for idx_slice in gen_batches(n_samples, batch_size):
                self._em_step(
                    X[idx_slice, :],
                    total_samples=self.total_samples,
                    batch_update=False,
                    parallel=parallel,
                )

        return self

    def fit(self, X, y=None):
        """Learn model for the data X with variational Bayes method.

        When `learning_method` is 'online', use mini-batch update.
        Otherwise, use batch update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._check_params()
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.fit"
        )
        n_samples, n_features = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method

        batch_size = self.batch_size

        # initialize parameters
        self._init_latent_vars(n_features)
        # change to perplexity later
        last_bound = None
        n_jobs = effective_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1)) as parallel:
            for i in range(max_iter):
                if learning_method == "online":
                    for idx_slice in gen_batches(n_samples, batch_size):
                        self._em_step(
                            X[idx_slice, :],
                            total_samples=n_samples,
                            batch_update=False,
                            parallel=parallel,
                        )
                else:
                    # batch update
                    self._em_step(
                        X, total_samples=n_samples, batch_update=True, parallel=parallel
                    )

                # check perplexity
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    doc_topics_distr, _ = self._e_step(
                        X, cal_sstats=False, random_init=False, parallel=parallel
                    )
                    bound = self._perplexity_precomp_distr(
                        X, doc_topics_distr, sub_sampling=False
                    )
                    if self.verbose:
                        print(
                            "iteration: %d of max_iter: %d, perplexity: %.4f"
                            % (i + 1, max_iter, bound)
                        )

                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound

                elif self.verbose:
                    print("iteration: %d of max_iter: %d" % (i + 1, max_iter))
                self.n_iter_ += 1

        # calculate final perplexity value on train set
        doc_topics_distr, _ = self._e_step(
            X, cal_sstats=False, random_init=False, parallel=parallel
        )
        self.bound_ = self._perplexity_precomp_distr(
            X, doc_topics_distr, sub_sampling=False
        )

        return self

    def _unnormalized_transform(self, X):
        """Transform data X according to fitted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        doc_topic_distr, _ = self._e_step(X, cal_sstats=False, random_init=False)

        return doc_topic_distr

    def transform(self, X):
        """Transform data X according to the fitted model.

           .. versionchanged:: 0.18
              *doc_topic_distr* is now normalized

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution for X.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.transform"
        )
        doc_topic_distr = self._unnormalized_transform(X)
        doc_topic_distr /= doc_topic_distr.sum(axis=1)[:, np.newaxis]
        return doc_topic_distr

    def _approx_bound(self, X, doc_topic_distr, sub_sampling):
        """Estimate the variational bound.

        Estimate the variational bound over "all documents" using only the
        documents passed in as X. Since log-likelihood of each word cannot
        be computed directly, we use this bound to estimate it.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document topic distribution. In the literature, this is called
            gamma.

        sub_sampling : bool, default=False
            Compensate for subsampling of documents.
            It is used in calculate bound in online learning.

        Returns
        -------
        score : float

        """

        def _loglikelihood(prior, distr, dirichlet_distr, size):
            # calculate log-likelihood
            score = np.sum((prior - distr) * dirichlet_distr)
            score += np.sum(gammaln(distr) - gammaln(prior))
            score += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, 1)))
            return score

        is_sparse_x = sp.issparse(X)
        n_samples, n_components = doc_topic_distr.shape
        n_features = self.components_.shape[1]
        score = 0

        dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic_distr)
        dirichlet_component_ = _dirichlet_expectation_2d(self.components_)
        doc_topic_prior = self.doc_topic_prior_
        topic_word_prior = self.topic_word_prior_

        if is_sparse_x:
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

        # E[log p(docs | theta, beta)]
        for idx_d in range(0, n_samples):
            if is_sparse_x:
                ids = X_indices[X_indptr[idx_d] : X_indptr[idx_d + 1]]
                cnts = X_data[X_indptr[idx_d] : X_indptr[idx_d + 1]]
            else:
                ids = np.nonzero(X[idx_d, :])[0]
                cnts = X[idx_d, ids]
            temp = (
                dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component_[:, ids]
            )
            norm_phi = logsumexp(temp, axis=0)
            score += np.dot(cnts, norm_phi)

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        score += _loglikelihood(
            doc_topic_prior, doc_topic_distr, dirichlet_doc_topic, self.n_components
        )

        # Compensate for the subsampling of the population of documents
        if sub_sampling:
            doc_ratio = float(self.total_samples) / n_samples
            score *= doc_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]
        score += _loglikelihood(
            topic_word_prior, self.components_, dirichlet_component_, n_features
        )

        return score

    def score(self, X, y=None):
        """Calculate approximate log-likelihood as score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        score : float
            Use approximate bound as score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=False, whom="LatentDirichletAllocation.score"
        )

        doc_topic_distr = self._unnormalized_transform(X)
        score = self._approx_bound(X, doc_topic_distr, sub_sampling=False)
        return score

    def _perplexity_precomp_distr(self, X, doc_topic_distr=None, sub_sampling=False):
        """Calculate approximate perplexity for data X with ability to accept
        precomputed doc_topic_distr

        Perplexity is defined as exp(-1. * log-likelihood per word)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        doc_topic_distr : ndarray of shape (n_samples, n_components), \
                default=None
            Document topic distribution.
            If it is None, it will be generated by applying transform on X.

        Returns
        -------
        score : float
            Perplexity score.
        """
        if doc_topic_distr is None:
            doc_topic_distr = self._unnormalized_transform(X)
        else:
            n_samples, n_components = doc_topic_distr.shape
            if n_samples != X.shape[0]:
                raise ValueError(
                    "Number of samples in X and doc_topic_distr do not match."
                )

            if n_components != self.n_components:
                raise ValueError("Number of topics does not match.")

        current_samples = X.shape[0]
        bound = self._approx_bound(X, doc_topic_distr, sub_sampling)

        if sub_sampling:
            word_cnt = X.sum() * (float(self.total_samples) / current_samples)
        else:
            word_cnt = X.sum()
        perword_bound = bound / word_cnt

        return np.exp(-1.0 * perword_bound)

    def perplexity(self, X, sub_sampling=False):
        """Calculate approximate perplexity for data X.

        Perplexity is defined as exp(-1. * log-likelihood per word)

        .. versionchanged:: 0.19
           *doc_topic_distr* argument has been deprecated and is ignored
           because user no longer has access to unnormalized distribution

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        sub_sampling : bool
            Do sub-sampling or not.

        Returns
        -------
        score : float
            Perplexity score.
        """
        check_is_fitted(self)
        X = self._check_non_neg_array(
            X, reset_n_features=True, whom="LatentDirichletAllocation.perplexity"
        )
        return self._perplexity_precomp_distr(X, sub_sampling=sub_sampling)
