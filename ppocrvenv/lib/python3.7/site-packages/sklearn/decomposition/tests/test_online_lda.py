import sys

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.special import psi

import pytest

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._lda import (
    _dirichlet_expectation_1d,
    _dirichlet_expectation_2d,
)

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import if_safe_multiprocessing_with_blas

from sklearn.exceptions import NotFittedError
from io import StringIO


def _build_sparse_mtx():
    # Create 3 topics and each topic has 3 distinct words.
    # (Each word only belongs to a single topic.)
    n_components = 3
    block = np.full((3, 3), n_components, dtype=int)
    blocks = [block] * n_components
    X = block_diag(*blocks)
    X = csr_matrix(X)
    return (n_components, X)


def test_lda_default_prior_params():
    # default prior parameter should be `1 / topics`
    # and verbose params should not affect result
    n_components, X = _build_sparse_mtx()
    prior = 1.0 / n_components
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=prior,
        topic_word_prior=prior,
        random_state=0,
    )
    lda_2 = LatentDirichletAllocation(n_components=n_components, random_state=0)
    topic_distr_1 = lda_1.fit_transform(X)
    topic_distr_2 = lda_2.fit_transform(X)
    assert_almost_equal(topic_distr_1, topic_distr_2)


def test_lda_fit_batch():
    # Test LDA batch learning_offset (`fit` method with 'batch' learning)
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        evaluate_every=1,
        learning_method="batch",
        random_state=rng,
    )
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for component in lda.components_:
        # Find top 3 words in each LDA component
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_fit_online():
    # Test LDA online learning (`fit` method with 'online' learning)
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=10.0,
        evaluate_every=1,
        learning_method="online",
        random_state=rng,
    )
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for component in lda.components_:
        # Find top 3 words in each LDA component
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_partial_fit():
    # Test LDA online learning (`partial_fit` method)
    # (same as test_lda_batch)
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=10.0,
        total_samples=100,
        random_state=rng,
    )
    for i in range(3):
        lda.partial_fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_dense_input():
    # Test LDA with dense input.
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components, learning_method="batch", random_state=rng
    )
    lda.fit(X.toarray())

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for component in lda.components_:
        # Find top 3 words in each LDA component
        top_idx = set(component.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_transform():
    # Test LDA transform.
    # Transform result cannot be negative and should be normalized
    rng = np.random.RandomState(0)
    X = rng.randint(5, size=(20, 10))
    n_components = 3
    lda = LatentDirichletAllocation(n_components=n_components, random_state=rng)
    X_trans = lda.fit_transform(X)
    assert (X_trans > 0.0).any()
    assert_array_almost_equal(np.sum(X_trans, axis=1), np.ones(X_trans.shape[0]))


@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_fit_transform(method):
    # Test LDA fit_transform & transform
    # fit_transform and transform result should be the same
    rng = np.random.RandomState(0)
    X = rng.randint(10, size=(50, 20))
    lda = LatentDirichletAllocation(
        n_components=5, learning_method=method, random_state=rng
    )
    X_fit = lda.fit_transform(X)
    X_trans = lda.transform(X)
    assert_array_almost_equal(X_fit, X_trans, 4)


def test_invalid_params():
    # test `_check_params` method
    X = np.ones((5, 10))

    invalid_models = (
        ("n_components", LatentDirichletAllocation(n_components=0)),
        ("learning_method", LatentDirichletAllocation(learning_method="unknown")),
        ("total_samples", LatentDirichletAllocation(total_samples=0)),
        ("learning_offset", LatentDirichletAllocation(learning_offset=-1)),
    )
    for param, model in invalid_models:
        regex = r"^Invalid %r parameter" % param
        with pytest.raises(ValueError, match=regex):
            model.fit(X)


def test_lda_negative_input():
    # test pass dense matrix with sparse negative input.
    X = np.full((5, 10), -1.0)
    lda = LatentDirichletAllocation()
    regex = r"^Negative values in data passed"
    with pytest.raises(ValueError, match=regex):
        lda.fit(X)


def test_lda_no_component_error():
    # test `perplexity` before `fit`
    rng = np.random.RandomState(0)
    X = rng.randint(4, size=(20, 10))
    lda = LatentDirichletAllocation()
    regex = (
        "This LatentDirichletAllocation instance is not fitted yet. "
        "Call 'fit' with appropriate arguments before using this "
        "estimator."
    )
    with pytest.raises(NotFittedError, match=regex):
        lda.perplexity(X)


@if_safe_multiprocessing_with_blas
@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_multi_jobs(method):
    n_components, X = _build_sparse_mtx()
    # Test LDA batch training with multi CPU
    rng = np.random.RandomState(0)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        n_jobs=2,
        learning_method=method,
        evaluate_every=1,
        random_state=rng,
    )
    lda.fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


@if_safe_multiprocessing_with_blas
def test_lda_partial_fit_multi_jobs():
    # Test LDA online training with multi CPU
    rng = np.random.RandomState(0)
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        n_jobs=2,
        learning_offset=5.0,
        total_samples=30,
        random_state=rng,
    )
    for i in range(2):
        lda.partial_fit(X)

    correct_idx_grps = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    for c in lda.components_:
        top_idx = set(c.argsort()[-3:][::-1])
        assert tuple(sorted(top_idx)) in correct_idx_grps


def test_lda_preplexity_mismatch():
    # test dimension mismatch in `perplexity` method
    rng = np.random.RandomState(0)
    n_components = rng.randint(3, 6)
    n_samples = rng.randint(6, 10)
    X = np.random.randint(4, size=(n_samples, 10))
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_offset=5.0,
        total_samples=20,
        random_state=rng,
    )
    lda.fit(X)
    # invalid samples
    invalid_n_samples = rng.randint(4, size=(n_samples + 1, n_components))
    with pytest.raises(ValueError, match=r"Number of samples"):
        lda._perplexity_precomp_distr(X, invalid_n_samples)
    # invalid topic number
    invalid_n_components = rng.randint(4, size=(n_samples, n_components + 1))
    with pytest.raises(ValueError, match=r"Number of topics"):
        lda._perplexity_precomp_distr(X, invalid_n_components)


@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_perplexity(method):
    # Test LDA perplexity for batch training
    # perplexity should be lower after each iteration
    n_components, X = _build_sparse_mtx()
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    lda_2 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    lda_1.fit(X)
    perp_1 = lda_1.perplexity(X, sub_sampling=False)

    lda_2.fit(X)
    perp_2 = lda_2.perplexity(X, sub_sampling=False)
    assert perp_1 >= perp_2

    perp_1_subsampling = lda_1.perplexity(X, sub_sampling=True)
    perp_2_subsampling = lda_2.perplexity(X, sub_sampling=True)
    assert perp_1_subsampling >= perp_2_subsampling


@pytest.mark.parametrize("method", ("online", "batch"))
def test_lda_score(method):
    # Test LDA score for batch training
    # score should be higher after each iteration
    n_components, X = _build_sparse_mtx()
    lda_1 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    lda_2 = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method=method,
        total_samples=100,
        random_state=0,
    )
    lda_1.fit_transform(X)
    score_1 = lda_1.score(X)

    lda_2.fit_transform(X)
    score_2 = lda_2.score(X)
    assert score_2 >= score_1


def test_perplexity_input_format():
    # Test LDA perplexity for sparse and dense input
    # score should be the same for both dense and sparse input
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method="batch",
        total_samples=100,
        random_state=0,
    )
    lda.fit(X)
    perp_1 = lda.perplexity(X)
    perp_2 = lda.perplexity(X.toarray())
    assert_almost_equal(perp_1, perp_2)


def test_lda_score_perplexity():
    # Test the relationship between LDA score and perplexity
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components, max_iter=10, random_state=0
    )
    lda.fit(X)
    perplexity_1 = lda.perplexity(X, sub_sampling=False)

    score = lda.score(X)
    perplexity_2 = np.exp(-1.0 * (score / np.sum(X.data)))
    assert_almost_equal(perplexity_1, perplexity_2)


def test_lda_fit_perplexity():
    # Test that the perplexity computed during fit is consistent with what is
    # returned by the perplexity method
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=1,
        learning_method="batch",
        random_state=0,
        evaluate_every=1,
    )
    lda.fit(X)

    # Perplexity computed at end of fit method
    perplexity1 = lda.bound_

    # Result of perplexity method on the train set
    perplexity2 = lda.perplexity(X)

    assert_almost_equal(perplexity1, perplexity2)


def test_lda_empty_docs():
    """Test LDA on empty document (all-zero rows)."""
    Z = np.zeros((5, 4))
    for X in [Z, csr_matrix(Z)]:
        lda = LatentDirichletAllocation(max_iter=750).fit(X)
        assert_almost_equal(
            lda.components_.sum(axis=0), np.ones(lda.components_.shape[1])
        )


def test_dirichlet_expectation():
    """Test Cython version of Dirichlet expectation calculation."""
    x = np.logspace(-100, 10, 10000)
    expectation = np.empty_like(x)
    _dirichlet_expectation_1d(x, 0, expectation)
    assert_allclose(expectation, np.exp(psi(x) - psi(np.sum(x))), atol=1e-19)

    x = x.reshape(100, 100)
    assert_allclose(
        _dirichlet_expectation_2d(x),
        psi(x) - psi(np.sum(x, axis=1)[:, np.newaxis]),
        rtol=1e-11,
        atol=3e-9,
    )


def check_verbosity(verbose, evaluate_every, expected_lines, expected_perplexities):
    n_components, X = _build_sparse_mtx()
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=3,
        learning_method="batch",
        verbose=verbose,
        evaluate_every=evaluate_every,
        random_state=0,
    )
    out = StringIO()
    old_out, sys.stdout = sys.stdout, out
    try:
        lda.fit(X)
    finally:
        sys.stdout = old_out

    n_lines = out.getvalue().count("\n")
    n_perplexity = out.getvalue().count("perplexity")
    assert expected_lines == n_lines
    assert expected_perplexities == n_perplexity


@pytest.mark.parametrize(
    "verbose,evaluate_every,expected_lines,expected_perplexities",
    [
        (False, 1, 0, 0),
        (False, 0, 0, 0),
        (True, 0, 3, 0),
        (True, 1, 3, 3),
        (True, 2, 3, 1),
    ],
)
def test_verbosity(verbose, evaluate_every, expected_lines, expected_perplexities):
    check_verbosity(verbose, evaluate_every, expected_lines, expected_perplexities)
