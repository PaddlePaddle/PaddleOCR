"""Tests for _sketches.py."""

import numpy as np
from numpy.testing import assert_, assert_equal
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from scipy.sparse import issparse, rand
from scipy.sparse.linalg import norm


class TestClarksonWoodruffTransform:
    """
    Testing the Clarkson Woodruff Transform
    """
    # set seed for generating test matrices
    rng = np.random.RandomState(seed=1179103485)

    # Test matrix parameters
    n_rows = 2000
    n_cols = 100
    density = 0.1

    # Sketch matrix dimensions
    n_sketch_rows = 200

    # Seeds to test with
    seeds = [1755490010, 934377150, 1391612830, 1752708722, 2008891431,
             1302443994, 1521083269, 1501189312, 1126232505, 1533465685]

    A_dense = rng.randn(n_rows, n_cols)
    A_csc = rand(
        n_rows, n_cols, density=density, format='csc', random_state=rng,
    )
    A_csr = rand(
        n_rows, n_cols, density=density, format='csr', random_state=rng,
    )
    A_coo = rand(
        n_rows, n_cols, density=density, format='coo', random_state=rng,
    )

    # Collect the test matrices
    test_matrices = [
        A_dense, A_csc, A_csr, A_coo,
    ]

    # Test vector with norm ~1
    x = rng.randn(n_rows, 1) / np.sqrt(n_rows)

    def test_sketch_dimensions(self):
        for A in self.test_matrices:
            for seed in self.seeds:
                sketch = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )
                assert_(sketch.shape == (self.n_sketch_rows, self.n_cols))

    def test_seed_returns_identical_transform_matrix(self):
        for A in self.test_matrices:
            for seed in self.seeds:
                S1 = cwt_matrix(
                    self.n_sketch_rows, self.n_rows, seed=seed
                ).todense()
                S2 = cwt_matrix(
                    self.n_sketch_rows, self.n_rows, seed=seed
                ).todense()
                assert_equal(S1, S2)

    def test_seed_returns_identically(self):
        for A in self.test_matrices:
            for seed in self.seeds:
                sketch1 = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )
                sketch2 = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed
                )
                if issparse(sketch1):
                    sketch1 = sketch1.todense()
                if issparse(sketch2):
                    sketch2 = sketch2.todense()
                assert_equal(sketch1, sketch2)

    def test_sketch_preserves_frobenius_norm(self):
        # Given the probabilistic nature of the sketches
        # we run the test multiple times and check that
        # we pass all/almost all the tries.
        n_errors = 0
        for A in self.test_matrices:
            if issparse(A):
                true_norm = norm(A)
            else:
                true_norm = np.linalg.norm(A)
            for seed in self.seeds:
                sketch = clarkson_woodruff_transform(
                    A, self.n_sketch_rows, seed=seed,
                )
                if issparse(sketch):
                    sketch_norm = norm(sketch)
                else:
                    sketch_norm = np.linalg.norm(sketch)

                if np.abs(true_norm - sketch_norm) > 0.1 * true_norm:
                    n_errors += 1
        assert_(n_errors == 0)

    def test_sketch_preserves_vector_norm(self):
        n_errors = 0
        n_sketch_rows = int(np.ceil(2. / (0.01 * 0.5**2)))
        true_norm = np.linalg.norm(self.x)
        for seed in self.seeds:
            sketch = clarkson_woodruff_transform(
                self.x, n_sketch_rows, seed=seed,
            )
            sketch_norm = np.linalg.norm(sketch)

            if np.abs(true_norm - sketch_norm) > 0.5 * true_norm:
                n_errors += 1
        assert_(n_errors == 0)
