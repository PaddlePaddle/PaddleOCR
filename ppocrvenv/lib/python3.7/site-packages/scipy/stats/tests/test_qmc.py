import os
from collections import Counter

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_equal, assert_array_almost_equal,
                           assert_array_equal)
from scipy.stats import shapiro

from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (van_der_corput, n_primes, primes_from_2_to,
                              update_discrepancy, QMCEngine)
from scipy._lib._pep440 import Version


class TestUtils:
    def test_scale(self):
        # 1d scalar
        space = [[0], [1], [0.5]]
        out = [[-2], [6], [2]]
        scaled_space = qmc.scale(space, l_bounds=-2, u_bounds=6)

        assert_allclose(scaled_space, out)

        # 2d space
        space = [[0, 0], [1, 1], [0.5, 0.5]]
        bounds = np.array([[-2, 0], [6, 5]])
        out = [[-2, 0], [6, 5], [2, 2.5]]

        scaled_space = qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        assert_allclose(scaled_space, out)

        scaled_back_space = qmc.scale(scaled_space, l_bounds=bounds[0],
                                      u_bounds=bounds[1], reverse=True)
        assert_allclose(scaled_back_space, space)

        # broadcast
        space = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]
        l_bounds, u_bounds = 0, [6, 5, 3]
        out = [[0, 0, 0], [6, 5, 3], [3, 2.5, 1.5]]

        scaled_space = qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        assert_allclose(scaled_space, out)

    def test_scale_random(self):
        np.random.seed(0)
        sample = np.random.rand(30, 10)
        a = -np.random.rand(10) * 10
        b = np.random.rand(10) * 10
        scaled = qmc.scale(sample, a, b, reverse=False)
        unscaled = qmc.scale(scaled, a, b, reverse=True)
        assert_allclose(unscaled, sample)

    def test_scale_errors(self):
        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            space = [0, 1, 0.5]
            qmc.scale(space, l_bounds=-2, u_bounds=6)

        with pytest.raises(ValueError, match=r"Bounds are not consistent"
                                             r" a < b"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 6], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"shape mismatch: objects cannot "
                                             r"be broadcast to a "
                                             r"single shape"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            l_bounds, u_bounds = [-2, 0, 2], [6, 5]
            qmc.scale(space, l_bounds=l_bounds, u_bounds=u_bounds)

        with pytest.raises(ValueError, match=r"Sample dimension is different "
                                             r"than bounds dimension"):
            space = [[0, 0], [1, 1], [0.5, 0.5]]
            bounds = np.array([[-2, 0, 2], [6, 5, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"Sample is not in unit "
                                             r"hypercube"):
            space = [[0, 0], [1, 1.5], [0.5, 0.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(space, l_bounds=bounds[0], u_bounds=bounds[1])

        with pytest.raises(ValueError, match=r"Sample is out of bounds"):
            out = [[-2, 0], [6, 5], [8, 2.5]]
            bounds = np.array([[-2, 0], [6, 5]])
            qmc.scale(out, l_bounds=bounds[0], u_bounds=bounds[1],
                      reverse=True)

    def test_discrepancy(self):
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)
        space_2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]])
        space_2 = (2.0 * space_2 - 1.0) / (2.0 * 6.0)

        # From Fang et al. Design and modeling for computer experiments, 2006
        assert_allclose(qmc.discrepancy(space_1), 0.0081, atol=1e-4)
        assert_allclose(qmc.discrepancy(space_2), 0.0105, atol=1e-4)

        # From Zhou Y.-D. et al. Mixture discrepancy for quasi-random point
        # sets. Journal of Complexity, 29 (3-4), pp. 283-301, 2013.
        # Example 4 on Page 298
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        assert_allclose(qmc.discrepancy(sample, method='MD'), 2.5000,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD'), 1.3680,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD'), 0.3172,
                        atol=1e-4)

        # From Tim P. et al. Minimizing the L2 and Linf star discrepancies
        # of a single point in the unit hypercube. JCAM, 2005
        # Table 1 on Page 283
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3**(-dim))
            assert_allclose(qmc.discrepancy(np.array([[1]*dim]),
                                            method='L2-star'), ref)

    def test_discrepancy_errors(self):
        sample = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])

        with pytest.raises(
            ValueError, match=r"Sample is not in unit hypercube"
        ):
            qmc.discrepancy(sample)

        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            qmc.discrepancy([1, 3])

        sample = [[0, 0], [1, 1], [0.5, 0.5]]
        with pytest.raises(ValueError, match=r"'toto' is not a valid ..."):
            qmc.discrepancy(sample, method="toto")

    def test_discrepancy_parallel(self, monkeypatch):
        sample = np.array([[2, 1, 1, 2, 2, 2],
                           [1, 2, 2, 2, 2, 2],
                           [2, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 2, 2],
                           [1, 2, 2, 2, 1, 1],
                           [2, 2, 2, 2, 1, 1],
                           [2, 2, 2, 1, 2, 2]])
        sample = (2.0 * sample - 1.0) / (2.0 * 2.0)

        assert_allclose(qmc.discrepancy(sample, method='MD', workers=8),
                        2.5000,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='WD', workers=8),
                        1.3680,
                        atol=1e-4)
        assert_allclose(qmc.discrepancy(sample, method='CD', workers=8),
                        0.3172,
                        atol=1e-4)

        # From Tim P. et al. Minimizing the L2 and Linf star discrepancies
        # of a single point in the unit hypercube. JCAM, 2005
        # Table 1 on Page 283
        for dim in [2, 4, 8, 16, 32, 64]:
            ref = np.sqrt(3 ** (-dim))
            assert_allclose(qmc.discrepancy(np.array([[1] * dim]),
                                            method='L2-star', workers=-1), ref)

        monkeypatch.setattr(os, 'cpu_count', lambda: None)
        with pytest.raises(NotImplementedError, match="Cannot determine the"):
            qmc.discrepancy(sample, workers=-1)

        with pytest.raises(ValueError, match="Invalid number of workers..."):
            qmc.discrepancy(sample, workers=-2)

    @pytest.mark.skipif(Version(np.__version__) < Version('1.17'),
                        reason='default_rng not available for numpy, < 1.17')
    def test_update_discrepancy(self):
        # From Fang et al. Design and modeling for computer experiments, 2006
        space_1 = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
        space_1 = (2.0 * space_1 - 1.0) / (2.0 * 6.0)

        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        assert_allclose(disc_iter, 0.0081, atol=1e-4)

        # n<d
        rng = np.random.default_rng(241557431858162136881731220526394276199)
        space_1 = rng.random((4, 10))

        disc_ref = qmc.discrepancy(space_1)
        disc_init = qmc.discrepancy(space_1[:-1], iterative=True)
        disc_iter = update_discrepancy(space_1[-1], space_1[:-1], disc_init)

        assert_allclose(disc_iter, disc_ref, atol=1e-4)

        # errors
        with pytest.raises(ValueError, match=r"Sample is not in unit "
                                             r"hypercube"):
            update_discrepancy(space_1[-1], space_1[:-1] + 1, disc_init)

        with pytest.raises(ValueError, match=r"Sample is not a 2D array"):
            update_discrepancy(space_1[-1], space_1[0], disc_init)

        x_new = [1, 3]
        with pytest.raises(ValueError, match=r"x_new is not in unit "
                                             r"hypercube"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        x_new = [[0.5, 0.5]]
        with pytest.raises(ValueError, match=r"x_new is not a 1D array"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

        x_new = [0.3, 0.1, 0]
        with pytest.raises(ValueError, match=r"x_new and sample must be "
                                             r"broadcastable"):
            update_discrepancy(x_new, space_1[:-1], disc_init)

    def test_discrepancy_alternative_implementation(self):
        """Alternative definitions from Matt Haberland."""
        def disc_c2(x):
            n, s = x.shape
            xij = x
            disc1 = np.sum(np.prod((1
                                    + 1/2*np.abs(xij-0.5)
                                    - 1/2*np.abs(xij-0.5)**2), axis=1))
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc2 = np.sum(np.sum(np.prod(1
                                          + 1/2*np.abs(xij - 0.5)
                                          + 1/2*np.abs(xkj - 0.5)
                                          - 1/2*np.abs(xij - xkj), axis=2),
                                  axis=0))
            return (13/12)**s - 2/n * disc1 + 1/n**2*disc2

        def disc_wd(x):
            n, s = x.shape
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc = np.sum(np.sum(np.prod(3/2
                                         - np.abs(xij - xkj)
                                         + np.abs(xij - xkj)**2, axis=2),
                                 axis=0))
            return -(4/3)**s + 1/n**2 * disc

        def disc_md(x):
            n, s = x.shape
            xij = x
            disc1 = np.sum(np.prod((5/3
                                    - 1/4*np.abs(xij-0.5)
                                    - 1/4*np.abs(xij-0.5)**2), axis=1))
            xij = x[None, :, :]
            xkj = x[:, None, :]
            disc2 = np.sum(np.sum(np.prod(15/8
                                          - 1/4*np.abs(xij - 0.5)
                                          - 1/4*np.abs(xkj - 0.5)
                                          - 3/4*np.abs(xij - xkj)
                                          + 1/2*np.abs(xij - xkj)**2,
                                          axis=2), axis=0))
            return (19/12)**s - 2/n * disc1 + 1/n**2*disc2

        def disc_star_l2(x):
            n, s = x.shape
            return np.sqrt(
                3 ** (-s) - 2 ** (1 - s) / n
                * np.sum(np.prod(1 - x ** 2, axis=1))
                + np.sum([
                    np.prod(1 - np.maximum(x[k, :], x[j, :]))
                    for k in range(n) for j in range(n)
                ]) / n ** 2
            )

        np.random.seed(0)
        sample = np.random.rand(30, 10)

        disc_curr = qmc.discrepancy(sample, method='CD')
        disc_alt = disc_c2(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='WD')
        disc_alt = disc_wd(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='MD')
        disc_alt = disc_md(sample)
        assert_allclose(disc_curr, disc_alt)

        disc_curr = qmc.discrepancy(sample, method='L2-star')
        disc_alt = disc_star_l2(sample)
        assert_allclose(disc_curr, disc_alt)

    def test_n_primes(self):
        primes = n_primes(10)
        assert primes[-1] == 29

        primes = n_primes(168)
        assert primes[-1] == 997

        primes = n_primes(350)
        assert primes[-1] == 2357

    def test_primes(self):
        primes = primes_from_2_to(50)
        out = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        assert_allclose(primes, out)


class TestVDC:
    def test_van_der_corput(self):
        seed = np.random.RandomState(12345)
        sample = van_der_corput(10, seed=seed)
        out = [0., 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625]
        assert_almost_equal(sample, out)

        sample = van_der_corput(7, start_index=3, seed=seed)
        assert_almost_equal(sample, out[3:])

    def test_van_der_corput_scramble(self):
        seed = np.random.RandomState(123456)
        out = van_der_corput(10, scramble=True, seed=seed)

        seed = np.random.RandomState(123456)
        sample = van_der_corput(7, start_index=3, scramble=True,
                                seed=seed)
        assert_almost_equal(sample, out[3:])


class RandomEngine(qmc.QMCEngine):
    def __init__(self, d, seed):
        super().__init__(d=d, seed=seed)

    def random(self, n=1):
        self.num_generated += n
        try:
            sample = self.rng.random((n, self.d))
        except AttributeError:
            sample = self.rng.random_sample((n, self.d))
        return sample


def test_subclassing_QMCEngine():
    seed = np.random.RandomState(123456)
    engine = RandomEngine(2, seed=seed)

    sample_1 = engine.random(n=5)
    sample_2 = engine.random(n=7)
    assert engine.num_generated == 12

    # reset and re-sample
    engine.reset()
    assert engine.num_generated == 0

    sample_1_test = engine.random(n=5)
    assert_equal(sample_1, sample_1_test)

    # repeat reset and fast forward
    engine.reset()
    engine.fast_forward(n=5)
    sample_2_test = engine.random(n=7)

    assert_equal(sample_2, sample_2_test)
    assert engine.num_generated == 12

    # input validation
    with pytest.raises(ValueError, match=r"d must be an integer value"):
        RandomEngine((2,), seed=seed)  # noqa


class QMCEngineTests:
    """Generic tests for QMC engines."""
    qmce = NotImplemented
    can_scramble = NotImplemented
    unscramble_nd = NotImplemented
    scramble_nd = NotImplemented

    scramble = [True, False]
    ids = ["Scrambled", "Unscrambled"]

    def engine(self, scramble: bool, **kwargs) -> QMCEngine:
        seed = np.random.RandomState(123456)
        if self.can_scramble:
            return self.qmce(scramble=scramble, seed=seed, **kwargs)
        else:
            if scramble:
                pytest.skip()
            else:
                return self.qmce(seed=seed, **kwargs)

    def reference(self, scramble: bool) -> np.ndarray:
        return self.scramble_nd if scramble else self.unscramble_nd

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_0dim(self, scramble):
        engine = self.engine(d=0, scramble=scramble)
        sample = engine.random(4)
        assert_array_equal(np.empty((4, 0)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_0sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(0)
        assert_array_equal(np.empty((0, 2)), sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_1sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(1)
        assert (1, 2) == sample.shape

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_bounds(self, scramble):
        engine = self.engine(d=100, scramble=scramble)
        sample = engine.random(512)
        assert np.all(sample >= 0)
        assert np.all(sample <= 1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_sample(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(n=len(ref_sample))

        assert_almost_equal(sample, ref_sample, decimal=1)
        assert engine.num_generated == len(ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_continuing(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        engine = self.engine(d=2, scramble=scramble)

        n_half = len(ref_sample) // 2

        _ = engine.random(n=n_half)
        sample = engine.random(n=n_half)
        assert_almost_equal(sample, ref_sample[n_half:], decimal=1)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_reset(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        engine.reset()
        assert engine.num_generated == 0

        sample = engine.random(n=8)
        assert_allclose(sample, ref_sample)

    @pytest.mark.parametrize("scramble", scramble, ids=ids)
    def test_fast_forward(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)

        engine = self.engine(d=2, scramble=scramble)

        engine.fast_forward(4)
        sample = engine.random(n=4)

        assert_almost_equal(sample, ref_sample[4:], decimal=1)

        # alternate fast forwarding with sampling
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.random())
            else:
                engine.fast_forward(1)
        assert_almost_equal(
            ref_sample[[i for i in range(8) if i % 2 == 0]],
            np.concatenate(even_draws),
            decimal=5
        )

    @pytest.mark.parametrize("scramble", [True])
    def test_distribution(self, scramble):
        d = 50
        engine = self.engine(d=d, scramble=scramble)
        sample = engine.random(1024)
        assert_array_almost_equal(
            np.mean(sample, axis=0), np.repeat(0.5, d), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 25, axis=0), np.repeat(0.25, d), decimal=2
        )
        assert_array_almost_equal(
            np.percentile(sample, 75, axis=0), np.repeat(0.75, d), decimal=2
        )


class TestHalton(QMCEngineTests):
    qmce = qmc.Halton
    can_scramble = True
    # theoretical values known from Van der Corput
    unscramble_nd = np.array([[0, 0], [1 / 2, 1 / 3],
                              [1 / 4, 2 / 3], [3 / 4, 1 / 9],
                              [1 / 8, 4 / 9], [5 / 8, 7 / 9],
                              [3 / 8, 2 / 9], [7 / 8, 5 / 9]])
    # theoretical values unknown: convergence properties checked
    scramble_nd = np.array([[0.34229571, 0.89178423],
                            [0.84229571, 0.07696942],
                            [0.21729571, 0.41030275],
                            [0.71729571, 0.74363609],
                            [0.46729571, 0.18808053],
                            [0.96729571, 0.52141386],
                            [0.06104571, 0.8547472],
                            [0.56104571, 0.29919164]])


class TestLHS(QMCEngineTests):
    qmce = qmc.LatinHypercube
    can_scramble = False

    def test_continuing(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    def test_fast_forward(self, *args):
        pytest.skip("Not applicable: not a sequence.")

    def test_sample(self, *args):
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    def test_sample_stratified(self):
        d, n = 4, 20
        expected1d = (np.arange(n) + 0.5) / n
        expected = np.broadcast_to(expected1d, (d, n)).T

        engine = self.engine(d=d, scramble=False, centered=True)
        sample = engine.random(n=n)
        sorted_sample = np.sort(sample, axis=0)

        assert_equal(sorted_sample, expected)
        assert np.any(sample != expected)

        engine = self.engine(d=d, scramble=False, centered=False)
        sample = engine.random(n=n)
        sorted_sample = np.sort(sample, axis=0)
        assert_allclose(sorted_sample, expected, atol=0.5 / n)
        assert np.any(sample - expected > 0.5 / n)


class TestSobol(QMCEngineTests):
    qmce = qmc.Sobol
    can_scramble = True
    # theoretical values from Joe Kuo2010
    unscramble_nd = np.array([[0., 0.],
                              [0.5, 0.5],
                              [0.75, 0.25],
                              [0.25, 0.75],
                              [0.375, 0.375],
                              [0.875, 0.875],
                              [0.625, 0.125],
                              [0.125, 0.625]])

    # theoretical values unknown: convergence properties checked
    scramble_nd = np.array([[0.50860737, 0.29320504],
                            [0.07116939, 0.89594537],
                            [0.49354145, 0.11524881],
                            [0.93097717, 0.70244044],
                            [0.87266153, 0.23887917],
                            [0.31021884, 0.57600391],
                            [0.13687253, 0.42054182],
                            [0.69931293, 0.77336788]])

    def test_warning(self):
        with pytest.warns(UserWarning, match=r"The balance properties of "
                                             r"Sobol' points"):
            seed = np.random.RandomState(12345)
            engine = qmc.Sobol(1, seed=seed)
            engine.random(10)

    def test_random_base2(self):
        seed = np.random.RandomState(12345)
        engine = qmc.Sobol(2, scramble=False, seed=seed)
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[:4],
                           sample)

        # resampling still having N=2**n
        sample = engine.random_base2(2)
        assert_array_equal(self.unscramble_nd[4:8],
                           sample)

        # resampling again but leading to N!=2**n
        with pytest.raises(ValueError, match=r"The balance properties of "
                                             r"Sobol' points"):
            engine.random_base2(2)

    def test_raise(self):
        seed = np.random.RandomState(12345)
        with pytest.raises(ValueError, match=r"Maximum supported "
                                             r"dimensionality"):
            qmc.Sobol(qmc.Sobol.MAXDIM + 1, seed=seed)

    def test_high_dim(self):
        seed = np.random.RandomState(12345)
        engine = qmc.Sobol(1111, scramble=False, seed=seed)
        count1 = Counter(engine.random().flatten().tolist())
        count2 = Counter(engine.random().flatten().tolist())
        assert_equal(count1, Counter({0.0: 1111}))
        assert_equal(count2, Counter({0.5: 1111}))


class TestMultinomialQMC:
    def test_validations(self):
        # negative Ps
        p = np.array([0.12, 0.26, -0.05, 0.35, 0.22])
        with pytest.raises(ValueError, match=r"Elements of pvals must "
                                             r"be non-negative."):
            qmc.MultinomialQMC(p)

        # sum of P too large
        p = np.array([0.12, 0.26, 0.1, 0.35, 0.22])
        message = r"Elements of pvals must sum to 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p)

        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        seed = np.random.RandomState(12345)
        message = r"Dimension of `engine` must be 1."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, engine=qmc.Sobol(d=2, seed=seed))

        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, engine=np.random.RandomState)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_MultinomialBasicDraw(self):
        seed = np.random.RandomState(12345)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        expected = np.array([12, 25, 6, 35, 22])
        engine = qmc.MultinomialQMC(p, seed=seed)
        assert_array_equal(engine.random(100), expected)

    def test_MultinomialDistribution(self):
        seed = np.random.RandomState(12345)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        engine = qmc.MultinomialQMC(p, seed=seed)
        draws = engine.random(8192)
        assert_array_almost_equal(draws / np.sum(draws), p, decimal=4)

    def test_FindIndex(self):
        p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
        size = len(p_cumulative)
        assert_equal(_test_find_index(p_cumulative, size, 0.0), 0)
        assert_equal(_test_find_index(p_cumulative, size, 0.4), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.44999), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.45001), 3)
        assert_equal(_test_find_index(p_cumulative, size, 1.0), size - 1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_other_engine(self):
        # same as test_MultinomialBasicDraw with different engine
        seed = np.random.RandomState(12345)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        expected = np.array([12, 25, 6, 35, 22])
        base_engine = qmc.Sobol(1, scramble=True, seed=seed)
        engine = qmc.MultinomialQMC(p, engine=base_engine, seed=seed)
        assert_array_equal(engine.random(100), expected)

    def test_reset(self):
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        seed = np.random.RandomState(12345)
        engine = qmc.MultinomialQMC(p, seed=seed)
        samples = engine.random(2)
        engine.reset()
        samples_reset = engine.random(2)
        assert_array_equal(samples, samples_reset)


def _wrapper_mv_qmc(*args, **kwargs):
    d = kwargs.pop("d")
    return qmc.MultivariateNormalQMC(mean=np.zeros(d), **kwargs)


class TestMultivariateNormalQMCEngine(QMCEngineTests):
    qmce = _wrapper_mv_qmc
    can_scramble = False

    def test_sample(self, *args):
        pytest.skip("Not applicable: the value of reference sample is"
                    " implementation dependent.")

    def test_bounds(self, *args):
        pytest.skip("Not applicable: normal is not bounded.")


class TestNormalQMC:
    def test_NormalQMC(self):
        # d = 1
        seed = np.random.RandomState(123456)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(1), seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        # d = 2
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_NormalQMCInvTransform(self):
        # d = 1
        seed = np.random.RandomState(123456)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(1), inv_transform=True, seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))
        # d = 2
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=True, seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

    def test_other_engine(self):
        seed = np.random.RandomState(123456)
        base_engine = qmc.Sobol(d=2, scramble=False, seed=seed)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2),
                                           engine=base_engine,
                                           inv_transform=True, seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))

    def test_NormalQMCSeeded(self):
        # test even dimension
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [[-0.943472, 0.405116], [-0.63099602, -1.32950772]]
        )
        assert_array_almost_equal(samples, samples_expected)

        # test odd dimension
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [
                [-0.943472, 0.405116, 0.268828],
                [1.83169884, -1.40473647, 0.24334828],
            ]
        )
        assert_array_almost_equal(samples, samples_expected)

    def test_NormalQMCSeededInvTransform(self):
        # test even dimension
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [[0.228309, -0.162516], [-0.41622922, 0.46622792]]
        )
        assert_array_almost_equal(samples, samples_expected)

        # test odd dimension
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(3), seed=seed, inv_transform=True)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [
                [0.228309, -0.162516, 0.167352],
                [-1.40525266, 1.37652443, -0.8519666],
            ]
        )
        assert_array_almost_equal(samples, samples_expected)

    def test_NormalQMCShapiro(self):
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(2), seed=seed)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

    def test_NormalQMCShapiroInvTransform(self):
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=np.zeros(2), seed=seed, inv_transform=True)
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2


class TestMultivariateNormalQMC:

    def test_validations(self):
        seed = np.random.RandomState()

        message = r"Dimension of `engine` must be consistent"
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], engine=qmc.Sobol(d=2, seed=seed),
                                      seed=seed)

        message = r"`engine` must be an instance of..."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], engine=np.random.RandomState,
                                      seed=seed)

        message = r"Covariance matrix not PSD."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 2], [2, 1]], seed=seed)

        message = r"Covariance matrix is not symmetric."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0, 0], [[1, 0], [2, 1]], seed=seed)

        message = r"Dimension mismatch between mean and covariance."
        with pytest.raises(ValueError, match=message):
            qmc.MultivariateNormalQMC([0], [[1, 0], [0, 1]], seed=seed)

    def test_MultivariateNormalQMCNonPD(self):
        # try with non-pd but psd cov; should work
        seed = np.random.RandomState(123456)
        engine = qmc.MultivariateNormalQMC(
            [0, 0, 0], [[1, 0, 1], [0, 1, 1], [1, 1, 2]],
            seed=seed
        )
        assert engine._corr_matrix is not None

    def test_MultivariateNormalQMC(self):
        # d = 1 scalar
        seed = np.random.RandomState(123456)
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5, seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # d = 2 list
        engine = qmc.MultivariateNormalQMC(mean=[0, 1], cov=[[1, 0], [0, 1]],
                                           seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov, seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCInvTransform(self):
        # d = 1 scalar
        seed = np.random.RandomState(123456)
        engine = qmc.MultivariateNormalQMC(mean=0, cov=5, inv_transform=True,
                                           seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 1))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 1))

        # d = 2 list
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 1], cov=[[1, 0], [0, 1]], inv_transform=True,
            seed=seed
        )
        samples = engine.random()
        assert_equal(samples.shape, (1, 2))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 2))

        # d = 3 np.array
        mean = np.array([0, 1, 2])
        cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        engine = qmc.MultivariateNormalQMC(mean, cov, inv_transform=True,
                                           seed=seed)
        samples = engine.random()
        assert_equal(samples.shape, (1, 3))
        samples = engine.random(n=5)
        assert_equal(samples.shape, (5, 3))

    def test_MultivariateNormalQMCSeeded(self):
        # test even dimension
        seed = np.random.RandomState(12345)
        np.random.seed(54321)
        a = np.random.randn(2, 2)
        A = a @ a.transpose() + np.diag(np.random.rand(2))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0]), A,
                                           inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [[-1.010703, -0.324223], [-0.67595995, -2.27437872]]
        )
        assert_array_almost_equal(samples, samples_expected)

        # test odd dimension
        seed = np.random.RandomState(12345)
        np.random.seed(54321)
        a = np.random.randn(3, 3)
        A = a @ a.transpose() + np.diag(np.random.rand(3))
        engine = qmc.MultivariateNormalQMC(np.array([0, 0, 0]), A,
                                           inv_transform=False, seed=seed)
        samples = engine.random(n=2)
        samples_expected = np.array(
            [
                [-1.056834, 2.493251, 0.114556],
                [2.05178452, -6.35744194, 0.67944512],
            ]
        )
        assert_array_almost_equal(samples, samples_expected)

    def test_MultivariateNormalQMCSeededInvTransform(self):
        # test even dimension
        seed = np.random.RandomState(12345)
        np.random.seed(54321)
        a = np.random.randn(2, 2)
        A = a @ a.transpose() + np.diag(np.random.rand(2))
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0]), A, seed=seed, inv_transform=True
        )
        samples = engine.random(n=2)
        samples_expected = np.array(
            [[0.244578, -0.004441], [-0.44588916, 0.22657776]]
        )
        assert_array_almost_equal(samples, samples_expected)

        # test odd dimension
        seed = np.random.RandomState(12345)
        np.random.seed(54321)
        a = np.random.randn(3, 3)
        A = a @ a.transpose() + np.diag(np.random.rand(3))
        engine = qmc.MultivariateNormalQMC(
            np.array([0, 0, 0]), A, seed=seed, inv_transform=True
        )
        samples = engine.random(n=2)
        samples_expected = np.array(
            [
                [0.255741, -0.761559, 0.234236],
                [-1.5740992, 5.61057598, -1.28218525],
            ]
        )
        assert_array_almost_equal(samples, samples_expected)

    def test_MultivariateNormalQMCShapiro(self):
        # test the standard case
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

        # test the correlated, non-zero mean case
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0], cov=[[1.5, 0.5], [0.5, 1.5]], seed=seed
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # check covariance
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 1e-2

    def test_MultivariateNormalQMCShapiroInvTransform(self):
        # test the standard case
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=[0, 0], cov=[[1, 0], [0, 1]], seed=seed, inv_transform=True
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - 1) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # make sure samples are uncorrelated
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2

        # test the correlated, non-zero mean case
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=[1.0, 2.0],
            cov=[[1.5, 0.5], [0.5, 1.5]],
            seed=seed,
            inv_transform=True,
        )
        samples = engine.random(n=256)
        assert all(np.abs(samples.mean(axis=0) - [1, 2]) < 1e-2)
        assert all(np.abs(samples.std(axis=0) - np.sqrt(1.5)) < 1e-2)
        # perform Shapiro-Wilk test for normality
        for i in (0, 1):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.9
        # check covariance
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1] - 0.5) < 1e-2

    def test_MultivariateNormalQMCDegenerate(self):
        # X, Y iid standard Normal and Z = X + Y, random vector (X, Y, Z)
        seed = np.random.RandomState(12345)
        engine = qmc.MultivariateNormalQMC(
            mean=[0.0, 0.0, 0.0],
            cov=[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
            seed=seed,
        )
        samples = engine.random(n=512)
        assert all(np.abs(samples.mean(axis=0)) < 1e-2)
        assert np.abs(np.std(samples[:, 0]) - 1) < 1e-2
        assert np.abs(np.std(samples[:, 1]) - 1) < 1e-2
        assert np.abs(np.std(samples[:, 2]) - np.sqrt(2)) < 1e-2
        for i in (0, 1, 2):
            _, pval = shapiro(samples[:, i])
            assert pval > 0.8
        cov = np.cov(samples.transpose())
        assert np.abs(cov[0, 1]) < 1e-2
        assert np.abs(cov[0, 2] - 1) < 1e-2
        # check to see if X + Y = Z almost exactly
        assert all(np.abs(samples[:, 0] + samples[:, 1] - samples[:, 2])
                   < 1e-5)
