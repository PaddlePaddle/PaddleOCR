# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause
import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)
from scipy.optimize import check_grad
import pytest

from sklearn._loss.glm_distribution import (
    TweedieDistribution,
    NormalDistribution,
    PoissonDistribution,
    GammaDistribution,
    InverseGaussianDistribution,
    DistributionBoundary,
)


@pytest.mark.parametrize(
    "family, expected",
    [
        (NormalDistribution(), [True, True, True]),
        (PoissonDistribution(), [False, True, True]),
        (TweedieDistribution(power=1.5), [False, True, True]),
        (GammaDistribution(), [False, False, True]),
        (InverseGaussianDistribution(), [False, False, True]),
        (TweedieDistribution(power=4.5), [False, False, True]),
    ],
)
def test_family_bounds(family, expected):
    """Test the valid range of distributions at -1, 0, 1."""
    result = family.in_y_range([-1, 0, 1])
    assert_array_equal(result, expected)


def test_invalid_distribution_bound():
    dist = TweedieDistribution()
    dist._lower_bound = 0
    with pytest.raises(TypeError, match="must be of type DistributionBoundary"):
        dist.in_y_range([-1, 0, 1])


def test_tweedie_distribution_power():
    msg = "distribution is only defined for power<=0 and power>=1"
    with pytest.raises(ValueError, match=msg):
        TweedieDistribution(power=0.5)

    with pytest.raises(TypeError, match="must be a real number"):
        TweedieDistribution(power=1j)

    with pytest.raises(TypeError, match="must be a real number"):
        dist = TweedieDistribution()
        dist.power = 1j

    dist = TweedieDistribution()
    assert isinstance(dist._lower_bound, DistributionBoundary)

    assert dist._lower_bound.inclusive is False
    dist.power = 1
    assert dist._lower_bound.value == 0.0
    assert dist._lower_bound.inclusive is True


@pytest.mark.parametrize(
    "family, chk_values",
    [
        (NormalDistribution(), [-1.5, -0.1, 0.1, 2.5]),
        (PoissonDistribution(), [0.1, 1.5]),
        (GammaDistribution(), [0.1, 1.5]),
        (InverseGaussianDistribution(), [0.1, 1.5]),
        (TweedieDistribution(power=-2.5), [0.1, 1.5]),
        (TweedieDistribution(power=-1), [0.1, 1.5]),
        (TweedieDistribution(power=1.5), [0.1, 1.5]),
        (TweedieDistribution(power=2.5), [0.1, 1.5]),
        (TweedieDistribution(power=-4), [0.1, 1.5]),
    ],
)
def test_deviance_zero(family, chk_values):
    """Test deviance(y,y) = 0 for different families."""
    for x in chk_values:
        assert_allclose(family.deviance(x, x), 0, atol=1e-9)


@pytest.mark.parametrize(
    "family",
    [
        NormalDistribution(),
        PoissonDistribution(),
        GammaDistribution(),
        InverseGaussianDistribution(),
        TweedieDistribution(power=-2.5),
        TweedieDistribution(power=-1),
        TweedieDistribution(power=1.5),
        TweedieDistribution(power=2.5),
        TweedieDistribution(power=-4),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_deviance_derivative(family):
    """Test deviance derivative for different families."""
    rng = np.random.RandomState(0)
    y_true = rng.rand(10)
    # make data positive
    y_true += np.abs(y_true.min()) + 1e-2

    y_pred = y_true + np.fmax(rng.rand(10), 0.0)

    dev = family.deviance(y_true, y_pred)
    assert isinstance(dev, float)
    dev_derivative = family.deviance_derivative(y_true, y_pred)
    assert dev_derivative.shape == y_pred.shape

    err = (
        check_grad(
            lambda y_pred: family.deviance(y_true, y_pred),
            lambda y_pred: family.deviance_derivative(y_true, y_pred),
            y_pred,
        )
        / np.linalg.norm(dev_derivative)
    )
    assert abs(err) < 1e-6
