# Authors: Christian Lorentzen <lorentzen.ch@gmail.com>
#
# License: BSD 3 clause
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import check_grad

from sklearn.linear_model._glm.link import (
    IdentityLink,
    LogLink,
    LogitLink,
)


LINK_FUNCTIONS = [IdentityLink, LogLink, LogitLink]


@pytest.mark.parametrize("Link", LINK_FUNCTIONS)
def test_link_properties(Link):
    """Test link inverse and derivative."""
    rng = np.random.RandomState(42)
    x = rng.rand(100) * 100
    link = Link()
    if isinstance(link, LogitLink):
        # careful for large x, note expit(36) = 1
        # limit max eta to 15
        x = x / 100 * 15
    assert_allclose(link(link.inverse(x)), x)
    # if g(h(x)) = x, then g'(h(x)) = 1/h'(x)
    # g = link, h = link.inverse
    assert_allclose(link.derivative(link.inverse(x)), 1 / link.inverse_derivative(x))


@pytest.mark.parametrize("Link", LINK_FUNCTIONS)
def test_link_derivative(Link):
    link = Link()
    x = np.random.RandomState(0).rand(1)
    err = check_grad(link, link.derivative, x) / link.derivative(x)
    assert abs(err) < 1e-6

    err = check_grad(link.inverse, link.inverse_derivative, x) / link.derivative(x)
    assert abs(err) < 1e-6
