import pytest
from numpy.testing import assert_allclose

from sklearn.utils import check_random_state
from sklearn.utils._arpack import _init_arpack_v0


@pytest.mark.parametrize("seed", range(100))
def test_init_arpack_v0(seed):
    # check that the initialization a sampling from an uniform distribution
    # where we can fix the random state
    size = 1000
    v0 = _init_arpack_v0(size, seed)

    rng = check_random_state(seed)
    assert_allclose(v0, rng.uniform(-1, 1, size=size))
