import numpy as np
from pytest import raises as assert_raises

from scipy.sparse.linalg import utils


def test_make_system_bad_shape():
    assert_raises(ValueError, utils.make_system, np.zeros((5,3)), None, np.zeros(4), np.zeros(4))
