"""Test Olivetti faces fetcher, if the data is available,
or if specifically requested via environment variable
(e.g. for travis cron job)."""

import numpy as np

from sklearn.utils import Bunch
from sklearn.datasets.tests.test_common import check_return_X_y

from sklearn.utils._testing import assert_array_equal


def test_olivetti_faces(fetch_olivetti_faces_fxt):
    data = fetch_olivetti_faces_fxt(shuffle=True, random_state=0)

    assert isinstance(data, Bunch)
    for expected_keys in ("data", "images", "target", "DESCR"):
        assert expected_keys in data.keys()

    assert data.data.shape == (400, 4096)
    assert data.images.shape == (400, 64, 64)
    assert data.target.shape == (400,)
    assert_array_equal(np.unique(np.sort(data.target)), np.arange(40))
    assert data.DESCR.startswith(".. _olivetti_faces_dataset:")

    # test the return_X_y option
    check_return_X_y(data, fetch_olivetti_faces_fxt)
