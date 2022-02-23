"""Test the rcv1 loader, if the data is available,
or if specifically requested via environment variable
(e.g. for travis cron job)."""

import scipy.sparse as sp
import numpy as np
from functools import partial
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_equal


def test_fetch_rcv1(fetch_rcv1_fxt):
    data1 = fetch_rcv1_fxt(shuffle=False)
    X1, Y1 = data1.data, data1.target
    cat_list, s1 = data1.target_names.tolist(), data1.sample_id

    # test sparsity
    assert sp.issparse(X1)
    assert sp.issparse(Y1)
    assert 60915113 == X1.data.size
    assert 2606875 == Y1.data.size

    # test shapes
    assert (804414, 47236) == X1.shape
    assert (804414, 103) == Y1.shape
    assert (804414,) == s1.shape
    assert 103 == len(cat_list)

    # test descr
    assert data1.DESCR.startswith(".. _rcv1_dataset:")

    # test ordering of categories
    first_categories = ["C11", "C12", "C13", "C14", "C15", "C151"]
    assert_array_equal(first_categories, cat_list[:6])

    # test number of sample for some categories
    some_categories = ("GMIL", "E143", "CCAT")
    number_non_zero_in_cat = (5, 1206, 381327)
    for num, cat in zip(number_non_zero_in_cat, some_categories):
        j = cat_list.index(cat)
        assert num == Y1[:, j].data.size

    # test shuffling and subset
    data2 = fetch_rcv1_fxt(shuffle=True, subset="train", random_state=77)
    X2, Y2 = data2.data, data2.target
    s2 = data2.sample_id

    # test return_X_y option
    fetch_func = partial(fetch_rcv1_fxt, shuffle=False, subset="train")
    check_return_X_y(data2, fetch_func)

    # The first 23149 samples are the training samples
    assert_array_equal(np.sort(s1[:23149]), np.sort(s2))

    # test some precise values
    some_sample_ids = (2286, 3274, 14042)
    for sample_id in some_sample_ids:
        idx1 = s1.tolist().index(sample_id)
        idx2 = s2.tolist().index(sample_id)

        feature_values_1 = X1[idx1, :].toarray()
        feature_values_2 = X2[idx2, :].toarray()
        assert_almost_equal(feature_values_1, feature_values_2)

        target_values_1 = Y1[idx1, :].toarray()
        target_values_2 = Y2[idx2, :].toarray()
        assert_almost_equal(target_values_1, target_values_2)
