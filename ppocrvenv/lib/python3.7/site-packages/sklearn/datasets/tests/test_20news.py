"""Test the 20news downloader, if the data is available,
or if specifically requested via environment variable
(e.g. for travis cron job)."""
from functools import partial
from unittest.mock import patch

import pytest

import numpy as np
import scipy.sparse as sp

from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.datasets.tests.test_common import check_pandas_dependency_message
from sklearn.datasets.tests.test_common import check_return_X_y
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.preprocessing import normalize


def test_20news(fetch_20newsgroups_fxt):
    data = fetch_20newsgroups_fxt(subset="all", shuffle=False)
    assert data.DESCR.startswith(".. _20newsgroups_dataset:")

    # Extract a reduced dataset
    data2cats = fetch_20newsgroups_fxt(
        subset="all", categories=data.target_names[-1:-3:-1], shuffle=False
    )
    # Check that the ordering of the target_names is the same
    # as the ordering in the full dataset
    assert data2cats.target_names == data.target_names[-2:]
    # Assert that we have only 0 and 1 as labels
    assert np.unique(data2cats.target).tolist() == [0, 1]

    # Check that the number of filenames is consistent with data/target
    assert len(data2cats.filenames) == len(data2cats.target)
    assert len(data2cats.filenames) == len(data2cats.data)

    # Check that the first entry of the reduced dataset corresponds to
    # the first entry of the corresponding category in the full dataset
    entry1 = data2cats.data[0]
    category = data2cats.target_names[data2cats.target[0]]
    label = data.target_names.index(category)
    entry2 = data.data[np.where(data.target == label)[0][0]]
    assert entry1 == entry2

    # check that return_X_y option
    X, y = fetch_20newsgroups_fxt(subset="all", shuffle=False, return_X_y=True)
    assert len(X) == len(data.data)
    assert y.shape == data.target.shape


def test_20news_length_consistency(fetch_20newsgroups_fxt):
    """Checks the length consistencies within the bunch

    This is a non-regression test for a bug present in 0.16.1.
    """
    # Extract the full dataset
    data = fetch_20newsgroups_fxt(subset="all")
    assert len(data["data"]) == len(data.data)
    assert len(data["target"]) == len(data.target)
    assert len(data["filenames"]) == len(data.filenames)


def test_20news_vectorized(fetch_20newsgroups_vectorized_fxt):
    # test subset = train
    bunch = fetch_20newsgroups_vectorized_fxt(subset="train")
    assert sp.isspmatrix_csr(bunch.data)
    assert bunch.data.shape == (11314, 130107)
    assert bunch.target.shape[0] == 11314
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")

    # test subset = test
    bunch = fetch_20newsgroups_vectorized_fxt(subset="test")
    assert sp.isspmatrix_csr(bunch.data)
    assert bunch.data.shape == (7532, 130107)
    assert bunch.target.shape[0] == 7532
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")

    # test return_X_y option
    fetch_func = partial(fetch_20newsgroups_vectorized_fxt, subset="test")
    check_return_X_y(bunch, fetch_func)

    # test subset = all
    bunch = fetch_20newsgroups_vectorized_fxt(subset="all")
    assert sp.isspmatrix_csr(bunch.data)
    assert bunch.data.shape == (11314 + 7532, 130107)
    assert bunch.target.shape[0] == 11314 + 7532
    assert bunch.data.dtype == np.float64
    assert bunch.DESCR.startswith(".. _20newsgroups_dataset:")


def test_20news_normalization(fetch_20newsgroups_vectorized_fxt):
    X = fetch_20newsgroups_vectorized_fxt(normalize=False)
    X_ = fetch_20newsgroups_vectorized_fxt(normalize=True)
    X_norm = X_["data"][:100]
    X = X["data"][:100]

    assert_allclose_dense_sparse(X_norm, normalize(X))
    assert np.allclose(np.linalg.norm(X_norm.todense(), axis=1), 1)


def test_20news_as_frame(fetch_20newsgroups_vectorized_fxt):
    pd = pytest.importorskip("pandas")

    bunch = fetch_20newsgroups_vectorized_fxt(as_frame=True)
    check_as_frame(bunch, fetch_20newsgroups_vectorized_fxt)

    frame = bunch.frame
    assert frame.shape == (11314, 130108)
    assert all([isinstance(col, pd.SparseDtype) for col in bunch.data.dtypes])

    # Check a small subset of features
    for expected_feature in [
        "beginner",
        "beginners",
        "beginning",
        "beginnings",
        "begins",
        "begley",
        "begone",
    ]:
        assert expected_feature in frame.keys()
    assert "category_class" in frame.keys()
    assert bunch.target.name == "category_class"


def test_as_frame_no_pandas(fetch_20newsgroups_vectorized_fxt, hide_available_pandas):
    check_pandas_dependency_message(fetch_20newsgroups_vectorized_fxt)


def test_outdated_pickle(fetch_20newsgroups_vectorized_fxt):
    with patch("os.path.exists") as mock_is_exist:
        with patch("joblib.load") as mock_load:
            # mock that the dataset was cached
            mock_is_exist.return_value = True
            # mock that we have an outdated pickle with only X and y returned
            mock_load.return_value = ("X", "y")
            err_msg = "The cached dataset located in"
            with pytest.raises(ValueError, match=err_msg):
                fetch_20newsgroups_vectorized_fxt(as_frame=True)
