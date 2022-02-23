import numpy as np
from numpy.testing import assert_array_equal
import pytest

from sklearn.feature_extraction import FeatureHasher
from sklearn.utils._testing import ignore_warnings, fails_if_pypy

pytestmark = fails_if_pypy


def test_feature_hasher_dicts():
    h = FeatureHasher(n_features=16)
    assert "dict" == h.input_type

    raw_X = [{"foo": "bar", "dada": 42, "tzara": 37}, {"foo": "baz", "gaga": "string1"}]
    X1 = FeatureHasher(n_features=16).transform(raw_X)
    gen = (iter(d.items()) for d in raw_X)
    X2 = FeatureHasher(n_features=16, input_type="pair").transform(gen)
    assert_array_equal(X1.toarray(), X2.toarray())


def test_feature_hasher_strings():
    # mix byte and Unicode strings; note that "foo" is a duplicate in row 0
    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    for lg_n_features in (7, 9, 11, 16, 22):
        n_features = 2 ** lg_n_features

        it = (x for x in raw_X)  # iterable

        h = FeatureHasher(
            n_features=n_features, input_type="string", alternate_sign=False
        )
        X = h.transform(it)

        assert X.shape[0] == len(raw_X)
        assert X.shape[1] == n_features

        assert X[0].sum() == 4
        assert X[1].sum() == 3

        assert X.nnz == 6


def test_hashing_transform_seed():
    # check the influence of the seed when computing the hashes
    # import is here to avoid importing on pypy
    from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform

    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices, indptr, _ = _hashing_transform(raw_X_, 2 ** 7, str, False)

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices_0, indptr_0, _ = _hashing_transform(raw_X_, 2 ** 7, str, False, seed=0)
    assert_array_equal(indices, indices_0)
    assert_array_equal(indptr, indptr_0)

    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    indices_1, _, _ = _hashing_transform(raw_X_, 2 ** 7, str, False, seed=1)
    with pytest.raises(AssertionError):
        assert_array_equal(indices, indices_1)


def test_feature_hasher_pairs():
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": 2}, {"baz": 3, "quux": 4, "foo": -1}]
    )
    h = FeatureHasher(n_features=16, input_type="pair")
    x1, x2 = h.transform(raw_X).toarray()
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    assert [1, 2] == x1_nz
    assert [1, 3, 4] == x2_nz


def test_feature_hasher_pairs_with_string_values():
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": "a"}, {"baz": "abc", "quux": 4, "foo": -1}]
    )
    h = FeatureHasher(n_features=16, input_type="pair")
    x1, x2 = h.transform(raw_X).toarray()
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    assert [1, 1] == x1_nz
    assert [1, 1, 4] == x2_nz

    raw_X = (iter(d.items()) for d in [{"bax": "abc"}, {"bax": "abc"}])
    x1, x2 = h.transform(raw_X).toarray()
    x1_nz = np.abs(x1[x1 != 0])
    x2_nz = np.abs(x2[x2 != 0])
    assert [1] == x1_nz
    assert [1] == x2_nz
    assert_array_equal(x1, x2)


def test_hash_empty_input():
    n_features = 16
    raw_X = [[], (), iter(range(0))]

    h = FeatureHasher(n_features=n_features, input_type="string")
    X = h.transform(raw_X)

    assert_array_equal(X.A, np.zeros((len(raw_X), n_features)))


def test_hasher_invalid_input():
    with pytest.raises(ValueError):
        FeatureHasher(input_type="gobbledygook")
    with pytest.raises(ValueError):
        FeatureHasher(n_features=-1)
    with pytest.raises(ValueError):
        FeatureHasher(n_features=0)
    with pytest.raises(TypeError):
        FeatureHasher(n_features="ham")

    h = FeatureHasher(n_features=np.uint16(2 ** 6))
    with pytest.raises(ValueError):
        h.transform([])
    with pytest.raises(Exception):
        h.transform([[5.5]])
    with pytest.raises(Exception):
        h.transform([[None]])


def test_hasher_set_params():
    # Test delayed input validation in fit (useful for grid search).
    hasher = FeatureHasher()
    hasher.set_params(n_features=np.inf)
    with pytest.raises(TypeError):
        hasher.fit()


def test_hasher_zeros():
    # Assert that no zeros are materialized in the output.
    X = FeatureHasher().transform([{"foo": 0}])
    assert X.data.shape == (0,)


@ignore_warnings(category=FutureWarning)
def test_hasher_alternate_sign():
    X = [list("Thequickbrownfoxjumped")]

    Xt = FeatureHasher(alternate_sign=True, input_type="string").fit_transform(X)
    assert Xt.data.min() < 0 and Xt.data.max() > 0

    Xt = FeatureHasher(alternate_sign=False, input_type="string").fit_transform(X)
    assert Xt.data.min() > 0


def test_hash_collisions():
    X = [list("Thequickbrownfoxjumped")]

    Xt = FeatureHasher(
        alternate_sign=True, n_features=1, input_type="string"
    ).fit_transform(X)
    # check that some of the hashed tokens are added
    # with an opposite sign and cancel out
    assert abs(Xt.data[0]) < len(X[0])

    Xt = FeatureHasher(
        alternate_sign=False, n_features=1, input_type="string"
    ).fit_transform(X)
    assert Xt.data[0] == len(X[0])
