import pytest

from sklearn.base import BaseEstimator
from sklearn.utils._tags import (
    _DEFAULT_TAGS,
    _safe_tags,
)


class NoTagsEstimator:
    pass


class MoreTagsEstimator:
    def _more_tags(self):
        return {"allow_nan": True}


@pytest.mark.parametrize(
    "estimator, err_msg",
    [
        (BaseEstimator(), "The key xxx is not defined in _get_tags"),
        (NoTagsEstimator(), "The key xxx is not defined in _DEFAULT_TAGS"),
    ],
)
def test_safe_tags_error(estimator, err_msg):
    # Check that safe_tags raises error in ambiguous case.
    with pytest.raises(ValueError, match=err_msg):
        _safe_tags(estimator, key="xxx")


@pytest.mark.parametrize(
    "estimator, key, expected_results",
    [
        (NoTagsEstimator(), None, _DEFAULT_TAGS),
        (NoTagsEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
        (MoreTagsEstimator(), None, {**_DEFAULT_TAGS, **{"allow_nan": True}}),
        (MoreTagsEstimator(), "allow_nan", True),
        (BaseEstimator(), None, _DEFAULT_TAGS),
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
    ],
)
def test_safe_tags_no_get_tags(estimator, key, expected_results):
    # check the behaviour of _safe_tags when an estimator does not implement
    # _get_tags
    assert _safe_tags(estimator, key=key) == expected_results
