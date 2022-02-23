""""
Test module for testing ``pandas._testing.assert_produces_warning``.
"""
import warnings

import pytest

from pandas.errors import (
    DtypeWarning,
    PerformanceWarning,
)

import pandas._testing as tm


@pytest.fixture(
    params=[
        RuntimeWarning,
        ResourceWarning,
        UserWarning,
        FutureWarning,
        DeprecationWarning,
        PerformanceWarning,
        DtypeWarning,
    ],
)
def category(request):
    """
    Return unique warning.

    Useful for testing behavior of tm.assert_produces_warning with various categories.
    """
    return request.param


@pytest.fixture(
    params=[
        (RuntimeWarning, UserWarning),
        (UserWarning, FutureWarning),
        (FutureWarning, RuntimeWarning),
        (DeprecationWarning, PerformanceWarning),
        (PerformanceWarning, FutureWarning),
        (DtypeWarning, DeprecationWarning),
        (ResourceWarning, DeprecationWarning),
        (FutureWarning, DeprecationWarning),
    ],
    ids=lambda x: type(x).__name__,
)
def pair_different_warnings(request):
    """
    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    """
    return request.param


def f():
    warnings.warn("f1", FutureWarning)
    warnings.warn("f2", RuntimeWarning)


@pytest.mark.filterwarnings("ignore:f1:FutureWarning")
def test_assert_produces_warning_honors_filter():
    # Raise by default.
    msg = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()

    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()


@pytest.mark.parametrize(
    "message, match",
    [
        ("", None),
        ("", ""),
        ("Warning message", r".*"),
        ("Warning message", "War"),
        ("Warning message", r"[Ww]arning"),
        ("Warning message", "age"),
        ("Warning message", r"age$"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"^Mes.*\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}\s\S+"),
        ("Message, which we do not match", None),
    ],
)
def test_catch_warning_category_and_match(category, message, match):
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)


@pytest.mark.parametrize(
    "message, match",
    [
        ("Warning message", "Not this message"),
        ("Warning message", "warning"),
        ("Warning message", r"\d+"),
    ],
)
def test_fail_to_match(category, message, match):
    msg = f"Did not see warning {repr(category.__name__)} matching"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn(message, category)


def test_fail_to_catch_actual_warning(pair_different_warnings):
    expected_category, actual_category = pair_different_warnings
    match = "Did not see expected warning of class"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("warning message", actual_category)


def test_ignore_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        warnings.warn("Expected warning", expected_category)
        warnings.warn("Unexpected warning OK", extra_category)


def test_raise_on_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    match = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn("Expected warning", expected_category)
            warnings.warn("Unexpected warning NOT OK", extra_category)


def test_same_category_different_messages_first_match():
    category = UserWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Match this", category)
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)


def test_same_category_different_messages_last_match():
    category = DeprecationWarning
    with tm.assert_produces_warning(category, match=r"^Match this"):
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)
        warnings.warn("Match this", category)


def test_right_category_wrong_match_raises(pair_different_warnings):
    target_category, other_category = pair_different_warnings
    with pytest.raises(AssertionError, match="Did not see warning.*matching"):
        with tm.assert_produces_warning(target_category, match=r"^Match this"):
            warnings.warn("Do not match it", target_category)
            warnings.warn("Match this", other_category)


@pytest.mark.parametrize("false_or_none", [False, None])
class TestFalseOrNoneExpectedWarning:
    def test_raise_on_warning(self, false_or_none):
        msg = r"Caused unexpected warning\(s\)"
        with pytest.raises(AssertionError, match=msg):
            with tm.assert_produces_warning(false_or_none):
                f()

    def test_no_raise_without_warning(self, false_or_none):
        with tm.assert_produces_warning(false_or_none):
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            f()
