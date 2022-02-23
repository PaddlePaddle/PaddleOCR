"""
Some utility functions that are only used for unittests.
Placing them in test/ directory seems to be against convention, so they are part of the library.

"""
from __future__ import print_function, division, absolute_import

import random
import copy
import warnings
import tempfile
import shutil
import re
import sys

import numpy as np
import six.moves as sm
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock
try:
    import cPickle as pickle
except ImportError:
    import pickle

import imgaug as ia
import imgaug.random as iarandom
from imgaug.augmentables.kps import KeypointsOnImage


class ArgCopyingMagicMock(mock.MagicMock):
    """A MagicMock that copies its call args/kwargs before storing the call.

    This is useful for imgaug as many augmentation methods change data
    in-place.

    Taken from https://stackoverflow.com/a/23264042/3760780

    """

    def _mock_call(self, *args, **kwargs):
        args_copy = copy.deepcopy(args)
        kwargs_copy = copy.deepcopy(kwargs)
        return super(ArgCopyingMagicMock, self)._mock_call(
            *args_copy, **kwargs_copy)


# Added in 0.4.0.
def assert_cbaois_equal(observed, expected, max_distance=1e-4):
    # pylint: disable=unidiomatic-typecheck
    if isinstance(observed, list) or isinstance(expected, list):
        assert isinstance(observed, list)
        assert isinstance(expected, list)
        assert len(observed) == len(expected)
        for observed_i, expected_i in zip(observed, expected):
            assert_cbaois_equal(observed_i, expected_i,
                                max_distance=max_distance)
    else:
        assert type(observed) == type(expected)
        assert len(observed.items) == len(expected.items)
        assert observed.shape == expected.shape
        for item_a, item_b in zip(observed.items, expected.items):
            assert item_a.coords_almost_equals(item_b,
                                               max_distance=max_distance)
        if isinstance(expected, ia.PolygonsOnImage):
            for item_obs, item_exp in zip(observed.items, expected.items):
                if item_exp.is_valid:
                    assert item_obs.is_valid


def create_random_images(size):
    return np.random.uniform(0, 255, size).astype(np.uint8)


def create_random_keypoints(size_images, nb_keypoints_per_img):
    result = []
    for _ in sm.xrange(size_images[0]):
        kps = []
        height, width = size_images[1], size_images[2]
        for _ in sm.xrange(nb_keypoints_per_img):
            x = np.random.randint(0, width-1)
            y = np.random.randint(0, height-1)
            kps.append(ia.Keypoint(x=x, y=y))
        result.append(ia.KeypointsOnImage(kps, shape=size_images[1:]))
    return result


def array_equal_lists(list1, list2):
    assert isinstance(list1, list), (
        "Expected list1 to be a list, got type %s." % (type(list1),))
    assert isinstance(list2, list), (
        "Expected list2 to be a list, got type %s." % (type(list2),))

    if len(list1) != len(list2):
        return False

    for arr1, arr2 in zip(list1, list2):
        if not np.array_equal(arr1, arr2):
            return False

    return True


def keypoints_equal(kpsois1, kpsois2, eps=0.001):
    if isinstance(kpsois1, KeypointsOnImage):
        assert isinstance(kpsois2, KeypointsOnImage)
        kpsois1 = [kpsois1]
        kpsois2 = [kpsois2]

    if len(kpsois1) != len(kpsois2):
        return False

    for kpsoi1, kpsoi2 in zip(kpsois1, kpsois2):
        kps1 = kpsoi1.keypoints
        kps2 = kpsoi2.keypoints
        if len(kps1) != len(kps2):
            return False

        for kp1, kp2 in zip(kps1, kps2):
            x_equal = (float(kp2.x) - eps
                       <= float(kp1.x)
                       <= float(kp2.x) + eps)
            y_equal = (float(kp2.y) - eps
                       <= float(kp1.y)
                       <= float(kp2.y) + eps)
            if not x_equal or not y_equal:
                return False

    return True


def reseed(seed=0):
    iarandom.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Added in 0.4.0.
def runtest_pickleable_uint8_img(augmenter, shape=(15, 15, 3), iterations=3):
    image = np.mod(np.arange(int(np.prod(shape))), 256).astype(np.uint8)
    image = image.reshape(shape)
    augmenter_pkl = pickle.loads(pickle.dumps(augmenter, protocol=-1))

    for _ in np.arange(iterations):
        image_aug = augmenter(image=image)
        image_aug_pkl = augmenter_pkl(image=image)
        assert np.array_equal(image_aug, image_aug_pkl)


def wrap_shift_deprecation(func, *args, **kwargs):
    """Helper for tests of CBA shift() functions.

    Added in 0.4.0.

    """
    # No deprecated arguments? Just call the functions directly.
    deprecated_kwargs = ["top", "right", "bottom", "left"]
    if not any([kwname in kwargs for kwname in deprecated_kwargs]):
        return func()

    # Deprecated arguments? Log warnings and assume that there was a
    # deprecation warning with expected message.
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        result = func()

        assert (
            "These are deprecated. Use `x` and `y` instead."
            in str(caught_warnings[-1].message)
        )

        return result


class TemporaryDirectory(object):
    """Create a context for a temporary directory.

    The directory is automatically removed at the end of the context.
    This context is available in ``tmpfile.TemporaryDirectory``, but only
    from 3.2+.

    Added in 0.4.0.

    """

    def __init__(self, suffix="", prefix="tmp", dir=None):
        # pylint: disable=redefined-builtin
        self.name = tempfile.mkdtemp(suffix, prefix, dir)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.name)


# Copied from
# https://github.com/python/cpython/blob/master/Lib/unittest/case.py
# at commit 293dd23 (Nov 19, 2019).
# Required at least to enable assertWarns() in python <3.2.
# Added in 0.4.0.
def _is_subtype(expected, basetype):
    if isinstance(expected, tuple):
        return all(_is_subtype(e, basetype) for e in expected)
    return isinstance(expected, type) and issubclass(expected, basetype)


# Copied from
# https://github.com/python/cpython/blob/master/Lib/unittest/case.py
# at commit 293dd23 (Nov 19, 2019).
# Required at least to enable assertWarns() in python <3.2.
# Added in 0.4.0.
class _BaseTestCaseContext:
    # Added in 0.4.0.
    def __init__(self, test_case):
        self.test_case = test_case

    # Added in 0.4.0.
    def _raiseFailure(self, standardMsg):
        # pylint: disable=invalid-name, protected-access, no-member
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)


# Copied from
# https://github.com/python/cpython/blob/master/Lib/unittest/case.py
# at commit 293dd23 (Nov 19, 2019).
# Required at least to enable assertWarns() in python <3.2.
# Added in 0.4.0.
class _AssertRaisesBaseContext(_BaseTestCaseContext):
    # Added in 0.4.0.
    def __init__(self, expected, test_case, expected_regex=None):
        _BaseTestCaseContext.__init__(self, test_case)
        self.expected = expected
        self.test_case = test_case
        if expected_regex is not None:
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.obj_name = None
        self.msg = None

    # Added in 0.4.0.
    # pylint: disable=inconsistent-return-statements
    def handle(self, name, args, kwargs):
        """
        If args is empty, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If args is not empty, call a callable passing positional and keyword
        arguments.
        """
        # pylint: disable=no-member, self-cls-assignment, not-context-manager
        try:
            if not _is_subtype(self.expected, self._base_type):
                raise TypeError('%s() arg 1 must be %s' %
                                (name, self._base_type_str))
            if not args:
                self.msg = kwargs.pop('msg', None)
                if kwargs:
                    raise TypeError('%r is an invalid keyword argument for '
                                    'this function' % (next(iter(kwargs)),))
                return self

            callable_obj = args[0]
            args = args[1:]

            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
            with self:
                callable_obj(*args, **kwargs)
        finally:
            # bpo-23890: manually break a reference cycle
            self = None
    # pylint: enable=inconsistent-return-statements


# Copied from
# https://github.com/python/cpython/blob/master/Lib/unittest/case.py
# at commit 293dd23 (Nov 19, 2019).
# Required at least to enable assertWarns() in python <3.2.
# Added in 0.4.0.
class _AssertWarnsContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertWarns* methods."""

    _base_type = Warning
    _base_type_str = 'a warning type or tuple of warning types'

    # Added in 0.4.0.
    def __enter__(self):
        # The __warningregistry__'s need to be in a pristine state for tests
        # to work properly.
        # pylint: disable=invalid-name, attribute-defined-outside-init
        for v in sys.modules.values():
            if getattr(v, '__warningregistry__', None):
                v.__warningregistry__ = {}
        self.warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self.warnings_manager.__enter__()
        warnings.simplefilter("always", self.expected)
        return self

    # Added in 0.4.0.
    def __exit__(self, exc_type, exc_value, tb):
        # pylint: disable=invalid-name, attribute-defined-outside-init
        self.warnings_manager.__exit__(exc_type, exc_value, tb)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return
        try:
            exc_name = self.expected.__name__
        except AttributeError:
            exc_name = str(self.expected)
        first_matching = None
        for m in self.warnings:
            w = m.message
            if not isinstance(w, self.expected):
                continue
            if first_matching is None:
                first_matching = w
            if (self.expected_regex is not None and
                    not self.expected_regex.search(str(w))):
                continue
            # store warning for later retrieval
            self.warning = w
            self.filename = m.filename
            self.lineno = m.lineno
            return
        # Now we simply try to choose a helpful failure message
        if first_matching is not None:
            self._raiseFailure('"{}" does not match "{}"'.format(
                self.expected_regex.pattern, str(first_matching)))
        if self.obj_name:
            self._raiseFailure("{} not triggered by {}".format(exc_name,
                                                               self.obj_name))
        else:
            self._raiseFailure("{} not triggered".format(exc_name))


# Partially copied from
# https://github.com/python/cpython/blob/master/Lib/unittest/case.py
# at commit 293dd23 (Nov 19, 2019).
# Required at least to enable assertWarns() in python <3.2.
def assertWarns(testcase, expected_warning, *args, **kwargs):
    """Context with same functionality as ``assertWarns`` in ``unittest``.

    Note that ``assertWarns`` is only available in python 3.2+.

    Added in 0.4.0.

    """
    # pylint: disable=invalid-name
    context = _AssertWarnsContext(expected_warning, testcase)
    return context.handle('assertWarns', args, kwargs)
