import re

import numpy as np
import pytest

from matplotlib import _api


@pytest.mark.parametrize('target,test_shape',
                         [((None, ), (1, 3)),
                          ((None, 3), (1,)),
                          ((None, 3), (1, 2)),
                          ((1, 5), (1, 9)),
                          ((None, 2, None), (1, 3, 1))
                          ])
def test_check_shape(target, test_shape):
    error_pattern = (f"^'aardvark' must be {len(target)}D.*" +
                     re.escape(f'has shape {test_shape}'))
    data = np.zeros(test_shape)
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)


def test_classproperty_deprecation():
    class A:
        @_api.deprecated("0.0.0")
        @_api.classproperty
        def f(cls):
            pass
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        A.f
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        a = A()
        a.f


def test_deprecate_privatize_attribute():
    class C:
        def __init__(self): self._attr = 1
        def _meth(self, arg): return arg
        attr = _api.deprecate_privatize_attribute("0.0")
        meth = _api.deprecate_privatize_attribute("0.0")

    c = C()
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        assert c.attr == 1
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        c.attr = 2
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        assert c.attr == 2
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        assert c.meth(42) == 42


def test_delete_parameter():
    @_api.delete_parameter("3.0", "foo")
    def func1(foo=None):
        pass

    @_api.delete_parameter("3.0", "foo")
    def func2(**kwargs):
        pass

    for func in [func1, func2]:
        func()  # No warning.
        with pytest.warns(_api.MatplotlibDeprecationWarning):
            func(foo="bar")

    def pyplot_wrapper(foo=_api.deprecation._deprecated_parameter):
        func1(foo)

    pyplot_wrapper()  # No warning.
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        func(foo="bar")


def test_make_keyword_only():
    @_api.make_keyword_only("3.0", "arg")
    def func(pre, arg, post=None):
        pass

    func(1, arg=2)  # Check that no warning is emitted.

    with pytest.warns(_api.MatplotlibDeprecationWarning):
        func(1, 2)
    with pytest.warns(_api.MatplotlibDeprecationWarning):
        func(1, 2, 3)
