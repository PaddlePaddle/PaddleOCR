"""
An implementation of the basestring type for Python 3

Example use:

>>> s = b'abc'
>>> assert isinstance(s, basestring)
>>> from past.types import str as oldstr
>>> s2 = oldstr(b'abc')
>>> assert isinstance(s2, basestring)

"""

import sys

from past.utils import with_metaclass, PY2

if PY2:
    str = unicode

ver = sys.version_info[:2]


class BaseBaseString(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, (bytes, str))

    def __subclasshook__(cls, thing):
        # TODO: What should go here?
        raise NotImplemented


class basestring(with_metaclass(BaseBaseString)):
    """
    A minimal backport of the Python 2 basestring type to Py3
    """


__all__ = ['basestring']
