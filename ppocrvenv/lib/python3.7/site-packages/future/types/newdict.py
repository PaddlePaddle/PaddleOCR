"""
A dict subclass for Python 2 that behaves like Python 3's dict

Example use:

>>> from builtins import dict
>>> d1 = dict()    # instead of {} for an empty dict
>>> d2 = dict(key1='value1', key2='value2')

The keys, values and items methods now return iterators on Python 2.x
(with set-like behaviour on Python 2.7).

>>> for d in (d1, d2):
...     assert not isinstance(d.keys(), list)
...     assert not isinstance(d.values(), list)
...     assert not isinstance(d.items(), list)
"""

import sys

from future.utils import with_metaclass
from future.types.newobject import newobject


_builtin_dict = dict
ver = sys.version_info[:2]


class BaseNewDict(type):
    def __instancecheck__(cls, instance):
        if cls == newdict:
            return isinstance(instance, _builtin_dict)
        else:
            return issubclass(instance.__class__, cls)


class newdict(with_metaclass(BaseNewDict, _builtin_dict)):
    """
    A backport of the Python 3 dict object to Py2
    """
    def items(self):
        """
        On Python 2.7+:
            D.items() -> a set-like object providing a view on D's items
        On Python 2.6:
            D.items() -> an iterator over D's items
        """
        if ver == (2, 7):
            return self.viewitems()
        elif ver == (2, 6):
            return self.iteritems()
        elif ver >= (3, 0):
            return self.items()

    def keys(self):
        """
        On Python 2.7+:
            D.keys() -> a set-like object providing a view on D's keys
        On Python 2.6:
            D.keys() -> an iterator over D's keys
        """
        if ver == (2, 7):
            return self.viewkeys()
        elif ver == (2, 6):
            return self.iterkeys()
        elif ver >= (3, 0):
            return self.keys()

    def values(self):
        """
        On Python 2.7+:
            D.values() -> a set-like object providing a view on D's values
        On Python 2.6:
            D.values() -> an iterator over D's values
        """
        if ver == (2, 7):
            return self.viewvalues()
        elif ver == (2, 6):
            return self.itervalues()
        elif ver >= (3, 0):
            return self.values()

    def __new__(cls, *args, **kwargs):
        """
        dict() -> new empty dictionary
        dict(mapping) -> new dictionary initialized from a mapping object's
            (key, value) pairs
        dict(iterable) -> new dictionary initialized as if via:
            d = {}
            for k, v in iterable:
                d[k] = v
        dict(**kwargs) -> new dictionary initialized with the name=value pairs
            in the keyword argument list.  For example:  dict(one=1, two=2)
        """

        if len(args) == 0:
            return super(newdict, cls).__new__(cls)
        elif type(args[0]) == newdict:
            value = args[0]
        else:
            value = args[0]
        return super(newdict, cls).__new__(cls, value)

    def __native__(self):
        """
        Hook for the future.utils.native() function
        """
        return dict(self)


__all__ = ['newdict']
