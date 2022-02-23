"""
A list subclass for Python 2 that behaves like Python 3's list.

The primary difference is that lists have a .copy() method in Py3.

Example use:

>>> from builtins import list
>>> l1 = list()    # instead of {} for an empty list
>>> l1.append('hello')
>>> l2 = l1.copy()

"""

import sys
import copy

from future.utils import with_metaclass
from future.types.newobject import newobject


_builtin_list = list
ver = sys.version_info[:2]


class BaseNewList(type):
    def __instancecheck__(cls, instance):
        if cls == newlist:
            return isinstance(instance, _builtin_list)
        else:
            return issubclass(instance.__class__, cls)


class newlist(with_metaclass(BaseNewList, _builtin_list)):
    """
    A backport of the Python 3 list object to Py2
    """
    def copy(self):
        """
        L.copy() -> list -- a shallow copy of L
        """
        return copy.copy(self)

    def clear(self):
        """L.clear() -> None -- remove all items from L"""
        for i in range(len(self)):
            self.pop()

    def __new__(cls, *args, **kwargs):
        """
        list() -> new empty list
        list(iterable) -> new list initialized from iterable's items
        """

        if len(args) == 0:
            return super(newlist, cls).__new__(cls)
        elif type(args[0]) == newlist:
            value = args[0]
        else:
            value = args[0]
        return super(newlist, cls).__new__(cls, value)

    def __add__(self, value):
        return newlist(super(newlist, self).__add__(value))

    def __radd__(self, left):
        " left + self "
        try:
            return newlist(left) + self
        except:
            return NotImplemented

    def __getitem__(self, y):
        """
        x.__getitem__(y) <==> x[y]

        Warning: a bug in Python 2.x prevents indexing via a slice from
        returning a newlist object.
        """
        if isinstance(y, slice):
            return newlist(super(newlist, self).__getitem__(y))
        else:
            return super(newlist, self).__getitem__(y)

    def __native__(self):
        """
        Hook for the future.utils.native() function
        """
        return list(self)

    def __nonzero__(self):
        return len(self) > 0


__all__ = ['newlist']
