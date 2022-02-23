"""
This module contains backports the data types that were significantly changed
in the transition from Python 2 to Python 3.

- an implementation of Python 3's bytes object (pure Python subclass of
  Python 2's builtin 8-bit str type)
- an implementation of Python 3's str object (pure Python subclass of
  Python 2's builtin unicode type)
- a backport of the range iterator from Py3 with slicing support

It is used as follows::

    from __future__ import division, absolute_import, print_function
    from builtins import bytes, dict, int, range, str

to bring in the new semantics for these functions from Python 3. And
then, for example::

    b = bytes(b'ABCD')
    assert list(b) == [65, 66, 67, 68]
    assert repr(b) == "b'ABCD'"
    assert [65, 66] in b

    # These raise TypeErrors:
    # b + u'EFGH'
    # b.split(u'B')
    # bytes(b',').join([u'Fred', u'Bill'])


    s = str(u'ABCD')

    # These raise TypeErrors:
    # s.join([b'Fred', b'Bill'])
    # s.startswith(b'A')
    # b'B' in s
    # s.find(b'A')
    # s.replace(u'A', b'a')

    # This raises an AttributeError:
    # s.decode('utf-8')

    assert repr(s) == 'ABCD'      # consistent repr with Py3 (no u prefix)


    for i in range(10**11)[:10]:
        pass

and::

    class VerboseList(list):
        def append(self, item):
            print('Adding an item')
            super().append(item)        # new simpler super() function

For more information:
---------------------

- future.types.newbytes
- future.types.newdict
- future.types.newint
- future.types.newobject
- future.types.newrange
- future.types.newstr


Notes
=====

range()
-------
``range`` is a custom class that backports the slicing behaviour from
Python 3 (based on the ``xrange`` module by Dan Crosta). See the
``newrange`` module docstring for more details.


super()
-------
``super()`` is based on Ryan Kelly's ``magicsuper`` module. See the
``newsuper`` module docstring for more details.


round()
-------
Python 3 modifies the behaviour of ``round()`` to use "Banker's Rounding".
See http://stackoverflow.com/a/10825998. See the ``newround`` module
docstring for more details.

"""

from __future__ import absolute_import, division, print_function

import functools
from numbers import Integral

from future import utils


# Some utility functions to enforce strict type-separation of unicode str and
# bytes:
def disallow_types(argnums, disallowed_types):
    """
    A decorator that raises a TypeError if any of the given numbered
    arguments is of the corresponding given type (e.g. bytes or unicode
    string).

    For example:

        @disallow_types([0, 1], [unicode, bytes])
        def f(a, b):
            pass

    raises a TypeError when f is called if a unicode object is passed as
    `a` or a bytes object is passed as `b`.

    This also skips over keyword arguments, so

        @disallow_types([0, 1], [unicode, bytes])
        def g(a, b=None):
            pass

    doesn't raise an exception if g is called with only one argument a,
    e.g.:

        g(b'Byte string')

    Example use:

    >>> class newbytes(object):
    ...     @disallow_types([1], [unicode])
    ...     def __add__(self, other):
    ...          pass

    >>> newbytes('1234') + u'1234'      #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TypeError: can't concat 'bytes' to (unicode) str
    """

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # These imports are just for this decorator, and are defined here
            # to prevent circular imports:
            from .newbytes import newbytes
            from .newint import newint
            from .newstr import newstr

            errmsg = "argument can't be {0}"
            for (argnum, mytype) in zip(argnums, disallowed_types):
                # Handle the case where the type is passed as a string like 'newbytes'.
                if isinstance(mytype, str) or isinstance(mytype, bytes):
                    mytype = locals()[mytype]

                # Only restrict kw args only if they are passed:
                if len(args) <= argnum:
                    break

                # Here we use type() rather than isinstance() because
                # __instancecheck__ is being overridden. E.g.
                # isinstance(b'abc', newbytes) is True on Py2.
                if type(args[argnum]) == mytype:
                    raise TypeError(errmsg.format(mytype))

            return function(*args, **kwargs)
        return wrapper
    return decorator


def no(mytype, argnums=(1,)):
    """
    A shortcut for the disallow_types decorator that disallows only one type
    (in any position in argnums).

    Example use:

    >>> class newstr(object):
    ...     @no('bytes')
    ...     def __add__(self, other):
    ...          pass

    >>> newstr(u'1234') + b'1234'     #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TypeError: argument can't be bytes

    The object can also be passed directly, but passing the string helps
    to prevent circular import problems.
    """
    if isinstance(argnums, Integral):
        argnums = (argnums,)
    disallowed_types = [mytype] * len(argnums)
    return disallow_types(argnums, disallowed_types)


def issubset(list1, list2):
    """
    Examples:

    >>> issubset([], [65, 66, 67])
    True
    >>> issubset([65], [65, 66, 67])
    True
    >>> issubset([65, 66], [65, 66, 67])
    True
    >>> issubset([65, 67], [65, 66, 67])
    False
    """
    n = len(list1)
    for startpos in range(len(list2) - n + 1):
        if list2[startpos:startpos+n] == list1:
            return True
    return False


if utils.PY3:
    import builtins
    bytes = builtins.bytes
    dict = builtins.dict
    int = builtins.int
    list = builtins.list
    object = builtins.object
    range = builtins.range
    str = builtins.str

    # The identity mapping
    newtypes = {bytes: bytes,
                dict: dict,
                int: int,
                list: list,
                object: object,
                range: range,
                str: str}

    __all__ = ['newtypes']

else:

    from .newbytes import newbytes
    from .newdict import newdict
    from .newint import newint
    from .newlist import newlist
    from .newrange import newrange
    from .newobject import newobject
    from .newstr import newstr

    newtypes = {bytes: newbytes,
                dict: newdict,
                int: newint,
                long: newint,
                list: newlist,
                object: newobject,
                range: newrange,
                str: newbytes,
                unicode: newstr}

    __all__ = ['newbytes', 'newdict', 'newint', 'newlist', 'newrange', 'newstr', 'newtypes']
