"""
A selection of cross-compatible functions for Python 2 and 3.

This module exports useful functions for 2/3 compatible code:

    * bind_method: binds functions to classes
    * ``native_str_to_bytes`` and ``bytes_to_native_str``
    * ``native_str``: always equal to the native platform string object (because
      this may be shadowed by imports from future.builtins)
    * lists: lrange(), lmap(), lzip(), lfilter()
    * iterable method compatibility:
        - iteritems, iterkeys, itervalues
        - viewitems, viewkeys, viewvalues

        These use the original method if available, otherwise they use items,
        keys, values.

    * types:

        * text_type: unicode in Python 2, str in Python 3
        * string_types: basestring in Python 2, str in Python 3
        * binary_type: str in Python 2, bytes in Python 3
        * integer_types: (int, long) in Python 2, int in Python 3
        * class_types: (type, types.ClassType) in Python 2, type in Python 3

    * bchr(c):
        Take an integer and make a 1-character byte string
    * bord(c)
        Take the result of indexing on a byte string and make an integer
    * tobytes(s)
        Take a text string, a byte string, or a sequence of characters taken
        from a byte string, and make a byte string.

    * raise_from()
    * raise_with_traceback()

This module also defines these decorators:

    * ``python_2_unicode_compatible``
    * ``with_metaclass``
    * ``implements_iterator``

Some of the functions in this module come from the following sources:

    * Jinja2 (BSD licensed: see
      https://github.com/mitsuhiko/jinja2/blob/master/LICENSE)
    * Pandas compatibility module pandas.compat
    * six.py by Benjamin Peterson
    * Django
"""

import types
import sys
import numbers
import functools
import copy
import inspect


PY3 = sys.version_info[0] >= 3
PY34_PLUS = sys.version_info[0:2] >= (3, 4)
PY35_PLUS = sys.version_info[0:2] >= (3, 5)
PY36_PLUS = sys.version_info[0:2] >= (3, 6)
PY2 = sys.version_info[0] == 2
PY26 = sys.version_info[0:2] == (2, 6)
PY27 = sys.version_info[0:2] == (2, 7)
PYPY = hasattr(sys, 'pypy_translation_info')


def python_2_unicode_compatible(cls):
    """
    A decorator that defines __unicode__ and __str__ methods under Python
    2. Under Python 3, this decorator is a no-op.

    To support Python 2 and 3 with a single code base, define a __str__
    method returning unicode text and apply this decorator to the class, like
    this::

    >>> from future.utils import python_2_unicode_compatible

    >>> @python_2_unicode_compatible
    ... class MyClass(object):
    ...     def __str__(self):
    ...         return u'Unicode string: \u5b54\u5b50'

    >>> a = MyClass()

    Then, after this import:

    >>> from future.builtins import str

    the following is ``True`` on both Python 3 and 2::

    >>> str(a) == a.encode('utf-8').decode('utf-8')
    True

    and, on a Unicode-enabled terminal with the right fonts, these both print the
    Chinese characters for Confucius::

    >>> print(a)
    >>> print(str(a))

    The implementation comes from django.utils.encoding.
    """
    if not PY3:
        cls.__unicode__ = cls.__str__
        cls.__str__ = lambda self: self.__unicode__().encode('utf-8')
    return cls


def with_metaclass(meta, *bases):
    """
    Function from jinja2/_compat.py. License: BSD.

    Use it like this::

        class BaseForm(object):
            pass

        class FormType(type):
            pass

        class Form(with_metaclass(FormType, BaseForm)):
            pass

    This requires a bit of explanation: the basic idea is to make a
    dummy metaclass for one level of class instantiation that replaces
    itself with the actual metaclass.  Because of internal type checks
    we also need to make sure that we downgrade the custom metaclass
    for one level to something closer to type (that's why __call__ and
    __init__ comes back from type etc.).

    This has the advantage over six.with_metaclass of not introducing
    dummy classes into the final MRO.
    """
    class metaclass(meta):
        __call__ = type.__call__
        __init__ = type.__init__
        def __new__(cls, name, this_bases, d):
            if this_bases is None:
                return type.__new__(cls, name, (), d)
            return meta(name, bases, d)
    return metaclass('temporary_class', None, {})


# Definitions from pandas.compat and six.py follow:
if PY3:
    def bchr(s):
        return bytes([s])
    def bstr(s):
        if isinstance(s, str):
            return bytes(s, 'latin-1')
        else:
            return bytes(s)
    def bord(s):
        return s

    string_types = str,
    integer_types = int,
    class_types = type,
    text_type = str
    binary_type = bytes

else:
    # Python 2
    def bchr(s):
        return chr(s)
    def bstr(s):
        return str(s)
    def bord(s):
        return ord(s)

    string_types = basestring,
    integer_types = (int, long)
    class_types = (type, types.ClassType)
    text_type = unicode
    binary_type = str

###

if PY3:
    def tobytes(s):
        if isinstance(s, bytes):
            return s
        else:
            if isinstance(s, str):
                return s.encode('latin-1')
            else:
                return bytes(s)
else:
    # Python 2
    def tobytes(s):
        if isinstance(s, unicode):
            return s.encode('latin-1')
        else:
            return ''.join(s)

tobytes.__doc__ = """
    Encodes to latin-1 (where the first 256 chars are the same as
    ASCII.)
    """

if PY3:
    def native_str_to_bytes(s, encoding='utf-8'):
        return s.encode(encoding)

    def bytes_to_native_str(b, encoding='utf-8'):
        return b.decode(encoding)

    def text_to_native_str(t, encoding=None):
        return t
else:
    # Python 2
    def native_str_to_bytes(s, encoding=None):
        from future.types import newbytes    # to avoid a circular import
        return newbytes(s)

    def bytes_to_native_str(b, encoding=None):
        return native(b)

    def text_to_native_str(t, encoding='ascii'):
        """
        Use this to create a Py2 native string when "from __future__ import
        unicode_literals" is in effect.
        """
        return unicode(t).encode(encoding)

native_str_to_bytes.__doc__ = """
    On Py3, returns an encoded string.
    On Py2, returns a newbytes type, ignoring the ``encoding`` argument.
    """

if PY3:
    # list-producing versions of the major Python iterating functions
    def lrange(*args, **kwargs):
        return list(range(*args, **kwargs))

    def lzip(*args, **kwargs):
        return list(zip(*args, **kwargs))

    def lmap(*args, **kwargs):
        return list(map(*args, **kwargs))

    def lfilter(*args, **kwargs):
        return list(filter(*args, **kwargs))
else:
    import __builtin__
    # Python 2-builtin ranges produce lists
    lrange = __builtin__.range
    lzip = __builtin__.zip
    lmap = __builtin__.map
    lfilter = __builtin__.filter


def isidentifier(s, dotted=False):
    '''
    A function equivalent to the str.isidentifier method on Py3
    '''
    if dotted:
        return all(isidentifier(a) for a in s.split('.'))
    if PY3:
        return s.isidentifier()
    else:
        import re
        _name_re = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*$")
        return bool(_name_re.match(s))


def viewitems(obj, **kwargs):
    """
    Function for iterating over dictionary items with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewitems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def viewkeys(obj, **kwargs):
    """
    Function for iterating over dictionary keys with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewkeys", None)
    if not func:
        func = obj.keys
    return func(**kwargs)


def viewvalues(obj, **kwargs):
    """
    Function for iterating over dictionary values with the same set-like
    behaviour on Py2.7 as on Py3.

    Passes kwargs to method."""
    func = getattr(obj, "viewvalues", None)
    if not func:
        func = obj.values
    return func(**kwargs)


def iteritems(obj, **kwargs):
    """Use this only if compatibility with Python versions before 2.7 is
    required. Otherwise, prefer viewitems().
    """
    func = getattr(obj, "iteritems", None)
    if not func:
        func = obj.items
    return func(**kwargs)


def iterkeys(obj, **kwargs):
    """Use this only if compatibility with Python versions before 2.7 is
    required. Otherwise, prefer viewkeys().
    """
    func = getattr(obj, "iterkeys", None)
    if not func:
        func = obj.keys
    return func(**kwargs)


def itervalues(obj, **kwargs):
    """Use this only if compatibility with Python versions before 2.7 is
    required. Otherwise, prefer viewvalues().
    """
    func = getattr(obj, "itervalues", None)
    if not func:
        func = obj.values
    return func(**kwargs)


def bind_method(cls, name, func):
    """Bind a method to class, python 2 and python 3 compatible.

    Parameters
    ----------

    cls : type
        class to receive bound method
    name : basestring
        name of method on class instance
    func : function
        function to be bound as method

    Returns
    -------
    None
    """
    # only python 2 has an issue with bound/unbound methods
    if not PY3:
        setattr(cls, name, types.MethodType(func, None, cls))
    else:
        setattr(cls, name, func)


def getexception():
    return sys.exc_info()[1]


def _get_caller_globals_and_locals():
    """
    Returns the globals and locals of the calling frame.

    Is there an alternative to frame hacking here?
    """
    caller_frame = inspect.stack()[2]
    myglobals = caller_frame[0].f_globals
    mylocals = caller_frame[0].f_locals
    return myglobals, mylocals


def _repr_strip(mystring):
    """
    Returns the string without any initial or final quotes.
    """
    r = repr(mystring)
    if r.startswith("'") and r.endswith("'"):
        return r[1:-1]
    else:
        return r


if PY3:
    def raise_from(exc, cause):
        """
        Equivalent to:

            raise EXCEPTION from CAUSE

        on Python 3. (See PEP 3134).
        """
        myglobals, mylocals = _get_caller_globals_and_locals()

        # We pass the exception and cause along with other globals
        # when we exec():
        myglobals = myglobals.copy()
        myglobals['__python_future_raise_from_exc'] = exc
        myglobals['__python_future_raise_from_cause'] = cause
        execstr = "raise __python_future_raise_from_exc from __python_future_raise_from_cause"
        exec(execstr, myglobals, mylocals)

    def raise_(tp, value=None, tb=None):
        """
        A function that matches the Python 2.x ``raise`` statement. This
        allows re-raising exceptions with the cls value and traceback on
        Python 2 and 3.
        """
        if isinstance(tp, BaseException):
            # If the first object is an instance, the type of the exception
            # is the class of the instance, the instance itself is the value,
            # and the second object must be None.
            if value is not None:
                raise TypeError("instance exception may not have a separate value")
            exc = tp
        elif isinstance(tp, type) and not issubclass(tp, BaseException):
            # If the first object is a class, it becomes the type of the
            # exception.
            raise TypeError("class must derive from BaseException, not %s" % tp.__name__)
        else:
            # The second object is used to determine the exception value: If it
            # is an instance of the class, the instance becomes the exception
            # value. If the second object is a tuple, it is used as the argument
            # list for the class constructor; if it is None, an empty argument
            # list is used, and any other object is treated as a single argument
            # to the constructor. The instance so created by calling the
            # constructor is used as the exception value.
            if isinstance(value, tp):
                exc = value
            elif isinstance(value, tuple):
                exc = tp(*value)
            elif value is None:
                exc = tp()
            else:
                exc = tp(value)

        if exc.__traceback__ is not tb:
            raise exc.with_traceback(tb)
        raise exc

    def raise_with_traceback(exc, traceback=Ellipsis):
        if traceback == Ellipsis:
            _, _, traceback = sys.exc_info()
        raise exc.with_traceback(traceback)

else:
    def raise_from(exc, cause):
        """
        Equivalent to:

            raise EXCEPTION from CAUSE

        on Python 3. (See PEP 3134).
        """
        # Is either arg an exception class (e.g. IndexError) rather than
        # instance (e.g. IndexError('my message here')? If so, pass the
        # name of the class undisturbed through to "raise ... from ...".
        if isinstance(exc, type) and issubclass(exc, Exception):
            e = exc()
            # exc = exc.__name__
            # execstr = "e = " + _repr_strip(exc) + "()"
            # myglobals, mylocals = _get_caller_globals_and_locals()
            # exec(execstr, myglobals, mylocals)
        else:
            e = exc
        e.__suppress_context__ = False
        if isinstance(cause, type) and issubclass(cause, Exception):
            e.__cause__ = cause()
            e.__cause__.__traceback__ = sys.exc_info()[2]
            e.__suppress_context__ = True
        elif cause is None:
            e.__cause__ = None
            e.__suppress_context__ = True
        elif isinstance(cause, BaseException):
            e.__cause__ = cause
            object.__setattr__(e.__cause__,  '__traceback__', sys.exc_info()[2])
            e.__suppress_context__ = True
        else:
            raise TypeError("exception causes must derive from BaseException")
        e.__context__ = sys.exc_info()[1]
        raise e

    exec('''
def raise_(tp, value=None, tb=None):
    raise tp, value, tb

def raise_with_traceback(exc, traceback=Ellipsis):
    if traceback == Ellipsis:
        _, _, traceback = sys.exc_info()
    raise exc, None, traceback
'''.strip())


raise_with_traceback.__doc__ = (
"""Raise exception with existing traceback.
If traceback is not passed, uses sys.exc_info() to get traceback."""
)


# Deprecated alias for backward compatibility with ``future`` versions < 0.11:
reraise = raise_


def implements_iterator(cls):
    '''
    From jinja2/_compat.py. License: BSD.

    Use as a decorator like this::

        @implements_iterator
        class UppercasingIterator(object):
            def __init__(self, iterable):
                self._iter = iter(iterable)
            def __iter__(self):
                return self
            def __next__(self):
                return next(self._iter).upper()

    '''
    if PY3:
        return cls
    else:
        cls.next = cls.__next__
        del cls.__next__
        return cls

if PY3:
    get_next = lambda x: x.next
else:
    get_next = lambda x: x.__next__


def encode_filename(filename):
    if PY3:
        return filename
    else:
        if isinstance(filename, unicode):
            return filename.encode('utf-8')
        return filename


def is_new_style(cls):
    """
    Python 2.7 has both new-style and old-style classes. Old-style classes can
    be pesky in some circumstances, such as when using inheritance.  Use this
    function to test for whether a class is new-style. (Python 3 only has
    new-style classes.)
    """
    return hasattr(cls, '__class__') and ('__dict__' in dir(cls)
                                          or hasattr(cls, '__slots__'))

# The native platform string and bytes types. Useful because ``str`` and
# ``bytes`` are redefined on Py2 by ``from future.builtins import *``.
native_str = str
native_bytes = bytes


def istext(obj):
    """
    Deprecated. Use::
        >>> isinstance(obj, str)
    after this import:
        >>> from future.builtins import str
    """
    return isinstance(obj, type(u''))


def isbytes(obj):
    """
    Deprecated. Use::
        >>> isinstance(obj, bytes)
    after this import:
        >>> from future.builtins import bytes
    """
    return isinstance(obj, type(b''))


def isnewbytes(obj):
    """
    Equivalent to the result of ``type(obj)  == type(newbytes)``
    in other words, it is REALLY a newbytes instance, not a Py2 native str
    object?

    Note that this does not cover subclasses of newbytes, and it is not
    equivalent to ininstance(obj, newbytes)
    """
    return type(obj).__name__ == 'newbytes'


def isint(obj):
    """
    Deprecated. Tests whether an object is a Py3 ``int`` or either a Py2 ``int`` or
    ``long``.

    Instead of using this function, you can use:

        >>> from future.builtins import int
        >>> isinstance(obj, int)

    The following idiom is equivalent:

        >>> from numbers import Integral
        >>> isinstance(obj, Integral)
    """

    return isinstance(obj, numbers.Integral)


def native(obj):
    """
    On Py3, this is a no-op: native(obj) -> obj

    On Py2, returns the corresponding native Py2 types that are
    superclasses for backported objects from Py3:

    >>> from builtins import str, bytes, int

    >>> native(str(u'ABC'))
    u'ABC'
    >>> type(native(str(u'ABC')))
    unicode

    >>> native(bytes(b'ABC'))
    b'ABC'
    >>> type(native(bytes(b'ABC')))
    bytes

    >>> native(int(10**20))
    100000000000000000000L
    >>> type(native(int(10**20)))
    long

    Existing native types on Py2 will be returned unchanged:

    >>> type(native(u'ABC'))
    unicode
    """
    if hasattr(obj, '__native__'):
        return obj.__native__()
    else:
        return obj


# Implementation of exec_ is from ``six``:
if PY3:
    import builtins
    exec_ = getattr(builtins, "exec")
else:
    def exec_(code, globs=None, locs=None):
        """Execute code in a namespace."""
        if globs is None:
            frame = sys._getframe(1)
            globs = frame.f_globals
            if locs is None:
                locs = frame.f_locals
            del frame
        elif locs is None:
            locs = globs
        exec("""exec code in globs, locs""")


# Defined here for backward compatibility:
def old_div(a, b):
    """
    DEPRECATED: import ``old_div`` from ``past.utils`` instead.

    Equivalent to ``a / b`` on Python 2 without ``from __future__ import
    division``.

    TODO: generalize this to other objects (like arrays etc.)
    """
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return a // b
    else:
        return a / b


def as_native_str(encoding='utf-8'):
    '''
    A decorator to turn a function or method call that returns text, i.e.
    unicode, into one that returns a native platform str.

    Use it as a decorator like this::

        from __future__ import unicode_literals

        class MyClass(object):
            @as_native_str(encoding='ascii')
            def __repr__(self):
                return next(self._iter).upper()
    '''
    if PY3:
        return lambda f: f
    else:
        def encoder(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs).encode(encoding=encoding)
            return wrapper
        return encoder

# listvalues and listitems definitions from Nick Coghlan's (withdrawn)
# PEP 496:
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def listvalues(d):
        return list(d.values())
    def listitems(d):
        return list(d.items())
else:
    # Python 2
    def listvalues(d):
        return d.values()
    def listitems(d):
        return d.items()

if PY3:
    def ensure_new_type(obj):
        return obj
else:
    def ensure_new_type(obj):
        from future.types.newbytes import newbytes
        from future.types.newstr import newstr
        from future.types.newint import newint
        from future.types.newdict import newdict

        native_type = type(native(obj))

        # Upcast only if the type is already a native (non-future) type
        if issubclass(native_type, type(obj)):
            # Upcast
            if native_type == str:  # i.e. Py2 8-bit str
                return newbytes(obj)
            elif native_type == unicode:
                return newstr(obj)
            elif native_type == int:
                return newint(obj)
            elif native_type == long:
                return newint(obj)
            elif native_type == dict:
                return newdict(obj)
            else:
                return obj
        else:
            # Already a new type
            assert type(obj) in [newbytes, newstr]
            return obj


__all__ = ['PY2', 'PY26', 'PY3', 'PYPY',
           'as_native_str', 'binary_type', 'bind_method', 'bord', 'bstr',
           'bytes_to_native_str', 'class_types', 'encode_filename',
           'ensure_new_type', 'exec_', 'get_next', 'getexception',
           'implements_iterator', 'integer_types', 'is_new_style', 'isbytes',
           'isidentifier', 'isint', 'isnewbytes', 'istext', 'iteritems',
           'iterkeys', 'itervalues', 'lfilter', 'listitems', 'listvalues',
           'lmap', 'lrange', 'lzip', 'native', 'native_bytes', 'native_str',
           'native_str_to_bytes', 'old_div',
           'python_2_unicode_compatible', 'raise_',
           'raise_with_traceback', 'reraise', 'string_types',
           'text_to_native_str', 'text_type', 'tobytes', 'viewitems',
           'viewkeys', 'viewvalues', 'with_metaclass'
           ]
