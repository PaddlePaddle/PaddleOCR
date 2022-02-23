""":mod:`wand.compat` --- Compatibility layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module provides several subtle things to support
multiple Python versions (2.7, 3.3+) and VM implementations
(CPython, PyPy).

"""
import collections
try:
    import collections.abc
except ImportError:
    pass
import contextlib
import io
import sys
import types

__all__ = ('PY3', 'abc', 'binary', 'binary_type', 'encode_filename',
           'file_types', 'nested', 'string_type', 'text', 'text_type',
           'to_bytes', 'xrange')


#: (:class:`bool`) Whether it is Python 3.x or not.
PY3 = sys.version_info >= (3,)

#: (:class:`module`) Module containing abstract base classes.
#: :mod:`collections` in Python 2 and :mod:`collections.abc` in Python 3.
abc = collections.abc if PY3 else collections

#: (:class:`type`) Type for representing binary data.  :class:`str` in Python 2
#: and :class:`bytes` in Python 3.
binary_type = bytes if PY3 else str

#: (:class:`type`) Type for text data.  :class:`basestring` in Python 2
#: and :class:`str` in Python 3.
string_type = str if PY3 else basestring  # noqa

#: (:class:`type`) Type for representing Unicode textual data.
#: :class:`unicode` in Python 2 and :class:`str` in Python 3.
text_type = str if PY3 else unicode  # noqa


def binary(string, var=None):
    """Makes ``string`` to :class:`str` in Python 2.
    Makes ``string`` to :class:`bytes` in Python 3.

    :param string: a string to cast it to :data:`binary_type`
    :type string: :class:`bytes`, :class:`str`, :class:`unicode`
    :param var: an optional variable name to be used for error message
    :type var: :class:`str`

    """
    if isinstance(string, text_type):
        return string.encode()
    elif isinstance(string, binary_type):
        return string
    if var:
        raise TypeError('{0} must be a string, not {1!r}'.format(var, string))
    raise TypeError('expected a string, not ' + repr(string))


def to_bytes(value, string_pattern='{0}'):
    """Short-cut method to allow mixed value types to be converted to bytes.

    :param value: Value to be cast to bytes
    :type value: :class:`basestring`, :class:`int`, :class:`float`
    :param string_pattern: String format to allow printf style control of
                           bytes output.
    :type string_pattern: :class:`basestring`

    .. versionadded:: 0.6.4
    """
    return string_pattern.format(value).encode()


if PY3:
    def text(string):
        if isinstance(string, bytes):
            return string.decode('utf-8')
        return string
else:
    def text(string):
        """Makes ``string`` to :class:`str` in Python 3.
        Does nothing in Python 2.

        :param string: a string to cast it to :data:`text_type`
        :type string: :class:`bytes`, :class:`str`, :class:`unicode`

        """
        return string


#: The :func:`xrange()` function.  Alias for :func:`range()` in Python 3.
xrange = range if PY3 else xrange  # noqa


#: (:class:`type`, :class:`tuple`) Types for file objects that have
#: ``fileno()``.
file_types = io.RawIOBase if PY3 else (io.RawIOBase, types.FileType)


def encode_filename(filename):
    """If ``filename`` is a :data:`text_type`, encode it to
    :data:`binary_type` according to filesystem's default encoding.

    .. versionchanged:: 0.5.3
       Added support for PEP-519 https://github.com/emcconville/wand/pull/339
    """
    if hasattr(filename, "__fspath__"):  # PEP 519
        filename = filename.__fspath__()
    if isinstance(filename, text_type):
        return filename.encode(sys.getfilesystemencoding())
    return filename


try:
    nested = contextlib.nested
except AttributeError:
    # http://hg.python.org/cpython/file/v2.7.6/Lib/contextlib.py#l88
    @contextlib.contextmanager
    def nested(*managers):
        exits = []
        vars = []
        exc = (None, None, None)
        try:
            for mgr in managers:
                exit = mgr.__exit__
                enter = mgr.__enter__
                vars.append(enter())
                exits.append(exit)
            yield vars
        except:  # noqa: E722
            exc = sys.exc_info()
        finally:
            while exits:
                exit = exits.pop()
                try:
                    if exit(*exc):
                        exc = (None, None, None)
                except:  # noqa: E722
                    exc = sys.exc_info()
            if exc != (None, None, None):
                # PEP 3109
                e = exc[0](exc[1])
                e.__traceback__ = e[2]
                raise e
