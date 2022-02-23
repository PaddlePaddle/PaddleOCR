"""
This module redefines ``str`` on Python 2.x to be a subclass of the Py2
``unicode`` type that behaves like the Python 3.x ``str``.

The main differences between ``newstr`` and Python 2.x's ``unicode`` type are
the stricter type-checking and absence of a `u''` prefix in the representation.

It is designed to be used together with the ``unicode_literals`` import
as follows:

    >>> from __future__ import unicode_literals
    >>> from builtins import str, isinstance

On Python 3.x and normally on Python 2.x, these expressions hold

    >>> str('blah') is 'blah'
    True
    >>> isinstance('blah', str)
    True

However, on Python 2.x, with this import:

    >>> from __future__ import unicode_literals

the same expressions are False:

    >>> str('blah') is 'blah'
    False
    >>> isinstance('blah', str)
    False

This module is designed to be imported together with ``unicode_literals`` on
Python 2 to bring the meaning of ``str`` back into alignment with unprefixed
string literals (i.e. ``unicode`` subclasses).

Note that ``str()`` (and ``print()``) would then normally call the
``__unicode__`` method on objects in Python 2. To define string
representations of your objects portably across Py3 and Py2, use the
:func:`python_2_unicode_compatible` decorator in  :mod:`future.utils`.

"""

from numbers import Number

from future.utils import PY3, istext, with_metaclass, isnewbytes
from future.types import no, issubset
from future.types.newobject import newobject


if PY3:
    # We'll probably never use newstr on Py3 anyway...
    unicode = str
    from collections.abc import Iterable
else:
    from collections import Iterable


class BaseNewStr(type):
    def __instancecheck__(cls, instance):
        if cls == newstr:
            return isinstance(instance, unicode)
        else:
            return issubclass(instance.__class__, cls)


class newstr(with_metaclass(BaseNewStr, unicode)):
    """
    A backport of the Python 3 str object to Py2
    """
    no_convert_msg = "Can't convert '{0}' object to str implicitly"

    def __new__(cls, *args, **kwargs):
        """
        From the Py3 str docstring:

          str(object='') -> str
          str(bytes_or_buffer[, encoding[, errors]]) -> str

          Create a new string object from the given object. If encoding or
          errors is specified, then the object must expose a data buffer
          that will be decoded using the given encoding and error handler.
          Otherwise, returns the result of object.__str__() (if defined)
          or repr(object).
          encoding defaults to sys.getdefaultencoding().
          errors defaults to 'strict'.

        """
        if len(args) == 0:
            return super(newstr, cls).__new__(cls)
        # Special case: If someone requests str(str(u'abc')), return the same
        # object (same id) for consistency with Py3.3. This is not true for
        # other objects like list or dict.
        elif type(args[0]) == newstr and cls == newstr:
            return args[0]
        elif isinstance(args[0], unicode):
            value = args[0]
        elif isinstance(args[0], bytes):   # i.e. Py2 bytes or newbytes
            if 'encoding' in kwargs or len(args) > 1:
                value = args[0].decode(*args[1:], **kwargs)
            else:
                value = args[0].__str__()
        else:
            value = args[0]
        return super(newstr, cls).__new__(cls, value)

    def __repr__(self):
        """
        Without the u prefix
        """

        value = super(newstr, self).__repr__()
        # assert value[0] == u'u'
        return value[1:]

    def __getitem__(self, y):
        """
        Warning: Python <= 2.7.6 has a bug that causes this method never to be called
        when y is a slice object. Therefore the type of newstr()[:2] is wrong
        (unicode instead of newstr).
        """
        return newstr(super(newstr, self).__getitem__(y))

    def __contains__(self, key):
        errmsg = "'in <string>' requires string as left operand, not {0}"
        # Don't use isinstance() here because we only want to catch
        # newstr, not Python 2 unicode:
        if type(key) == newstr:
            newkey = key
        elif isinstance(key, unicode) or isinstance(key, bytes) and not isnewbytes(key):
            newkey = newstr(key)
        else:
            raise TypeError(errmsg.format(type(key)))
        return issubset(list(newkey), list(self))

    @no('newbytes')
    def __add__(self, other):
        return newstr(super(newstr, self).__add__(other))

    @no('newbytes')
    def __radd__(self, left):
        " left + self "
        try:
            return newstr(left) + self
        except:
            return NotImplemented

    def __mul__(self, other):
        return newstr(super(newstr, self).__mul__(other))

    def __rmul__(self, other):
        return newstr(super(newstr, self).__rmul__(other))

    def join(self, iterable):
        errmsg = 'sequence item {0}: expected unicode string, found bytes'
        for i, item in enumerate(iterable):
            # Here we use type() rather than isinstance() because
            # __instancecheck__ is being overridden. E.g.
            # isinstance(b'abc', newbytes) is True on Py2.
            if isnewbytes(item):
                raise TypeError(errmsg.format(i))
        # Support use as a staticmethod: str.join('-', ['a', 'b'])
        if type(self) == newstr:
            return newstr(super(newstr, self).join(iterable))
        else:
            return newstr(super(newstr, newstr(self)).join(iterable))

    @no('newbytes')
    def find(self, sub, *args):
        return super(newstr, self).find(sub, *args)

    @no('newbytes')
    def rfind(self, sub, *args):
        return super(newstr, self).rfind(sub, *args)

    @no('newbytes', (1, 2))
    def replace(self, old, new, *args):
        return newstr(super(newstr, self).replace(old, new, *args))

    def decode(self, *args):
        raise AttributeError("decode method has been disabled in newstr")

    def encode(self, encoding='utf-8', errors='strict'):
        """
        Returns bytes

        Encode S using the codec registered for encoding. Default encoding
        is 'utf-8'. errors may be given to set a different error
        handling scheme. Default is 'strict' meaning that encoding errors raise
        a UnicodeEncodeError. Other possible values are 'ignore', 'replace' and
        'xmlcharrefreplace' as well as any other name registered with
        codecs.register_error that can handle UnicodeEncodeErrors.
        """
        from future.types.newbytes import newbytes
        # Py2 unicode.encode() takes encoding and errors as optional parameter,
        # not keyword arguments as in Python 3 str.

        # For the surrogateescape error handling mechanism, the
        # codecs.register_error() function seems to be inadequate for an
        # implementation of it when encoding. (Decoding seems fine, however.)
        # For example, in the case of
        #     u'\udcc3'.encode('ascii', 'surrogateescape_handler')
        # after registering the ``surrogateescape_handler`` function in
        # future.utils.surrogateescape, both Python 2.x and 3.x raise an
        # exception anyway after the function is called because the unicode
        # string it has to return isn't encodable strictly as ASCII.

        if errors == 'surrogateescape':
            if encoding == 'utf-16':
                # Known to fail here. See test_encoding_works_normally()
                raise NotImplementedError('FIXME: surrogateescape handling is '
                                          'not yet implemented properly')
            # Encode char by char, building up list of byte-strings
            mybytes = []
            for c in self:
                code = ord(c)
                if 0xD800 <= code <= 0xDCFF:
                    mybytes.append(newbytes([code - 0xDC00]))
                else:
                    mybytes.append(c.encode(encoding=encoding))
            return newbytes(b'').join(mybytes)
        return newbytes(super(newstr, self).encode(encoding, errors))

    @no('newbytes', 1)
    def startswith(self, prefix, *args):
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).startswith(prefix, *args)

    @no('newbytes', 1)
    def endswith(self, prefix, *args):
        # Note we need the decorator above as well as the isnewbytes()
        # check because prefix can be either a bytes object or e.g. a
        # tuple of possible prefixes. (If it's a bytes object, each item
        # in it is an int.)
        if isinstance(prefix, Iterable):
            for thing in prefix:
                if isnewbytes(thing):
                    raise TypeError(self.no_convert_msg.format(type(thing)))
        return super(newstr, self).endswith(prefix, *args)

    @no('newbytes', 1)
    def split(self, sep=None, maxsplit=-1):
        # Py2 unicode.split() takes maxsplit as an optional parameter,
        # not as a keyword argument as in Python 3 str.
        parts = super(newstr, self).split(sep, maxsplit)
        return [newstr(part) for part in parts]

    @no('newbytes', 1)
    def rsplit(self, sep=None, maxsplit=-1):
        # Py2 unicode.rsplit() takes maxsplit as an optional parameter,
        # not as a keyword argument as in Python 3 str.
        parts = super(newstr, self).rsplit(sep, maxsplit)
        return [newstr(part) for part in parts]

    @no('newbytes', 1)
    def partition(self, sep):
        parts = super(newstr, self).partition(sep)
        return tuple(newstr(part) for part in parts)

    @no('newbytes', 1)
    def rpartition(self, sep):
        parts = super(newstr, self).rpartition(sep)
        return tuple(newstr(part) for part in parts)

    @no('newbytes', 1)
    def index(self, sub, *args):
        """
        Like newstr.find() but raise ValueError when the substring is not
        found.
        """
        pos = self.find(sub, *args)
        if pos == -1:
            raise ValueError('substring not found')
        return pos

    def splitlines(self, keepends=False):
        """
        S.splitlines(keepends=False) -> list of strings

        Return a list of the lines in S, breaking at line boundaries.
        Line breaks are not included in the resulting list unless keepends
        is given and true.
        """
        # Py2 unicode.splitlines() takes keepends as an optional parameter,
        # not as a keyword argument as in Python 3 str.
        parts = super(newstr, self).splitlines(keepends)
        return [newstr(part) for part in parts]

    def __eq__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__eq__(other)
        else:
            return NotImplemented

    def __hash__(self):
        if (isinstance(self, unicode) or
            isinstance(self, bytes) and not isnewbytes(self)):
            return super(newstr, self).__hash__()
        else:
            raise NotImplementedError()

    def __ne__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__ne__(other)
        else:
            return True

    unorderable_err = 'unorderable types: str() and {0}'

    def __lt__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__lt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __le__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__le__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __gt__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__gt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __ge__(self, other):
        if (isinstance(other, unicode) or
            isinstance(other, bytes) and not isnewbytes(other)):
            return super(newstr, self).__ge__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __getattribute__(self, name):
        """
        A trick to cause the ``hasattr`` builtin-fn to return False for
        the 'decode' method on Py2.
        """
        if name in ['decode', u'decode']:
            raise AttributeError("decode method has been disabled in newstr")
        return super(newstr, self).__getattribute__(name)

    def __native__(self):
        """
        A hook for the future.utils.native() function.
        """
        return unicode(self)

    @staticmethod
    def maketrans(x, y=None, z=None):
        """
        Return a translation table usable for str.translate().

        If there is only one argument, it must be a dictionary mapping Unicode
        ordinals (integers) or characters to Unicode ordinals, strings or None.
        Character keys will be then converted to ordinals.
        If there are two arguments, they must be strings of equal length, and
        in the resulting dictionary, each character in x will be mapped to the
        character at the same position in y. If there is a third argument, it
        must be a string, whose characters will be mapped to None in the result.
        """

        if y is None:
            assert z is None
            if not isinstance(x, dict):
                raise TypeError('if you give only one argument to maketrans it must be a dict')
            result = {}
            for (key, value) in x.items():
                if len(key) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(key)] = value
        else:
            if not isinstance(x, unicode) and isinstance(y, unicode):
                raise TypeError('x and y must be unicode strings')
            if not len(x) == len(y):
                raise ValueError('the first two maketrans arguments must have equal length')
            result = {}
            for (xi, yi) in zip(x, y):
                if len(xi) > 1:
                    raise ValueError('keys in translate table must be strings or integers')
                result[ord(xi)] = ord(yi)

        if z is not None:
            for char in z:
                result[ord(char)] = None
        return result

    def translate(self, table):
        """
        S.translate(table) -> str

        Return a copy of the string S, where all characters have been mapped
        through the given translation table, which must be a mapping of
        Unicode ordinals to Unicode ordinals, strings, or None.
        Unmapped characters are left untouched. Characters mapped to None
        are deleted.
        """
        l = []
        for c in self:
            if ord(c) in table:
                val = table[ord(c)]
                if val is None:
                    continue
                elif isinstance(val, unicode):
                    l.append(val)
                else:
                    l.append(chr(val))
            else:
                l.append(c)
        return ''.join(l)

    def isprintable(self):
        raise NotImplementedError('fixme')

    def isidentifier(self):
        raise NotImplementedError('fixme')

    def format_map(self):
        raise NotImplementedError('fixme')


__all__ = ['newstr']
