"""
Pure-Python implementation of a Python 3-like bytes object for Python 2.

Why do this? Without it, the Python 2 bytes object is a very, very
different beast to the Python 3 bytes object.
"""

from numbers import Integral
import string
import copy

from future.utils import istext, isbytes, PY2, PY3, with_metaclass
from future.types import no, issubset
from future.types.newobject import newobject

if PY2:
    from collections import Iterable
else:
    from collections.abc import Iterable


_builtin_bytes = bytes

if PY3:
    # We'll probably never use newstr on Py3 anyway...
    unicode = str


class BaseNewBytes(type):
    def __instancecheck__(cls, instance):
        if cls == newbytes:
            return isinstance(instance, _builtin_bytes)
        else:
            return issubclass(instance.__class__, cls)


def _newchr(x):
    if isinstance(x, str):  # this happens on pypy
        return x.encode('ascii')
    else:
        return chr(x)


class newbytes(with_metaclass(BaseNewBytes, _builtin_bytes)):
    """
    A backport of the Python 3 bytes object to Py2
    """
    def __new__(cls, *args, **kwargs):
        """
        From the Py3 bytes docstring:

        bytes(iterable_of_ints) -> bytes
        bytes(string, encoding[, errors]) -> bytes
        bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer
        bytes(int) -> bytes object of size given by the parameter initialized with null bytes
        bytes() -> empty bytes object

        Construct an immutable array of bytes from:
          - an iterable yielding integers in range(256)
          - a text string encoded using the specified encoding
          - any object implementing the buffer API.
          - an integer
        """

        encoding = None
        errors = None

        if len(args) == 0:
            return super(newbytes, cls).__new__(cls)
        elif len(args) >= 2:
            args = list(args)
            if len(args) == 3:
                errors = args.pop()
            encoding=args.pop()
        # Was: elif isinstance(args[0], newbytes):
        # We use type() instead of the above because we're redefining
        # this to be True for all unicode string subclasses. Warning:
        # This may render newstr un-subclassable.
        if type(args[0]) == newbytes:
            # Special-case: for consistency with Py3.3, we return the same object
            # (with the same id) if a newbytes object is passed into the
            # newbytes constructor.
            return args[0]
        elif isinstance(args[0], _builtin_bytes):
            value = args[0]
        elif isinstance(args[0], unicode):
            try:
                if 'encoding' in kwargs:
                    assert encoding is None
                    encoding = kwargs['encoding']
                if 'errors' in kwargs:
                    assert errors is None
                    errors = kwargs['errors']
            except AssertionError:
                raise TypeError('Argument given by name and position')
            if encoding is None:
                raise TypeError('unicode string argument without an encoding')
            ###
            # Was:   value = args[0].encode(**kwargs)
            # Python 2.6 string encode() method doesn't take kwargs:
            # Use this instead:
            newargs = [encoding]
            if errors is not None:
                newargs.append(errors)
            value = args[0].encode(*newargs)
            ###
        elif hasattr(args[0], '__bytes__'):
            value = args[0].__bytes__()
        elif isinstance(args[0], Iterable):
            if len(args[0]) == 0:
                # This could be an empty list or tuple. Return b'' as on Py3.
                value = b''
            else:
                # Was: elif len(args[0])>0 and isinstance(args[0][0], Integral):
                #      # It's a list of integers
                # But then we can't index into e.g. frozensets. Try to proceed
                # anyway.
                try:
                    value = bytearray([_newchr(x) for x in args[0]])
                except:
                    raise ValueError('bytes must be in range(0, 256)')
        elif isinstance(args[0], Integral):
            if args[0] < 0:
                raise ValueError('negative count')
            value = b'\x00' * args[0]
        else:
            value = args[0]
        if type(value) == newbytes:
            # Above we use type(...) rather than isinstance(...) because the
            # newbytes metaclass overrides __instancecheck__.
            # oldbytes(value) gives the wrong thing on Py2: the same
            # result as str(value) on Py3, e.g. "b'abc'". (Issue #193).
            # So we handle this case separately:
            return copy.copy(value)
        else:
            return super(newbytes, cls).__new__(cls, value)

    def __repr__(self):
        return 'b' + super(newbytes, self).__repr__()

    def __str__(self):
        return 'b' + "'{0}'".format(super(newbytes, self).__str__())

    def __getitem__(self, y):
        value = super(newbytes, self).__getitem__(y)
        if isinstance(y, Integral):
            return ord(value)
        else:
            return newbytes(value)

    def __getslice__(self, *args):
        return self.__getitem__(slice(*args))

    def __contains__(self, key):
        if isinstance(key, int):
            newbyteskey = newbytes([key])
        # Don't use isinstance() here because we only want to catch
        # newbytes, not Python 2 str:
        elif type(key) == newbytes:
            newbyteskey = key
        else:
            newbyteskey = newbytes(key)
        return issubset(list(newbyteskey), list(self))

    @no(unicode)
    def __add__(self, other):
        return newbytes(super(newbytes, self).__add__(other))

    @no(unicode)
    def __radd__(self, left):
        return newbytes(left) + self

    @no(unicode)
    def __mul__(self, other):
        return newbytes(super(newbytes, self).__mul__(other))

    @no(unicode)
    def __rmul__(self, other):
        return newbytes(super(newbytes, self).__rmul__(other))

    def __mod__(self, vals):
        if isinstance(vals, newbytes):
            vals = _builtin_bytes.__str__(vals)

        elif isinstance(vals, tuple):
            newvals = []
            for v in vals:
                if isinstance(v, newbytes):
                    v = _builtin_bytes.__str__(v)
                newvals.append(v)
            vals = tuple(newvals)

        elif (hasattr(vals.__class__, '__getitem__') and
                hasattr(vals.__class__, 'iteritems')):
            for k, v in vals.iteritems():
                if isinstance(v, newbytes):
                    vals[k] = _builtin_bytes.__str__(v)

        return _builtin_bytes.__mod__(self, vals)

    def __imod__(self, other):
        return self.__mod__(other)

    def join(self, iterable_of_bytes):
        errmsg = 'sequence item {0}: expected bytes, {1} found'
        if isbytes(iterable_of_bytes) or istext(iterable_of_bytes):
            raise TypeError(errmsg.format(0, type(iterable_of_bytes)))
        for i, item in enumerate(iterable_of_bytes):
            if istext(item):
                raise TypeError(errmsg.format(i, type(item)))
        return newbytes(super(newbytes, self).join(iterable_of_bytes))

    @classmethod
    def fromhex(cls, string):
        # Only on Py2:
        return cls(string.replace(' ', '').decode('hex'))

    @no(unicode)
    def find(self, sub, *args):
        return super(newbytes, self).find(sub, *args)

    @no(unicode)
    def rfind(self, sub, *args):
        return super(newbytes, self).rfind(sub, *args)

    @no(unicode, (1, 2))
    def replace(self, old, new, *args):
        return newbytes(super(newbytes, self).replace(old, new, *args))

    def encode(self, *args):
        raise AttributeError("encode method has been disabled in newbytes")

    def decode(self, encoding='utf-8', errors='strict'):
        """
        Returns a newstr (i.e. unicode subclass)

        Decode B using the codec registered for encoding. Default encoding
        is 'utf-8'. errors may be given to set a different error
        handling scheme.  Default is 'strict' meaning that encoding errors raise
        a UnicodeDecodeError.  Other possible values are 'ignore' and 'replace'
        as well as any other name registered with codecs.register_error that is
        able to handle UnicodeDecodeErrors.
        """
        # Py2 str.encode() takes encoding and errors as optional parameter,
        # not keyword arguments as in Python 3 str.

        from future.types.newstr import newstr

        if errors == 'surrogateescape':
            from future.utils.surrogateescape import register_surrogateescape
            register_surrogateescape()

        return newstr(super(newbytes, self).decode(encoding, errors))

        # This is currently broken:
        # # We implement surrogateescape error handling here in addition rather
        # # than relying on the custom error handler from
        # # future.utils.surrogateescape to be registered globally, even though
        # # that is fine in the case of decoding. (But not encoding: see the
        # # comments in newstr.encode()``.)
        #
        # if errors == 'surrogateescape':
        #     # Decode char by char
        #     mybytes = []
        #     for code in self:
        #         # Code is an int
        #         if 0x80 <= code <= 0xFF:
        #             b = 0xDC00 + code
        #         elif code <= 0x7F:
        #             b = _unichr(c).decode(encoding=encoding)
        #         else:
        #             # # It may be a bad byte
        #             # FIXME: What to do in this case? See the Py3 docs / tests.
        #             # # Try swallowing it.
        #             # continue
        #             # print("RAISE!")
        #             raise NotASurrogateError
        #         mybytes.append(b)
        #     return newbytes(mybytes)
        # return newbytes(super(newstr, self).decode(encoding, errors))

    @no(unicode)
    def startswith(self, prefix, *args):
        return super(newbytes, self).startswith(prefix, *args)

    @no(unicode)
    def endswith(self, prefix, *args):
        return super(newbytes, self).endswith(prefix, *args)

    @no(unicode)
    def split(self, sep=None, maxsplit=-1):
        # Py2 str.split() takes maxsplit as an optional parameter, not as a
        # keyword argument as in Python 3 bytes.
        parts = super(newbytes, self).split(sep, maxsplit)
        return [newbytes(part) for part in parts]

    def splitlines(self, keepends=False):
        """
        B.splitlines([keepends]) -> list of lines

        Return a list of the lines in B, breaking at line boundaries.
        Line breaks are not included in the resulting list unless keepends
        is given and true.
        """
        # Py2 str.splitlines() takes keepends as an optional parameter,
        # not as a keyword argument as in Python 3 bytes.
        parts = super(newbytes, self).splitlines(keepends)
        return [newbytes(part) for part in parts]

    @no(unicode)
    def rsplit(self, sep=None, maxsplit=-1):
        # Py2 str.rsplit() takes maxsplit as an optional parameter, not as a
        # keyword argument as in Python 3 bytes.
        parts = super(newbytes, self).rsplit(sep, maxsplit)
        return [newbytes(part) for part in parts]

    @no(unicode)
    def partition(self, sep):
        parts = super(newbytes, self).partition(sep)
        return tuple(newbytes(part) for part in parts)

    @no(unicode)
    def rpartition(self, sep):
        parts = super(newbytes, self).rpartition(sep)
        return tuple(newbytes(part) for part in parts)

    @no(unicode, (1,))
    def rindex(self, sub, *args):
        '''
        S.rindex(sub [,start [,end]]) -> int

        Like S.rfind() but raise ValueError when the substring is not found.
        '''
        pos = self.rfind(sub, *args)
        if pos == -1:
            raise ValueError('substring not found')

    @no(unicode)
    def index(self, sub, *args):
        '''
        Returns index of sub in bytes.
        Raises ValueError if byte is not in bytes and TypeError if can't
        be converted bytes or its length is not 1.
        '''
        if isinstance(sub, int):
            if len(args) == 0:
                start, end = 0, len(self)
            elif len(args) == 1:
                start = args[0]
            elif len(args) == 2:
                start, end = args
            else:
                raise TypeError('takes at most 3 arguments')
            return list(self)[start:end].index(sub)
        if not isinstance(sub, bytes):
            try:
                sub = self.__class__(sub)
            except (TypeError, ValueError):
                raise TypeError("can't convert sub to bytes")
        try:
            return super(newbytes, self).index(sub, *args)
        except ValueError:
            raise ValueError('substring not found')

    def __eq__(self, other):
        if isinstance(other, (_builtin_bytes, bytearray)):
            return super(newbytes, self).__eq__(other)
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__ne__(other)
        else:
            return True

    unorderable_err = 'unorderable types: bytes() and {0}'

    def __lt__(self, other):
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__lt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __le__(self, other):
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__le__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __gt__(self, other):
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__gt__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __ge__(self, other):
        if isinstance(other, _builtin_bytes):
            return super(newbytes, self).__ge__(other)
        raise TypeError(self.unorderable_err.format(type(other)))

    def __native__(self):
        # We can't just feed a newbytes object into str(), because
        # newbytes.__str__() returns e.g. "b'blah'", consistent with Py3 bytes.
        return super(newbytes, self).__str__()

    def __getattribute__(self, name):
        """
        A trick to cause the ``hasattr`` builtin-fn to return False for
        the 'encode' method on Py2.
        """
        if name in ['encode', u'encode']:
            raise AttributeError("encode method has been disabled in newbytes")
        return super(newbytes, self).__getattribute__(name)

    @no(unicode)
    def rstrip(self, bytes_to_strip=None):
        """
        Strip trailing bytes contained in the argument.
        If the argument is omitted, strip trailing ASCII whitespace.
        """
        return newbytes(super(newbytes, self).rstrip(bytes_to_strip))

    @no(unicode)
    def strip(self, bytes_to_strip=None):
        """
        Strip leading and trailing bytes contained in the argument.
        If the argument is omitted, strip trailing ASCII whitespace.
        """
        return newbytes(super(newbytes, self).strip(bytes_to_strip))

    def lower(self):
        """
        b.lower() -> copy of b

        Return a copy of b with all ASCII characters converted to lowercase.
        """
        return newbytes(super(newbytes, self).lower())

    @no(unicode)
    def upper(self):
        """
        b.upper() -> copy of b

        Return a copy of b with all ASCII characters converted to uppercase.
        """
        return newbytes(super(newbytes, self).upper())

    @classmethod
    @no(unicode)
    def maketrans(cls, frm, to):
        """
        B.maketrans(frm, to) -> translation table

        Return a translation table (a bytes object of length 256) suitable
        for use in the bytes or bytearray translate method where each byte
        in frm is mapped to the byte at the same position in to.
        The bytes objects frm and to must be of the same length.
        """
        return newbytes(string.maketrans(frm, to))


__all__ = ['newbytes']
