"""
Backport of Python 3's int, based on Py2's long.

They are very similar. The most notable difference is:

- representation: trailing L in Python 2 removed in Python 3
"""
from __future__ import division

import struct

from future.types.newbytes import newbytes
from future.types.newobject import newobject
from future.utils import PY3, isint, istext, isbytes, with_metaclass, native


if PY3:
    long = int
    from collections.abc import Iterable
else:
    from collections import Iterable


class BaseNewInt(type):
    def __instancecheck__(cls, instance):
        if cls == newint:
            # Special case for Py2 short or long int
            return isinstance(instance, (int, long))
        else:
            return issubclass(instance.__class__, cls)


class newint(with_metaclass(BaseNewInt, long)):
    """
    A backport of the Python 3 int object to Py2
    """
    def __new__(cls, x=0, base=10):
        """
        From the Py3 int docstring:

        |  int(x=0) -> integer
        |  int(x, base=10) -> integer
        |
        |  Convert a number or string to an integer, or return 0 if no
        |  arguments are given.  If x is a number, return x.__int__().  For
        |  floating point numbers, this truncates towards zero.
        |
        |  If x is not a number or if base is given, then x must be a string,
        |  bytes, or bytearray instance representing an integer literal in the
        |  given base.  The literal can be preceded by '+' or '-' and be
        |  surrounded by whitespace.  The base defaults to 10.  Valid bases are
        |  0 and 2-36. Base 0 means to interpret the base from the string as an
        |  integer literal.
        |  >>> int('0b100', base=0)
        |  4

        """
        try:
            val = x.__int__()
        except AttributeError:
            val = x
        else:
            if not isint(val):
                raise TypeError('__int__ returned non-int ({0})'.format(
                    type(val)))

        if base != 10:
            # Explicit base
            if not (istext(val) or isbytes(val) or isinstance(val, bytearray)):
                raise TypeError(
                    "int() can't convert non-string with explicit base")
            try:
                return super(newint, cls).__new__(cls, val, base)
            except TypeError:
                return super(newint, cls).__new__(cls, newbytes(val), base)
        # After here, base is 10
        try:
            return super(newint, cls).__new__(cls, val)
        except TypeError:
            # Py2 long doesn't handle bytearray input with an explicit base, so
            # handle this here.
            # Py3: int(bytearray(b'10'), 2) == 2
            # Py2: int(bytearray(b'10'), 2) == 2 raises TypeError
            # Py2: long(bytearray(b'10'), 2) == 2 raises TypeError
            try:
                return super(newint, cls).__new__(cls, newbytes(val))
            except:
                raise TypeError("newint argument must be a string or a number,"
                                "not '{0}'".format(type(val)))

    def __repr__(self):
        """
        Without the L suffix
        """
        value = super(newint, self).__repr__()
        assert value[-1] == 'L'
        return value[:-1]

    def __add__(self, other):
        value = super(newint, self).__add__(other)
        if value is NotImplemented:
            return long(self) + other
        return newint(value)

    def __radd__(self, other):
        value = super(newint, self).__radd__(other)
        if value is NotImplemented:
            return other + long(self)
        return newint(value)

    def __sub__(self, other):
        value = super(newint, self).__sub__(other)
        if value is NotImplemented:
            return long(self) - other
        return newint(value)

    def __rsub__(self, other):
        value = super(newint, self).__rsub__(other)
        if value is NotImplemented:
            return other - long(self)
        return newint(value)

    def __mul__(self, other):
        value = super(newint, self).__mul__(other)
        if isint(value):
            return newint(value)
        elif value is NotImplemented:
            return long(self) * other
        return value

    def __rmul__(self, other):
        value = super(newint, self).__rmul__(other)
        if isint(value):
            return newint(value)
        elif value is NotImplemented:
            return other * long(self)
        return value

    def __div__(self, other):
        # We override this rather than e.g. relying on object.__div__ or
        # long.__div__ because we want to wrap the value in a newint()
        # call if other is another int
        value = long(self) / other
        if isinstance(other, (int, long)):
            return newint(value)
        else:
            return value

    def __rdiv__(self, other):
        value = other / long(self)
        if isinstance(other, (int, long)):
            return newint(value)
        else:
            return value

    def __idiv__(self, other):
        # long has no __idiv__ method. Use __itruediv__ and cast back to
        # newint:
        value = self.__itruediv__(other)
        if isinstance(other, (int, long)):
            return newint(value)
        else:
            return value

    def __truediv__(self, other):
        value = super(newint, self).__truediv__(other)
        if value is NotImplemented:
            value = long(self) / other
        return value

    def __rtruediv__(self, other):
        return super(newint, self).__rtruediv__(other)

    def __itruediv__(self, other):
        # long has no __itruediv__ method
        mylong = long(self)
        mylong /= other
        return mylong

    def __floordiv__(self, other):
        return newint(super(newint, self).__floordiv__(other))

    def __rfloordiv__(self, other):
        return newint(super(newint, self).__rfloordiv__(other))

    def __ifloordiv__(self, other):
        # long has no __ifloordiv__ method
        mylong = long(self)
        mylong //= other
        return newint(mylong)

    def __mod__(self, other):
        value = super(newint, self).__mod__(other)
        if value is NotImplemented:
            return long(self) % other
        return newint(value)

    def __rmod__(self, other):
        value = super(newint, self).__rmod__(other)
        if value is NotImplemented:
            return other % long(self)
        return newint(value)

    def __divmod__(self, other):
        value = super(newint, self).__divmod__(other)
        if value is NotImplemented:
            mylong = long(self)
            return (mylong // other, mylong % other)
        return (newint(value[0]), newint(value[1]))

    def __rdivmod__(self, other):
        value = super(newint, self).__rdivmod__(other)
        if value is NotImplemented:
            mylong = long(self)
            return (other // mylong, other % mylong)
        return (newint(value[0]), newint(value[1]))

    def __pow__(self, other):
        value = super(newint, self).__pow__(other)
        if value is NotImplemented:
            return long(self) ** other
        return newint(value)

    def __rpow__(self, other):
        value = super(newint, self).__rpow__(other)
        if value is NotImplemented:
            return other ** long(self)
        return newint(value)

    def __lshift__(self, other):
        if not isint(other):
            raise TypeError(
                "unsupported operand type(s) for <<: '%s' and '%s'" %
                (type(self).__name__, type(other).__name__))
        return newint(super(newint, self).__lshift__(other))

    def __rshift__(self, other):
        if not isint(other):
            raise TypeError(
                "unsupported operand type(s) for >>: '%s' and '%s'" %
                (type(self).__name__, type(other).__name__))
        return newint(super(newint, self).__rshift__(other))

    def __and__(self, other):
        if not isint(other):
            raise TypeError(
                "unsupported operand type(s) for &: '%s' and '%s'" %
                (type(self).__name__, type(other).__name__))
        return newint(super(newint, self).__and__(other))

    def __or__(self, other):
        if not isint(other):
            raise TypeError(
                "unsupported operand type(s) for |: '%s' and '%s'" %
                (type(self).__name__, type(other).__name__))
        return newint(super(newint, self).__or__(other))

    def __xor__(self, other):
        if not isint(other):
            raise TypeError(
                "unsupported operand type(s) for ^: '%s' and '%s'" %
                (type(self).__name__, type(other).__name__))
        return newint(super(newint, self).__xor__(other))

    def __neg__(self):
        return newint(super(newint, self).__neg__())

    def __pos__(self):
        return newint(super(newint, self).__pos__())

    def __abs__(self):
        return newint(super(newint, self).__abs__())

    def __invert__(self):
        return newint(super(newint, self).__invert__())

    def __int__(self):
        return self

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        """
        So subclasses can override this, Py3-style
        """
        return super(newint, self).__nonzero__()

    def __native__(self):
        return long(self)

    def to_bytes(self, length, byteorder='big', signed=False):
        """
        Return an array of bytes representing an integer.

        The integer is represented using length bytes.  An OverflowError is
        raised if the integer is not representable with the given number of
        bytes.

        The byteorder argument determines the byte order used to represent the
        integer.  If byteorder is 'big', the most significant byte is at the
        beginning of the byte array.  If byteorder is 'little', the most
        significant byte is at the end of the byte array.  To request the native
        byte order of the host system, use `sys.byteorder' as the byte order value.

        The signed keyword-only argument determines whether two's complement is
        used to represent the integer.  If signed is False and a negative integer
        is given, an OverflowError is raised.
        """
        if length < 0:
            raise ValueError("length argument must be non-negative")
        if length == 0 and self == 0:
            return newbytes()
        if signed and self < 0:
            bits = length * 8
            num = (2**bits) + self
            if num <= 0:
                raise OverflowError("int too smal to convert")
        else:
            if self < 0:
                raise OverflowError("can't convert negative int to unsigned")
            num = self
        if byteorder not in ('little', 'big'):
            raise ValueError("byteorder must be either 'little' or 'big'")
        h = b'%x' % num
        s = newbytes((b'0'*(len(h) % 2) + h).zfill(length*2).decode('hex'))
        if signed:
            high_set = s[0] & 0x80
            if self > 0 and high_set:
                raise OverflowError("int too big to convert")
            if self < 0 and not high_set:
                raise OverflowError("int too small to convert")
        if len(s) > length:
            raise OverflowError("int too big to convert")
        return s if byteorder == 'big' else s[::-1]

    @classmethod
    def from_bytes(cls, mybytes, byteorder='big', signed=False):
        """
        Return the integer represented by the given array of bytes.

        The mybytes argument must either support the buffer protocol or be an
        iterable object producing bytes.  Bytes and bytearray are examples of
        built-in objects that support the buffer protocol.

        The byteorder argument determines the byte order used to represent the
        integer.  If byteorder is 'big', the most significant byte is at the
        beginning of the byte array.  If byteorder is 'little', the most
        significant byte is at the end of the byte array.  To request the native
        byte order of the host system, use `sys.byteorder' as the byte order value.

        The signed keyword-only argument indicates whether two's complement is
        used to represent the integer.
        """
        if byteorder not in ('little', 'big'):
            raise ValueError("byteorder must be either 'little' or 'big'")
        if isinstance(mybytes, unicode):
            raise TypeError("cannot convert unicode objects to bytes")
        # mybytes can also be passed as a sequence of integers on Py3.
        # Test for this:
        elif isinstance(mybytes, Iterable):
            mybytes = newbytes(mybytes)
        b = mybytes if byteorder == 'big' else mybytes[::-1]
        if len(b) == 0:
            b = b'\x00'
        # The encode() method has been disabled by newbytes, but Py2's
        # str has it:
        num = int(native(b).encode('hex'), 16)
        if signed and (b[0] & 0x80):
            num = num - (2 ** (len(b)*8))
        return cls(num)


# def _twos_comp(val, bits):
#     """compute the 2's compliment of int value val"""
#     if( (val&(1<<(bits-1))) != 0 ):
#         val = val - (1<<bits)
#     return val


__all__ = ['newint']
