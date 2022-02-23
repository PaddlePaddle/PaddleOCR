# ===================================================================
#
# Copyright (c) 2018, Helder Eijs <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

import abc

from Crypto.Util.py3compat import iter_range, bord, bchr, ABC

from Crypto import Random


class IntegerBase(ABC):

    # Conversions
    @abc.abstractmethod
    def __int__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def to_bytes(self, block_size=0):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_bytes(byte_string):
        pass

    # Relations
    @abc.abstractmethod
    def __eq__(self, term):
        pass

    @abc.abstractmethod
    def __ne__(self, term):
        pass

    @abc.abstractmethod
    def __lt__(self, term):
        pass

    @abc.abstractmethod
    def __le__(self, term):
        pass

    @abc.abstractmethod
    def __gt__(self, term):
        pass

    @abc.abstractmethod
    def __ge__(self, term):
        pass

    @abc.abstractmethod
    def __nonzero__(self):
        pass
    __bool__ = __nonzero__

    @abc.abstractmethod
    def is_negative(self):
        pass

    # Arithmetic operations
    @abc.abstractmethod
    def __add__(self, term):
        pass

    @abc.abstractmethod
    def __sub__(self, term):
        pass

    @abc.abstractmethod
    def __mul__(self, factor):
        pass

    @abc.abstractmethod
    def __floordiv__(self, divisor):
        pass

    @abc.abstractmethod
    def __mod__(self, divisor):
        pass

    @abc.abstractmethod
    def inplace_pow(self, exponent, modulus=None):
        pass

    @abc.abstractmethod
    def __pow__(self, exponent, modulus=None):
        pass

    @abc.abstractmethod
    def __abs__(self):
        pass

    @abc.abstractmethod
    def sqrt(self, modulus=None):
        pass

    @abc.abstractmethod
    def __iadd__(self, term):
        pass

    @abc.abstractmethod
    def __isub__(self, term):
        pass

    @abc.abstractmethod
    def __imul__(self, term):
        pass

    @abc.abstractmethod
    def __imod__(self, term):
        pass

    # Boolean/bit operations
    @abc.abstractmethod
    def __and__(self, term):
        pass

    @abc.abstractmethod
    def __or__(self, term):
        pass

    @abc.abstractmethod
    def __rshift__(self, pos):
        pass

    @abc.abstractmethod
    def __irshift__(self, pos):
        pass

    @abc.abstractmethod
    def __lshift__(self, pos):
        pass

    @abc.abstractmethod
    def __ilshift__(self, pos):
        pass

    @abc.abstractmethod
    def get_bit(self, n):
        pass

    # Extra
    @abc.abstractmethod
    def is_odd(self):
        pass

    @abc.abstractmethod
    def is_even(self):
        pass

    @abc.abstractmethod
    def size_in_bits(self):
        pass

    @abc.abstractmethod
    def size_in_bytes(self):
        pass

    @abc.abstractmethod
    def is_perfect_square(self):
        pass

    @abc.abstractmethod
    def fail_if_divisible_by(self, small_prime):
        pass

    @abc.abstractmethod
    def multiply_accumulate(self, a, b):
        pass

    @abc.abstractmethod
    def set(self, source):
        pass

    @abc.abstractmethod
    def inplace_inverse(self, modulus):
        pass

    @abc.abstractmethod
    def inverse(self, modulus):
        pass

    @abc.abstractmethod
    def gcd(self, term):
        pass

    @abc.abstractmethod
    def lcm(self, term):
        pass

    @staticmethod
    @abc.abstractmethod
    def jacobi_symbol(a, n):
        pass
    
    @staticmethod
    def _tonelli_shanks(n, p):
        """Tonelli-shanks algorithm for computing the square root
        of n modulo a prime p.

        n must be in the range [0..p-1].
        p must be at least even.

        The return value r is the square root of modulo p. If non-zero,
        another solution will also exist (p-r).

        Note we cannot assume that p is really a prime: if it's not,
        we can either raise an exception or return the correct value.
        """

        # See https://rosettacode.org/wiki/Tonelli-Shanks_algorithm

        if n in (0, 1):
            return n

        if p % 4 == 3:
            root = pow(n, (p + 1) // 4, p)
            if pow(root, 2, p) != n:
                raise ValueError("Cannot compute square root")
            return root

        s = 1
        q = (p - 1) // 2
        while not (q & 1):
            s += 1
            q >>= 1

        z = n.__class__(2)
        while True:
            euler = pow(z, (p - 1) // 2, p)
            if euler == 1:
                z += 1
                continue
            if euler == p - 1:
                break
            # Most probably p is not a prime
            raise ValueError("Cannot compute square root")

        m = s
        c = pow(z, q, p)
        t = pow(n, q, p)
        r = pow(n, (q + 1) // 2, p)

        while t != 1:
            for i in iter_range(0, m):
                if pow(t, 2**i, p) == 1:
                    break
            if i == m:
                raise ValueError("Cannot compute square root of %d mod %d" % (n, p))
            b = pow(c, 2**(m - i - 1), p)
            m = i
            c = b**2 % p
            t = (t * b**2) % p
            r = (r * b) % p

        if pow(r, 2, p) != n:
            raise ValueError("Cannot compute square root")

        return r

    @classmethod
    def random(cls, **kwargs):
        """Generate a random natural integer of a certain size.

        :Keywords:
          exact_bits : positive integer
            The length in bits of the resulting random Integer number.
            The number is guaranteed to fulfil the relation:

                2^bits > result >= 2^(bits - 1)

          max_bits : positive integer
            The maximum length in bits of the resulting random Integer number.
            The number is guaranteed to fulfil the relation:

                2^bits > result >=0

          randfunc : callable
            A function that returns a random byte string. The length of the
            byte string is passed as parameter. Optional.
            If not provided (or ``None``), randomness is read from the system RNG.

        :Return: a Integer object
        """

        exact_bits = kwargs.pop("exact_bits", None)
        max_bits = kwargs.pop("max_bits", None)
        randfunc = kwargs.pop("randfunc", None)

        if randfunc is None:
            randfunc = Random.new().read

        if exact_bits is None and max_bits is None:
            raise ValueError("Either 'exact_bits' or 'max_bits' must be specified")

        if exact_bits is not None and max_bits is not None:
            raise ValueError("'exact_bits' and 'max_bits' are mutually exclusive")

        bits = exact_bits or max_bits
        bytes_needed = ((bits - 1) // 8) + 1
        significant_bits_msb = 8 - (bytes_needed * 8 - bits)
        msb = bord(randfunc(1)[0])
        if exact_bits is not None:
            msb |= 1 << (significant_bits_msb - 1)
        msb &= (1 << significant_bits_msb) - 1

        return cls.from_bytes(bchr(msb) + randfunc(bytes_needed - 1))

    @classmethod
    def random_range(cls, **kwargs):
        """Generate a random integer within a given internal.

        :Keywords:
          min_inclusive : integer
            The lower end of the interval (inclusive).
          max_inclusive : integer
            The higher end of the interval (inclusive).
          max_exclusive : integer
            The higher end of the interval (exclusive).
          randfunc : callable
            A function that returns a random byte string. The length of the
            byte string is passed as parameter. Optional.
            If not provided (or ``None``), randomness is read from the system RNG.
        :Returns:
            An Integer randomly taken in the given interval.
        """

        min_inclusive = kwargs.pop("min_inclusive", None)
        max_inclusive = kwargs.pop("max_inclusive", None)
        max_exclusive = kwargs.pop("max_exclusive", None)
        randfunc = kwargs.pop("randfunc", None)

        if kwargs:
            raise ValueError("Unknown keywords: " + str(kwargs.keys))
        if None not in (max_inclusive, max_exclusive):
            raise ValueError("max_inclusive and max_exclusive cannot be both"
                         " specified")
        if max_exclusive is not None:
            max_inclusive = max_exclusive - 1
        if None in (min_inclusive, max_inclusive):
            raise ValueError("Missing keyword to identify the interval")

        if randfunc is None:
            randfunc = Random.new().read

        norm_maximum = max_inclusive - min_inclusive
        bits_needed = cls(norm_maximum).size_in_bits()

        norm_candidate = -1
        while not 0 <= norm_candidate <= norm_maximum:
            norm_candidate = cls.random(
                                    max_bits=bits_needed,
                                    randfunc=randfunc
                                    )
        return norm_candidate + min_inclusive

