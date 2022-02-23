#
# SecretSharing.py : distribute a secret amongst a group of participants
#
# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
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

from Crypto.Util.py3compat import is_native_int
from Crypto.Util import number
from Crypto.Util.number import long_to_bytes, bytes_to_long
from Crypto.Random import get_random_bytes as rng


def _mult_gf2(f1, f2):
    """Multiply two polynomials in GF(2)"""

    # Ensure f2 is the smallest
    if f2 > f1:
        f1, f2 = f2, f1
    z = 0
    while f2:
        if f2 & 1:
            z ^= f1
        f1 <<= 1
        f2 >>= 1
    return z


def _div_gf2(a, b):
    """
    Compute division of polynomials over GF(2).
    Given a and b, it finds two polynomials q and r such that:

    a = b*q + r with deg(r)<deg(b)
    """

    if (a < b):
        return 0, a

    deg = number.size
    q = 0
    r = a
    d = deg(b)
    while deg(r) >= d:
        s = 1 << (deg(r) - d)
        q ^= s
        r ^= _mult_gf2(b, s)
    return (q, r)


class _Element(object):
    """Element of GF(2^128) field"""

    # The irreducible polynomial defining this field is 1+x+x^2+x^7+x^128
    irr_poly = 1 + 2 + 4 + 128 + 2 ** 128

    def __init__(self, encoded_value):
        """Initialize the element to a certain value.

        The value passed as parameter is internally encoded as
        a 128-bit integer, where each bit represents a polynomial
        coefficient. The LSB is the constant coefficient.
        """

        if is_native_int(encoded_value):
            self._value = encoded_value
        elif len(encoded_value) == 16:
            self._value = bytes_to_long(encoded_value)
        else:
            raise ValueError("The encoded value must be an integer or a 16 byte string")

    def __eq__(self, other):
        return self._value == other._value

    def __int__(self):
        """Return the field element, encoded as a 128-bit integer."""
        return self._value

    def encode(self):
        """Return the field element, encoded as a 16 byte string."""
        return long_to_bytes(self._value, 16)

    def __mul__(self, factor):

        f1 = self._value
        f2 = factor._value

        # Make sure that f2 is the smallest, to speed up the loop
        if f2 > f1:
            f1, f2 = f2, f1

        if self.irr_poly in (f1, f2):
            return _Element(0)

        mask1 = 2 ** 128
        v, z = f1, 0
        while f2:
            # if f2 ^ 1: z ^= v
            mask2 = int(bin(f2 & 1)[2:] * 128, base=2)
            z = (mask2 & (z ^ v)) | ((mask1 - mask2 - 1) & z)
            v <<= 1
            # if v & mask1: v ^= self.irr_poly
            mask3 = int(bin((v >> 128) & 1)[2:] * 128, base=2)
            v = (mask3 & (v ^ self.irr_poly)) | ((mask1 - mask3 - 1) & v)
            f2 >>= 1
        return _Element(z)

    def __add__(self, term):
        return _Element(self._value ^ term._value)

    def inverse(self):
        """Return the inverse of this element in GF(2^128)."""

        # We use the Extended GCD algorithm
        # http://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor

        if self._value == 0:
            raise ValueError("Inversion of zero")

        r0, r1 = self._value, self.irr_poly
        s0, s1 = 1, 0
        while r1 > 0:
            q = _div_gf2(r0, r1)[0]
            r0, r1 = r1, r0 ^ _mult_gf2(q, r1)
            s0, s1 = s1, s0 ^ _mult_gf2(q, s1)
        return _Element(s0)

    def __pow__(self, exponent):
        result = _Element(self._value)
        for _ in range(exponent - 1):
            result = result * self
        return result


class Shamir(object):
    """Shamir's secret sharing scheme.

    A secret is split into ``n`` shares, and it is sufficient to collect
    ``k`` of them to reconstruct the secret.
    """

    @staticmethod
    def split(k, n, secret, ssss=False):
        """Split a secret into ``n`` shares.

        The secret can be reconstructed later using just ``k`` shares
        out of the original ``n``.
        Each share must be kept confidential to the person it was
        assigned to.

        Each share is associated to an index (starting from 1).

        Args:
          k (integer):
            The sufficient number of shares to reconstruct the secret (``k < n``).
          n (integer):
            The number of shares that this method will create.
          secret (byte string):
            A byte string of 16 bytes (e.g. the AES 128 key).
          ssss (bool):
            If ``True``, the shares can be used with the ``ssss`` utility.
            Default: ``False``.

        Return (tuples):
            ``n`` tuples. A tuple is meant for each participant and it contains two items:

            1. the unique index (an integer)
            2. the share (a byte string, 16 bytes)
        """

        #
        # We create a polynomial with random coefficients in GF(2^128):
        #
        # p(x) = \sum_{i=0}^{k-1} c_i * x^i
        #
        # c_0 is the encoded secret
        #

        coeffs = [_Element(rng(16)) for i in range(k - 1)]
        coeffs.append(_Element(secret))

        # Each share is y_i = p(x_i) where x_i is the public index
        # associated to each of the n users.

        def make_share(user, coeffs, ssss):
            idx = _Element(user)
            share = _Element(0)
            for coeff in coeffs:
                share = idx * share + coeff
            if ssss:
                share += _Element(user) ** len(coeffs)
            return share.encode()

        return [(i, make_share(i, coeffs, ssss)) for i in range(1, n + 1)]

    @staticmethod
    def combine(shares, ssss=False):
        """Recombine a secret, if enough shares are presented.

        Args:
          shares (tuples):
            The *k* tuples, each containin the index (an integer) and
            the share (a byte string, 16 bytes long) that were assigned to
            a participant.
          ssss (bool):
            If ``True``, the shares were produced by the ``ssss`` utility.
            Default: ``False``.

        Return:
            The original secret, as a byte string (16 bytes long).
        """

        #
        # Given k points (x,y), the interpolation polynomial of degree k-1 is:
        #
        # L(x) = \sum_{j=0}^{k-1} y_i * l_j(x)
        #
        # where:
        #
        # l_j(x) = \prod_{ \overset{0 \le m \le k-1}{m \ne j} }
        #          \frac{x - x_m}{x_j - x_m}
        #
        # However, in this case we are purely interested in the constant
        # coefficient of L(x).
        #

        k = len(shares)

        gf_shares = []
        for x in shares:
            idx = _Element(x[0])
            value = _Element(x[1])
            if any(y[0] == idx for y in gf_shares):
                raise ValueError("Duplicate share")
            if ssss:
                value += idx ** k
            gf_shares.append((idx, value))

        result = _Element(0)
        for j in range(k):
            x_j, y_j = gf_shares[j]

            numerator = _Element(1)
            denominator = _Element(1)

            for m in range(k):
                x_m = gf_shares[m][0]
                if m != j:
                    numerator *= x_m
                    denominator *= x_j + x_m
            result += y_j * numerator * denominator.inverse()
        return result.encode()
