#
#   ElGamal.py : ElGamal encryption/decryption and signatures
#
#  Part of the Python Cryptography Toolkit
#
#  Originally written by: A.M. Kuchling
#
# ===================================================================
# The contents of this file are dedicated to the public domain.  To
# the extent that dedication to the public domain is not available,
# everyone is granted a worldwide, perpetual, royalty-free,
# non-exclusive license to exercise all rights associated with the
# contents of this file for any purpose whatsoever.
# No rights are reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================

__all__ = ['generate', 'construct', 'ElGamalKey']

from Crypto import Random
from Crypto.Math.Primality import ( generate_probable_safe_prime,
                                    test_probable_prime, COMPOSITE )
from Crypto.Math.Numbers import Integer

# Generate an ElGamal key with N bits
def generate(bits, randfunc):
    """Randomly generate a fresh, new ElGamal key.

    The key will be safe for use for both encryption and signature
    (although it should be used for **only one** purpose).

    Args:
      bits (int):
        Key length, or size (in bits) of the modulus *p*.
        The recommended value is 2048.
      randfunc (callable):
        Random number generation function; it should accept
        a single integer *N* and return a string of random
        *N* random bytes.

    Return:
        an :class:`ElGamalKey` object
    """

    obj=ElGamalKey()

    # Generate a safe prime p
    # See Algorithm 4.86 in Handbook of Applied Cryptography
    obj.p = generate_probable_safe_prime(exact_bits=bits, randfunc=randfunc)
    q = (obj.p - 1) >> 1

    # Generate generator g
    while 1:
        # Choose a square residue; it will generate a cyclic group of order q.
        obj.g = pow(Integer.random_range(min_inclusive=2,
                                     max_exclusive=obj.p,
                                     randfunc=randfunc), 2, obj.p)

        # We must avoid g=2 because of Bleichenbacher's attack described
        # in "Generating ElGamal signatures without knowning the secret key",
        # 1996
        if obj.g in (1, 2):
            continue

        # Discard g if it divides p-1 because of the attack described
        # in Note 11.67 (iii) in HAC
        if (obj.p - 1) % obj.g == 0:
            continue

        # g^{-1} must not divide p-1 because of Khadir's attack
        # described in "Conditions of the generator for forging ElGamal
        # signature", 2011
        ginv = obj.g.inverse(obj.p)
        if (obj.p - 1) % ginv == 0:
            continue

        # Found
        break

    # Generate private key x
    obj.x = Integer.random_range(min_inclusive=2,
                                 max_exclusive=obj.p-1,
                                 randfunc=randfunc)
    # Generate public key y
    obj.y = pow(obj.g, obj.x, obj.p)
    return obj

def construct(tup):
    r"""Construct an ElGamal key from a tuple of valid ElGamal components.

    The modulus *p* must be a prime.
    The following conditions must apply:

    .. math::

        \begin{align}
        &1 < g < p-1 \\
        &g^{p-1} = 1 \text{ mod } 1 \\
        &1 < x < p-1 \\
        &g^x = y \text{ mod } p
        \end{align}

    Args:
      tup (tuple):
        A tuple with either 3 or 4 integers,
        in the following order:

        1. Modulus (*p*).
        2. Generator (*g*).
        3. Public key (*y*).
        4. Private key (*x*). Optional.

    Raises:
        ValueError: when the key being imported fails the most basic ElGamal validity checks.

    Returns:
        an :class:`ElGamalKey` object
    """

    obj=ElGamalKey()
    if len(tup) not in [3,4]:
        raise ValueError('argument for construct() wrong length')
    for i in range(len(tup)):
        field = obj._keydata[i]
        setattr(obj, field, Integer(tup[i]))

    fmt_error = test_probable_prime(obj.p) == COMPOSITE
    fmt_error |= obj.g<=1 or obj.g>=obj.p
    fmt_error |= pow(obj.g, obj.p-1, obj.p)!=1
    fmt_error |= obj.y<1 or obj.y>=obj.p
    if len(tup)==4:
        fmt_error |= obj.x<=1 or obj.x>=obj.p
        fmt_error |= pow(obj.g, obj.x, obj.p)!=obj.y

    if fmt_error:
        raise ValueError("Invalid ElGamal key components")

    return obj

class ElGamalKey(object):
    r"""Class defining an ElGamal key.
    Do not instantiate directly.
    Use :func:`generate` or :func:`construct` instead.

    :ivar p: Modulus
    :vartype d: integer

    :ivar g: Generator
    :vartype e: integer

    :ivar y: Public key component
    :vartype y: integer

    :ivar x: Private key component
    :vartype x: integer
    """

    #: Dictionary of ElGamal parameters.
    #:
    #: A public key will only have the following entries:
    #:
    #:  - **y**, the public key.
    #:  - **g**, the generator.
    #:  - **p**, the modulus.
    #:
    #: A private key will also have:
    #:
    #:  - **x**, the private key.
    _keydata=['p', 'g', 'y', 'x']

    def __init__(self, randfunc=None):
        if randfunc is None:
            randfunc = Random.new().read
        self._randfunc = randfunc

    def _encrypt(self, M, K):
        a=pow(self.g, K, self.p)
        b=( pow(self.y, K, self.p)*M ) % self.p
        return [int(a), int(b)]

    def _decrypt(self, M):
        if (not hasattr(self, 'x')):
            raise TypeError('Private key not available in this object')
        r = Integer.random_range(min_inclusive=2,
                                 max_exclusive=self.p-1,
                                 randfunc=self._randfunc)
        a_blind = (pow(self.g, r, self.p) * M[0]) % self.p
        ax=pow(a_blind, self.x, self.p)
        plaintext_blind = (ax.inverse(self.p) * M[1] ) % self.p
        plaintext = (plaintext_blind * pow(self.y, r, self.p)) % self.p
        return int(plaintext)

    def _sign(self, M, K):
        if (not hasattr(self, 'x')):
            raise TypeError('Private key not available in this object')
        p1=self.p-1
        K = Integer(K)
        if (K.gcd(p1)!=1):
            raise ValueError('Bad K value: GCD(K,p-1)!=1')
        a=pow(self.g, K, self.p)
        t=(Integer(M)-self.x*a) % p1
        while t<0: t=t+p1
        b=(t*K.inverse(p1)) % p1
        return [int(a), int(b)]

    def _verify(self, M, sig):
        sig = [Integer(x) for x in sig]
        if sig[0]<1 or sig[0]>self.p-1:
            return 0
        v1=pow(self.y, sig[0], self.p)
        v1=(v1*pow(sig[0], sig[1], self.p)) % self.p
        v2=pow(self.g, M, self.p)
        if v1==v2:
            return 1
        return 0

    def has_private(self):
        """Whether this is an ElGamal private key"""

        if hasattr(self, 'x'):
            return 1
        else:
            return 0

    def can_encrypt(self):
        return True

    def can_sign(self):
        return True

    def publickey(self):
        """A matching ElGamal public key.

        Returns:
            a new :class:`ElGamalKey` object
        """
        return construct((self.p, self.g, self.y))

    def __eq__(self, other):
        if bool(self.has_private()) != bool(other.has_private()):
            return False

        result = True
        for comp in self._keydata:
            result = result and (getattr(self.key, comp, None) ==
                                 getattr(other.key, comp, None))
        return result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        # ElGamal key is not pickable
        from pickle import PicklingError
        raise PicklingError

    # Methods defined in PyCrypto that we don't support anymore

    def sign(self, M, K):
        raise NotImplementedError

    def verify(self, M, signature):
        raise NotImplementedError

    def encrypt(self, plaintext, K):
        raise NotImplementedError

    def decrypt(self, ciphertext):
        raise NotImplementedError

    def blind(self, M, B):
        raise NotImplementedError

    def unblind(self, M, B):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError
