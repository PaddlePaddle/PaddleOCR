# -*- coding: utf-8 -*-
#
#  PublicKey/DSA.py : DSA signature primitive
#
# Written in 2008 by Dwayne C. Litzenberger <dlitz@dlitz.net>
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

__all__ = ['generate', 'construct', 'DsaKey', 'import_key' ]

import binascii
import struct
import itertools

from Crypto.Util.py3compat import bchr, bord, tobytes, tostr, iter_range

from Crypto import Random
from Crypto.IO import PKCS8, PEM
from Crypto.Hash import SHA256
from Crypto.Util.asn1 import (
                DerObject, DerSequence,
                DerInteger, DerObjectId,
                DerBitString,
                )

from Crypto.Math.Numbers import Integer
from Crypto.Math.Primality import (test_probable_prime, COMPOSITE,
                                   PROBABLY_PRIME)

from Crypto.PublicKey import (_expand_subject_public_key_info,
                              _create_subject_public_key_info,
                              _extract_subject_public_key_info)

#   ; The following ASN.1 types are relevant for DSA
#
#   SubjectPublicKeyInfo    ::=     SEQUENCE {
#       algorithm   AlgorithmIdentifier,
#       subjectPublicKey BIT STRING
#   }
#
#   id-dsa ID ::= { iso(1) member-body(2) us(840) x9-57(10040) x9cm(4) 1 }
#
#   ; See RFC3279
#   Dss-Parms  ::=  SEQUENCE  {
#       p INTEGER,
#       q INTEGER,
#       g INTEGER
#   }
#
#   DSAPublicKey ::= INTEGER
#
#   DSSPrivatKey_OpenSSL ::= SEQUENCE
#       version INTEGER,
#       p INTEGER,
#       q INTEGER,
#       g INTEGER,
#       y INTEGER,
#       x INTEGER
#   }
#

class DsaKey(object):
    r"""Class defining an actual DSA key.
    Do not instantiate directly.
    Use :func:`generate`, :func:`construct` or :func:`import_key` instead.

    :ivar p: DSA modulus
    :vartype p: integer

    :ivar q: Order of the subgroup
    :vartype q: integer

    :ivar g: Generator
    :vartype g: integer

    :ivar y: Public key
    :vartype y: integer

    :ivar x: Private key
    :vartype x: integer

    :undocumented: exportKey, publickey
    """

    _keydata = ['y', 'g', 'p', 'q', 'x']

    def __init__(self, key_dict):
        input_set = set(key_dict.keys())
        public_set = set(('y' , 'g', 'p', 'q'))
        if not public_set.issubset(input_set):
            raise ValueError("Some DSA components are missing = %s" %
                             str(public_set - input_set))
        extra_set = input_set - public_set
        if extra_set and extra_set != set(('x',)):
            raise ValueError("Unknown DSA components = %s" %
                             str(extra_set - set(('x',))))
        self._key = dict(key_dict)

    def _sign(self, m, k):
        if not self.has_private():
            raise TypeError("DSA public key cannot be used for signing")
        if not (1 < k < self.q):
            raise ValueError("k is not between 2 and q-1")

        x, q, p, g = [self._key[comp] for comp in ['x', 'q', 'p', 'g']]

        blind_factor = Integer.random_range(min_inclusive=1,
                                           max_exclusive=q)
        inv_blind_k = (blind_factor * k).inverse(q)
        blind_x = x * blind_factor

        r = pow(g, k, p) % q  # r = (g**k mod p) mod q
        s = (inv_blind_k * (blind_factor * m + blind_x * r)) % q
        return map(int, (r, s))

    def _verify(self, m, sig):
        r, s = sig
        y, q, p, g = [self._key[comp] for comp in ['y', 'q', 'p', 'g']]
        if not (0 < r < q) or not (0 < s < q):
            return False
        w = Integer(s).inverse(q)
        u1 = (w * m) % q
        u2 = (w * r) % q
        v = (pow(g, u1, p) * pow(y, u2, p) % p) % q
        return v == r

    def has_private(self):
        """Whether this is a DSA private key"""

        return 'x' in self._key

    def can_encrypt(self):  # legacy
        return False

    def can_sign(self):     # legacy
        return True

    def public_key(self):
        """A matching DSA public key.

        Returns:
            a new :class:`DsaKey` object
        """

        public_components = dict((k, self._key[k]) for k in ('y', 'g', 'p', 'q'))
        return DsaKey(public_components)

    def __eq__(self, other):
        if bool(self.has_private()) != bool(other.has_private()):
            return False

        result = True
        for comp in self._keydata:
            result = result and (getattr(self._key, comp, None) ==
                                 getattr(other._key, comp, None))
        return result

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        # DSA key is not pickable
        from pickle import PicklingError
        raise PicklingError

    def domain(self):
        """The DSA domain parameters.

        Returns
            tuple : (p,q,g)
        """

        return [int(self._key[comp]) for comp in ('p', 'q', 'g')]

    def __repr__(self):
        attrs = []
        for k in self._keydata:
            if k == 'p':
                bits = Integer(self.p).size_in_bits()
                attrs.append("p(%d)" % (bits,))
            elif hasattr(self, k):
                attrs.append(k)
        if self.has_private():
            attrs.append("private")
        # PY3K: This is meant to be text, do not change to bytes (data)
        return "<%s @0x%x %s>" % (self.__class__.__name__, id(self), ",".join(attrs))

    def __getattr__(self, item):
        try:
            return int(self._key[item])
        except KeyError:
            raise AttributeError(item)

    def export_key(self, format='PEM', pkcs8=None, passphrase=None,
                  protection=None, randfunc=None):
        """Export this DSA key.

        Args:
          format (string):
            The encoding for the output:

            - *'PEM'* (default). ASCII as per `RFC1421`_/ `RFC1423`_.
            - *'DER'*. Binary ASN.1 encoding.
            - *'OpenSSH'*. ASCII one-liner as per `RFC4253`_.
              Only suitable for public keys, not for private keys.

          passphrase (string):
            *Private keys only*. The pass phrase to protect the output.

          pkcs8 (boolean):
            *Private keys only*. If ``True`` (default), the key is encoded
            with `PKCS#8`_. If ``False``, it is encoded in the custom
            OpenSSL/OpenSSH container.

          protection (string):
            *Only in combination with a pass phrase*.
            The encryption scheme to use to protect the output.

            If :data:`pkcs8` takes value ``True``, this is the PKCS#8
            algorithm to use for deriving the secret and encrypting
            the private DSA key.
            For a complete list of algorithms, see :mod:`Crypto.IO.PKCS8`.
            The default is *PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC*.

            If :data:`pkcs8` is ``False``, the obsolete PEM encryption scheme is
            used. It is based on MD5 for key derivation, and Triple DES for
            encryption. Parameter :data:`protection` is then ignored.

            The combination ``format='DER'`` and ``pkcs8=False`` is not allowed
            if a passphrase is present.

          randfunc (callable):
            A function that returns random bytes.
            By default it is :func:`Crypto.Random.get_random_bytes`.

        Returns:
          byte string : the encoded key

        Raises:
          ValueError : when the format is unknown or when you try to encrypt a private
            key with *DER* format and OpenSSL/OpenSSH.

        .. warning::
            If you don't provide a pass phrase, the private key will be
            exported in the clear!

        .. _RFC1421:    http://www.ietf.org/rfc/rfc1421.txt
        .. _RFC1423:    http://www.ietf.org/rfc/rfc1423.txt
        .. _RFC4253:    http://www.ietf.org/rfc/rfc4253.txt
        .. _`PKCS#8`:   http://www.ietf.org/rfc/rfc5208.txt
        """

        if passphrase is not None:
            passphrase = tobytes(passphrase)

        if randfunc is None:
            randfunc = Random.get_random_bytes

        if format == 'OpenSSH':
            tup1 = [self._key[x].to_bytes() for x in ('p', 'q', 'g', 'y')]

            def func(x):
                if (bord(x[0]) & 0x80):
                    return bchr(0) + x
                else:
                    return x

            tup2 = [func(x) for x in tup1]
            keyparts = [b'ssh-dss'] + tup2
            keystring = b''.join(
                            [struct.pack(">I", len(kp)) + kp for kp in keyparts]
                            )
            return b'ssh-dss ' + binascii.b2a_base64(keystring)[:-1]

        # DER format is always used, even in case of PEM, which simply
        # encodes it into BASE64.
        params = DerSequence([self.p, self.q, self.g])
        if self.has_private():
            if pkcs8 is None:
                pkcs8 = True
            if pkcs8:
                if not protection:
                    protection = 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC'
                private_key = DerInteger(self.x).encode()
                binary_key = PKCS8.wrap(
                                private_key, oid, passphrase,
                                protection, key_params=params,
                                randfunc=randfunc
                                )
                if passphrase:
                    key_type = 'ENCRYPTED PRIVATE'
                else:
                    key_type = 'PRIVATE'
                passphrase = None
            else:
                if format != 'PEM' and passphrase:
                    raise ValueError("DSA private key cannot be encrypted")
                ints = [0, self.p, self.q, self.g, self.y, self.x]
                binary_key = DerSequence(ints).encode()
                key_type = "DSA PRIVATE"
        else:
            if pkcs8:
                raise ValueError("PKCS#8 is only meaningful for private keys")

            binary_key = _create_subject_public_key_info(oid,
                                DerInteger(self.y), params)
            key_type = "PUBLIC"

        if format == 'DER':
            return binary_key
        if format == 'PEM':
            pem_str = PEM.encode(
                                binary_key, key_type + " KEY",
                                passphrase, randfunc
                            )
            return tobytes(pem_str)
        raise ValueError("Unknown key format '%s'. Cannot export the DSA key." % format)

    # Backward-compatibility
    exportKey = export_key
    publickey = public_key

    # Methods defined in PyCrypto that we don't support anymore

    def sign(self, M, K):
        raise NotImplementedError("Use module Crypto.Signature.DSS instead")

    def verify(self, M, signature):
        raise NotImplementedError("Use module Crypto.Signature.DSS instead")

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


def _generate_domain(L, randfunc):
    """Generate a new set of DSA domain parameters"""

    N = { 1024:160, 2048:224, 3072:256 }.get(L)
    if N is None:
        raise ValueError("Invalid modulus length (%d)" % L)

    outlen = SHA256.digest_size * 8
    n = (L + outlen - 1) // outlen - 1  # ceil(L/outlen) -1
    b_ = L - 1 - (n * outlen)

    # Generate q (A.1.1.2)
    q = Integer(4)
    upper_bit = 1 << (N - 1)
    while test_probable_prime(q, randfunc) != PROBABLY_PRIME:
        seed = randfunc(64)
        U = Integer.from_bytes(SHA256.new(seed).digest()) & (upper_bit - 1)
        q = U | upper_bit | 1

    assert(q.size_in_bits() == N)

    # Generate p (A.1.1.2)
    offset = 1
    upper_bit = 1 << (L - 1)
    while True:
        V = [ SHA256.new(seed + Integer(offset + j).to_bytes()).digest()
              for j in iter_range(n + 1) ]
        V = [ Integer.from_bytes(v) for v in V ]
        W = sum([V[i] * (1 << (i * outlen)) for i in iter_range(n)],
                (V[n] & ((1 << b_) - 1)) * (1 << (n * outlen)))

        X = Integer(W + upper_bit) # 2^{L-1} < X < 2^{L}
        assert(X.size_in_bits() == L)

        c = X % (q * 2)
        p = X - (c - 1)  # 2q divides (p-1)
        if p.size_in_bits() == L and \
           test_probable_prime(p, randfunc) == PROBABLY_PRIME:
               break
        offset += n + 1

    # Generate g (A.2.3, index=1)
    e = (p - 1) // q
    for count in itertools.count(1):
        U = seed + b"ggen" + bchr(1) + Integer(count).to_bytes()
        W = Integer.from_bytes(SHA256.new(U).digest())
        g = pow(W, e, p)
        if g != 1:
            break

    return (p, q, g, seed)


def generate(bits, randfunc=None, domain=None):
    """Generate a new DSA key pair.

    The algorithm follows Appendix A.1/A.2 and B.1 of `FIPS 186-4`_,
    respectively for domain generation and key pair generation.

    Args:
      bits (integer):
        Key length, or size (in bits) of the DSA modulus *p*.
        It must be 1024, 2048 or 3072.

      randfunc (callable):
        Random number generation function; it accepts a single integer N
        and return a string of random data N bytes long.
        If not specified, :func:`Crypto.Random.get_random_bytes` is used.

      domain (tuple):
        The DSA domain parameters *p*, *q* and *g* as a list of 3
        integers. Size of *p* and *q* must comply to `FIPS 186-4`_.
        If not specified, the parameters are created anew.

    Returns:
      :class:`DsaKey` : a new DSA key object

    Raises:
      ValueError : when **bits** is too little, too big, or not a multiple of 64.

    .. _FIPS 186-4: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    """

    if randfunc is None:
        randfunc = Random.get_random_bytes

    if domain:
        p, q, g = map(Integer, domain)

        ## Perform consistency check on domain parameters
        # P and Q must be prime
        fmt_error = test_probable_prime(p) == COMPOSITE
        fmt_error |= test_probable_prime(q) == COMPOSITE
        # Verify Lagrange's theorem for sub-group
        fmt_error |= ((p - 1) % q) != 0
        fmt_error |= g <= 1 or g >= p
        fmt_error |= pow(g, q, p) != 1
        if fmt_error:
            raise ValueError("Invalid DSA domain parameters")
    else:
        p, q, g, _ = _generate_domain(bits, randfunc)

    L = p.size_in_bits()
    N = q.size_in_bits()

    if L != bits:
        raise ValueError("Mismatch between size of modulus (%d)"
                         " and 'bits' parameter (%d)" % (L, bits))

    if (L, N) not in [(1024, 160), (2048, 224),
                      (2048, 256), (3072, 256)]:
        raise ValueError("Lengths of p and q (%d, %d) are not compatible"
                         "to FIPS 186-3" % (L, N))

    if not 1 < g < p:
        raise ValueError("Incorrent DSA generator")

    # B.1.1
    c = Integer.random(exact_bits=N + 64, randfunc=randfunc)
    x = c % (q - 1) + 1 # 1 <= x <= q-1
    y = pow(g, x, p)

    key_dict = { 'y':y, 'g':g, 'p':p, 'q':q, 'x':x }
    return DsaKey(key_dict)


def construct(tup, consistency_check=True):
    """Construct a DSA key from a tuple of valid DSA components.

    Args:
      tup (tuple):
        A tuple of long integers, with 4 or 5 items
        in the following order:

            1. Public key (*y*).
            2. Sub-group generator (*g*).
            3. Modulus, finite field order (*p*).
            4. Sub-group order (*q*).
            5. Private key (*x*). Optional.

      consistency_check (boolean):
        If ``True``, the library will verify that the provided components
        fulfil the main DSA properties.

    Raises:
      ValueError: when the key being imported fails the most basic DSA validity checks.

    Returns:
      :class:`DsaKey` : a DSA key object
    """

    key_dict = dict(zip(('y', 'g', 'p', 'q', 'x'), map(Integer, tup)))
    key = DsaKey(key_dict)

    fmt_error = False
    if consistency_check:
        # P and Q must be prime
        fmt_error = test_probable_prime(key.p) == COMPOSITE
        fmt_error |= test_probable_prime(key.q) == COMPOSITE
        # Verify Lagrange's theorem for sub-group
        fmt_error |= ((key.p - 1) % key.q) != 0
        fmt_error |= key.g <= 1 or key.g >= key.p
        fmt_error |= pow(key.g, key.q, key.p) != 1
        # Public key
        fmt_error |= key.y <= 0 or key.y >= key.p
        if hasattr(key, 'x'):
            fmt_error |= key.x <= 0 or key.x >= key.q
            fmt_error |= pow(key.g, key.x, key.p) != key.y

    if fmt_error:
        raise ValueError("Invalid DSA key components")

    return key


# Dss-Parms  ::=  SEQUENCE  {
#       p       OCTET STRING,
#       q       OCTET STRING,
#       g       OCTET STRING
# }
# DSAPublicKey ::= INTEGER --  public key, y

def _import_openssl_private(encoded, passphrase, params):
    if params:
        raise ValueError("DSA private key already comes with parameters")
    der = DerSequence().decode(encoded, nr_elements=6, only_ints_expected=True)
    if der[0] != 0:
        raise ValueError("No version found")
    tup = [der[comp] for comp in (4, 3, 1, 2, 5)]
    return construct(tup)


def _import_subjectPublicKeyInfo(encoded, passphrase, params):

    algoid, encoded_key, emb_params =  _expand_subject_public_key_info(encoded)
    if algoid != oid:
        raise ValueError("No DSA subjectPublicKeyInfo")
    if params and emb_params:
        raise ValueError("Too many DSA parameters")

    y = DerInteger().decode(encoded_key).value
    p, q, g = list(DerSequence().decode(params or emb_params))
    tup = (y, g, p, q)
    return construct(tup)


def _import_x509_cert(encoded, passphrase, params):

    sp_info = _extract_subject_public_key_info(encoded)
    return _import_subjectPublicKeyInfo(sp_info, None, params)


def _import_pkcs8(encoded, passphrase, params):
    if params:
        raise ValueError("PKCS#8 already includes parameters")
    k = PKCS8.unwrap(encoded, passphrase)
    if k[0] != oid:
        raise ValueError("No PKCS#8 encoded DSA key")
    x = DerInteger().decode(k[1]).value
    p, q, g = list(DerSequence().decode(k[2]))
    tup = (pow(g, x, p), g, p, q, x)
    return construct(tup)


def _import_key_der(key_data, passphrase, params):
    """Import a DSA key (public or private half), encoded in DER form."""

    decodings = (_import_openssl_private,
                 _import_subjectPublicKeyInfo,
                 _import_x509_cert,
                 _import_pkcs8)

    for decoding in decodings:
        try:
            return decoding(key_data, passphrase, params)
        except ValueError:
            pass

    raise ValueError("DSA key format is not supported")


def import_key(extern_key, passphrase=None):
    """Import a DSA key.

    Args:
      extern_key (string or byte string):
        The DSA key to import.

        The following formats are supported for a DSA **public** key:

        - X.509 certificate (binary DER or PEM)
        - X.509 ``subjectPublicKeyInfo`` (binary DER or PEM)
        - OpenSSH (ASCII one-liner, see `RFC4253`_)

        The following formats are supported for a DSA **private** key:

        - `PKCS#8`_ ``PrivateKeyInfo`` or ``EncryptedPrivateKeyInfo``
          DER SEQUENCE (binary or PEM)
        - OpenSSL/OpenSSH custom format (binary or PEM)

        For details about the PEM encoding, see `RFC1421`_/`RFC1423`_.

      passphrase (string):
        In case of an encrypted private key, this is the pass phrase
        from which the decryption key is derived.

        Encryption may be applied either at the `PKCS#8`_ or at the PEM level.

    Returns:
      :class:`DsaKey` : a DSA key object

    Raises:
      ValueError : when the given key cannot be parsed (possibly because
        the pass phrase is wrong).

    .. _RFC1421: http://www.ietf.org/rfc/rfc1421.txt
    .. _RFC1423: http://www.ietf.org/rfc/rfc1423.txt
    .. _RFC4253: http://www.ietf.org/rfc/rfc4253.txt
    .. _PKCS#8: http://www.ietf.org/rfc/rfc5208.txt
    """

    extern_key = tobytes(extern_key)
    if passphrase is not None:
        passphrase = tobytes(passphrase)

    if extern_key.startswith(b'-----'):
        # This is probably a PEM encoded key
        (der, marker, enc_flag) = PEM.decode(tostr(extern_key), passphrase)
        if enc_flag:
            passphrase = None
        return _import_key_der(der, passphrase, None)

    if extern_key.startswith(b'ssh-dss '):
        # This is probably a public OpenSSH key
        keystring = binascii.a2b_base64(extern_key.split(b' ')[1])
        keyparts = []
        while len(keystring) > 4:
            length = struct.unpack(">I", keystring[:4])[0]
            keyparts.append(keystring[4:4 + length])
            keystring = keystring[4 + length:]
        if keyparts[0] == b"ssh-dss":
            tup = [Integer.from_bytes(keyparts[x]) for x in (4, 3, 1, 2)]
            return construct(tup)

    if len(extern_key) > 0 and bord(extern_key[0]) == 0x30:
        # This is probably a DER encoded key
        return _import_key_der(extern_key, passphrase, None)

    raise ValueError("DSA key format is not supported")


# Backward compatibility
importKey = import_key

#: `Object ID`_ for a DSA key.
#:
#: id-dsa ID ::= { iso(1) member-body(2) us(840) x9-57(10040) x9cm(4) 1 }
#:
#: .. _`Object ID`: http://www.alvestrand.no/objectid/1.2.840.10040.4.1.html
oid = "1.2.840.10040.4.1"
