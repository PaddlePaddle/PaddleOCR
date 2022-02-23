#
#  Signature/DSS.py : DSS.py
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

from Crypto.Util.asn1 import DerSequence
from Crypto.Util.number import long_to_bytes
from Crypto.Math.Numbers import Integer

from Crypto.Hash import HMAC
from Crypto.PublicKey.ECC import EccKey
from Crypto.PublicKey.DSA import DsaKey

__all__ = ['DssSigScheme', 'new']


class DssSigScheme(object):
    """A (EC)DSA signature object.
    Do not instantiate directly.
    Use :func:`Crypto.Signature.DSS.new`.
    """

    def __init__(self, key, encoding, order):
        """Create a new Digital Signature Standard (DSS) object.

        Do not instantiate this object directly,
        use `Crypto.Signature.DSS.new` instead.
        """

        self._key = key
        self._encoding = encoding
        self._order = order

        self._order_bits = self._order.size_in_bits()
        self._order_bytes = (self._order_bits - 1) // 8 + 1

    def can_sign(self):
        """Return ``True`` if this signature object can be used
        for signing messages."""

        return self._key.has_private()

    def _compute_nonce(self, msg_hash):
        raise NotImplementedError("To be provided by subclasses")

    def _valid_hash(self, msg_hash):
        raise NotImplementedError("To be provided by subclasses")

    def sign(self, msg_hash):
        """Compute the DSA/ECDSA signature of a message.

        Args:
          msg_hash (hash object):
            The hash that was carried out over the message.
            The object belongs to the :mod:`Crypto.Hash` package.
            Under mode ``'fips-186-3'``, the hash must be a FIPS
            approved secure hash (SHA-2 or SHA-3).

        :return: The signature as ``bytes``
        :raise ValueError: if the hash algorithm is incompatible to the (EC)DSA key
        :raise TypeError: if the (EC)DSA key has no private half
        """

        if not self._key.has_private():
            raise TypeError("Private key is needed to sign")

        if not self._valid_hash(msg_hash):
            raise ValueError("Hash is not sufficiently strong")

        # Generate the nonce k (critical!)
        nonce = self._compute_nonce(msg_hash)

        # Perform signature using the raw API
        z = Integer.from_bytes(msg_hash.digest()[:self._order_bytes])
        sig_pair = self._key._sign(z, nonce)

        # Encode the signature into a single byte string
        if self._encoding == 'binary':
            output = b"".join([long_to_bytes(x, self._order_bytes)
                               for x in sig_pair])
        else:
            # Dss-sig  ::=  SEQUENCE  {
            #   r   INTEGER,
            #   s   INTEGER
            # }
            # Ecdsa-Sig-Value  ::=  SEQUENCE  {
            #   r   INTEGER,
            #   s   INTEGER
            # }
            output = DerSequence(sig_pair).encode()

        return output

    def verify(self, msg_hash, signature):
        """Check if a certain (EC)DSA signature is authentic.

        Args:
          msg_hash (hash object):
            The hash that was carried out over the message.
            This is an object belonging to the :mod:`Crypto.Hash` module.
            Under mode ``'fips-186-3'``, the hash must be a FIPS
            approved secure hash (SHA-2 or SHA-3).

          signature (``bytes``):
            The signature that needs to be validated.

        :raise ValueError: if the signature is not authentic
        """

        if not self._valid_hash(msg_hash):
            raise ValueError("Hash is not sufficiently strong")

        if self._encoding == 'binary':
            if len(signature) != (2 * self._order_bytes):
                raise ValueError("The signature is not authentic (length)")
            r_prime, s_prime = [Integer.from_bytes(x)
                                for x in (signature[:self._order_bytes],
                                          signature[self._order_bytes:])]
        else:
            try:
                der_seq = DerSequence().decode(signature, strict=True)
            except (ValueError, IndexError):
                raise ValueError("The signature is not authentic (DER)")
            if len(der_seq) != 2 or not der_seq.hasOnlyInts():
                raise ValueError("The signature is not authentic (DER content)")
            r_prime, s_prime = Integer(der_seq[0]), Integer(der_seq[1])

        if not (0 < r_prime < self._order) or not (0 < s_prime < self._order):
            raise ValueError("The signature is not authentic (d)")

        z = Integer.from_bytes(msg_hash.digest()[:self._order_bytes])
        result = self._key._verify(z, (r_prime, s_prime))
        if not result:
            raise ValueError("The signature is not authentic")
        # Make PyCrypto code to fail
        return False


class DeterministicDsaSigScheme(DssSigScheme):
    # Also applicable to ECDSA

    def __init__(self, key, encoding, order, private_key):
        super(DeterministicDsaSigScheme, self).__init__(key, encoding, order)
        self._private_key = private_key

    def _bits2int(self, bstr):
        """See 2.3.2 in RFC6979"""

        result = Integer.from_bytes(bstr)
        q_len = self._order.size_in_bits()
        b_len = len(bstr) * 8
        if b_len > q_len:
            # Only keep leftmost q_len bits
            result >>= (b_len - q_len)
        return result

    def _int2octets(self, int_mod_q):
        """See 2.3.3 in RFC6979"""

        assert 0 < int_mod_q < self._order
        return long_to_bytes(int_mod_q, self._order_bytes)

    def _bits2octets(self, bstr):
        """See 2.3.4 in RFC6979"""

        z1 = self._bits2int(bstr)
        if z1 < self._order:
            z2 = z1
        else:
            z2 = z1 - self._order
        return self._int2octets(z2)

    def _compute_nonce(self, mhash):
        """Generate k in a deterministic way"""

        # See section 3.2 in RFC6979.txt
        # Step a
        h1 = mhash.digest()
        # Step b
        mask_v = b'\x01' * mhash.digest_size
        # Step c
        nonce_k = b'\x00' * mhash.digest_size

        for int_oct in (b'\x00', b'\x01'):
            # Step d/f
            nonce_k = HMAC.new(nonce_k,
                               mask_v + int_oct +
                               self._int2octets(self._private_key) +
                               self._bits2octets(h1), mhash).digest()
            # Step e/g
            mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()

        nonce = -1
        while not (0 < nonce < self._order):
            # Step h.C (second part)
            if nonce != -1:
                nonce_k = HMAC.new(nonce_k, mask_v + b'\x00',
                                   mhash).digest()
                mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()

            # Step h.A
            mask_t = b""

            # Step h.B
            while len(mask_t) < self._order_bytes:
                mask_v = HMAC.new(nonce_k, mask_v, mhash).digest()
                mask_t += mask_v

            # Step h.C (first part)
            nonce = self._bits2int(mask_t)
        return nonce

    def _valid_hash(self, msg_hash):
        return True


class FipsDsaSigScheme(DssSigScheme):

    #: List of L (bit length of p) and N (bit length of q) combinations
    #: that are allowed by FIPS 186-3. The security level is provided in
    #: Table 2 of FIPS 800-57 (rev3).
    _fips_186_3_L_N = (
                        (1024, 160),    # 80 bits  (SHA-1 or stronger)
                        (2048, 224),    # 112 bits (SHA-224 or stronger)
                        (2048, 256),    # 128 bits (SHA-256 or stronger)
                        (3072, 256)     # 256 bits (SHA-512)
                      )

    def __init__(self, key, encoding, order, randfunc):
        super(FipsDsaSigScheme, self).__init__(key, encoding, order)
        self._randfunc = randfunc

        L = Integer(key.p).size_in_bits()
        if (L, self._order_bits) not in self._fips_186_3_L_N:
            error = ("L/N (%d, %d) is not compliant to FIPS 186-3"
                     % (L, self._order_bits))
            raise ValueError(error)

    def _compute_nonce(self, msg_hash):
        # hash is not used
        return Integer.random_range(min_inclusive=1,
                                    max_exclusive=self._order,
                                    randfunc=self._randfunc)

    def _valid_hash(self, msg_hash):
        """Verify that SHA-1, SHA-2 or SHA-3 are used"""
        return (msg_hash.oid == "1.3.14.3.2.26" or
                msg_hash.oid.startswith("2.16.840.1.101.3.4.2."))


class FipsEcDsaSigScheme(DssSigScheme):

    def __init__(self, key, encoding, order, randfunc):
        super(FipsEcDsaSigScheme, self).__init__(key, encoding, order)
        self._randfunc = randfunc

    def _compute_nonce(self, msg_hash):
        return Integer.random_range(min_inclusive=1,
                                    max_exclusive=self._key._curve.order,
                                    randfunc=self._randfunc)

    def _valid_hash(self, msg_hash):
        """Verify that the strength of the hash matches or exceeds
        the strength of the EC. We fail if the hash is too weak."""

        modulus_bits = self._key.pointQ.size_in_bits()

        # SHS: SHA-2, SHA-3, truncated SHA-512
        sha224 = ("2.16.840.1.101.3.4.2.4", "2.16.840.1.101.3.4.2.7", "2.16.840.1.101.3.4.2.5")
        sha256 = ("2.16.840.1.101.3.4.2.1", "2.16.840.1.101.3.4.2.8", "2.16.840.1.101.3.4.2.6")
        sha384 = ("2.16.840.1.101.3.4.2.2", "2.16.840.1.101.3.4.2.9")
        sha512 = ("2.16.840.1.101.3.4.2.3", "2.16.840.1.101.3.4.2.10")
        shs = sha224 + sha256 + sha384 + sha512

        try:
            result = msg_hash.oid in shs
        except AttributeError:
            result = False
        return result


def new(key, mode, encoding='binary', randfunc=None):
    """Create a signature object :class:`DssSigScheme` that
    can perform (EC)DSA signature or verification.

    .. note::
        Refer to `NIST SP 800 Part 1 Rev 4`_ (or newer release) for an
        overview of the recommended key lengths.

    Args:
        key (:class:`Crypto.PublicKey.DSA` or :class:`Crypto.PublicKey.ECC`):
            The key to use for computing the signature (*private* keys only)
            or for verifying one.
            For DSA keys, let ``L`` and ``N`` be the bit lengths of the modulus ``p``
            and of ``q``: the pair ``(L,N)`` must appear in the following list,
            in compliance to section 4.2 of `FIPS 186-4`_:

            - (1024, 160) *legacy only; do not create new signatures with this*
            - (2048, 224) *deprecated; do not create new signatures with this*
            - (2048, 256)
            - (3072, 256)

            For ECC, only keys over P-224, P-256, P-384, and P-521 are accepted.

        mode (string):
            The parameter can take these values:

            - ``'fips-186-3'``. The signature generation is randomized and carried out
              according to `FIPS 186-3`_: the nonce ``k`` is taken from the RNG.
            - ``'deterministic-rfc6979'``. The signature generation is not
              randomized. See RFC6979_.

        encoding (string):
            How the signature is encoded. This value determines the output of
            :meth:`sign` and the input to :meth:`verify`.

            The following values are accepted:

            - ``'binary'`` (default), the signature is the raw concatenation
              of ``r`` and ``s``. It is defined in the IEEE P.1363 standard.
              For DSA, the size in bytes of the signature is ``N/4`` bytes
              (e.g. 64 for ``N=256``).
              For ECDSA, the signature is always twice the length of a point
              coordinate (e.g. 64 bytes for P-256).

            - ``'der'``, the signature is a ASN.1 DER SEQUENCE
              with two INTEGERs (``r`` and ``s``). It is defined in RFC3279_.
              The size of the signature is variable.

        randfunc (callable):
            A function that returns random ``bytes``, of a given length.
            If omitted, the internal RNG is used.
            Only applicable for the *'fips-186-3'* mode.

    .. _FIPS 186-3: http://csrc.nist.gov/publications/fips/fips186-3/fips_186-3.pdf
    .. _FIPS 186-4: http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf
    .. _NIST SP 800 Part 1 Rev 4: http://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r4.pdf
    .. _RFC6979: http://tools.ietf.org/html/rfc6979
    .. _RFC3279: https://tools.ietf.org/html/rfc3279#section-2.2.2
    """

    # The goal of the 'mode' parameter is to avoid to
    # have the current version of the standard as default.
    #
    # Over time, such version will be superseded by (for instance)
    # FIPS 186-4 and it will be odd to have -3 as default.

    if encoding not in ('binary', 'der'):
        raise ValueError("Unknown encoding '%s'" % encoding)

    if isinstance(key, EccKey):
        order = key._curve.order
        private_key_attr = 'd'
    elif isinstance(key, DsaKey):
        order = Integer(key.q)
        private_key_attr = 'x'
    else:
        raise ValueError("Unsupported key type " + str(type(key)))

    if key.has_private():
        private_key = getattr(key, private_key_attr)
    else:
        private_key = None

    if mode == 'deterministic-rfc6979':
        return DeterministicDsaSigScheme(key, encoding, order, private_key)
    elif mode == 'fips-186-3':
        if isinstance(key, EccKey):
            return FipsEcDsaSigScheme(key, encoding, order, randfunc)
        else:
            return FipsDsaSigScheme(key, encoding, order, randfunc)
    else:
        raise ValueError("Unknown DSS mode '%s'" % mode)
