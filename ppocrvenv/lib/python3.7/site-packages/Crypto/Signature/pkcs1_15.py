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

import Crypto.Util.number
from Crypto.Util.number import ceil_div, bytes_to_long, long_to_bytes
from Crypto.Util.asn1 import DerSequence, DerNull, DerOctetString, DerObjectId

class PKCS115_SigScheme:
    """A signature object for ``RSASSA-PKCS1-v1_5``.
    Do not instantiate directly.
    Use :func:`Crypto.Signature.pkcs1_15.new`.
    """

    def __init__(self, rsa_key):
        """Initialize this PKCS#1 v1.5 signature scheme object.

        :Parameters:
          rsa_key : an RSA key object
            Creation of signatures is only possible if this is a *private*
            RSA key. Verification of signatures is always possible.
        """
        self._key = rsa_key

    def can_sign(self):
        """Return ``True`` if this object can be used to sign messages."""
        return self._key.has_private()

    def sign(self, msg_hash):
        """Create the PKCS#1 v1.5 signature of a message.

        This function is also called ``RSASSA-PKCS1-V1_5-SIGN`` and
        it is specified in
        `section 8.2.1 of RFC8017 <https://tools.ietf.org/html/rfc8017#page-36>`_.

        :parameter msg_hash:
            This is an object from the :mod:`Crypto.Hash` package.
            It has been used to digest the message to sign.
        :type msg_hash: hash object

        :return: the signature encoded as a *byte string*.
        :raise ValueError: if the RSA key is not long enough for the given hash algorithm.
        :raise TypeError: if the RSA key has no private half.
        """

        # See 8.2.1 in RFC3447
        modBits = Crypto.Util.number.size(self._key.n)
        k = ceil_div(modBits,8) # Convert from bits to bytes

        # Step 1
        em = _EMSA_PKCS1_V1_5_ENCODE(msg_hash, k)
        # Step 2a (OS2IP)
        em_int = bytes_to_long(em)
        # Step 2b (RSASP1)
        m_int = self._key._decrypt(em_int)
        # Step 2c (I2OSP)
        signature = long_to_bytes(m_int, k)
        return signature

    def verify(self, msg_hash, signature):
        """Check if the  PKCS#1 v1.5 signature over a message is valid.

        This function is also called ``RSASSA-PKCS1-V1_5-VERIFY`` and
        it is specified in
        `section 8.2.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#page-37>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Crypto.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: byte string

        :raise ValueError: if the signature is not valid.
        """

        # See 8.2.2 in RFC3447
        modBits = Crypto.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8) # Convert from bits to bytes

        # Step 1
        if len(signature) != k:
            raise ValueError("Invalid signature")
        # Step 2a (O2SIP)
        signature_int = bytes_to_long(signature)
        # Step 2b (RSAVP1)
        em_int = self._key._encrypt(signature_int)
        # Step 2c (I2OSP)
        em1 = long_to_bytes(em_int, k)
        # Step 3
        try:
            possible_em1 = [ _EMSA_PKCS1_V1_5_ENCODE(msg_hash, k, True) ]
            # MD2/4/5 hashes always require NULL params in AlgorithmIdentifier.
            # For all others, it is optional.
            try:
                algorithm_is_md = msg_hash.oid.startswith('1.2.840.113549.2.')
            except AttributeError:
                algorithm_is_md = False
            if not algorithm_is_md:  # MD2/MD4/MD5
                possible_em1.append(_EMSA_PKCS1_V1_5_ENCODE(msg_hash, k, False))
        except ValueError:
            raise ValueError("Invalid signature")
        # Step 4
        # By comparing the full encodings (as opposed to checking each
        # of its components one at a time) we avoid attacks to the padding
        # scheme like Bleichenbacher's (see http://www.mail-archive.com/cryptography@metzdowd.com/msg06537).
        #
        if em1 not in possible_em1:
            raise ValueError("Invalid signature")
        pass


def _EMSA_PKCS1_V1_5_ENCODE(msg_hash, emLen, with_hash_parameters=True):
    """
    Implement the ``EMSA-PKCS1-V1_5-ENCODE`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.2).

    ``_EMSA-PKCS1-V1_5-ENCODE`` actually accepts the message ``M`` as input,
    and hash it internally. Here, we expect that the message has already
    been hashed instead.

    :Parameters:
     msg_hash : hash object
            The hash object that holds the digest of the message being signed.
     emLen : int
            The length the final encoding must have, in bytes.
     with_hash_parameters : bool
            If True (default), include NULL parameters for the hash
            algorithm in the ``digestAlgorithm`` SEQUENCE.

    :attention: the early standard (RFC2313) stated that ``DigestInfo``
        had to be BER-encoded. This means that old signatures
        might have length tags in indefinite form, which
        is not supported in DER. Such encoding cannot be
        reproduced by this function.

    :Return: An ``emLen`` byte long string that encodes the hash.
    """

    # First, build the ASN.1 DER object DigestInfo:
    #
    #   DigestInfo ::= SEQUENCE {
    #       digestAlgorithm AlgorithmIdentifier,
    #       digest OCTET STRING
    #   }
    #
    # where digestAlgorithm identifies the hash function and shall be an
    # algorithm ID with an OID in the set PKCS1-v1-5DigestAlgorithms.
    #
    #   PKCS1-v1-5DigestAlgorithms    ALGORITHM-IDENTIFIER ::= {
    #       { OID id-md2 PARAMETERS NULL    }|
    #       { OID id-md5 PARAMETERS NULL    }|
    #       { OID id-sha1 PARAMETERS NULL   }|
    #       { OID id-sha256 PARAMETERS NULL }|
    #       { OID id-sha384 PARAMETERS NULL }|
    #       { OID id-sha512 PARAMETERS NULL }
    #   }
    #
    # Appendix B.1 also says that for SHA-1/-2 algorithms, the parameters
    # should be omitted. They may be present, but when they are, they shall
    # have NULL value.

    digestAlgo = DerSequence([ DerObjectId(msg_hash.oid).encode() ])

    if with_hash_parameters:
        digestAlgo.append(DerNull().encode())

    digest      = DerOctetString(msg_hash.digest())
    digestInfo  = DerSequence([
                    digestAlgo.encode(),
                    digest.encode()
                    ]).encode()

    # We need at least 11 bytes for the remaining data: 3 fixed bytes and
    # at least 8 bytes of padding).
    if emLen<len(digestInfo)+11:
        raise TypeError("Selected hash algorithm has a too long digest (%d bytes)." % len(digest))
    PS = b'\xFF' * (emLen - len(digestInfo) - 3)
    return b'\x00\x01' + PS + b'\x00' + digestInfo

def new(rsa_key):
    """Create a signature object for creating
    or verifying PKCS#1 v1.5 signatures.

    :parameter rsa_key:
      The RSA key to use for signing or verifying the message.
      This is a :class:`Crypto.PublicKey.RSA` object.
      Signing is only possible when ``rsa_key`` is a **private** RSA key.
    :type rsa_key: RSA object

    :return: a :class:`PKCS115_SigScheme` signature object
    """
    return PKCS115_SigScheme(rsa_key)

