#
#  PublicKey/PKCS8.py : PKCS#8 functions
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


from Crypto.Util.py3compat import *

from Crypto.Util.asn1 import (
            DerNull,
            DerSequence,
            DerObjectId,
            DerOctetString,
            )

from Crypto.IO._PBES import PBES1, PBES2, PbesError


__all__ = ['wrap', 'unwrap']


def wrap(private_key, key_oid, passphrase=None, protection=None,
         prot_params=None, key_params=None, randfunc=None):
    """Wrap a private key into a PKCS#8 blob (clear or encrypted).

    Args:

      private_key (byte string):
        The private key encoded in binary form. The actual encoding is
        algorithm specific. In most cases, it is DER.

      key_oid (string):
        The object identifier (OID) of the private key to wrap.
        It is a dotted string, like ``1.2.840.113549.1.1.1`` (for RSA keys).

      passphrase (bytes string or string):
        The secret passphrase from which the wrapping key is derived.
        Set it only if encryption is required.

      protection (string):
        The identifier of the algorithm to use for securely wrapping the key.
        The default value is ``PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC``.

      prot_params (dictionary):
        Parameters for the protection algorithm.

        +------------------+-----------------------------------------------+
        | Key              | Description                                   |
        +==================+===============================================+
        | iteration_count  | The KDF algorithm is repeated several times to|
        |                  | slow down brute force attacks on passwords    |
        |                  | (called *N* or CPU/memory cost in scrypt).    |
        |                  | The default value for PBKDF2 is 1000.         |
        |                  | The default value for scrypt is 16384.        |
        +------------------+-----------------------------------------------+
        | salt_size        | Salt is used to thwart dictionary and rainbow |
        |                  | attacks on passwords. The default value is 8  |
        |                  | bytes.                                        |
        +------------------+-----------------------------------------------+
        | block_size       | *(scrypt only)* Memory-cost (r). The default  |
        |                  | value is 8.                                   |
        +------------------+-----------------------------------------------+
        | parallelization  | *(scrypt only)* CPU-cost (p). The default     |
        |                  | value is 1.                                   |
        +------------------+-----------------------------------------------+

      key_params (DER object):
        The algorithm parameters associated to the private key.
        It is required for algorithms like DSA, but not for others like RSA.

      randfunc (callable):
        Random number generation function; it should accept a single integer
        N and return a string of random data, N bytes long.
        If not specified, a new RNG will be instantiated
        from :mod:`Crypto.Random`.

    Return:
      The PKCS#8-wrapped private key (possibly encrypted), as a byte string.
    """

    if key_params is None:
        key_params = DerNull()

    #
    #   PrivateKeyInfo ::= SEQUENCE {
    #       version                 Version,
    #       privateKeyAlgorithm     PrivateKeyAlgorithmIdentifier,
    #       privateKey              PrivateKey,
    #       attributes              [0]  IMPLICIT Attributes OPTIONAL
    #   }
    #
    pk_info = DerSequence([
                0,
                DerSequence([
                    DerObjectId(key_oid),
                    key_params
                ]),
                DerOctetString(private_key)
            ])
    pk_info_der = pk_info.encode()

    if passphrase is None:
        return pk_info_der

    if not passphrase:
        raise ValueError("Empty passphrase")

    # Encryption with PBES2
    passphrase = tobytes(passphrase)
    if protection is None:
        protection = 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC'
    return PBES2.encrypt(pk_info_der, passphrase,
                         protection, prot_params, randfunc)


def unwrap(p8_private_key, passphrase=None):
    """Unwrap a private key from a PKCS#8 blob (clear or encrypted).

    Args:
      p8_private_key (byte string):
        The private key wrapped into a PKCS#8 blob, DER encoded.
      passphrase (byte string or string):
        The passphrase to use to decrypt the blob (if it is encrypted).

    Return:
      A tuple containing

       #. the algorithm identifier of the wrapped key (OID, dotted string)
       #. the private key (byte string, DER encoded)
       #. the associated parameters (byte string, DER encoded) or ``None``

    Raises:
      ValueError : if decoding fails
    """

    if passphrase:
        passphrase = tobytes(passphrase)

        found = False
        try:
            p8_private_key = PBES1.decrypt(p8_private_key, passphrase)
            found = True
        except PbesError as e:
            error_str = "PBES1[%s]" % str(e)
        except ValueError:
            error_str = "PBES1[Invalid]"

        if not found:
            try:
                p8_private_key = PBES2.decrypt(p8_private_key, passphrase)
                found = True
            except PbesError as e:
                error_str += ",PBES2[%s]" % str(e)
            except ValueError:
                error_str += ",PBES2[Invalid]"

        if not found:
            raise ValueError("Error decoding PKCS#8 (%s)" % error_str)

    pk_info = DerSequence().decode(p8_private_key, nr_elements=(2, 3, 4))
    if len(pk_info) == 2 and not passphrase:
        raise ValueError("Not a valid clear PKCS#8 structure "
                         "(maybe it is encrypted?)")

    #
    #   PrivateKeyInfo ::= SEQUENCE {
    #       version                 Version,
    #       privateKeyAlgorithm     PrivateKeyAlgorithmIdentifier,
    #       privateKey              PrivateKey,
    #       attributes              [0]  IMPLICIT Attributes OPTIONAL
    #   }
    #   Version ::= INTEGER
    if pk_info[0] != 0:
        raise ValueError("Not a valid PrivateKeyInfo SEQUENCE")

    # PrivateKeyAlgorithmIdentifier ::= AlgorithmIdentifier
    #
    #   EncryptedPrivateKeyInfo ::= SEQUENCE {
    #       encryptionAlgorithm  EncryptionAlgorithmIdentifier,
    #       encryptedData        EncryptedData
    #   }
    #   EncryptionAlgorithmIdentifier ::= AlgorithmIdentifier

    #   AlgorithmIdentifier  ::=  SEQUENCE  {
    #       algorithm   OBJECT IDENTIFIER,
    #       parameters  ANY DEFINED BY algorithm OPTIONAL
    #   }

    algo = DerSequence().decode(pk_info[1], nr_elements=(1, 2))
    algo_oid = DerObjectId().decode(algo[0]).value
    if len(algo) == 1:
        algo_params = None
    else:
        try:
            DerNull().decode(algo[1])
            algo_params = None
        except:
            algo_params = algo[1]

    #   EncryptedData ::= OCTET STRING
    private_key = DerOctetString().decode(pk_info[2]).payload

    return (algo_oid, private_key, algo_params)
