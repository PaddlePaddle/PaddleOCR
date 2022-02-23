#
#  PublicKey/_PBES.py : Password-Based Encryption functions
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

from Crypto import Random
from Crypto.Util.asn1 import (
            DerSequence, DerOctetString,
            DerObjectId, DerInteger,
            )

from Crypto.Util.Padding import pad, unpad
from Crypto.Hash import MD5, SHA1, SHA224, SHA256, SHA384, SHA512
from Crypto.Cipher import DES, ARC2, DES3, AES
from Crypto.Protocol.KDF import PBKDF1, PBKDF2, scrypt

_OID_PBE_WITH_MD5_AND_DES_CBC = "1.2.840.113549.1.5.3"
_OID_PBE_WITH_MD5_AND_RC2_CBC = "1.2.840.113549.1.5.6"
_OID_PBE_WITH_SHA1_AND_DES_CBC = "1.2.840.113549.1.5.10"
_OID_PBE_WITH_SHA1_AND_RC2_CBC = "1.2.840.113549.1.5.11"

_OID_PBES2 = "1.2.840.113549.1.5.13"

_OID_PBKDF2 = "1.2.840.113549.1.5.12"
_OID_SCRYPT = "1.3.6.1.4.1.11591.4.11"

_OID_HMAC_SHA1 = "1.2.840.113549.2.7"
_OID_HMAC_SHA224 = "1.2.840.113549.2.8"
_OID_HMAC_SHA256 = "1.2.840.113549.2.9"
_OID_HMAC_SHA384 = "1.2.840.113549.2.10"
_OID_HMAC_SHA512 = "1.2.840.113549.2.11"

_OID_DES_EDE3_CBC = "1.2.840.113549.3.7"
_OID_AES128_CBC = "2.16.840.1.101.3.4.1.2"
_OID_AES192_CBC = "2.16.840.1.101.3.4.1.22"
_OID_AES256_CBC = "2.16.840.1.101.3.4.1.42"


class PbesError(ValueError):
    pass

# These are the ASN.1 definitions used by the PBES1/2 logic:
#
# EncryptedPrivateKeyInfo ::= SEQUENCE {
#   encryptionAlgorithm  EncryptionAlgorithmIdentifier,
#   encryptedData        EncryptedData
# }
#
# EncryptionAlgorithmIdentifier ::= AlgorithmIdentifier
#
# EncryptedData ::= OCTET STRING
#
# AlgorithmIdentifier  ::=  SEQUENCE  {
#       algorithm   OBJECT IDENTIFIER,
#       parameters  ANY DEFINED BY algorithm OPTIONAL
# }
#
# PBEParameter ::= SEQUENCE {
#       salt OCTET STRING (SIZE(8)),
#       iterationCount INTEGER
# }
#
# PBES2-params ::= SEQUENCE {
#       keyDerivationFunc AlgorithmIdentifier {{PBES2-KDFs}},
#       encryptionScheme AlgorithmIdentifier {{PBES2-Encs}}
# }
#
# PBKDF2-params ::= SEQUENCE {
#   salt CHOICE {
#       specified OCTET STRING,
#       otherSource AlgorithmIdentifier {{PBKDF2-SaltSources}}
#       },
#   iterationCount INTEGER (1..MAX),
#   keyLength INTEGER (1..MAX) OPTIONAL,
#   prf AlgorithmIdentifier {{PBKDF2-PRFs}} DEFAULT algid-hmacWithSHA1
#   }
#
# scrypt-params ::= SEQUENCE {
#       salt OCTET STRING,
#       costParameter INTEGER (1..MAX),
#       blockSize INTEGER (1..MAX),
#       parallelizationParameter INTEGER (1..MAX),
#       keyLength INTEGER (1..MAX) OPTIONAL
#   }

class PBES1(object):
    """Deprecated encryption scheme with password-based key derivation
    (originally defined in PKCS#5 v1.5, but still present in `v2.0`__).

    .. __: http://www.ietf.org/rfc/rfc2898.txt
    """

    @staticmethod
    def decrypt(data, passphrase):
        """Decrypt a piece of data using a passphrase and *PBES1*.

        The algorithm to use is automatically detected.

        :Parameters:
          data : byte string
            The piece of data to decrypt.
          passphrase : byte string
            The passphrase to use for decrypting the data.
        :Returns:
          The decrypted data, as a binary string.
        """

        enc_private_key_info = DerSequence().decode(data)
        encrypted_algorithm = DerSequence().decode(enc_private_key_info[0])
        encrypted_data = DerOctetString().decode(enc_private_key_info[1]).payload

        pbe_oid = DerObjectId().decode(encrypted_algorithm[0]).value
        cipher_params = {}
        if pbe_oid == _OID_PBE_WITH_MD5_AND_DES_CBC:
            # PBE_MD5_DES_CBC
            hashmod = MD5
            ciphermod = DES
        elif pbe_oid == _OID_PBE_WITH_MD5_AND_RC2_CBC:
            # PBE_MD5_RC2_CBC
            hashmod = MD5
            ciphermod = ARC2
            cipher_params['effective_keylen'] = 64
        elif pbe_oid == _OID_PBE_WITH_SHA1_AND_DES_CBC:
            # PBE_SHA1_DES_CBC
            hashmod = SHA1
            ciphermod = DES
        elif pbe_oid == _OID_PBE_WITH_SHA1_AND_RC2_CBC:
            # PBE_SHA1_RC2_CBC
            hashmod = SHA1
            ciphermod = ARC2
            cipher_params['effective_keylen'] = 64
        else:
            raise PbesError("Unknown OID for PBES1")

        pbe_params = DerSequence().decode(encrypted_algorithm[1], nr_elements=2)
        salt = DerOctetString().decode(pbe_params[0]).payload
        iterations = pbe_params[1]

        key_iv = PBKDF1(passphrase, salt, 16, iterations, hashmod)
        key, iv = key_iv[:8], key_iv[8:]

        cipher = ciphermod.new(key, ciphermod.MODE_CBC, iv, **cipher_params)
        pt = cipher.decrypt(encrypted_data)
        return unpad(pt, cipher.block_size)


class PBES2(object):
    """Encryption scheme with password-based key derivation
    (defined in `PKCS#5 v2.0`__).

    .. __: http://www.ietf.org/rfc/rfc2898.txt."""

    @staticmethod
    def encrypt(data, passphrase, protection, prot_params=None, randfunc=None):
        """Encrypt a piece of data using a passphrase and *PBES2*.

        :Parameters:
          data : byte string
            The piece of data to encrypt.
          passphrase : byte string
            The passphrase to use for encrypting the data.
          protection : string
            The identifier of the encryption algorithm to use.
            The default value is '``PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC``'.
          prot_params : dictionary
            Parameters of the protection algorithm.

            +------------------+-----------------------------------------------+
            | Key              | Description                                   |
            +==================+===============================================+
            | iteration_count  | The KDF algorithm is repeated several times to|
            |                  | slow down brute force attacks on passwords    |
            |                  | (called *N* or CPU/memory cost in scrypt).    |
            |                  |                                               |
            |                  | The default value for PBKDF2 is 1 000.        |
            |                  | The default value for scrypt is 16 384.       |
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


          randfunc : callable
            Random number generation function; it should accept
            a single integer N and return a string of random data,
            N bytes long. If not specified, a new RNG will be
            instantiated from ``Crypto.Random``.

        :Returns:
          The encrypted data, as a binary string.
        """

        if prot_params is None:
            prot_params = {}

        if randfunc is None:
            randfunc = Random.new().read

        if protection == 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC':
            key_size = 24
            module = DES3
            cipher_mode = DES3.MODE_CBC
            enc_oid = _OID_DES_EDE3_CBC
        elif protection in ('PBKDF2WithHMAC-SHA1AndAES128-CBC',
                'scryptAndAES128-CBC'):
            key_size = 16
            module = AES
            cipher_mode = AES.MODE_CBC
            enc_oid = _OID_AES128_CBC
        elif protection in ('PBKDF2WithHMAC-SHA1AndAES192-CBC',
                'scryptAndAES192-CBC'):
            key_size = 24
            module = AES
            cipher_mode = AES.MODE_CBC
            enc_oid = _OID_AES192_CBC
        elif protection in ('PBKDF2WithHMAC-SHA1AndAES256-CBC',
                'scryptAndAES256-CBC'):
            key_size = 32
            module = AES
            cipher_mode = AES.MODE_CBC
            enc_oid = _OID_AES256_CBC
        else:
            raise ValueError("Unknown PBES2 mode")

        # Get random data
        iv = randfunc(module.block_size)
        salt = randfunc(prot_params.get("salt_size", 8))

        # Derive key from password
        if protection.startswith('PBKDF2'):
            count = prot_params.get("iteration_count", 1000)
            key = PBKDF2(passphrase, salt, key_size, count)
            kdf_info = DerSequence([
                    DerObjectId(_OID_PBKDF2),   # PBKDF2
                    DerSequence([
                        DerOctetString(salt),
                        DerInteger(count)
                    ])
            ])
        else:
            # It must be scrypt
            count = prot_params.get("iteration_count", 16384)
            scrypt_r = prot_params.get('block_size', 8)
            scrypt_p = prot_params.get('parallelization', 1)
            key = scrypt(passphrase, salt, key_size,
                         count, scrypt_r, scrypt_p)
            kdf_info = DerSequence([
                    DerObjectId(_OID_SCRYPT),  # scrypt
                    DerSequence([
                        DerOctetString(salt),
                        DerInteger(count),
                        DerInteger(scrypt_r),
                        DerInteger(scrypt_p)
                    ])
            ])

        # Create cipher and use it
        cipher = module.new(key, cipher_mode, iv)
        encrypted_data = cipher.encrypt(pad(data, cipher.block_size))
        enc_info = DerSequence([
                DerObjectId(enc_oid),
                DerOctetString(iv)
        ])

        # Result
        enc_private_key_info = DerSequence([
            # encryptionAlgorithm
            DerSequence([
                DerObjectId(_OID_PBES2),
                DerSequence([
                    kdf_info,
                    enc_info
                ]),
            ]),
            DerOctetString(encrypted_data)
        ])
        return enc_private_key_info.encode()

    @staticmethod
    def decrypt(data, passphrase):
        """Decrypt a piece of data using a passphrase and *PBES2*.

        The algorithm to use is automatically detected.

        :Parameters:
          data : byte string
            The piece of data to decrypt.
          passphrase : byte string
            The passphrase to use for decrypting the data.
        :Returns:
          The decrypted data, as a binary string.
        """

        enc_private_key_info = DerSequence().decode(data, nr_elements=2)
        enc_algo = DerSequence().decode(enc_private_key_info[0])
        encrypted_data = DerOctetString().decode(enc_private_key_info[1]).payload

        pbe_oid = DerObjectId().decode(enc_algo[0]).value
        if pbe_oid != _OID_PBES2:
            raise PbesError("Not a PBES2 object")

        pbes2_params = DerSequence().decode(enc_algo[1], nr_elements=2)

        ### Key Derivation Function selection
        kdf_info = DerSequence().decode(pbes2_params[0], nr_elements=2)
        kdf_oid = DerObjectId().decode(kdf_info[0]).value

        kdf_key_length = None

        # We only support PBKDF2 or scrypt
        if kdf_oid == _OID_PBKDF2:

            pbkdf2_params = DerSequence().decode(kdf_info[1], nr_elements=(2, 3, 4))
            salt = DerOctetString().decode(pbkdf2_params[0]).payload
            iteration_count = pbkdf2_params[1]

            left = len(pbkdf2_params) - 2
            idx = 2

            if left > 0:
                try:
                    kdf_key_length = pbkdf2_params[idx] - 0
                    left -= 1
                    idx += 1
                except TypeError:
                    pass

            # Default is HMAC-SHA1
            pbkdf2_prf_oid = "1.2.840.113549.2.7"
            if left > 0:
                pbkdf2_prf_algo_id = DerSequence().decode(pbkdf2_params[idx])
                pbkdf2_prf_oid = DerObjectId().decode(pbkdf2_prf_algo_id[0]).value

        elif kdf_oid == _OID_SCRYPT:

            scrypt_params = DerSequence().decode(kdf_info[1], nr_elements=(4, 5))
            salt = DerOctetString().decode(scrypt_params[0]).payload
            iteration_count, scrypt_r, scrypt_p = [scrypt_params[x]
                                                   for x in (1, 2, 3)]
            if len(scrypt_params) > 4:
                kdf_key_length = scrypt_params[4]
            else:
                kdf_key_length = None
        else:
            raise PbesError("Unsupported PBES2 KDF")

        ### Cipher selection
        enc_info = DerSequence().decode(pbes2_params[1])
        enc_oid = DerObjectId().decode(enc_info[0]).value

        if enc_oid == _OID_DES_EDE3_CBC:
            # DES_EDE3_CBC
            ciphermod = DES3
            key_size = 24
        elif enc_oid == _OID_AES128_CBC:
            # AES128_CBC
            ciphermod = AES
            key_size = 16
        elif enc_oid == _OID_AES192_CBC:
            # AES192_CBC
            ciphermod = AES
            key_size = 24
        elif enc_oid == _OID_AES256_CBC:
            # AES256_CBC
            ciphermod = AES
            key_size = 32
        else:
            raise PbesError("Unsupported PBES2 cipher")

        if kdf_key_length and kdf_key_length != key_size:
            raise PbesError("Mismatch between PBES2 KDF parameters"
                            " and selected cipher")

        IV = DerOctetString().decode(enc_info[1]).payload

        # Create cipher
        if kdf_oid == _OID_PBKDF2:
            if pbkdf2_prf_oid == _OID_HMAC_SHA1:
                hmac_hash_module = SHA1
            elif pbkdf2_prf_oid == _OID_HMAC_SHA224:
                hmac_hash_module = SHA224
            elif pbkdf2_prf_oid == _OID_HMAC_SHA256:
                hmac_hash_module = SHA256
            elif pbkdf2_prf_oid == _OID_HMAC_SHA384:
                hmac_hash_module = SHA384
            elif pbkdf2_prf_oid == _OID_HMAC_SHA512:
                hmac_hash_module = SHA512
            else:
                raise PbesError("Unsupported HMAC %s" % pbkdf2_prf_oid)

            key = PBKDF2(passphrase, salt, key_size, iteration_count,
                         hmac_hash_module=hmac_hash_module)
        else:
            key = scrypt(passphrase, salt, key_size, iteration_count,
                         scrypt_r, scrypt_p)
        cipher = ciphermod.new(key, ciphermod.MODE_CBC, IV)

        # Decrypt data
        pt = cipher.decrypt(encrypted_data)
        return unpad(pt, cipher.block_size)
