#
#  Util/PEM.py : Privacy Enhanced Mail utilities
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

__all__ = ['encode', 'decode']

import re
from binascii import a2b_base64, b2a_base64, hexlify, unhexlify

from Crypto.Hash import MD5
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import DES, DES3, AES
from Crypto.Protocol.KDF import PBKDF1
from Crypto.Random import get_random_bytes
from Crypto.Util.py3compat import tobytes, tostr


def encode(data, marker, passphrase=None, randfunc=None):
    """Encode a piece of binary data into PEM format.

    Args:
      data (byte string):
        The piece of binary data to encode.
      marker (string):
        The marker for the PEM block (e.g. "PUBLIC KEY").
        Note that there is no official master list for all allowed markers.
        Still, you can refer to the OpenSSL_ source code.
      passphrase (byte string):
        If given, the PEM block will be encrypted. The key is derived from
        the passphrase.
      randfunc (callable):
        Random number generation function; it accepts an integer N and returns
        a byte string of random data, N bytes long. If not given, a new one is
        instantiated.

    Returns:
      The PEM block, as a string.

    .. _OpenSSL: https://github.com/openssl/openssl/blob/master/include/openssl/pem.h
    """

    if randfunc is None:
        randfunc = get_random_bytes

    out = "-----BEGIN %s-----\n" % marker
    if passphrase:
        # We only support 3DES for encryption
        salt = randfunc(8)
        key = PBKDF1(passphrase, salt, 16, 1, MD5)
        key += PBKDF1(key + passphrase, salt, 8, 1, MD5)
        objenc = DES3.new(key, DES3.MODE_CBC, salt)
        out += "Proc-Type: 4,ENCRYPTED\nDEK-Info: DES-EDE3-CBC,%s\n\n" %\
            tostr(hexlify(salt).upper())
        # Encrypt with PKCS#7 padding
        data = objenc.encrypt(pad(data, objenc.block_size))
    elif passphrase is not None:
        raise ValueError("Empty password")

    # Each BASE64 line can take up to 64 characters (=48 bytes of data)
    # b2a_base64 adds a new line character!
    chunks = [tostr(b2a_base64(data[i:i + 48]))
              for i in range(0, len(data), 48)]
    out += "".join(chunks)
    out += "-----END %s-----" % marker
    return out


def _EVP_BytesToKey(data, salt, key_len):
    d = [ b'' ]
    m = (key_len + 15 ) // 16
    for _ in range(m):
        nd = MD5.new(d[-1] + data + salt).digest()
        d.append(nd)
    return b"".join(d)[:key_len]


def decode(pem_data, passphrase=None):
    """Decode a PEM block into binary.

    Args:
      pem_data (string):
        The PEM block.
      passphrase (byte string):
        If given and the PEM block is encrypted,
        the key will be derived from the passphrase.

    Returns:
      A tuple with the binary data, the marker string, and a boolean to
      indicate if decryption was performed.

    Raises:
      ValueError: if decoding fails, if the PEM file is encrypted and no passphrase has
                  been provided or if the passphrase is incorrect.
    """

    # Verify Pre-Encapsulation Boundary
    r = re.compile(r"\s*-----BEGIN (.*)-----\s+")
    m = r.match(pem_data)
    if not m:
        raise ValueError("Not a valid PEM pre boundary")
    marker = m.group(1)

    # Verify Post-Encapsulation Boundary
    r = re.compile(r"-----END (.*)-----\s*$")
    m = r.search(pem_data)
    if not m or m.group(1) != marker:
        raise ValueError("Not a valid PEM post boundary")

    # Removes spaces and slit on lines
    lines = pem_data.replace(" ", '').split()

    # Decrypts, if necessary
    if lines[1].startswith('Proc-Type:4,ENCRYPTED'):
        if not passphrase:
            raise ValueError("PEM is encrypted, but no passphrase available")
        DEK = lines[2].split(':')
        if len(DEK) != 2 or DEK[0] != 'DEK-Info':
            raise ValueError("PEM encryption format not supported.")
        algo, salt = DEK[1].split(',')
        salt = unhexlify(tobytes(salt))

        padding = True

        if algo == "DES-CBC":
            key = _EVP_BytesToKey(passphrase, salt, 8)
            objdec = DES.new(key, DES.MODE_CBC, salt)
        elif algo == "DES-EDE3-CBC":
            key = _EVP_BytesToKey(passphrase, salt, 24)
            objdec = DES3.new(key, DES3.MODE_CBC, salt)
        elif algo == "AES-128-CBC":
            key = _EVP_BytesToKey(passphrase, salt[:8], 16)
            objdec = AES.new(key, AES.MODE_CBC, salt)
        elif algo == "AES-192-CBC":
            key = _EVP_BytesToKey(passphrase, salt[:8], 24)
            objdec = AES.new(key, AES.MODE_CBC, salt)
        elif algo == "AES-256-CBC":
            key = _EVP_BytesToKey(passphrase, salt[:8], 32)
            objdec = AES.new(key, AES.MODE_CBC, salt)
        elif algo.lower() == "id-aes256-gcm":
            key = _EVP_BytesToKey(passphrase, salt[:8], 32)
            objdec = AES.new(key, AES.MODE_GCM, nonce=salt)
            padding = False
        else:
            raise ValueError("Unsupport PEM encryption algorithm (%s)." % algo)
        lines = lines[2:]
    else:
        objdec = None

    # Decode body
    data = a2b_base64(''.join(lines[1:-1]))
    enc_flag = False
    if objdec:
        if padding:
            data = unpad(objdec.decrypt(data), objdec.block_size)
        else:
            # There is no tag, so we don't use decrypt_and_verify
            data = objdec.decrypt(data)
        enc_flag = True

    return (data, marker, enc_flag)
