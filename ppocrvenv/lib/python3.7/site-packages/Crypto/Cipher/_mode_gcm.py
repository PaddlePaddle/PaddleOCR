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

"""
Galois/Counter Mode (GCM).
"""

__all__ = ['GcmMode']

from binascii import unhexlify

from Crypto.Util.py3compat import bord, _copy_bytes

from Crypto.Util._raw_api import is_buffer

from Crypto.Util.number import long_to_bytes, bytes_to_long
from Crypto.Hash import BLAKE2s
from Crypto.Random import get_random_bytes

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
                                  create_string_buffer, get_raw_buffer,
                                  SmartPointer, c_size_t, c_uint8_ptr)

from Crypto.Util import _cpu_features


# C API by module implementing GHASH
_ghash_api_template = """
    int ghash_%imp%(uint8_t y_out[16],
                    const uint8_t block_data[],
                    size_t len,
                    const uint8_t y_in[16],
                    const void *exp_key);
    int ghash_expand_%imp%(const uint8_t h[16],
                           void **ghash_tables);
    int ghash_destroy_%imp%(void *ghash_tables);
"""

def _build_impl(lib, postfix):
    from collections import namedtuple

    funcs = ( "ghash", "ghash_expand", "ghash_destroy" )
    GHASH_Imp = namedtuple('_GHash_Imp', funcs)
    try:
        imp_funcs = [ getattr(lib, x + "_" + postfix) for x in funcs ]
    except AttributeError:      # Make sphinx stop complaining with its mocklib
        imp_funcs = [ None ] * 3
    params = dict(zip(funcs, imp_funcs))
    return GHASH_Imp(**params)


def _get_ghash_portable():
    api = _ghash_api_template.replace("%imp%", "portable")
    lib = load_pycryptodome_raw_lib("Crypto.Hash._ghash_portable", api)
    result = _build_impl(lib, "portable")
    return result
_ghash_portable = _get_ghash_portable()


def _get_ghash_clmul():
    """Return None if CLMUL implementation is not available"""

    if not _cpu_features.have_clmul():
        return None
    try:
        api = _ghash_api_template.replace("%imp%", "clmul")
        lib = load_pycryptodome_raw_lib("Crypto.Hash._ghash_clmul", api)
        result = _build_impl(lib, "clmul")
    except OSError:
        result = None
    return result
_ghash_clmul = _get_ghash_clmul()


class _GHASH(object):
    """GHASH function defined in NIST SP 800-38D, Algorithm 2.

    If X_1, X_2, .. X_m are the blocks of input data, the function
    computes:

       X_1*H^{m} + X_2*H^{m-1} + ... + X_m*H

    in the Galois field GF(2^256) using the reducing polynomial
    (x^128 + x^7 + x^2 + x + 1).
    """

    def __init__(self, subkey, ghash_c):
        assert len(subkey) == 16

        self.ghash_c = ghash_c

        self._exp_key = VoidPointer()
        result = ghash_c.ghash_expand(c_uint8_ptr(subkey),
                                      self._exp_key.address_of())
        if result:
            raise ValueError("Error %d while expanding the GHASH key" % result)

        self._exp_key = SmartPointer(self._exp_key.get(),
                                     ghash_c.ghash_destroy)

        # create_string_buffer always returns a string of zeroes
        self._last_y = create_string_buffer(16)

    def update(self, block_data):
        assert len(block_data) % 16 == 0

        result = self.ghash_c.ghash(self._last_y,
                                    c_uint8_ptr(block_data),
                                    c_size_t(len(block_data)),
                                    self._last_y,
                                    self._exp_key.get())
        if result:
            raise ValueError("Error %d while updating GHASH" % result)

        return self

    def digest(self):
        return get_raw_buffer(self._last_y)


def enum(**enums):
    return type('Enum', (), enums)


MacStatus = enum(PROCESSING_AUTH_DATA=1, PROCESSING_CIPHERTEXT=2)


class GcmMode(object):
    """Galois Counter Mode (GCM).

    This is an Authenticated Encryption with Associated Data (`AEAD`_) mode.
    It provides both confidentiality and authenticity.

    The header of the message may be left in the clear, if needed, and it will
    still be subject to authentication. The decryption step tells the receiver
    if the message comes from a source that really knowns the secret key.
    Additionally, decryption detects if any part of the message - including the
    header - has been modified or corrupted.

    This mode requires a *nonce*.

    This mode is only available for ciphers that operate on 128 bits blocks
    (e.g. AES but not TDES).

    See `NIST SP800-38D`_.

    .. _`NIST SP800-38D`: http://csrc.nist.gov/publications/nistpubs/800-38D/SP-800-38D.pdf
    .. _AEAD: http://blog.cryptographyengineering.com/2012/05/how-to-choose-authenticated-encryption.html

    :undocumented: __init__
    """

    def __init__(self, factory, key, nonce, mac_len, cipher_params, ghash_c):

        self.block_size = factory.block_size
        if self.block_size != 16:
            raise ValueError("GCM mode is only available for ciphers"
                             " that operate on 128 bits blocks")

        if len(nonce) == 0:
            raise ValueError("Nonce cannot be empty")
        
        if not is_buffer(nonce):
            raise TypeError("Nonce must be bytes, bytearray or memoryview")

        # See NIST SP 800 38D, 5.2.1.1
        if len(nonce) > 2**64 - 1:
            raise ValueError("Nonce exceeds maximum length")


        self.nonce = _copy_bytes(None, None, nonce)
        """Nonce"""

        self._factory = factory
        self._key = _copy_bytes(None, None, key)
        self._tag = None  # Cache for MAC tag

        self._mac_len = mac_len
        if not (4 <= mac_len <= 16):
            raise ValueError("Parameter 'mac_len' must be in the range 4..16")

        # Allowed transitions after initialization
        self._next = [self.update, self.encrypt, self.decrypt,
                      self.digest, self.verify]

        self._no_more_assoc_data = False

        # Length of associated data
        self._auth_len = 0

        # Length of the ciphertext or plaintext
        self._msg_len = 0

        # Step 1 in SP800-38D, Algorithm 4 (encryption) - Compute H
        # See also Algorithm 5 (decryption)
        hash_subkey = factory.new(key,
                                  self._factory.MODE_ECB,
                                  **cipher_params
                                  ).encrypt(b'\x00' * 16)

        # Step 2 - Compute J0
        if len(self.nonce) == 12:
            j0 = self.nonce + b"\x00\x00\x00\x01"
        else:
            fill = (16 - (len(nonce) % 16)) % 16 + 8
            ghash_in = (self.nonce +
                        b'\x00' * fill +
                        long_to_bytes(8 * len(nonce), 8))
            j0 = _GHASH(hash_subkey, ghash_c).update(ghash_in).digest()

        # Step 3 - Prepare GCTR cipher for encryption/decryption
        nonce_ctr = j0[:12]
        iv_ctr = (bytes_to_long(j0) + 1) & 0xFFFFFFFF
        self._cipher = factory.new(key,
                                   self._factory.MODE_CTR,
                                   initial_value=iv_ctr,
                                   nonce=nonce_ctr,
                                   **cipher_params)

        # Step 5 - Bootstrat GHASH
        self._signer = _GHASH(hash_subkey, ghash_c)

        # Step 6 - Prepare GCTR cipher for GMAC
        self._tag_cipher = factory.new(key,
                                       self._factory.MODE_CTR,
                                       initial_value=j0,
                                       nonce=b"",
                                       **cipher_params)

        # Cache for data to authenticate
        self._cache = b""

        self._status = MacStatus.PROCESSING_AUTH_DATA

    def update(self, assoc_data):
        """Protect associated data

        If there is any associated data, the caller has to invoke
        this function one or more times, before using
        ``decrypt`` or ``encrypt``.

        By *associated data* it is meant any data (e.g. packet headers) that
        will not be encrypted and will be transmitted in the clear.
        However, the receiver is still able to detect any modification to it.
        In GCM, the *associated data* is also called
        *additional authenticated data* (AAD).

        If there is no associated data, this method must not be called.

        The caller may split associated data in segments of any size, and
        invoke this method multiple times, each time with the next segment.

        :Parameters:
          assoc_data : bytes/bytearray/memoryview
            A piece of associated data. There are no restrictions on its size.
        """

        if self.update not in self._next:
            raise TypeError("update() can only be called"
                            " immediately after initialization")

        self._next = [self.update, self.encrypt, self.decrypt,
                      self.digest, self.verify]

        self._update(assoc_data)
        self._auth_len += len(assoc_data)

        # See NIST SP 800 38D, 5.2.1.1
        if self._auth_len > 2**64 - 1:
            raise ValueError("Additional Authenticated Data exceeds maximum length")

        return self

    def _update(self, data):
        assert(len(self._cache) < 16)

        if len(self._cache) > 0:
            filler = min(16 - len(self._cache), len(data))
            self._cache += _copy_bytes(None, filler, data)
            data = data[filler:]

            if len(self._cache) < 16:
                return

            # The cache is exactly one block
            self._signer.update(self._cache)
            self._cache = b""

        update_len = len(data) // 16 * 16
        self._cache = _copy_bytes(update_len, None, data)
        if update_len > 0:
            self._signer.update(data[:update_len])

    def _pad_cache_and_update(self):
        assert(len(self._cache) < 16)

        # The authenticated data A is concatenated to the minimum
        # number of zero bytes (possibly none) such that the
        # - ciphertext C is aligned to the 16 byte boundary.
        #   See step 5 in section 7.1
        # - ciphertext C is aligned to the 16 byte boundary.
        #   See step 6 in section 7.2
        len_cache = len(self._cache)
        if len_cache > 0:
            self._update(b'\x00' * (16 - len_cache))

    def encrypt(self, plaintext, output=None):
        """Encrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have encrypted a message
        you cannot encrypt (or decrypt) another message using the same
        object.

        The data to encrypt can be broken up in two or
        more pieces and `encrypt` can be called multiple times.

        That is, the statement:

            >>> c.encrypt(a) + c.encrypt(b)

        is equivalent to:

             >>> c.encrypt(a+b)

        This function does not add any padding to the plaintext.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The piece of data to encrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
          If ``output`` is ``None``, the ciphertext as ``bytes``.
          Otherwise, ``None``.
        """

        if self.encrypt not in self._next:
            raise TypeError("encrypt() can only be called after"
                            " initialization or an update()")
        self._next = [self.encrypt, self.digest]

        ciphertext = self._cipher.encrypt(plaintext, output=output)

        if self._status == MacStatus.PROCESSING_AUTH_DATA:
            self._pad_cache_and_update()
            self._status = MacStatus.PROCESSING_CIPHERTEXT

        self._update(ciphertext if output is None else output)
        self._msg_len += len(plaintext)

        # See NIST SP 800 38D, 5.2.1.1
        if self._msg_len > 2**39 - 256:
            raise ValueError("Plaintext exceeds maximum length")

        return ciphertext

    def decrypt(self, ciphertext, output=None):
        """Decrypt data with the key and the parameters set at initialization.

        A cipher object is stateful: once you have decrypted a message
        you cannot decrypt (or encrypt) another message with the same
        object.

        The data to decrypt can be broken up in two or
        more pieces and `decrypt` can be called multiple times.

        That is, the statement:

            >>> c.decrypt(a) + c.decrypt(b)

        is equivalent to:

             >>> c.decrypt(a+b)

        This function does not remove any padding from the plaintext.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
            It can be of any length.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return:
          If ``output`` is ``None``, the plaintext as ``bytes``.
          Otherwise, ``None``.
        """

        if self.decrypt not in self._next:
            raise TypeError("decrypt() can only be called"
                            " after initialization or an update()")
        self._next = [self.decrypt, self.verify]

        if self._status == MacStatus.PROCESSING_AUTH_DATA:
            self._pad_cache_and_update()
            self._status = MacStatus.PROCESSING_CIPHERTEXT

        self._update(ciphertext)
        self._msg_len += len(ciphertext)

        return self._cipher.decrypt(ciphertext, output=output)

    def digest(self):
        """Compute the *binary* MAC tag in an AEAD mode.

        The caller invokes this function at the very end.

        This method returns the MAC that shall be sent to the receiver,
        together with the ciphertext.

        :Return: the MAC, as a byte string.
        """

        if self.digest not in self._next:
            raise TypeError("digest() cannot be called when decrypting"
                            " or validating a message")
        self._next = [self.digest]

        return self._compute_mac()

    def _compute_mac(self):
        """Compute MAC without any FSM checks."""

        if self._tag:
            return self._tag

        # Step 5 in NIST SP 800-38D, Algorithm 4 - Compute S
        self._pad_cache_and_update()
        self._update(long_to_bytes(8 * self._auth_len, 8))
        self._update(long_to_bytes(8 * self._msg_len, 8))
        s_tag = self._signer.digest()

        # Step 6 - Compute T
        self._tag = self._tag_cipher.encrypt(s_tag)[:self._mac_len]

        return self._tag

    def hexdigest(self):
        """Compute the *printable* MAC tag.

        This method is like `digest`.

        :Return: the MAC, as a hexadecimal string.
        """
        return "".join(["%02x" % bord(x) for x in self.digest()])

    def verify(self, received_mac_tag):
        """Validate the *binary* MAC tag.

        The caller invokes this function at the very end.

        This method checks if the decrypted message is indeed valid
        (that is, if the key is correct) and it has not been
        tampered with while in transit.

        :Parameters:
          received_mac_tag : bytes/bytearray/memoryview
            This is the *binary* MAC, as received from the sender.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        if self.verify not in self._next:
            raise TypeError("verify() cannot be called"
                            " when encrypting a message")
        self._next = [self.verify]

        secret = get_random_bytes(16)

        mac1 = BLAKE2s.new(digest_bits=160, key=secret,
                           data=self._compute_mac())
        mac2 = BLAKE2s.new(digest_bits=160, key=secret,
                           data=received_mac_tag)

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexverify(self, hex_mac_tag):
        """Validate the *printable* MAC tag.

        This method is like `verify`.

        :Parameters:
          hex_mac_tag : string
            This is the *printable* MAC, as received from the sender.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        self.verify(unhexlify(hex_mac_tag))

    def encrypt_and_digest(self, plaintext, output=None):
        """Perform encrypt() and digest() in one step.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The piece of data to encrypt.
        :Keywords:
          output : bytearray/memoryview
            The location where the ciphertext must be written to.
            If ``None``, the ciphertext is returned.
        :Return:
            a tuple with two items:

            - the ciphertext, as ``bytes``
            - the MAC tag, as ``bytes``

            The first item becomes ``None`` when the ``output`` parameter
            specified a location for the result.
        """

        return self.encrypt(plaintext, output=output), self.digest()

    def decrypt_and_verify(self, ciphertext, received_mac_tag, output=None):
        """Perform decrypt() and verify() in one step.

        :Parameters:
          ciphertext : bytes/bytearray/memoryview
            The piece of data to decrypt.
          received_mac_tag : byte string
            This is the *binary* MAC, as received from the sender.
        :Keywords:
          output : bytearray/memoryview
            The location where the plaintext must be written to.
            If ``None``, the plaintext is returned.
        :Return: the plaintext as ``bytes`` or ``None`` when the ``output``
            parameter specified a location for the result.
        :Raises ValueError:
            if the MAC does not match. The message has been tampered with
            or the key is incorrect.
        """

        plaintext = self.decrypt(ciphertext, output=output)
        self.verify(received_mac_tag)
        return plaintext


def _create_gcm_cipher(factory, **kwargs):
    """Create a new block cipher, configured in Galois Counter Mode (GCM).

    :Parameters:
      factory : module
        A block cipher module, taken from `Crypto.Cipher`.
        The cipher must have block length of 16 bytes.
        GCM has been only defined for `Crypto.Cipher.AES`.

    :Keywords:
      key : bytes/bytearray/memoryview
        The secret key to use in the symmetric cipher.
        It must be 16 (e.g. *AES-128*), 24 (e.g. *AES-192*)
        or 32 (e.g. *AES-256*) bytes long.

      nonce : bytes/bytearray/memoryview
        A value that must never be reused for any other encryption.

        There are no restrictions on its length,
        but it is recommended to use at least 16 bytes.

        The nonce shall never repeat for two
        different messages encrypted with the same key,
        but it does not need to be random.

        If not provided, a 16 byte nonce will be randomly created.

      mac_len : integer
        Length of the MAC, in bytes.
        It must be no larger than 16 bytes (which is the default).
    """

    try:
        key = kwargs.pop("key")
    except KeyError as e:
        raise TypeError("Missing parameter:" + str(e))

    nonce = kwargs.pop("nonce", None)
    if nonce is None:
        nonce = get_random_bytes(16)
    mac_len = kwargs.pop("mac_len", 16)

    # Not documented - only used for testing
    use_clmul = kwargs.pop("use_clmul", True)
    if use_clmul and _ghash_clmul:
        ghash_c = _ghash_clmul
    else:
        ghash_c = _ghash_portable

    return GcmMode(factory, key, nonce, mac_len, kwargs, ghash_c)
