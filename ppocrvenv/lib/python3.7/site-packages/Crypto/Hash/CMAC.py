# -*- coding: utf-8 -*-
#
# Hash/CMAC.py - Implements the CMAC algorithm
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

from binascii import unhexlify

from Crypto.Hash import BLAKE2s
from Crypto.Util.strxor import strxor
from Crypto.Util.number import long_to_bytes, bytes_to_long
from Crypto.Util.py3compat import bord, tobytes, _copy_bytes
from Crypto.Random import get_random_bytes


# The size of the authentication tag produced by the MAC.
digest_size = None


def _shift_bytes(bs, xor_lsb=0):
    num = (bytes_to_long(bs) << 1) ^ xor_lsb
    return long_to_bytes(num, len(bs))[-len(bs):]


class CMAC(object):
    """A CMAC hash object.
    Do not instantiate directly. Use the :func:`new` function.

    :ivar digest_size: the size in bytes of the resulting MAC tag
    :vartype digest_size: integer
    """

    digest_size = None

    def __init__(self, key, msg, ciphermod, cipher_params, mac_len,
                 update_after_digest):

        self.digest_size = mac_len

        self._key = _copy_bytes(None, None, key)
        self._factory = ciphermod
        self._cipher_params = cipher_params
        self._block_size = bs = ciphermod.block_size
        self._mac_tag = None
        self._update_after_digest = update_after_digest

        # Section 5.3 of NIST SP 800 38B and Appendix B
        if bs == 8:
            const_Rb = 0x1B
            self._max_size = 8 * (2 ** 21)
        elif bs == 16:
            const_Rb = 0x87
            self._max_size = 16 * (2 ** 48)
        else:
            raise TypeError("CMAC requires a cipher with a block size"
                            " of 8 or 16 bytes, not %d" % bs)

        # Compute sub-keys
        zero_block = b'\x00' * bs
        self._ecb = ciphermod.new(key,
                                  ciphermod.MODE_ECB,
                                  **self._cipher_params)
        L = self._ecb.encrypt(zero_block)
        if bord(L[0]) & 0x80:
            self._k1 = _shift_bytes(L, const_Rb)
        else:
            self._k1 = _shift_bytes(L)
        if bord(self._k1[0]) & 0x80:
            self._k2 = _shift_bytes(self._k1, const_Rb)
        else:
            self._k2 = _shift_bytes(self._k1)

        # Initialize CBC cipher with zero IV
        self._cbc = ciphermod.new(key,
                                  ciphermod.MODE_CBC,
                                  zero_block,
                                  **self._cipher_params)

        # Cache for outstanding data to authenticate
        self._cache = bytearray(bs)
        self._cache_n = 0

        # Last piece of ciphertext produced
        self._last_ct = zero_block

        # Last block that was encrypted with AES
        self._last_pt = None

        # Counter for total message size
        self._data_size = 0

        if msg:
            self.update(msg)

    def update(self, msg):
        """Authenticate the next chunk of message.

        Args:
            data (byte string/byte array/memoryview): The next chunk of data
        """

        if self._mac_tag is not None and not self._update_after_digest:
            raise TypeError("update() cannot be called after digest() or verify()")

        self._data_size += len(msg)
        bs = self._block_size

        if self._cache_n > 0:
            filler = min(bs - self._cache_n, len(msg))
            self._cache[self._cache_n:self._cache_n+filler] = msg[:filler]
            self._cache_n += filler

            if self._cache_n < bs:
                return self

            msg = memoryview(msg)[filler:]
            self._update(self._cache)
            self._cache_n = 0

        remain = len(msg) % bs
        if remain > 0:
            self._update(msg[:-remain])
            self._cache[:remain] = msg[-remain:]
        else:
            self._update(msg)
        self._cache_n = remain
        return self

    def _update(self, data_block):
        """Update a block aligned to the block boundary"""
        
        bs = self._block_size
        assert len(data_block) % bs == 0

        if len(data_block) == 0:
            return

        ct = self._cbc.encrypt(data_block)
        if len(data_block) == bs:
            second_last = self._last_ct
        else:
            second_last = ct[-bs*2:-bs]
        self._last_ct = ct[-bs:]
        self._last_pt = strxor(second_last, data_block[-bs:])

    def copy(self):
        """Return a copy ("clone") of the CMAC object.

        The copy will have the same internal state as the original CMAC
        object.
        This can be used to efficiently compute the MAC tag of byte
        strings that share a common initial substring.

        :return: An :class:`CMAC`
        """

        obj = self.__new__(CMAC)
        obj.__dict__ = self.__dict__.copy()
        obj._cbc = self._factory.new(self._key,
                                     self._factory.MODE_CBC,
                                     self._last_ct,
                                     **self._cipher_params)
        obj._cache = self._cache[:]
        obj._last_ct = self._last_ct[:]
        return obj

    def digest(self):
        """Return the **binary** (non-printable) MAC tag of the message
        that has been authenticated so far.

        :return: The MAC tag, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        bs = self._block_size

        if self._mac_tag is not None and not self._update_after_digest:
            return self._mac_tag

        if self._data_size > self._max_size:
            raise ValueError("MAC is unsafe for this message")

        if self._cache_n == 0 and self._data_size > 0:
            # Last block was full
            pt = strxor(self._last_pt, self._k1)
        else:
            # Last block is partial (or message length is zero)
            partial = self._cache[:]
            partial[self._cache_n:] = b'\x80' + b'\x00' * (bs - self._cache_n - 1)
            pt = strxor(strxor(self._last_ct, partial), self._k2)

        self._mac_tag = self._ecb.encrypt(pt)[:self.digest_size]

        return self._mac_tag

    def hexdigest(self):
        """Return the **printable** MAC tag of the message authenticated so far.

        :return: The MAC tag, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x)
                        for x in tuple(self.digest())])

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (byte string/byte array/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        secret = get_random_bytes(16)

        mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=mac_tag)
        mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=self.digest())

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexverify(self, hex_mac_tag):
        """Return the **printable** MAC tag of the message authenticated so far.

        :return: The MAC tag, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        self.verify(unhexlify(tobytes(hex_mac_tag)))


def new(key, msg=None, ciphermod=None, cipher_params=None, mac_len=None,
        update_after_digest=False):
    """Create a new MAC object.

    Args:
        key (byte string/byte array/memoryview):
            key for the CMAC object.
            The key must be valid for the underlying cipher algorithm.
            For instance, it must be 16 bytes long for AES-128.
        ciphermod (module):
            A cipher module from :mod:`Crypto.Cipher`.
            The cipher's block size has to be 128 bits,
            like :mod:`Crypto.Cipher.AES`, to reduce the probability
            of collisions.
        msg (byte string/byte array/memoryview):
            Optional. The very first chunk of the message to authenticate.
            It is equivalent to an early call to `CMAC.update`. Optional.
        cipher_params (dict):
            Optional. A set of parameters to use when instantiating a cipher
            object.
        mac_len (integer):
            Length of the MAC, in bytes.
            It must be at least 4 bytes long.
            The default (and recommended) length matches the size of a cipher block.
        update_after_digest (boolean):
            Optional. By default, a hash object cannot be updated anymore after
            the digest is computed. When this flag is ``True``, such check
            is no longer enforced.
    Returns:
        A :class:`CMAC` object
    """

    if ciphermod is None:
        raise TypeError("ciphermod must be specified (try AES)")

    cipher_params = {} if cipher_params is None else dict(cipher_params)

    if mac_len is None:
        mac_len = ciphermod.block_size
    
    if mac_len < 4:
        raise ValueError("MAC tag length must be at least 4 bytes long")
    
    if mac_len > ciphermod.block_size:
        raise ValueError("MAC tag length cannot be larger than a cipher block (%d) bytes" % ciphermod.block_size)

    return CMAC(key, msg, ciphermod, cipher_params, mac_len,
                update_after_digest)
