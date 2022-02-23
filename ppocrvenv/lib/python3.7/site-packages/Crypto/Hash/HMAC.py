#
# HMAC.py - Implements the HMAC algorithm as described by RFC 2104.
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

from Crypto.Util.py3compat import bord, tobytes

from binascii import unhexlify

from Crypto.Hash import MD5
from Crypto.Hash import BLAKE2s
from Crypto.Util.strxor import strxor
from Crypto.Random import get_random_bytes

__all__ = ['new', 'HMAC']


class HMAC(object):
    """An HMAC hash object.
    Do not instantiate directly. Use the :func:`new` function.

    :ivar digest_size: the size in bytes of the resulting MAC tag
    :vartype digest_size: integer
    """

    def __init__(self, key, msg=b"", digestmod=None):

        if digestmod is None:
            digestmod = MD5

        if msg is None:
            msg = b""

        # Size of the MAC tag
        self.digest_size = digestmod.digest_size

        self._digestmod = digestmod

        if isinstance(key, memoryview):
            key = key.tobytes()

        try:
            if len(key) <= digestmod.block_size:
                # Step 1 or 2
                key_0 = key + b"\x00" * (digestmod.block_size - len(key))
            else:
                # Step 3
                hash_k = digestmod.new(key).digest()
                key_0 = hash_k + b"\x00" * (digestmod.block_size - len(hash_k))
        except AttributeError:
            # Not all hash types have "block_size"
            raise ValueError("Hash type incompatible to HMAC")

        # Step 4
        key_0_ipad = strxor(key_0, b"\x36" * len(key_0))

        # Start step 5 and 6
        self._inner = digestmod.new(key_0_ipad)
        self._inner.update(msg)

        # Step 7
        key_0_opad = strxor(key_0, b"\x5c" * len(key_0))

        # Start step 8 and 9
        self._outer = digestmod.new(key_0_opad)

    def update(self, msg):
        """Authenticate the next chunk of message.

        Args:
            data (byte string/byte array/memoryview): The next chunk of data
        """

        self._inner.update(msg)
        return self

    def _pbkdf2_hmac_assist(self, first_digest, iterations):
        """Carry out the expensive inner loop for PBKDF2-HMAC"""

        result = self._digestmod._pbkdf2_hmac_assist(
                                    self._inner,
                                    self._outer,
                                    first_digest,
                                    iterations)
        return result

    def copy(self):
        """Return a copy ("clone") of the HMAC object.

        The copy will have the same internal state as the original HMAC
        object.
        This can be used to efficiently compute the MAC tag of byte
        strings that share a common initial substring.

        :return: An :class:`HMAC`
        """

        new_hmac = HMAC(b"fake key", digestmod=self._digestmod)

        # Syncronize the state
        new_hmac._inner = self._inner.copy()
        new_hmac._outer = self._outer.copy()

        return new_hmac

    def digest(self):
        """Return the **binary** (non-printable) MAC tag of the message
        authenticated so far.

        :return: The MAC tag digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        frozen_outer_hash = self._outer.copy()
        frozen_outer_hash.update(self._inner.digest())
        return frozen_outer_hash.digest()

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (byte string/byte string/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        secret = get_random_bytes(16)

        mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=mac_tag)
        mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=self.digest())

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexdigest(self):
        """Return the **printable** MAC tag of the message authenticated so far.

        :return: The MAC tag, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x)
                        for x in tuple(self.digest())])

    def hexverify(self, hex_mac_tag):
        """Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
            hex_mac_tag (string): the expected MAC of the message,
                as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        self.verify(unhexlify(tobytes(hex_mac_tag)))


def new(key, msg=b"", digestmod=None):
    """Create a new MAC object.

    Args:
        key (bytes/bytearray/memoryview):
            key for the MAC object.
            It must be long enough to match the expected security level of the
            MAC.
        msg (bytes/bytearray/memoryview):
            Optional. The very first chunk of the message to authenticate.
            It is equivalent to an early call to :meth:`HMAC.update`.
        digestmod (module):
            The hash to use to implement the HMAC.
            Default is :mod:`Crypto.Hash.MD5`.

    Returns:
        An :class:`HMAC` object
    """

    return HMAC(key, msg, digestmod)
