# ===================================================================
#
# Copyright (c) 2021, Legrandin <helderijs@gmail.com>
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

from binascii import unhexlify

from Crypto.Util.py3compat import bord, tobytes, is_bytes
from Crypto.Random import get_random_bytes

from . import cSHAKE128, SHA3_256
from .cSHAKE128 import _bytepad, _encode_str, _right_encode


class KMAC_Hash(object):
    """A KMAC hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, data, key, mac_len, custom,
                 oid_variant, cshake, rate):

        # See https://tools.ietf.org/html/rfc8702
        self.oid = "2.16.840.1.101.3.4.2." + oid_variant
        self.digest_size = mac_len

        self._mac = None

        partial_newX = _bytepad(_encode_str(tobytes(key)), rate)
        self._cshake = cshake._new(partial_newX, custom, b"KMAC")

        if data:
            self._cshake.update(data)

    def update(self, data):
        """Authenticate the next chunk of message.

        Args:
            data (bytes/bytearray/memoryview): The next chunk of the message to
            authenticate.
        """

        if self._mac:
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")

        self._cshake.update(data)
        return self

    def digest(self):
        """Return the **binary** (non-printable) MAC tag of the message.

        :return: The MAC tag. Binary form.
        :rtype: byte string
        """

        if not self._mac:
            self._cshake.update(_right_encode(self.digest_size * 8))
            self._mac = self._cshake.read(self.digest_size)

        return self._mac

    def hexdigest(self):
        """Return the **printable** MAC tag of the message.

        :return: The MAC tag. Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x) for x in tuple(self.digest())])

    def verify(self, mac_tag):
        """Verify that a given **binary** MAC (computed by another party)
        is valid.

        Args:
          mac_tag (bytes/bytearray/memoryview): the expected MAC of the message.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        secret = get_random_bytes(16)

        mac1 = SHA3_256.new(secret + mac_tag)
        mac2 = SHA3_256.new(secret + self.digest())

        if mac1.digest() != mac2.digest():
            raise ValueError("MAC check failed")

    def hexverify(self, hex_mac_tag):
        """Verify that a given **printable** MAC (computed by another party)
        is valid.

        Args:
            hex_mac_tag (string): the expected MAC of the message, as a hexadecimal string.

        Raises:
            ValueError: if the MAC does not match. It means that the message
                has been tampered with or that the MAC key is incorrect.
        """

        self.verify(unhexlify(tobytes(hex_mac_tag)))

    def new(self, **kwargs):
        """Return a new instance of a KMAC hash object.
        See :func:`new`.
        """

        if "mac_len" not in kwargs:
            kwargs["mac_len"] = self.digest_size

        return new(**kwargs)


def new(**kwargs):
    """Create a new KMAC128 object.

    Args:
        key (bytes/bytearray/memoryview):
            The key to use to compute the MAC.
            It must be at least 128 bits long (16 bytes).
        data (bytes/bytearray/memoryview):
            Optional. The very first chunk of the message to authenticate.
            It is equivalent to an early call to :meth:`KMAC_Hash.update`.
        mac_len (integer):
            Optional. The size of the authentication tag, in bytes.
            Default is 64. Minimum is 8.
        custom (bytes/bytearray/memoryview):
            Optional. A customization byte string (``S`` in SP 800-185).

    Returns:
        A :class:`KMAC_Hash` hash object
    """

    key = kwargs.pop("key", None)
    if not is_bytes(key):
        raise TypeError("You must pass a key to KMAC128")
    if len(key) < 16:
        raise ValueError("The key must be at least 128 bits long (16 bytes)")

    data = kwargs.pop("data", None)

    mac_len = kwargs.pop("mac_len", 64)
    if mac_len < 8:
        raise ValueError("'mac_len' must be 8 bytes or more")

    custom = kwargs.pop("custom", b"")

    if kwargs:
        raise TypeError("Unknown parameters: " + str(kwargs))

    return KMAC_Hash(data, key, mac_len, custom, "19", cSHAKE128, 168)
