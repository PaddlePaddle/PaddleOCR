# ===================================================================
#
# Copyright (c) 2015, Legrandin <helderijs@gmail.com>
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

from Crypto.Util.py3compat import bord

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr, c_ubyte)

_raw_keccak_lib = load_pycryptodome_raw_lib("Crypto.Hash._keccak",
                        """
                        int keccak_init(void **state,
                                        size_t capacity_bytes,
                                        uint8_t rounds);
                        int keccak_destroy(void *state);
                        int keccak_absorb(void *state,
                                          const uint8_t *in,
                                          size_t len);
                        int keccak_squeeze(const void *state,
                                           uint8_t *out,
                                           size_t len,
                                           uint8_t padding);
                        int keccak_digest(void *state,
                                          uint8_t *digest,
                                          size_t len,
                                          uint8_t padding);
                        int keccak_copy(const void *src, void *dst);
                        int keccak_reset(void *state);
                        """)

class Keccak_Hash(object):
    """A Keccak hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar digest_size: the size in bytes of the resulting hash
    :vartype digest_size: integer
    """

    def __init__(self, data, digest_bytes, update_after_digest):
        # The size of the resulting hash in bytes.
        self.digest_size = digest_bytes

        self._update_after_digest = update_after_digest
        self._digest_done = False
        self._padding = 0x01

        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(),
                                             c_size_t(self.digest_size * 2),
                                             c_ubyte(24))
        if result:
            raise ValueError("Error %d while instantiating keccak" % result)
        self._state = SmartPointer(state.get(),
                                   _raw_keccak_lib.keccak_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """

        if self._digest_done and not self._update_after_digest:
            raise TypeError("You can only call 'digest' or 'hexdigest' on this object")

        result = _raw_keccak_lib.keccak_absorb(self._state.get(),
                                               c_uint8_ptr(data),
                                               c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while updating keccak" % result)
        return self

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Binary form.
        :rtype: byte string
        """

        self._digest_done = True
        bfr = create_string_buffer(self.digest_size)
        result = _raw_keccak_lib.keccak_digest(self._state.get(),
                                               bfr,
                                               c_size_t(self.digest_size),
                                               c_ubyte(self._padding))
        if result:
            raise ValueError("Error %d while squeezing keccak" % result)

        return get_raw_buffer(bfr)

    def hexdigest(self):
        """Return the **printable** digest of the message that has been hashed so far.

        :return: The hash digest, computed over the data processed so far.
                 Hexadecimal encoded.
        :rtype: string
        """

        return "".join(["%02x" % bord(x) for x in self.digest()])

    def new(self, **kwargs):
        """Create a fresh Keccak hash object."""

        if "digest_bytes" not in kwargs and "digest_bits" not in kwargs:
            kwargs["digest_bytes"] = self.digest_size

        return new(**kwargs)


def new(**kwargs):
    """Create a new hash object.

    Args:
        data (bytes/bytearray/memoryview):
            The very first chunk of the message to hash.
            It is equivalent to an early call to :meth:`Keccak_Hash.update`.
        digest_bytes (integer):
            The size of the digest, in bytes (28, 32, 48, 64).
        digest_bits (integer):
            The size of the digest, in bits (224, 256, 384, 512).
        update_after_digest (boolean):
            Whether :meth:`Keccak.digest` can be followed by another
            :meth:`Keccak.update` (default: ``False``).

    :Return: A :class:`Keccak_Hash` hash object
    """

    data = kwargs.pop("data", None)
    update_after_digest = kwargs.pop("update_after_digest", False)

    digest_bytes = kwargs.pop("digest_bytes", None)
    digest_bits = kwargs.pop("digest_bits", None)
    if None not in (digest_bytes, digest_bits):
        raise TypeError("Only one digest parameter must be provided")
    if (None, None) == (digest_bytes, digest_bits):
        raise TypeError("Digest size (bits, bytes) not provided")
    if digest_bytes is not None:
        if digest_bytes not in (28, 32, 48, 64):
            raise ValueError("'digest_bytes' must be: 28, 32, 48 or 64")
    else:
        if digest_bits not in (224, 256, 384, 512):
            raise ValueError("'digest_bytes' must be: 224, 256, 384 or 512")
        digest_bytes = digest_bits // 8

    if kwargs:
        raise TypeError("Unknown parameters: " + str(kwargs))

    return Keccak_Hash(data, digest_bytes, update_after_digest)
