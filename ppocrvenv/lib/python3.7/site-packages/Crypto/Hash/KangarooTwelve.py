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

from Crypto.Util._raw_api import (VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr, c_ubyte)

from Crypto.Util.number import long_to_bytes
from Crypto.Util.py3compat import bchr

from .keccak import _raw_keccak_lib


def _length_encode(x):
    if x == 0:
        return b'\x00'

    S = long_to_bytes(x)
    return S + bchr(len(S))


# Possible states for a KangarooTwelve instance, which depend on the amount of data processed so far.
SHORT_MSG = 1       # Still within the first 8192 bytes, but it is not certain we will exceed them.
LONG_MSG_S0 = 2     # Still within the first 8192 bytes, and it is certain we will exceed them.
LONG_MSG_SX = 3     # Beyond the first 8192 bytes.
SQUEEZING = 4       # No more data to process.


class K12_XOF(object):
    """A KangarooTwelve hash object.
    Do not instantiate directly.
    Use the :func:`new` function.
    """

    def __init__(self, data, custom):

        if custom == None:
            custom = b''

        self._custom = custom + _length_encode(len(custom))
        self._state = SHORT_MSG
        self._padding = None        # Final padding is only decided in read()

        # Internal hash that consumes FinalNode
        self._hash1 = self._create_keccak()
        self._length1 = 0

        # Internal hash that produces CV_i (reset each time)
        self._hash2 = None
        self._length2 = 0

        # Incremented by one for each 8192-byte block
        self._ctr = 0

        if data:
            self.update(data)

    def _create_keccak(self):
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(),
                                             c_size_t(32),  # 32 bytes of capacity (256 bits)
                                             c_ubyte(12))   # Reduced number of rounds
        if result:
            raise ValueError("Error %d while instantiating KangarooTwelve"
                             % result)
        return SmartPointer(state.get(), _raw_keccak_lib.keccak_destroy)

    def _update(self, data, hash_obj):
        result = _raw_keccak_lib.keccak_absorb(hash_obj.get(),
                                               c_uint8_ptr(data),
                                               c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while updating KangarooTwelve state"
                             % result)

    def _squeeze(self, hash_obj, length, padding):
        bfr = create_string_buffer(length)
        result = _raw_keccak_lib.keccak_squeeze(hash_obj.get(),
                                                bfr,
                                                c_size_t(length),
                                                c_ubyte(padding))
        if result:
            raise ValueError("Error %d while extracting from KangarooTwelve"
                             % result)

        return get_raw_buffer(bfr)

    def _reset(self, hash_obj):
        result = _raw_keccak_lib.keccak_reset(hash_obj.get())
        if result:
            raise ValueError("Error %d while resetting KangarooTwelve state"
                             % result)

    def update(self, data):
        """Hash the next piece of data.

        .. note::
            For better performance, submit chunks with a length multiple of 8192 bytes.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the
            message to hash.
        """

        if self._state == SQUEEZING:
            raise TypeError("You cannot call 'update' after the first 'read'")

        if self._state == SHORT_MSG:
            next_length = self._length1 + len(data)

            if next_length + len(self._custom) <= 8192:
                self._length1 = next_length
                self._update(data, self._hash1)
                return self

            # Switch to tree hashing
            self._state = LONG_MSG_S0

        if self._state == LONG_MSG_S0:
            data_mem = memoryview(data)
            assert(self._length1 < 8192)
            dtc = min(len(data), 8192 - self._length1)
            self._update(data_mem[:dtc], self._hash1)
            self._length1 += dtc

            if self._length1 < 8192:
                return self

            # Finish hashing S_0 and start S_1
            assert(self._length1 == 8192)

            divider = b'\x03' + b'\x00' * 7
            self._update(divider, self._hash1)
            self._length1 += 8

            self._hash2 = self._create_keccak()
            self._length2 = 0
            self._ctr = 1

            self._state = LONG_MSG_SX
            return self.update(data_mem[dtc:])

        # LONG_MSG_SX
        assert(self._state == LONG_MSG_SX)
        index = 0
        len_data = len(data)

        # All iteractions could actually run in parallel
        data_mem = memoryview(data)
        while index < len_data:

            new_index = min(index + 8192 - self._length2, len_data)
            self._update(data_mem[index:new_index], self._hash2)
            self._length2 += new_index - index
            index = new_index

            if self._length2 == 8192:
                cv_i = self._squeeze(self._hash2, 32, 0x0B)
                self._update(cv_i, self._hash1)
                self._length1 += 32
                self._reset(self._hash2)
                self._length2 = 0
                self._ctr += 1

        return self

    def read(self, length):
        """
        Produce more bytes of the digest.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        """

        custom_was_consumed = False

        if self._state == SHORT_MSG:
            self._update(self._custom, self._hash1)
            self._padding = 0x07
            self._state = SQUEEZING

        if self._state == LONG_MSG_S0:
            self.update(self._custom)
            custom_was_consumed = True
            assert(self._state == LONG_MSG_SX)

        if self._state == LONG_MSG_SX:
            if not custom_was_consumed:
                self.update(self._custom)

            # Is there still some leftover data in hash2?
            if self._length2 > 0:
                cv_i = self._squeeze(self._hash2, 32, 0x0B)
                self._update(cv_i, self._hash1)
                self._length1 += 32
                self._reset(self._hash2)
                self._length2 = 0
                self._ctr += 1

            trailer = _length_encode(self._ctr - 1) + b'\xFF\xFF'
            self._update(trailer, self._hash1)

            self._padding = 0x06
            self._state = SQUEEZING

        return self._squeeze(self._hash1, length, self._padding)

    def new(self, data=None, custom=b''):
        return type(self)(data, custom)


def new(data=None, custom=None):
    """Return a fresh instance of a KangarooTwelve object.

    Args:
       data (bytes/bytearray/memoryview):
        Optional.
        The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`update`.
       custom (bytes):
        Optional.
        A customization byte string.

    :Return: A :class:`K12_XOF` object
    """

    return K12_XOF(data, custom)
