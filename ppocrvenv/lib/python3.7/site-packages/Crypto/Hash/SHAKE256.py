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

from Crypto.Hash.keccak import _raw_keccak_lib

class SHAKE256_XOF(object):
    """A SHAKE256 hash object.
    Do not instantiate directly.
    Use the :func:`new` function.

    :ivar oid: ASN.1 Object ID
    :vartype oid: string
    """

    # ASN.1 Object ID
    oid = "2.16.840.1.101.3.4.2.12"

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_keccak_lib.keccak_init(state.address_of(),
                                             c_size_t(64),
                                             c_ubyte(24))
        if result:
            raise ValueError("Error %d while instantiating SHAKE256"
                             % result)
        self._state = SmartPointer(state.get(),
                                   _raw_keccak_lib.keccak_destroy)
        self._is_squeezing = False
        self._padding = 0x1F

        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Args:
            data (byte string/byte array/memoryview): The next chunk of the message being hashed.
        """

        if self._is_squeezing:
            raise TypeError("You cannot call 'update' after the first 'read'")

        result = _raw_keccak_lib.keccak_absorb(self._state.get(),
                                               c_uint8_ptr(data),
                                               c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while updating SHAKE256 state"
                             % result)
        return self

    def read(self, length):
        """
        Compute the next piece of XOF output.

        .. note::
            You cannot use :meth:`update` anymore after the first call to
            :meth:`read`.

        Args:
            length (integer): the amount of bytes this method must return

        :return: the next piece of XOF output (of the given length)
        :rtype: byte string
        """

        self._is_squeezing = True
        bfr = create_string_buffer(length)
        result = _raw_keccak_lib.keccak_squeeze(self._state.get(),
                                                bfr,
                                                c_size_t(length),
                                                c_ubyte(self._padding))
        if result:
            raise ValueError("Error %d while extracting from SHAKE256"
                             % result)

        return get_raw_buffer(bfr)

    def new(self, data=None):
        return type(self)(data=data)


def new(data=None):
    """Return a fresh instance of a SHAKE256 object.

    Args:
       data (bytes/bytearray/memoryview):
        The very first chunk of the message to hash.
        It is equivalent to an early call to :meth:`update`.
        Optional.

    :Return: A :class:`SHAKE256_XOF` object
    """

    return SHAKE256_XOF(data=data)
