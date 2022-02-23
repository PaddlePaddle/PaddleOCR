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
MD4 is specified in RFC1320_ and produces the 128 bit digest of a message.

    >>> from Crypto.Hash import MD4
    >>>
    >>> h = MD4.new()
    >>> h.update(b'Hello')
    >>> print h.hexdigest()

MD4 stand for Message Digest version 4, and it was invented by Rivest in 1990.
This algorithm is insecure. Do not use it for new designs.

.. _RFC1320: http://tools.ietf.org/html/rfc1320
"""

from Crypto.Util.py3compat import bord

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t,
                                  c_uint8_ptr)

_raw_md4_lib = load_pycryptodome_raw_lib(
                        "Crypto.Hash._MD4",
                        """
                        int md4_init(void **shaState);
                        int md4_destroy(void *shaState);
                        int md4_update(void *hs,
                                          const uint8_t *buf,
                                          size_t len);
                        int md4_digest(const void *shaState,
                                          uint8_t digest[20]);
                        int md4_copy(const void *src, void *dst);
                        """)


class MD4Hash(object):
    """Class that implements an MD4 hash
    """

    #: The size of the resulting hash in bytes.
    digest_size = 16
    #: The internal block size of the hash algorithm in bytes.
    block_size = 64
    #: ASN.1 Object ID
    oid = "1.2.840.113549.2.4"

    def __init__(self, data=None):
        state = VoidPointer()
        result = _raw_md4_lib.md4_init(state.address_of())
        if result:
            raise ValueError("Error %d while instantiating MD4"
                             % result)
        self._state = SmartPointer(state.get(),
                                   _raw_md4_lib.md4_destroy)
        if data:
            self.update(data)

    def update(self, data):
        """Continue hashing of a message by consuming the next chunk of data.

        Repeated calls are equivalent to a single call with the concatenation
        of all the arguments. In other words:

           >>> m.update(a); m.update(b)

        is equivalent to:

           >>> m.update(a+b)

        :Parameters:
          data : byte string/byte array/memoryview
            The next chunk of the message being hashed.
        """

        result = _raw_md4_lib.md4_update(self._state.get(),
                                         c_uint8_ptr(data),
                                         c_size_t(len(data)))
        if result:
            raise ValueError("Error %d while instantiating MD4"
                             % result)

    def digest(self):
        """Return the **binary** (non-printable) digest of the message that
        has been hashed so far.

        This method does not change the state of the hash object.
        You can continue updating the object after calling this function.

        :Return: A byte string of `digest_size` bytes. It may contain non-ASCII
         characters, including null bytes.
        """

        bfr = create_string_buffer(self.digest_size)
        result = _raw_md4_lib.md4_digest(self._state.get(),
                                         bfr)
        if result:
            raise ValueError("Error %d while instantiating MD4"
                             % result)

        return get_raw_buffer(bfr)

    def hexdigest(self):
        """Return the **printable** digest of the message that has been
        hashed so far.

        This method does not change the state of the hash object.

        :Return: A string of 2* `digest_size` characters. It contains only
         hexadecimal ASCII digits.
        """

        return "".join(["%02x" % bord(x) for x in self.digest()])

    def copy(self):
        """Return a copy ("clone") of the hash object.

        The copy will have the same internal state as the original hash
        object.
        This can be used to efficiently compute the digests of strings that
        share a common initial substring.

        :Return: A hash object of the same type
        """

        clone = MD4Hash()
        result = _raw_md4_lib.md4_copy(self._state.get(),
                                       clone._state.get())
        if result:
            raise ValueError("Error %d while copying MD4" % result)
        return clone

    def new(self, data=None):
        return MD4Hash(data)


def new(data=None):
    """Return a fresh instance of the hash object.

    :Parameters:
       data : byte string/byte array/memoryview
        The very first chunk of the message to hash.
        It is equivalent to an early call to `MD4Hash.update()`.
        Optional.

    :Return: A `MD4Hash` object
    """
    return MD4Hash().new(data)

#: The size of the resulting hash in bytes.
digest_size = MD4Hash.digest_size

#: The internal block size of the hash algorithm in bytes.
block_size = MD4Hash.block_size
