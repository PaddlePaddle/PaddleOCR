# ===================================================================
#
# Copyright (c) 2019, Legrandin <helderijs@gmail.com>
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

import sys

from Crypto.Cipher import _create_cipher
from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  VoidPointer, SmartPointer, c_size_t,
                                  c_uint8_ptr, c_uint)

_raw_blowfish_lib = load_pycryptodome_raw_lib(
        "Crypto.Cipher._raw_eksblowfish",
        """
        int EKSBlowfish_start_operation(const uint8_t key[],
                                        size_t key_len,
                                        const uint8_t salt[16],
                                        size_t salt_len,
                                        unsigned cost,
                                        unsigned invert,
                                        void **pResult);
        int EKSBlowfish_encrypt(const void *state,
                                const uint8_t *in,
                                uint8_t *out,
                                size_t data_len);
        int EKSBlowfish_decrypt(const void *state,
                                const uint8_t *in,
                                uint8_t *out,
                                size_t data_len);
        int EKSBlowfish_stop_operation(void *state);
        """
        )


def _create_base_cipher(dict_parameters):
    """This method instantiates and returns a smart pointer to
    a low-level base cipher. It will absorb named parameters in
    the process."""

    try:
        key = dict_parameters.pop("key")
        salt = dict_parameters.pop("salt")
        cost = dict_parameters.pop("cost")
    except KeyError as e:
        raise TypeError("Missing EKSBlowfish parameter: " + str(e))
    invert = dict_parameters.pop("invert", True)

    if len(key) not in key_size:
        raise ValueError("Incorrect EKSBlowfish key length (%d bytes)" % len(key))

    start_operation = _raw_blowfish_lib.EKSBlowfish_start_operation
    stop_operation = _raw_blowfish_lib.EKSBlowfish_stop_operation

    void_p = VoidPointer()
    result = start_operation(c_uint8_ptr(key),
                             c_size_t(len(key)),
                             c_uint8_ptr(salt),
                             c_size_t(len(salt)),
                             c_uint(cost),
                             c_uint(int(invert)),
                             void_p.address_of())
    if result:
        raise ValueError("Error %X while instantiating the EKSBlowfish cipher"
                         % result)
    return SmartPointer(void_p.get(), stop_operation)


def new(key, mode, salt, cost, invert):
    """Create a new EKSBlowfish cipher
    
    Args:

      key (bytes, bytearray, memoryview):
        The secret key to use in the symmetric cipher.
        Its length can vary from 0 to 72 bytes.

      mode (one of the supported ``MODE_*`` constants):
        The chaining mode to use for encryption or decryption.

      salt (bytes, bytearray, memoryview):
        The salt that bcrypt uses to thwart rainbow table attacks

      cost (integer):
        The complexity factor in bcrypt

      invert (bool):
        If ``False``, in the inner loop use ``ExpandKey`` first over the salt
        and then over the key, as defined in
        the `original bcrypt specification <https://www.usenix.org/legacy/events/usenix99/provos/provos_html/node4.html>`_.
        If ``True``, reverse the order, as in the first implementation of
        `bcrypt` in OpenBSD.

    :Return: an EKSBlowfish object
    """

    kwargs = { 'salt':salt, 'cost':cost, 'invert':invert }
    return _create_cipher(sys.modules[__name__], key, mode, **kwargs)


MODE_ECB = 1

# Size of a data block (in bytes)
block_size = 8
# Size of a key (in bytes)
key_size = range(0, 72 + 1)
