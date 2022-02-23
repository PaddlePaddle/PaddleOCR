# -*- coding: utf-8 -*-
#
#  Cipher/PKCS1-v1_5.py : PKCS#1 v1.5
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

__all__ = ['new', 'PKCS115_Cipher']

from Crypto import Random
from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.Util.py3compat import bord, is_bytes, _copy_bytes

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib, c_size_t,
                                  c_uint8_ptr)


_raw_pkcs1_decode = load_pycryptodome_raw_lib("Crypto.Cipher._pkcs1_decode",
                        """
                        int pkcs1_decode(const uint8_t *em, size_t len_em,
                                         const uint8_t *sentinel, size_t len_sentinel,
                                         size_t expected_pt_len,
                                         uint8_t *output);
                        """)


def _pkcs1_decode(em, sentinel, expected_pt_len, output):
    if len(em) != len(output):
        raise ValueError("Incorrect output length")

    ret = _raw_pkcs1_decode.pkcs1_decode(c_uint8_ptr(em),
                                         c_size_t(len(em)),
                                         c_uint8_ptr(sentinel),
                                         c_size_t(len(sentinel)),
                                         c_size_t(expected_pt_len),
                                         c_uint8_ptr(output))
    return ret


class PKCS115_Cipher:
    """This cipher can perform PKCS#1 v1.5 RSA encryption or decryption.
    Do not instantiate directly. Use :func:`Crypto.Cipher.PKCS1_v1_5.new` instead."""

    def __init__(self, key, randfunc):
        """Initialize this PKCS#1 v1.5 cipher object.

        :Parameters:
         key : an RSA key object
          If a private half is given, both encryption and decryption are possible.
          If a public half is given, only encryption is possible.
         randfunc : callable
          Function that returns random bytes.
        """

        self._key = key
        self._randfunc = randfunc

    def can_encrypt(self):
        """Return True if this cipher object can be used for encryption."""
        return self._key.can_encrypt()

    def can_decrypt(self):
        """Return True if this cipher object can be used for decryption."""
        return self._key.can_decrypt()

    def encrypt(self, message):
        """Produce the PKCS#1 v1.5 encryption of a message.

        This function is named ``RSAES-PKCS1-V1_5-ENCRYPT``, and it is specified in
        `section 7.2.1 of RFC8017
        <https://tools.ietf.org/html/rfc8017#page-28>`_.

        :param message:
            The message to encrypt, also known as plaintext. It can be of
            variable length, but not longer than the RSA modulus (in bytes) minus 11.
        :type message: bytes/bytearray/memoryview

        :Returns: A byte string, the ciphertext in which the message is encrypted.
            It is as long as the RSA modulus (in bytes).

        :Raises ValueError:
            If the RSA key length is not sufficiently long to deal with the given
            message.
        """

        # See 7.2.1 in RFC8017
        k = self._key.size_in_bytes()
        mLen = len(message)

        # Step 1
        if mLen > k - 11:
            raise ValueError("Plaintext is too long.")
        # Step 2a
        ps = []
        while len(ps) != k - mLen - 3:
            new_byte = self._randfunc(1)
            if bord(new_byte[0]) == 0x00:
                continue
            ps.append(new_byte)
        ps = b"".join(ps)
        assert(len(ps) == k - mLen - 3)
        # Step 2b
        em = b'\x00\x02' + ps + b'\x00' + _copy_bytes(None, None, message)
        # Step 3a (OS2IP)
        em_int = bytes_to_long(em)
        # Step 3b (RSAEP)
        m_int = self._key._encrypt(em_int)
        # Step 3c (I2OSP)
        c = long_to_bytes(m_int, k)
        return c

    def decrypt(self, ciphertext, sentinel, expected_pt_len=0):
        r"""Decrypt a PKCS#1 v1.5 ciphertext.

        This is the function ``RSAES-PKCS1-V1_5-DECRYPT`` specified in
        `section 7.2.2 of RFC8017
        <https://tools.ietf.org/html/rfc8017#page-29>`_.

        Args:
          ciphertext (bytes/bytearray/memoryview):
            The ciphertext that contains the message to recover.
          sentinel (any type):
            The object to return whenever an error is detected.
          expected_pt_len (integer):
            The length the plaintext is known to have, or 0 if unknown.

        Returns (byte string):
            It is either the original message or the ``sentinel`` (in case of an error).

        .. warning::
            PKCS#1 v1.5 decryption is intrinsically vulnerable to timing
            attacks (see `Bleichenbacher's`__ attack).
            **Use PKCS#1 OAEP instead**.

            This implementation attempts to mitigate the risk
            with some constant-time constructs.
            However, they are not sufficient by themselves: the type of protocol you
            implement and the way you handle errors make a big difference.

            Specifically, you should make it very hard for the (malicious)
            party that submitted the ciphertext to quickly understand if decryption
            succeeded or not.

            To this end, it is recommended that your protocol only encrypts
            plaintexts of fixed length (``expected_pt_len``),
            that ``sentinel`` is a random byte string of the same length,
            and that processing continues for as long
            as possible even if ``sentinel`` is returned (i.e. in case of
            incorrect decryption).

            .. __: https://dx.doi.org/10.1007/BFb0055716
        """

        # See 7.2.2 in RFC8017
        k = self._key.size_in_bytes()

        # Step 1
        if len(ciphertext) != k:
            raise ValueError("Ciphertext with incorrect length (not %d bytes)" % k)

        # Step 2a (O2SIP)
        ct_int = bytes_to_long(ciphertext)

        # Step 2b (RSADP)
        m_int = self._key._decrypt(ct_int)

        # Complete step 2c (I2OSP)
        em = long_to_bytes(m_int, k)

        # Step 3 (not constant time when the sentinel is not a byte string)
        output = bytes(bytearray(k))
        if not is_bytes(sentinel) or len(sentinel) > k:
            size = _pkcs1_decode(em, b'', expected_pt_len, output)
            if size < 0:
                return sentinel
            else:
                return output[size:]

        # Step 3 (somewhat constant time)
        size = _pkcs1_decode(em, sentinel, expected_pt_len, output)
        return output[size:]


def new(key, randfunc=None):
    """Create a cipher for performing PKCS#1 v1.5 encryption or decryption.

    :param key:
      The key to use to encrypt or decrypt the message. This is a `Crypto.PublicKey.RSA` object.
      Decryption is only possible if *key* is a private RSA key.
    :type key: RSA key object

    :param randfunc:
      Function that return random bytes.
      The default is :func:`Crypto.Random.get_random_bytes`.
    :type randfunc: callable

    :returns: A cipher object `PKCS115_Cipher`.
    """

    if randfunc is None:
        randfunc = Random.get_random_bytes
    return PKCS115_Cipher(key, randfunc)
