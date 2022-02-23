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

from Crypto.Util.py3compat import bchr, bord, iter_range
import Crypto.Util.number
from Crypto.Util.number import (ceil_div,
                                long_to_bytes,
                                bytes_to_long
                                )
from Crypto.Util.strxor import strxor
from Crypto import Random


class PSS_SigScheme:
    """A signature object for ``RSASSA-PSS``.
    Do not instantiate directly.
    Use :func:`Crypto.Signature.pss.new`.
    """

    def __init__(self, key, mgfunc, saltLen, randfunc):
        """Initialize this PKCS#1 PSS signature scheme object.

        :Parameters:
          key : an RSA key object
            If a private half is given, both signature and
            verification are possible.
            If a public half is given, only verification is possible.
          mgfunc : callable
            A mask generation function that accepts two parameters:
            a string to use as seed, and the lenth of the mask to
            generate, in bytes.
          saltLen : integer
            Length of the salt, in bytes.
          randfunc : callable
            A function that returns random bytes.
        """

        self._key = key
        self._saltLen = saltLen
        self._mgfunc = mgfunc
        self._randfunc = randfunc

    def can_sign(self):
        """Return ``True`` if this object can be used to sign messages."""
        return self._key.has_private()

    def sign(self, msg_hash):
        """Create the PKCS#1 PSS signature of a message.

        This function is also called ``RSASSA-PSS-SIGN`` and
        it is specified in
        `section 8.1.1 of RFC8017 <https://tools.ietf.org/html/rfc8017#section-8.1.1>`_.

        :parameter msg_hash:
            This is an object from the :mod:`Crypto.Hash` package.
            It has been used to digest the message to sign.
        :type msg_hash: hash object

        :return: the signature encoded as a *byte string*.
        :raise ValueError: if the RSA key is not long enough for the given hash algorithm.
        :raise TypeError: if the RSA key has no private half.
        """

        # Set defaults for salt length and mask generation function
        if self._saltLen is None:
            sLen = msg_hash.digest_size
        else:
            sLen = self._saltLen

        if self._mgfunc is None:
            mgf = lambda x, y: MGF1(x, y, msg_hash)
        else:
            mgf = self._mgfunc

        modBits = Crypto.Util.number.size(self._key.n)

        # See 8.1.1 in RFC3447
        k = ceil_div(modBits, 8)  # k is length in bytes of the modulus
        # Step 1
        em = _EMSA_PSS_ENCODE(msg_hash, modBits-1, self._randfunc, mgf, sLen)
        # Step 2a (OS2IP)
        em_int = bytes_to_long(em)
        # Step 2b (RSASP1)
        m_int = self._key._decrypt(em_int)
        # Step 2c (I2OSP)
        signature = long_to_bytes(m_int, k)
        return signature

    def verify(self, msg_hash, signature):
        """Check if the  PKCS#1 PSS signature over a message is valid.

        This function is also called ``RSASSA-PSS-VERIFY`` and
        it is specified in
        `section 8.1.2 of RFC8037 <https://tools.ietf.org/html/rfc8017#section-8.1.2>`_.

        :parameter msg_hash:
            The hash that was carried out over the message. This is an object
            belonging to the :mod:`Crypto.Hash` module.
        :type parameter: hash object

        :parameter signature:
            The signature that needs to be validated.
        :type signature: bytes

        :raise ValueError: if the signature is not valid.
        """

        # Set defaults for salt length and mask generation function
        if self._saltLen is None:
            sLen = msg_hash.digest_size
        else:
            sLen = self._saltLen
        if self._mgfunc:
            mgf = self._mgfunc
        else:
            mgf = lambda x, y: MGF1(x, y, msg_hash)

        modBits = Crypto.Util.number.size(self._key.n)

        # See 8.1.2 in RFC3447
        k = ceil_div(modBits, 8)  # Convert from bits to bytes
        # Step 1
        if len(signature) != k:
            raise ValueError("Incorrect signature")
        # Step 2a (O2SIP)
        signature_int = bytes_to_long(signature)
        # Step 2b (RSAVP1)
        em_int = self._key._encrypt(signature_int)
        # Step 2c (I2OSP)
        emLen = ceil_div(modBits - 1, 8)
        em = long_to_bytes(em_int, emLen)
        # Step 3/4
        _EMSA_PSS_VERIFY(msg_hash, em, modBits-1, mgf, sLen)


def MGF1(mgfSeed, maskLen, hash_gen):
    """Mask Generation Function, described in `B.2.1 of RFC8017
    <https://tools.ietf.org/html/rfc8017>`_.

    :param mfgSeed:
        seed from which the mask is generated
    :type mfgSeed: byte string

    :param maskLen:
        intended length in bytes of the mask
    :type maskLen: integer

    :param hash_gen:
        A module or a hash object from :mod:`Crypto.Hash`
    :type hash_object:

    :return: the mask, as a *byte string*
    """
    
    T = b""
    for counter in iter_range(ceil_div(maskLen, hash_gen.digest_size)):
        c = long_to_bytes(counter, 4)
        hobj = hash_gen.new()
        hobj.update(mgfSeed + c)
        T = T + hobj.digest()
    assert(len(T) >= maskLen)
    return T[:maskLen]


def _EMSA_PSS_ENCODE(mhash, emBits, randFunc, mgf, sLen):
    r"""
    Implement the ``EMSA-PSS-ENCODE`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.1.1).

    The original ``EMSA-PSS-ENCODE`` actually accepts the message ``M``
    as input, and hash it internally. Here, we expect that the message
    has already been hashed instead.

    :Parameters:
      mhash : hash object
        The hash object that holds the digest of the message being signed.
      emBits : int
        Maximum length of the final encoding, in bits.
      randFunc : callable
        An RNG function that accepts as only parameter an int, and returns
        a string of random bytes, to be used as salt.
      mgf : callable
        A mask generation function that accepts two parameters: a string to
        use as seed, and the lenth of the mask to generate, in bytes.
      sLen : int
        Length of the salt, in bytes.

    :Return: An ``emLen`` byte long string that encodes the hash
      (with ``emLen = \ceil(emBits/8)``).

    :Raise ValueError:
        When digest or salt length are too big.
    """

    emLen = ceil_div(emBits, 8)

    # Bitmask of digits that fill up
    lmask = 0
    for i in iter_range(8*emLen-emBits):
        lmask = lmask >> 1 | 0x80

    # Step 1 and 2 have been already done
    # Step 3
    if emLen < mhash.digest_size+sLen+2:
        raise ValueError("Digest or salt length are too long"
                         " for given key size.")
    # Step 4
    salt = randFunc(sLen)
    # Step 5
    m_prime = bchr(0)*8 + mhash.digest() + salt
    # Step 6
    h = mhash.new()
    h.update(m_prime)
    # Step 7
    ps = bchr(0)*(emLen-sLen-mhash.digest_size-2)
    # Step 8
    db = ps + bchr(1) + salt
    # Step 9
    dbMask = mgf(h.digest(), emLen-mhash.digest_size-1)
    # Step 10
    maskedDB = strxor(db, dbMask)
    # Step 11
    maskedDB = bchr(bord(maskedDB[0]) & ~lmask) + maskedDB[1:]
    # Step 12
    em = maskedDB + h.digest() + bchr(0xBC)
    return em


def _EMSA_PSS_VERIFY(mhash, em, emBits, mgf, sLen):
    """
    Implement the ``EMSA-PSS-VERIFY`` function, as defined
    in PKCS#1 v2.1 (RFC3447, 9.1.2).

    ``EMSA-PSS-VERIFY`` actually accepts the message ``M`` as input,
    and hash it internally. Here, we expect that the message has already
    been hashed instead.

    :Parameters:
      mhash : hash object
        The hash object that holds the digest of the message to be verified.
      em : string
        The signature to verify, therefore proving that the sender really
        signed the message that was received.
      emBits : int
        Length of the final encoding (em), in bits.
      mgf : callable
        A mask generation function that accepts two parameters: a string to
        use as seed, and the lenth of the mask to generate, in bytes.
      sLen : int
        Length of the salt, in bytes.

    :Raise ValueError:
        When the encoding is inconsistent, or the digest or salt lengths
        are too big.
    """

    emLen = ceil_div(emBits, 8)

    # Bitmask of digits that fill up
    lmask = 0
    for i in iter_range(8*emLen-emBits):
        lmask = lmask >> 1 | 0x80

    # Step 1 and 2 have been already done
    # Step 3
    if emLen < mhash.digest_size+sLen+2:
        raise ValueError("Incorrect signature")
    # Step 4
    if ord(em[-1:]) != 0xBC:
        raise ValueError("Incorrect signature")
    # Step 5
    maskedDB = em[:emLen-mhash.digest_size-1]
    h = em[emLen-mhash.digest_size-1:-1]
    # Step 6
    if lmask & bord(em[0]):
        raise ValueError("Incorrect signature")
    # Step 7
    dbMask = mgf(h, emLen-mhash.digest_size-1)
    # Step 8
    db = strxor(maskedDB, dbMask)
    # Step 9
    db = bchr(bord(db[0]) & ~lmask) + db[1:]
    # Step 10
    if not db.startswith(bchr(0)*(emLen-mhash.digest_size-sLen-2) + bchr(1)):
        raise ValueError("Incorrect signature")
    # Step 11
    if sLen > 0:
        salt = db[-sLen:]
    else:
        salt = b""
    # Step 12
    m_prime = bchr(0)*8 + mhash.digest() + salt
    # Step 13
    hobj = mhash.new()
    hobj.update(m_prime)
    hp = hobj.digest()
    # Step 14
    if h != hp:
        raise ValueError("Incorrect signature")


def new(rsa_key, **kwargs):
    """Create an object for making or verifying PKCS#1 PSS signatures.

    :parameter rsa_key:
      The RSA key to use for signing or verifying the message.
      This is a :class:`Crypto.PublicKey.RSA` object.
      Signing is only possible when ``rsa_key`` is a **private** RSA key.
    :type rsa_key: RSA object

    :Keyword Arguments:

        *   *mask_func* (``callable``) --
            A function that returns the mask (as `bytes`).
            It must accept two parameters: a seed (as `bytes`)
            and the length of the data to return.

            If not specified, it will be the function :func:`MGF1` defined in
            `RFC8017 <https://tools.ietf.org/html/rfc8017#page-67>`_ and
            combined with the same hash algorithm applied to the
            message to sign or verify.

            If you want to use a different function, for instance still :func:`MGF1`
            but together with another hash, you can do::

                from Crypto.Hash import SHA256
                from Crypto.Signature.pss import MGF1
                mgf = lambda x, y: MGF1(x, y, SHA256)

        *   *salt_bytes* (``integer``) --
            Length of the salt, in bytes.
            It is a value between 0 and ``emLen - hLen - 2``, where ``emLen``
            is the size of the RSA modulus and ``hLen`` is the size of the digest
            applied to the message to sign or verify.

            The salt is generated internally, you don't need to provide it.

            If not specified, the salt length will be ``hLen``.
            If it is zero, the signature scheme becomes deterministic.

            Note that in some implementations such as OpenSSL the default
            salt length is ``emLen - hLen - 2`` (even though it is not more
            secure than ``hLen``).

        *   *rand_func* (``callable``) --
            A function that returns random ``bytes``, of the desired length.
            The default is :func:`Crypto.Random.get_random_bytes`.

    :return: a :class:`PSS_SigScheme` signature object
    """

    mask_func = kwargs.pop("mask_func", None)
    salt_len = kwargs.pop("salt_bytes", None)
    rand_func = kwargs.pop("rand_func", None)
    if rand_func is None:
        rand_func = Random.get_random_bytes
    if kwargs:
        raise ValueError("Unknown keywords: " + str(kwargs.keys()))
    return PSS_SigScheme(rsa_key, mask_func, salt_len, rand_func)
