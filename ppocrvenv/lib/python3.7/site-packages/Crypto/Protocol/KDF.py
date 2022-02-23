# coding=utf-8
#
#  KDF.py : a collection of Key Derivation Functions
#
# Part of the Python Cryptography Toolkit
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

import re
import struct
from functools import reduce

from Crypto.Util.py3compat import (tobytes, bord, _copy_bytes, iter_range,
                                  tostr, bchr, bstr)

from Crypto.Hash import SHA1, SHA256, HMAC, CMAC, BLAKE2s
from Crypto.Util.strxor import strxor
from Crypto.Random import get_random_bytes
from Crypto.Util.number import size as bit_size, long_to_bytes, bytes_to_long

from Crypto.Util._raw_api import (load_pycryptodome_raw_lib,
                                  create_string_buffer,
                                  get_raw_buffer, c_size_t)

_raw_salsa20_lib = load_pycryptodome_raw_lib("Crypto.Cipher._Salsa20",
                    """
                    int Salsa20_8_core(const uint8_t *x, const uint8_t *y,
                                       uint8_t *out);
                    """)

_raw_scrypt_lib = load_pycryptodome_raw_lib("Crypto.Protocol._scrypt",
                    """
                    typedef int (core_t)(const uint8_t [64], const uint8_t [64], uint8_t [64]);
                    int scryptROMix(const uint8_t *data_in, uint8_t *data_out,
                           size_t data_len, unsigned N, core_t *core);
                    """)


def PBKDF1(password, salt, dkLen, count=1000, hashAlgo=None):
    """Derive one key from a password (or passphrase).

    This function performs key derivation according to an old version of
    the PKCS#5 standard (v1.5) or `RFC2898
    <https://www.ietf.org/rfc/rfc2898.txt>`_.

    Args:
     password (string):
        The secret password to generate the key from.
     salt (byte string):
        An 8 byte string to use for better protection from dictionary attacks.
        This value does not need to be kept secret, but it should be randomly
        chosen for each derivation.
     dkLen (integer):
        The length of the desired key. The default is 16 bytes, suitable for
        instance for :mod:`Crypto.Cipher.AES`.
     count (integer):
        The number of iterations to carry out. The recommendation is 1000 or
        more.
     hashAlgo (module):
        The hash algorithm to use, as a module or an object from the :mod:`Crypto.Hash` package.
        The digest length must be no shorter than ``dkLen``.
        The default algorithm is :mod:`Crypto.Hash.SHA1`.

    Return:
        A byte string of length ``dkLen`` that can be used as key.
    """

    if not hashAlgo:
        hashAlgo = SHA1
    password = tobytes(password)
    pHash = hashAlgo.new(password+salt)
    digest = pHash.digest_size
    if dkLen > digest:
        raise TypeError("Selected hash algorithm has a too short digest (%d bytes)." % digest)
    if len(salt) != 8:
        raise ValueError("Salt is not 8 bytes long (%d bytes instead)." % len(salt))
    for i in iter_range(count-1):
        pHash = pHash.new(pHash.digest())
    return pHash.digest()[:dkLen]


def PBKDF2(password, salt, dkLen=16, count=1000, prf=None, hmac_hash_module=None):
    """Derive one or more keys from a password (or passphrase).

    This function performs key derivation according to the PKCS#5 standard (v2.0).

    Args:
     password (string or byte string):
        The secret password to generate the key from.
     salt (string or byte string):
        A (byte) string to use for better protection from dictionary attacks.
        This value does not need to be kept secret, but it should be randomly
        chosen for each derivation. It is recommended to use at least 16 bytes.
     dkLen (integer):
        The cumulative length of the keys to produce.

        Due to a flaw in the PBKDF2 design, you should not request more bytes
        than the ``prf`` can output. For instance, ``dkLen`` should not exceed
        20 bytes in combination with ``HMAC-SHA1``.
     count (integer):
        The number of iterations to carry out. The higher the value, the slower
        and the more secure the function becomes.

        You should find the maximum number of iterations that keeps the
        key derivation still acceptable on the slowest hardware you must support.

        Although the default value is 1000, **it is recommended to use at least
        1000000 (1 million) iterations**.
     prf (callable):
        A pseudorandom function. It must be a function that returns a
        pseudorandom byte string from two parameters: a secret and a salt.
        The slower the algorithm, the more secure the derivation function.
        If not specified, **HMAC-SHA1** is used.
     hmac_hash_module (module):
        A module from ``Crypto.Hash`` implementing a Merkle-Damgard cryptographic
        hash, which PBKDF2 must use in combination with HMAC.
        This parameter is mutually exclusive with ``prf``.

    Return:
        A byte string of length ``dkLen`` that can be used as key material.
        If you want multiple keys, just break up this string into segments of the desired length.
    """

    password = tobytes(password)
    salt = tobytes(salt)

    if prf and hmac_hash_module:
        raise ValueError("'prf' and 'hmac_hash_module' are mutually exlusive")

    if prf is None and hmac_hash_module is None:
        hmac_hash_module = SHA1

    if prf or not hasattr(hmac_hash_module, "_pbkdf2_hmac_assist"):
        # Generic (and slow) implementation

        if prf is None:
            prf = lambda p,s: HMAC.new(p, s, hmac_hash_module).digest()

        def link(s):
            s[0], s[1] = s[1], prf(password, s[1])
            return s[0]

        key = b''
        i = 1
        while len(key) < dkLen:
            s = [ prf(password, salt + struct.pack(">I", i)) ] * 2
            key += reduce(strxor, (link(s) for j in range(count)) )
            i += 1

    else:
        # Optimized implementation
        key = b''
        i = 1
        while len(key)<dkLen:
            base = HMAC.new(password, b"", hmac_hash_module)
            first_digest = base.copy().update(salt + struct.pack(">I", i)).digest()
            key += base._pbkdf2_hmac_assist(first_digest, count)
            i += 1

    return key[:dkLen]


class _S2V(object):
    """String-to-vector PRF as defined in `RFC5297`_.

    This class implements a pseudorandom function family
    based on CMAC that takes as input a vector of strings.

    .. _RFC5297: http://tools.ietf.org/html/rfc5297
    """

    def __init__(self, key, ciphermod, cipher_params=None):
        """Initialize the S2V PRF.

        :Parameters:
          key : byte string
            A secret that can be used as key for CMACs
            based on ciphers from ``ciphermod``.
          ciphermod : module
            A block cipher module from `Crypto.Cipher`.
          cipher_params : dictionary
            A set of extra parameters to use to create a cipher instance.
        """

        self._key = _copy_bytes(None, None, key)
        self._ciphermod = ciphermod
        self._last_string = self._cache = b'\x00' * ciphermod.block_size
        
        # Max number of update() call we can process
        self._n_updates = ciphermod.block_size * 8 - 1
        
        if cipher_params is None:
            self._cipher_params = {}
        else:
            self._cipher_params = dict(cipher_params)

    @staticmethod
    def new(key, ciphermod):
        """Create a new S2V PRF.

        :Parameters:
          key : byte string
            A secret that can be used as key for CMACs
            based on ciphers from ``ciphermod``.
          ciphermod : module
            A block cipher module from `Crypto.Cipher`.
        """
        return _S2V(key, ciphermod)

    def _double(self, bs):
        doubled = bytes_to_long(bs)<<1
        if bord(bs[0]) & 0x80:
            doubled ^= 0x87
        return long_to_bytes(doubled, len(bs))[-len(bs):]

    def update(self, item):
        """Pass the next component of the vector.

        The maximum number of components you can pass is equal to the block
        length of the cipher (in bits) minus 1.

        :Parameters:
          item : byte string
            The next component of the vector.
        :Raise TypeError: when the limit on the number of components has been reached.
        """

        if self._n_updates == 0:
            raise TypeError("Too many components passed to S2V")
        self._n_updates -= 1

        mac = CMAC.new(self._key,
                       msg=self._last_string,
                       ciphermod=self._ciphermod,
                       cipher_params=self._cipher_params)
        self._cache = strxor(self._double(self._cache), mac.digest())
        self._last_string = _copy_bytes(None, None, item)

    def derive(self):
        """"Derive a secret from the vector of components.

        :Return: a byte string, as long as the block length of the cipher.
        """

        if len(self._last_string) >= 16:
            # xorend
            final = self._last_string[:-16] + strxor(self._last_string[-16:], self._cache)
        else:
            # zero-pad & xor
            padded = (self._last_string + b'\x80' + b'\x00' * 15)[:16]
            final = strxor(padded, self._double(self._cache))
        mac = CMAC.new(self._key,
                       msg=final,
                       ciphermod=self._ciphermod,
                       cipher_params=self._cipher_params)
        return mac.digest()


def HKDF(master, key_len, salt, hashmod, num_keys=1, context=None):
    """Derive one or more keys from a master secret using
    the HMAC-based KDF defined in RFC5869_.

    Args:
     master (byte string):
        The unguessable value used by the KDF to generate the other keys.
        It must be a high-entropy secret, though not necessarily uniform.
        It must not be a password.
     salt (byte string):
        A non-secret, reusable value that strengthens the randomness
        extraction step.
        Ideally, it is as long as the digest size of the chosen hash.
        If empty, a string of zeroes in used.
     key_len (integer):
        The length in bytes of every derived key.
     hashmod (module):
        A cryptographic hash algorithm from :mod:`Crypto.Hash`.
        :mod:`Crypto.Hash.SHA512` is a good choice.
     num_keys (integer):
        The number of keys to derive. Every key is :data:`key_len` bytes long.
        The maximum cumulative length of all keys is
        255 times the digest size.
     context (byte string):
        Optional identifier describing what the keys are used for.

    Return:
        A byte string or a tuple of byte strings.

    .. _RFC5869: http://tools.ietf.org/html/rfc5869
    """

    output_len = key_len * num_keys
    if output_len > (255 * hashmod.digest_size):
        raise ValueError("Too much secret data to derive")
    if not salt:
        salt = b'\x00' * hashmod.digest_size
    if context is None:
        context = b""

    # Step 1: extract
    hmac = HMAC.new(salt, master, digestmod=hashmod)
    prk = hmac.digest()

    # Step 2: expand
    t = [ b"" ]
    n = 1
    tlen = 0
    while tlen < output_len:
        hmac = HMAC.new(prk, t[-1] + context + struct.pack('B', n), digestmod=hashmod)
        t.append(hmac.digest())
        tlen += hashmod.digest_size
        n += 1
    derived_output = b"".join(t)
    if num_keys == 1:
        return derived_output[:key_len]
    kol = [derived_output[idx:idx + key_len]
           for idx in iter_range(0, output_len, key_len)]
    return list(kol[:num_keys])



def scrypt(password, salt, key_len, N, r, p, num_keys=1):
    """Derive one or more keys from a passphrase.

    Args:
     password (string):
        The secret pass phrase to generate the keys from.
     salt (string):
        A string to use for better protection from dictionary attacks.
        This value does not need to be kept secret,
        but it should be randomly chosen for each derivation.
        It is recommended to be at least 16 bytes long.
     key_len (integer):
        The length in bytes of every derived key.
     N (integer):
        CPU/Memory cost parameter. It must be a power of 2 and less
        than :math:`2^{32}`.
     r (integer):
        Block size parameter.
     p (integer):
        Parallelization parameter.
        It must be no greater than :math:`(2^{32}-1)/(4r)`.
     num_keys (integer):
        The number of keys to derive. Every key is :data:`key_len` bytes long.
        By default, only 1 key is generated.
        The maximum cumulative length of all keys is :math:`(2^{32}-1)*32`
        (that is, 128TB).

    A good choice of parameters *(N, r , p)* was suggested
    by Colin Percival in his `presentation in 2009`__:

    - *( 2¹⁴, 8, 1 )* for interactive logins (≤100ms)
    - *( 2²⁰, 8, 1 )* for file encryption (≤5s)

    Return:
        A byte string or a tuple of byte strings.

    .. __: http://www.tarsnap.com/scrypt/scrypt-slides.pdf
    """

    if 2 ** (bit_size(N) - 1) != N:
        raise ValueError("N must be a power of 2")
    if N >= 2 ** 32:
        raise ValueError("N is too big")
    if p > ((2 ** 32 - 1) * 32)  // (128 * r):
        raise ValueError("p or r are too big")

    prf_hmac_sha256 = lambda p, s: HMAC.new(p, s, SHA256).digest()

    stage_1 = PBKDF2(password, salt, p * 128 * r, 1, prf=prf_hmac_sha256)

    scryptROMix = _raw_scrypt_lib.scryptROMix
    core = _raw_salsa20_lib.Salsa20_8_core

    # Parallelize into p flows
    data_out = []
    for flow in iter_range(p):
        idx = flow * 128 * r
        buffer_out = create_string_buffer(128 * r)
        result = scryptROMix(stage_1[idx : idx + 128 * r],
                             buffer_out,
                             c_size_t(128 * r),
                             N,
                             core)
        if result:
            raise ValueError("Error %X while running scrypt" % result)
        data_out += [ get_raw_buffer(buffer_out) ]

    dk = PBKDF2(password,
                b"".join(data_out),
                key_len * num_keys, 1,
                prf=prf_hmac_sha256)

    if num_keys == 1:
        return dk

    kol = [dk[idx:idx + key_len]
           for idx in iter_range(0, key_len * num_keys, key_len)]
    return kol


def _bcrypt_encode(data):
    s = "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    bits = []
    for c in data:
        bits_c = bin(bord(c))[2:].zfill(8)
        bits.append(bstr(bits_c))
    bits = b"".join(bits)

    bits6 = [ bits[idx:idx+6] for idx in range(0, len(bits), 6) ]

    result = []
    for g in bits6[:-1]:
        idx = int(g, 2)
        result.append(s[idx])

    g = bits6[-1]
    idx = int(g, 2) << (6 - len(g))
    result.append(s[idx])
    result = "".join(result)

    return tobytes(result)


def _bcrypt_decode(data):
    s = "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    bits = []
    for c in tostr(data):
        idx = s.find(c)
        bits6 = bin(idx)[2:].zfill(6)
        bits.append(bits6)
    bits = "".join(bits)

    modulo4 = len(data) % 4
    if modulo4 == 1:
        raise ValueError("Incorrect length")
    elif modulo4 == 2:
        bits = bits[:-4]
    elif modulo4 == 3:
        bits = bits[:-2]

    bits8 = [ bits[idx:idx+8] for idx in range(0, len(bits), 8) ]

    result = []
    for g in bits8:
        result.append(bchr(int(g, 2)))
    result = b"".join(result)

    return result


def _bcrypt_hash(password, cost, salt, constant, invert):
    from Crypto.Cipher import _EKSBlowfish

    if len(password) > 72:
        raise ValueError("The password is too long. It must be 72 bytes at most.")

    if not (4 <= cost <= 31):
        raise ValueError("bcrypt cost factor must be in the range 4..31")

    cipher = _EKSBlowfish.new(password, _EKSBlowfish.MODE_ECB, salt, cost, invert)
    ctext = constant
    for _ in range(64):
        ctext = cipher.encrypt(ctext)
    return ctext


def bcrypt(password, cost, salt=None):
    """Hash a password into a key, using the OpenBSD bcrypt protocol.

    Args:
      password (byte string or string):
        The secret password or pass phrase.
        It must be at most 72 bytes long.
        It must not contain the zero byte.
        Unicode strings will be encoded as UTF-8.
      cost (integer):
        The exponential factor that makes it slower to compute the hash.
        It must be in the range 4 to 31.
        A value of at least 12 is recommended.
      salt (byte string):
        Optional. Random byte string to thwarts dictionary and rainbow table
        attacks. It must be 16 bytes long.
        If not passed, a random value is generated.

    Return (byte string):
        The bcrypt hash

    Raises:
        ValueError: if password is longer than 72 bytes or if it contains the zero byte

   """

    password = tobytes(password, "utf-8")

    if password.find(bchr(0)[0]) != -1:
        raise ValueError("The password contains the zero byte")

    if len(password) < 72:
        password += b"\x00"

    if salt is None:
        salt = get_random_bytes(16)
    if len(salt) != 16:
        raise ValueError("bcrypt salt must be 16 bytes long")

    ctext = _bcrypt_hash(password, cost, salt, b"OrpheanBeholderScryDoubt", True)

    cost_enc = b"$" + bstr(str(cost).zfill(2))
    salt_enc = b"$" + _bcrypt_encode(salt)
    hash_enc = _bcrypt_encode(ctext[:-1])     # only use 23 bytes, not 24
    return b"$2a" + cost_enc + salt_enc + hash_enc


def bcrypt_check(password, bcrypt_hash):
    """Verify if the provided password matches the given bcrypt hash.

    Args:
      password (byte string or string):
        The secret password or pass phrase to test.
        It must be at most 72 bytes long.
        It must not contain the zero byte.
        Unicode strings will be encoded as UTF-8.
      bcrypt_hash (byte string, bytearray):
        The reference bcrypt hash the password needs to be checked against.

    Raises:
        ValueError: if the password does not match
    """

    bcrypt_hash = tobytes(bcrypt_hash)

    if len(bcrypt_hash) != 60:
        raise ValueError("Incorrect length of the bcrypt hash: %d bytes instead of 60" % len(bcrypt_hash))

    if bcrypt_hash[:4] != b'$2a$':
        raise ValueError("Unsupported prefix")

    p = re.compile(br'\$2a\$([0-9][0-9])\$([A-Za-z0-9./]{22,22})([A-Za-z0-9./]{31,31})')
    r = p.match(bcrypt_hash)
    if not r:
        raise ValueError("Incorrect bcrypt hash format")

    cost = int(r.group(1))
    if not (4 <= cost <= 31):
        raise ValueError("Incorrect cost")

    salt = _bcrypt_decode(r.group(2))

    bcrypt_hash2  = bcrypt(password, cost, salt)

    secret = get_random_bytes(16)

    mac1 = BLAKE2s.new(digest_bits=160, key=secret, data=bcrypt_hash).digest()
    mac2 = BLAKE2s.new(digest_bits=160, key=secret, data=bcrypt_hash2).digest()
    if mac1 != mac2:
        raise ValueError("Incorrect bcrypt hash")
