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

import unittest
from binascii import hexlify, unhexlify

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.Util.py3compat import tobytes, bchr
from Crypto.Cipher import AES, DES3
from Crypto.Hash import SHAKE128, SHA256
from Crypto.Util import Counter

def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)

class CtrTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    key_192 = get_tag_random("key_192", 24)
    nonce_32 = get_tag_random("nonce_32", 4)
    nonce_64 = get_tag_random("nonce_64", 8)
    ctr_64 = Counter.new(32, prefix=nonce_32)
    ctr_128 = Counter.new(64, prefix=nonce_64)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        pt = get_tag_random("plaintext", 8 * 100)
        ct = cipher.encrypt(pt)

        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_invalid_counter_parameter(self):
        # Counter object is required for ciphers with short block size
        self.assertRaises(TypeError, DES3.new, self.key_192, AES.MODE_CTR)
        # Positional arguments are not allowed (Counter must be passed as
        # keyword)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR, self.ctr_128)

    def test_nonce_attribute(self):
        # Nonce attribute is the prefix passed to Counter (DES3)
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(cipher.nonce, self.nonce_32)

        # Nonce attribute is the prefix passed to Counter (AES)
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(cipher.nonce, self.nonce_64)

        # Nonce attribute is not defined if suffix is used in Counter
        counter = Counter.new(64, prefix=self.nonce_32, suffix=self.nonce_32)
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertFalse(hasattr(cipher, "nonce"))

    def test_nonce_parameter(self):
        # Nonce parameter becomes nonce attribute
        cipher1 = AES.new(self.key_128, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertEqual(cipher1.nonce, self.nonce_64)

        counter = Counter.new(64, prefix=self.nonce_64, initial_value=0)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        pt = get_tag_random("plaintext", 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))

        # Nonce is implicitly created (for AES) when no parameters are passed
        nonce1 = AES.new(self.key_128, AES.MODE_CTR).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_CTR).nonce
        self.assertNotEqual(nonce1, nonce2)
        self.assertEqual(len(nonce1), 8)

        # Nonce can be zero-length
        cipher = AES.new(self.key_128, AES.MODE_CTR, nonce=b"")
        self.assertEqual(b"", cipher.nonce)
        cipher.encrypt(b'0'*300)

        # Nonce and Counter are mutually exclusive
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR,
                          counter=self.ctr_128, nonce=self.nonce_64)

    def test_initial_value_parameter(self):
        # Test with nonce parameter
        cipher1 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64, initial_value=0xFFFF)
        counter = Counter.new(64, prefix=self.nonce_64, initial_value=0xFFFF)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        pt = get_tag_random("plaintext", 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))

        # Test without nonce parameter
        cipher1 = AES.new(self.key_128, AES.MODE_CTR,
                          initial_value=0xFFFF)
        counter = Counter.new(64, prefix=cipher1.nonce, initial_value=0xFFFF)
        cipher2 = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        pt = get_tag_random("plaintext", 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))

        # Initial_value and Counter are mutually exclusive
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR,
                          counter=self.ctr_128, initial_value=0)

    def test_initial_value_bytes_parameter(self):
        # Same result as when passing an integer
        cipher1 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64,
                          initial_value=b"\x00"*6+b"\xFF\xFF")
        cipher2 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64, initial_value=0xFFFF)
        pt = get_tag_random("plaintext", 65536)
        self.assertEqual(cipher1.encrypt(pt), cipher2.encrypt(pt))

        # Fail if the iv is too large
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          initial_value=b"5"*17)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64, initial_value=b"5"*9)

        # Fail if the iv is too short
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          initial_value=b"5"*15)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64, initial_value=b"5"*7)

    def test_iv_with_matching_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          counter=Counter.new(120))
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CTR,
                          counter=Counter.new(136))

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_block_size_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(cipher.block_size, DES3.block_size)

    def test_unaligned_data_128(self):
        plaintexts = [ b"7777777" ] * 100

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [ b"7777777" ] * 100
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = DES3.new(self.key_192, AES.MODE_CTR, counter=self.ctr_64)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR,
                          7, counter=self.ctr_128)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CTR,
                          counter=self.ctr_128, unknown=7)
        # But some are only known by the base cipher (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128, use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
            result = getattr(cipher, func)(b"")
            self.assertEqual(result, b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=self.ctr_128)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_wrap_around(self):
        # Counter is only 8 bits, so we can only encrypt/decrypt 256 blocks (=4096 bytes)
        counter = Counter.new(8, prefix=bchr(9) * 15)
        max_bytes = 4096

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        cipher.encrypt(b'9' * max_bytes)
        self.assertRaises(OverflowError, cipher.encrypt, b'9')

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertRaises(OverflowError, cipher.encrypt, b'9' * (max_bytes + 1))

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        cipher.decrypt(b'9' * max_bytes)
        self.assertRaises(OverflowError, cipher.decrypt, b'9')

        cipher = AES.new(self.key_128, AES.MODE_CTR, counter=counter)
        self.assertRaises(OverflowError, cipher.decrypt, b'9' * (max_bytes + 1))

    def test_bytearray(self):
        data = b"1" * 16
        iv = b"\x00" * 6 + b"\xFF\xFF"

        # Encrypt
        cipher1 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64,
                          initial_value=iv)
        ref1 = cipher1.encrypt(data)

        cipher2 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=bytearray(self.nonce_64),
                          initial_value=bytearray(iv))
        ref2 = cipher2.encrypt(bytearray(data))

        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        cipher3 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=self.nonce_64,
                          initial_value=iv)
        ref3 = cipher3.decrypt(data)

        cipher4 = AES.new(self.key_128, AES.MODE_CTR,
                          nonce=bytearray(self.nonce_64),
                          initial_value=bytearray(iv))
        ref4 = cipher4.decrypt(bytearray(data))

        self.assertEqual(ref3, ref4)

    def test_very_long_data(self):
        cipher = AES.new(b'A' * 32, AES.MODE_CTR, nonce=b'')
        ct = cipher.encrypt(b'B' * 1000000)
        digest = SHA256.new(ct).hexdigest()
        self.assertEqual(digest, "96204fc470476561a3a8f3b6fe6d24be85c87510b638142d1d0fb90989f8a6a6")

    def test_output_param(self):

        pt = b'5' * 128
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)

        output = bytearray(128)
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):

        pt = b'5' * 128
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)

        output = memoryview(bytearray(128))
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128

        pt = b'5' * LEN_PT
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        ct = cipher.encrypt(pt)

        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)

        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)

        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(b'4'*16, AES.MODE_CTR, nonce=self.nonce_64)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


class SP800TestVectors(unittest.TestCase):
    """Class exercising the CTR test vectors found in Section F.5
    of NIST SP 800-38A"""

    def test_aes_128(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '874d6191b620e3261bef6864990db6ce' +\
                        '9806f66b7970fdff8617187bb9fffdff' +\
                        '5ae4df3edbd5d35e5b4f09020db03eab' +\
                        '1e031dda2fbe03d1792170a0f3009cee'
        key =           '2b7e151628aed2a6abf7158809cf4f3c'
        counter =       Counter.new(nbits=16,
                                    prefix=unhexlify('f0f1f2f3f4f5f6f7f8f9fafbfcfd'),
                                    initial_value=0xfeff)

        key = unhexlify(key)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_192(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '1abc932417521ca24f2b0459fe7e6e0b' +\
                        '090339ec0aa6faefd5ccc2c6f4ce8e94' +\
                        '1e36b26bd1ebc670d1bd1d665620abf7' +\
                        '4f78a7f6d29809585a97daec58c6b050'
        key =           '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
        counter =       Counter.new(nbits=16,
                                    prefix=unhexlify('f0f1f2f3f4f5f6f7f8f9fafbfcfd'),
                                    initial_value=0xfeff)

        key = unhexlify(key)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_256(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '601ec313775789a5b7a7f504bbf3d228' +\
                        'f443e3ca4d62b59aca84e990cacaf5c5' +\
                        '2b0930daa23de94ce87017ba2d84988d' +\
                        'dfc9c58db67aada613c2dd08457941a6'
        key =           '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4'
        counter =       Counter.new(nbits=16,
                                    prefix=unhexlify('f0f1f2f3f4f5f6f7f8f9fafbfcfd'),
                                    initial_value=0xfeff)
        key = unhexlify(key)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CTR, counter=counter)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)


class RFC3686TestVectors(unittest.TestCase):

    # Each item is a test vector with:
    # - plaintext
    # - ciphertext
    # - key (AES 128, 192 or 256 bits)
    # - counter prefix (4 byte nonce + 8 byte nonce)
    data = (
            ('53696e676c6520626c6f636b206d7367',
             'e4095d4fb7a7b3792d6175a3261311b8',
             'ae6852f8121067cc4bf7a5765577f39e',
             '000000300000000000000000'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
             '5104a106168a72d9790d41ee8edad388eb2e1efc46da57c8fce630df9141be28',
             '7e24067817fae0d743d6ce1f32539163',
             '006cb6dbc0543b59da48d90b'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223',
             'c1cf48a89f2ffdd9cf4652e9efdb72d74540a42bde6d7836d59a5ceaaef3105325b2072f',
             '7691be035e5020a8ac6e618529f9a0dc',
             '00e0017b27777f3f4a1786f0'),
            ('53696e676c6520626c6f636b206d7367',
             '4b55384fe259c9c84e7935a003cbe928',
             '16af5b145fc9f579c175f93e3bfb0eed863d06ccfdb78515',
             '0000004836733c147d6d93cb'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
             '453243fc609b23327edfaafa7131cd9f8490701c5ad4a79cfc1fe0ff42f4fb00',
             '7c5cb2401b3dc33c19e7340819e0f69c678c3db8e6f6a91a',
             '0096b03b020c6eadc2cb500d'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223',
             '96893fc55e5c722f540b7dd1ddf7e758d288bc95c69165884536c811662f2188abee0935',
             '02bf391ee8ecb159b959617b0965279bf59b60a786d3e0fe',
             '0007bdfd5cbd60278dcc0912'),
            ('53696e676c6520626c6f636b206d7367',
             '145ad01dbf824ec7560863dc71e3e0c0',
             '776beff2851db06f4c8a0542c8696f6c6a81af1eec96b4d37fc1d689e6c1c104',
             '00000060db5672c97aa8f0b2'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
             'f05e231b3894612c49ee000b804eb2a9b8306b508f839d6a5530831d9344af1c',
             'f6d66d6bd52d59bb0796365879eff886c66dd51a5b6a99744b50590c87a23884',
             '00faac24c1585ef15a43d875'),
            ('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223',
             'eb6c52821d0bbbf7ce7594462aca4faab407df866569fd07f48cc0b583d6071f1ec0e6b8',
             'ff7a617ce69148e4f1726e2f43581de2aa62d9f805532edff1eed687fb54153d',
             '001cc5b751a51d70a1c11148')
        )

    bindata = []
    for tv in data:
        bindata.append([unhexlify(x) for x in tv])

    def runTest(self):
        for pt, ct, key, prefix in self.bindata:
            counter = Counter.new(32, prefix=prefix)
            cipher = AES.new(key, AES.MODE_CTR, counter=counter)
            result = cipher.encrypt(pt)
            self.assertEqual(hexlify(ct), hexlify(result))


def get_tests(config={}):
    tests = []
    tests += list_test_cases(CtrTests)
    tests += list_test_cases(SP800TestVectors)
    tests += [ RFC3686TestVectors() ]
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
