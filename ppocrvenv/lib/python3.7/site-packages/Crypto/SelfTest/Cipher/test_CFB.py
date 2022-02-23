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

import unittest
from binascii import unhexlify

from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases
from Crypto.Util.py3compat import tobytes, is_string
from Crypto.Cipher import AES, DES3, DES
from Crypto.Hash import SHAKE128

from Crypto.SelfTest.Cipher.test_CBC import BlockChainingTests


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class CfbTests(BlockChainingTests):

    aes_mode = AES.MODE_CFB
    des3_mode = DES3.MODE_CFB

    # Redefine test_unaligned_data_128/64

    def test_unaligned_data_128(self):
        plaintexts = [ b"7777777" ] * 100

        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [ b"7777777" ] * 100
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        ciphertexts = [ cipher.encrypt(x) for x in plaintexts ]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    # Extra

    def test_segment_size_128(self):
        for bits in range(8, 129, 8):
            cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128,
                             segment_size=bits)

        for bits in 0, 7, 9, 127, 129:
            self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CFB,
                              self.iv_128,
                              segment_size=bits)

    def test_segment_size_64(self):
        for bits in range(8, 65, 8):
            cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64,
                              segment_size=bits)

        for bits in 0, 7, 9, 63, 65:
            self.assertRaises(ValueError, DES3.new, self.key_192, AES.MODE_CFB,
                              self.iv_64,
                              segment_size=bits)


class NistCfbVectors(unittest.TestCase):

    def _do_kat_aes_test(self, file_name, segment_size):

        test_vectors = load_test_vectors(("Cipher", "AES"),
                            file_name,
                            "AES CFB%d KAT" % segment_size,
                            { "count" : lambda x: int(x) } )
        if test_vectors is None:
            return

        direction = None
        for tv in test_vectors:

            # The test vector file contains some directive lines
            if is_string(tv):
                direction = tv
                continue

            self.description = tv.desc
            cipher = AES.new(tv.key, AES.MODE_CFB, tv.iv,
                             segment_size=segment_size)
            if direction == "[ENCRYPT]":
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == "[DECRYPT]":
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False

    # See Section 6.4.5 in AESAVS
    def _do_mct_aes_test(self, file_name, segment_size):

        test_vectors = load_test_vectors(("Cipher", "AES"),
                            file_name,
                            "AES CFB%d Montecarlo" % segment_size,
                            { "count" : lambda x: int(x) } )
        if test_vectors is None:
            return

        assert(segment_size in (8, 128))

        direction = None
        for tv in test_vectors:

            # The test vector file contains some directive lines
            if is_string(tv):
                direction = tv
                continue

            self.description = tv.desc
            cipher = AES.new(tv.key, AES.MODE_CFB, tv.iv,
                             segment_size=segment_size)

            def get_input(input_text, output_seq, j):
                # CFB128
                if segment_size == 128:
                    if j >= 2:
                        return output_seq[-2]
                    return [input_text, tv.iv][j]
                # CFB8
                if j == 0:
                    return input_text
                elif j <= 16:
                    return tv.iv[j - 1:j]
                return output_seq[j - 17]

            if direction == '[ENCRYPT]':
                cts = []
                for j in range(1000):
                    plaintext = get_input(tv.plaintext, cts, j)
                    cts.append(cipher.encrypt(plaintext))
                self.assertEqual(cts[-1], tv.ciphertext)
            elif direction == '[DECRYPT]':
                pts = []
                for j in range(1000):
                    ciphertext = get_input(tv.ciphertext, pts, j)
                    pts.append(cipher.decrypt(ciphertext))
                self.assertEqual(pts[-1], tv.plaintext)
            else:
                assert False

    def _do_tdes_test(self, file_name, segment_size):

        test_vectors = load_test_vectors(("Cipher", "TDES"),
                            file_name,
                            "TDES CFB%d KAT" % segment_size,
                            { "count" : lambda x: int(x) } )
        if test_vectors is None:
            return

        direction = None
        for tv in test_vectors:

            # The test vector file contains some directive lines
            if is_string(tv):
                direction = tv
                continue

            self.description = tv.desc
            if hasattr(tv, "keys"):
                cipher = DES.new(tv.keys, DES.MODE_CFB, tv.iv,
                                 segment_size=segment_size)
            else:
                if tv.key1 != tv.key3:
                    key = tv.key1 + tv.key2 + tv.key3  # Option 3
                else:
                    key = tv.key1 + tv.key2            # Option 2
                cipher = DES3.new(key, DES3.MODE_CFB, tv.iv,
                                  segment_size=segment_size)
            if direction == "[ENCRYPT]":
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == "[DECRYPT]":
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False


# Create one test method per file
nist_aes_kat_mmt_files = (
    # KAT
    "CFB?GFSbox128.rsp",
    "CFB?GFSbox192.rsp",
    "CFB?GFSbox256.rsp",
    "CFB?KeySbox128.rsp",
    "CFB?KeySbox192.rsp",
    "CFB?KeySbox256.rsp",
    "CFB?VarKey128.rsp",
    "CFB?VarKey192.rsp",
    "CFB?VarKey256.rsp",
    "CFB?VarTxt128.rsp",
    "CFB?VarTxt192.rsp",
    "CFB?VarTxt256.rsp",
    # MMT
    "CFB?MMT128.rsp",
    "CFB?MMT192.rsp",
    "CFB?MMT256.rsp",
    )
nist_aes_mct_files = (
    "CFB?MCT128.rsp",
    "CFB?MCT192.rsp",
    "CFB?MCT256.rsp",
    )

for file_gen_name in nist_aes_kat_mmt_files:
    for bits in "8", "128":
        file_name = file_gen_name.replace("?", bits)
        def new_func(self, file_name=file_name, bits=bits):
            self._do_kat_aes_test(file_name, int(bits))
        setattr(NistCfbVectors, "test_AES_" + file_name, new_func)

for file_gen_name in nist_aes_mct_files:
    for bits in "8", "128":
        file_name = file_gen_name.replace("?", bits)
        def new_func(self, file_name=file_name, bits=bits):
            self._do_mct_aes_test(file_name, int(bits))
        setattr(NistCfbVectors, "test_AES_" + file_name, new_func)
del file_name, new_func

nist_tdes_files = (
    "TCFB?MMT2.rsp",    # 2TDES
    "TCFB?MMT3.rsp",    # 3TDES
    "TCFB?invperm.rsp", # Single DES
    "TCFB?permop.rsp",
    "TCFB?subtab.rsp",
    "TCFB?varkey.rsp",
    "TCFB?vartext.rsp",
    )

for file_gen_name in nist_tdes_files:
    for bits in "8", "64":
        file_name = file_gen_name.replace("?", bits)
        def new_func(self, file_name=file_name, bits=bits):
            self._do_tdes_test(file_name, int(bits))
    setattr(NistCfbVectors, "test_TDES_" + file_name, new_func)

# END OF NIST CBC TEST VECTORS


class SP800TestVectors(unittest.TestCase):
    """Class exercising the CFB test vectors found in Section F.3
    of NIST SP 800-3A"""

    def test_aes_128_cfb8(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172aae2d'
        ciphertext =    '3b79424c9c0dd436bace9e0ed4586a4f32b9'
        key =           '2b7e151628aed2a6abf7158809cf4f3c'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_192_cfb8(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172aae2d'
        ciphertext =    'cda2521ef0a905ca44cd057cbf0d47a0678a'
        key =           '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_256_cfb8(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172aae2d'
        ciphertext =    'dc1f1a8520a64db55fcc8ac554844e889700'
        key =           '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=8)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_128_cfb128(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '3b3fd92eb72dad20333449f8e83cfb4a' +\
                        'c8a64537a0b3a93fcde3cdad9f1ce58b' +\
                        '26751f67a3cbb140b1808cf187a4f4df' +\
                        'c04b05357c5d1c0eeac4c66f9ff7f2e6'
        key =           '2b7e151628aed2a6abf7158809cf4f3c'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_192_cfb128(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    'cdc80d6fddf18cab34c25909c99a4174' +\
                        '67ce7f7f81173621961a2b70171d3d7a' +\
                        '2e1e8a1dd59b88b1c8e60fed1efac4c9' +\
                        'c05f9f9ca9834fa042ae8fba584b09ff'
        key =           '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_256_cfb128(self):
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'

        ciphertext =    'dc7e84bfda79164b7ecd8486985d3860' +\
                        '39ffed143b28b1c832113c6331e5407b' +\
                        'df10132415e54b92a13ed0a8267ae2f9' +\
                        '75a385741ab9cef82031623d55b1e471'
        key =           '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4'
        iv =            '000102030405060708090a0b0c0d0e0f'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(CfbTests)
    if config.get('slow_tests'):
        tests += list_test_cases(NistCfbVectors)
    tests += list_test_cases(SP800TestVectors)
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
