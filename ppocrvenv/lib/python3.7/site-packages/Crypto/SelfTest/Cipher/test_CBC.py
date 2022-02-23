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


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)

class BlockChainingTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    key_192 = get_tag_random("key_192", 24)
    iv_128 = get_tag_random("iv_128", 16)
    iv_64 = get_tag_random("iv_64", 8)
    data_128 = get_tag_random("data_128", 16)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        pt = get_tag_random("plaintext", 8 * 100)
        ct = cipher.encrypt(pt)

        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_iv(self):
        # If not passed, the iv is created randomly
        cipher = AES.new(self.key_128, self.aes_mode)
        iv1 = cipher.iv
        cipher = AES.new(self.key_128, self.aes_mode)
        iv2 = cipher.iv
        self.assertNotEqual(iv1, iv2)
        self.assertEqual(len(iv1), 16)

        # IV can be passed in uppercase or lowercase
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ct = cipher.encrypt(self.data_128)

        cipher = AES.new(self.key_128, self.aes_mode, iv=self.iv_128)
        self.assertEqual(ct, cipher.encrypt(self.data_128))

        cipher = AES.new(self.key_128, self.aes_mode, IV=self.iv_128)
        self.assertEqual(ct, cipher.encrypt(self.data_128))

    def test_iv_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode,
                          iv = u'test1234567890-*')

    def test_only_one_iv(self):
        # Only one IV/iv keyword allowed
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode,
                          iv=self.iv_128, IV=self.iv_128)

    def test_iv_with_matching_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode,
                          b"")
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode,
                          self.iv_128[:15])
        self.assertRaises(ValueError, AES.new, self.key_128, self.aes_mode,
                          self.iv_128 + b"0")

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_block_size_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        self.assertEqual(cipher.block_size, DES3.block_size)

    def test_unaligned_data_128(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        for wrong_length in range(1,16):
            self.assertRaises(ValueError, cipher.encrypt, b"5" * wrong_length)

        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        for wrong_length in range(1,16):
            self.assertRaises(ValueError, cipher.decrypt, b"5" * wrong_length)

    def test_unaligned_data_64(self):
        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        for wrong_length in range(1,8):
            self.assertRaises(ValueError, cipher.encrypt, b"5" * wrong_length)

        cipher = DES3.new(self.key_192, self.des3_mode, self.iv_64)
        for wrong_length in range(1,8):
            self.assertRaises(ValueError, cipher.decrypt, b"5" * wrong_length)

    def test_IV_iv_attributes(self):
        data = get_tag_random("data", 16 * 100)
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
            getattr(cipher, func)(data)
            self.assertEqual(cipher.iv, self.iv_128)
            self.assertEqual(cipher.IV, self.iv_128)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode,
                          self.iv_128, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, self.aes_mode,
                          iv=self.iv_128, unknown=7)
        # But some are only known by the base cipher (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_128, self.aes_mode, iv=self.iv_128, use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
            result = getattr(cipher, func)(b"")
            self.assertEqual(result, b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = AES.new(self.key_128, self.aes_mode, self.iv_128)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_bytearray(self):
        data = b"1" * 128
        data_ba = bytearray(data)

        # Encrypt
        key_ba = bytearray(self.key_128)
        iv_ba = bytearray(self.iv_128)

        cipher1 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref1 = cipher1.encrypt(data)

        cipher2 = AES.new(key_ba, self.aes_mode, iv_ba)
        key_ba[:3] = b'\xFF\xFF\xFF'
        iv_ba[:3] = b'\xFF\xFF\xFF'
        ref2 = cipher2.encrypt(data_ba)

        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.iv, cipher2.iv)

        # Decrypt
        key_ba = bytearray(self.key_128)
        iv_ba = bytearray(self.iv_128)

        cipher3 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref3 = cipher3.decrypt(data)

        cipher4 = AES.new(key_ba, self.aes_mode, iv_ba)
        key_ba[:3] = b'\xFF\xFF\xFF'
        iv_ba[:3] = b'\xFF\xFF\xFF'
        ref4 = cipher4.decrypt(data_ba)

        self.assertEqual(ref3, ref4)

    def test_memoryview(self):
        data = b"1" * 128
        data_mv = memoryview(bytearray(data))

        # Encrypt
        key_mv = memoryview(bytearray(self.key_128))
        iv_mv = memoryview(bytearray(self.iv_128))

        cipher1 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref1 = cipher1.encrypt(data)

        cipher2 = AES.new(key_mv, self.aes_mode, iv_mv)
        key_mv[:3] = b'\xFF\xFF\xFF'
        iv_mv[:3] = b'\xFF\xFF\xFF'
        ref2 = cipher2.encrypt(data_mv)

        self.assertEqual(ref1, ref2)
        self.assertEqual(cipher1.iv, cipher2.iv)

        # Decrypt
        key_mv = memoryview(bytearray(self.key_128))
        iv_mv = memoryview(bytearray(self.iv_128))

        cipher3 = AES.new(self.key_128, self.aes_mode, self.iv_128)
        ref3 = cipher3.decrypt(data)

        cipher4 = AES.new(key_mv, self.aes_mode, iv_mv)
        key_mv[:3] = b'\xFF\xFF\xFF'
        iv_mv[:3] = b'\xFF\xFF\xFF'
        ref4 = cipher4.decrypt(data_mv)

        self.assertEqual(ref3, ref4)

    def test_output_param(self):

        pt = b'5' * 128
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)

        output = bytearray(128)
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)


    def test_output_param_same_buffer(self):

        pt = b'5' * 128
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)

        pt_ba = bytearray(pt)
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        res = cipher.encrypt(pt_ba, output=pt_ba)
        self.assertEqual(ct, pt_ba)
        self.assertEqual(res, None)

        ct_ba = bytearray(ct)
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        res = cipher.decrypt(ct_ba, output=ct_ba)
        self.assertEqual(pt, ct_ba)
        self.assertEqual(res, None)


    def test_output_param_memoryview(self):

        pt = b'5' * 128
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)

        output = memoryview(bytearray(128))
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128

        pt = b'5' * LEN_PT
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        ct = cipher.encrypt(pt)

        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)

        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)

        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(b'4'*16, self.aes_mode, iv=self.iv_128)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


class CbcTests(BlockChainingTests):
    aes_mode = AES.MODE_CBC
    des3_mode = DES3.MODE_CBC


class NistBlockChainingVectors(unittest.TestCase):

    def _do_kat_aes_test(self, file_name):

        test_vectors = load_test_vectors(("Cipher", "AES"),
                            file_name,
                            "AES CBC KAT",
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

            cipher = AES.new(tv.key, self.aes_mode, tv.iv)
            if direction == "[ENCRYPT]":
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == "[DECRYPT]":
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False

    # See Section 6.4.2 in AESAVS
    def _do_mct_aes_test(self, file_name):

        test_vectors = load_test_vectors(("Cipher", "AES"),
                            file_name,
                            "AES CBC Montecarlo",
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
            cipher = AES.new(tv.key, self.aes_mode, tv.iv)

            if direction == '[ENCRYPT]':
                cts = [ tv.iv ]
                for count in range(1000):
                    cts.append(cipher.encrypt(tv.plaintext))
                    tv.plaintext = cts[-2]
                self.assertEqual(cts[-1], tv.ciphertext)
            elif direction == '[DECRYPT]':
                pts = [ tv.iv]
                for count in range(1000):
                    pts.append(cipher.decrypt(tv.ciphertext))
                    tv.ciphertext = pts[-2]
                self.assertEqual(pts[-1], tv.plaintext)
            else:
                assert False

    def _do_tdes_test(self, file_name):

        test_vectors = load_test_vectors(("Cipher", "TDES"),
                            file_name,
                            "TDES CBC KAT",
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
                cipher = DES.new(tv.keys, self.des_mode, tv.iv)
            else:
                if tv.key1 != tv.key3:
                    key = tv.key1 + tv.key2 + tv.key3  # Option 3
                else:
                    key = tv.key1 + tv.key2            # Option 2
                cipher = DES3.new(key, self.des3_mode, tv.iv)

            if direction == "[ENCRYPT]":
                self.assertEqual(cipher.encrypt(tv.plaintext), tv.ciphertext)
            elif direction == "[DECRYPT]":
                self.assertEqual(cipher.decrypt(tv.ciphertext), tv.plaintext)
            else:
                assert False


class NistCbcVectors(NistBlockChainingVectors):
    aes_mode = AES.MODE_CBC
    des_mode = DES.MODE_CBC
    des3_mode = DES3.MODE_CBC


# Create one test method per file
nist_aes_kat_mmt_files = (
    # KAT
    "CBCGFSbox128.rsp",
    "CBCGFSbox192.rsp",
    "CBCGFSbox256.rsp",
    "CBCKeySbox128.rsp",
    "CBCKeySbox192.rsp",
    "CBCKeySbox256.rsp",
    "CBCVarKey128.rsp",
    "CBCVarKey192.rsp",
    "CBCVarKey256.rsp",
    "CBCVarTxt128.rsp",
    "CBCVarTxt192.rsp",
    "CBCVarTxt256.rsp",
    # MMT
    "CBCMMT128.rsp",
    "CBCMMT192.rsp",
    "CBCMMT256.rsp",
    )
nist_aes_mct_files = (
    "CBCMCT128.rsp",
    "CBCMCT192.rsp",
    "CBCMCT256.rsp",
    )

for file_name in nist_aes_kat_mmt_files:
    def new_func(self, file_name=file_name):
        self._do_kat_aes_test(file_name)
    setattr(NistCbcVectors, "test_AES_" + file_name, new_func)

for file_name in nist_aes_mct_files:
    def new_func(self, file_name=file_name):
        self._do_mct_aes_test(file_name)
    setattr(NistCbcVectors, "test_AES_" + file_name, new_func)
del file_name, new_func

nist_tdes_files = (
    "TCBCMMT2.rsp",    # 2TDES
    "TCBCMMT3.rsp",    # 3TDES
    "TCBCinvperm.rsp", # Single DES
    "TCBCpermop.rsp",
    "TCBCsubtab.rsp",
    "TCBCvarkey.rsp",
    "TCBCvartext.rsp",
    )

for file_name in nist_tdes_files:
    def new_func(self, file_name=file_name):
        self._do_tdes_test(file_name)
    setattr(NistCbcVectors, "test_TDES_" + file_name, new_func)

# END OF NIST CBC TEST VECTORS


class SP800TestVectors(unittest.TestCase):
    """Class exercising the CBC test vectors found in Section F.2
    of NIST SP 800-3A"""

    def test_aes_128(self):
        key =           '2b7e151628aed2a6abf7158809cf4f3c'
        iv =            '000102030405060708090a0b0c0d0e0f'
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '7649abac8119b246cee98e9b12e9197d' +\
                        '5086cb9b507219ee95db113a917678b2' +\
                        '73bed6b8e3c1743b7116e69e22229516' +\
                        '3ff1caa1681fac09120eca307586e1a7'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_192(self):
        key =           '8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b'
        iv =            '000102030405060708090a0b0c0d0e0f'
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    '4f021db243bc633d7178183a9fa071e8' +\
                        'b4d9ada9ad7dedf4e5e738763f69145a' +\
                        '571b242012fb7ae07fa9baac3df102e0' +\
                        '08b0e27988598881d920a9e64f5615cd'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

    def test_aes_256(self):
        key =           '603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4'
        iv =            '000102030405060708090a0b0c0d0e0f'
        plaintext =     '6bc1bee22e409f96e93d7e117393172a' +\
                        'ae2d8a571e03ac9c9eb76fac45af8e51' +\
                        '30c81c46a35ce411e5fbc1191a0a52ef' +\
                        'f69f2445df4f9b17ad2b417be66c3710'
        ciphertext =    'f58c4c04d6e5f1ba779eabfb5f7bfbd6' +\
                        '9cfc4e967edb808d679f777bc6702c7d' +\
                        '39f23369a9d9bacfa530e26304231461' +\
                        'b2eb05e2c39be9fcda6c19078c6a9d1b'

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(CbcTests)
    if config.get('slow_tests'):
        tests += list_test_cases(NistCbcVectors)
    tests += list_test_cases(SP800TestVectors)
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
