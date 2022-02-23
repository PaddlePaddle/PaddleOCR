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

import json
import unittest
from binascii import unhexlify

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors_wycheproof

from Crypto.Util.py3compat import tobytes, bchr
from Crypto.Cipher import AES
from Crypto.Hash import SHAKE128

from Crypto.Util.strxor import strxor


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class SivTests(unittest.TestCase):

    key_256 = get_tag_random("key_256", 32)
    key_384 = get_tag_random("key_384", 48)
    key_512 = get_tag_random("key_512", 64)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_loopback_128(self):
        for key in self.key_256, self.key_384, self.key_512:
            cipher = AES.new(key, AES.MODE_SIV, nonce=self.nonce_96)
            pt = get_tag_random("plaintext", 16 * 100)
            ct, mac = cipher.encrypt_and_digest(pt)

            cipher = AES.new(key, AES.MODE_SIV, nonce=self.nonce_96)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)

    def test_nonce(self):
        # Deterministic encryption
        AES.new(self.key_256, AES.MODE_SIV)

        cipher = AES.new(self.key_256, AES.MODE_SIV, self.nonce_96)
        ct1, tag1 = cipher.encrypt_and_digest(self.data)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct2, tag2 = cipher.encrypt_and_digest(self.data)
        self.assertEqual(ct1 + tag1, ct2 + tag2)

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV,
                          nonce=u'test12345678')

    def test_nonce_length(self):
        # nonce can be of any length (but not empty)
        self.assertRaises(ValueError, AES.new, self.key_256, AES.MODE_SIV,
                          nonce=b"")

        for x in range(1, 128):
            cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=bchr(1) * x)
            cipher.encrypt_and_digest(b'\x01')

    def test_block_size_128(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)

        # By default, no nonce is randomly generated
        self.assertFalse(hasattr(AES.new(self.key_256, AES.MODE_SIV), "nonce"))

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV,
                          self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_256, AES.MODE_SIV,
                          nonce=self.nonce_96, unknown=7)

        # But some are only known by the base cipher
        # (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96,
                use_aesni=False)

    def test_encrypt_excludes_decrypt(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.decrypt, self.data)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.decrypt_and_verify,
                          self.data, self.data)

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt_and_verify,
                          u'test1234567890-*', b"xxxx")

    def test_mac_len(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Crypto.Util.strxor import strxor_c
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)

        invalid_mac = strxor_c(mac, 0x01)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct,
                          invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_bytearray(self):

        # Encrypt
        key = bytearray(self.key_256)
        nonce = bytearray(self.nonce_96)
        data = bytearray(self.data)
        header = bytearray(self.data)

        cipher1 = AES.new(self.key_256,
                          AES.MODE_SIV,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct, tag = cipher1.encrypt_and_digest(self.data)

        cipher2 = AES.new(key,
                          AES.MODE_SIV,
                          nonce=nonce)
        key[:3] = b'\xFF\xFF\xFF'
        nonce[:3] = b'\xFF\xFF\xFF'
        cipher2.update(header)
        header[:3] = b'\xFF\xFF\xFF'
        ct_test, tag_test = cipher2.encrypt_and_digest(data)

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key = bytearray(self.key_256)
        nonce = bytearray(self.nonce_96)
        header = bytearray(self.data)
        ct_ba = bytearray(ct)
        tag_ba = bytearray(tag)

        cipher3 = AES.new(key,
                          AES.MODE_SIV,
                          nonce=nonce)
        key[:3] = b'\xFF\xFF\xFF'
        nonce[:3] = b'\xFF\xFF\xFF'
        cipher3.update(header)
        header[:3] = b'\xFF\xFF\xFF'
        pt_test = cipher3.decrypt_and_verify(ct_ba, tag_ba)

        self.assertEqual(self.data, pt_test)

    def test_memoryview(self):

        # Encrypt
        key = memoryview(bytearray(self.key_256))
        nonce = memoryview(bytearray(self.nonce_96))
        data = memoryview(bytearray(self.data))
        header = memoryview(bytearray(self.data))

        cipher1 = AES.new(self.key_256,
                          AES.MODE_SIV,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct, tag = cipher1.encrypt_and_digest(self.data)

        cipher2 = AES.new(key,
                          AES.MODE_SIV,
                          nonce=nonce)
        key[:3] = b'\xFF\xFF\xFF'
        nonce[:3] = b'\xFF\xFF\xFF'
        cipher2.update(header)
        header[:3] = b'\xFF\xFF\xFF'
        ct_test, tag_test= cipher2.encrypt_and_digest(data)

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key = memoryview(bytearray(self.key_256))
        nonce = memoryview(bytearray(self.nonce_96))
        header = memoryview(bytearray(self.data))
        ct_ba = memoryview(bytearray(ct))
        tag_ba = memoryview(bytearray(tag))

        cipher3 = AES.new(key,
                          AES.MODE_SIV,
                          nonce=nonce)
        key[:3] = b'\xFF\xFF\xFF'
        nonce[:3] = b'\xFF\xFF\xFF'
        cipher3.update(header)
        header[:3] = b'\xFF\xFF\xFF'
        pt_test = cipher3.decrypt_and_verify(ct_ba, tag_ba)

        self.assertEqual(self.data, pt_test)

    def test_output_param(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)

        output = bytearray(128)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        res, tag_out = cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        self.assertEqual(tag, tag_out)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        res = cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)

        output = memoryview(bytearray(128))
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128

        pt = b'5' * LEN_PT
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(pt)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt_and_digest, pt, output=b'0' * LEN_PT)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, ct, tag, output=b'0' * LEN_PT)

        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.encrypt_and_digest, pt, output=shorter_output)
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct, tag, output=shorter_output)


class SivFSMTests(unittest.TestCase):

    key_256 = get_tag_random("key_256", 32)
    nonce_96 = get_tag_random("nonce_96", 12)
    data = get_tag_random("data", 128)

    def test_invalid_init_encrypt(self):
        # Path INIT->ENCRYPT fails
        cipher = AES.new(self.key_256, AES.MODE_SIV,
                         nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, b"xxx")

    def test_invalid_init_decrypt(self):
        # Path INIT->DECRYPT fails
        cipher = AES.new(self.key_256, AES.MODE_SIV,
                         nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, b"xxx")

    def test_valid_init_update_digest_verify(self):
        # No plaintext, fixed authenticated data
        # Verify path INIT->UPDATE->DIGEST
        cipher = AES.new(self.key_256, AES.MODE_SIV,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->VERIFY
        cipher = AES.new(self.key_256, AES.MODE_SIV,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        # Verify path INIT->DIGEST
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        # Verify path INIT->VERIFY
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        mac = cipher.digest()

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_digest_or_verify(self):
        # Multiple calls to digest
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())

        # Multiple calls to verify
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        # encrypt_and_digest
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)

        # decrypt_and_verify
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_multiple_encrypt_and_digest(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(self.data)
        self.assertRaises(TypeError, cipher.encrypt_and_digest, b'')

    def test_invalid_multiple_decrypt_and_verify(self):
        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        ct, tag = cipher.encrypt_and_digest(self.data)

        cipher = AES.new(self.key_256, AES.MODE_SIV, nonce=self.nonce_96)
        cipher.decrypt_and_verify(ct, tag)
        self.assertRaises(TypeError, cipher.decrypt_and_verify, ct, tag)


def transform(tv):
    new_tv = [[unhexlify(x) for x in tv[0].split("-")]]
    new_tv += [ unhexlify(x) for x in tv[1:5]]
    if tv[5]:
        nonce = unhexlify(tv[5])
    else:
        nonce = None
    new_tv += [ nonce ]
    return new_tv


class TestVectors(unittest.TestCase):
    """Class exercising the SIV test vectors found in RFC5297"""

    # This is a list of tuples with 5 items:
    #
    #  1. Header + '|' + plaintext
    #  2. Header + '|' + ciphertext + '|' + MAC
    #  3. AES-128 key
    #  4. Description
    #  5. Dictionary of parameters to be passed to AES.new().
    #     It must include the nonce.
    #
    #  A "Header" is a dash ('-') separated sequece of components.
    #
    test_vectors_hex = [
      (
        '101112131415161718191a1b1c1d1e1f2021222324252627',
        '112233445566778899aabbccddee',
        '40c02b9690c4dc04daef7f6afe5c',
        '85632d07c6e8f37f950acd320a2ecc93',
        'fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff',
        None
      ),
      (
        '00112233445566778899aabbccddeeffdeaddadadeaddadaffeeddccbbaa9988' +
        '7766554433221100-102030405060708090a0',
        '7468697320697320736f6d6520706c61696e7465787420746f20656e63727970' +
        '74207573696e67205349562d414553',
        'cb900f2fddbe404326601965c889bf17dba77ceb094fa663b7a3f748ba8af829' +
        'ea64ad544a272e9c485b62a3fd5c0d',
        '7bdb6e3b432667eb06f4d14bff2fbd0f',
        '7f7e7d7c7b7a79787776757473727170404142434445464748494a4b4c4d4e4f',
        '09f911029d74e35bd84156c5635688c0'
      ),
    ]

    test_vectors = [ transform(tv) for tv in test_vectors_hex ]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:

            # Encrypt
            cipher = AES.new(key, AES.MODE_SIV, nonce=nonce)
            for x in assoc_data:
                cipher.update(x)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)

            # Decrypt
            cipher = AES.new(key, AES.MODE_SIV, nonce=nonce)
            for x in assoc_data:
                cipher.update(x)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self):
        unittest.TestCase.__init__(self)
        self._id = "None"

    def setUp(self):
        self.tv = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                               "aes_siv_cmac_test.json",
                                               "Wycheproof AES SIV")

    def shortDescription(self):
        return self._id

    def test_encrypt(self, tv):
        self._id = "Wycheproof Encrypt AES-SIV Test #" + str(tv.id)

        cipher = AES.new(tv.key, AES.MODE_SIV)
        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(tag + ct, tv.ct)

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt AES_SIV Test #" + str(tv.id)

        cipher = AES.new(tv.key, AES.MODE_SIV)
        cipher.update(tv.aad)
        try:
            pt = cipher.decrypt_and_verify(tv.ct[16:], tv.ct[:16])
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)

    def runTest(self):

        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)


class TestVectorsWycheproof2(unittest.TestCase):

    def __init__(self):
        unittest.TestCase.__init__(self)
        self._id = "None"

    def setUp(self):
        self.tv = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                               "aead_aes_siv_cmac_test.json",
                                               "Wycheproof AEAD SIV")

    def shortDescription(self):
        return self._id

    def test_encrypt(self, tv):
        self._id = "Wycheproof Encrypt AEAD-AES-SIV Test #" + str(tv.id)

        cipher = AES.new(tv.key, AES.MODE_SIV, nonce=tv.iv)
        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(ct, tv.ct)
            self.assertEqual(tag, tv.tag)

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt AEAD-AES-SIV Test #" + str(tv.id)

        cipher = AES.new(tv.key, AES.MODE_SIV, nonce=tv.iv)
        cipher.update(tv.aad)
        try:
            pt = cipher.decrypt_and_verify(tv.ct, tv.tag)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)

    def runTest(self):

        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(SivTests)
    tests += list_test_cases(SivFSMTests)
    tests += [ TestVectors() ]
    tests += [ TestVectorsWycheproof() ]
    tests += [ TestVectorsWycheproof2() ]
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
