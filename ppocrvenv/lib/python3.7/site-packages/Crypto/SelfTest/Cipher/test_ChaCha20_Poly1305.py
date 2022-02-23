# ===================================================================
#
# Copyright (c) 2018, Helder Eijs <helderijs@gmail.com>
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

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors_wycheproof
from Crypto.Util.py3compat import tobytes
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Hash import SHAKE128

from Crypto.Util._file_system import pycryptodome_filename
from Crypto.Util.strxor import strxor


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class ChaCha20Poly1305Tests(unittest.TestCase):

    key_256 = get_tag_random("key_256", 32)
    nonce_96 = get_tag_random("nonce_96", 12)
    data_128 = get_tag_random("data_128", 16)

    def test_loopback(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        # Nonce can only be 8 or 12 bytes
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=b'H' * 8)
        self.assertEqual(len(cipher.nonce), 8)
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=b'H' * 12)
        self.assertEqual(len(cipher.nonce), 12)

        # If not passed, the nonce is created randomly
        cipher = ChaCha20_Poly1305.new(key=self.key_256)
        nonce1 = cipher.nonce
        cipher = ChaCha20_Poly1305.new(key=self.key_256)
        nonce2 = cipher.nonce
        self.assertEqual(len(nonce1), 12)
        self.assertNotEqual(nonce1, nonce2)

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data_128))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError,
                          ChaCha20_Poly1305.new,
                          key=self.key_256,
                          nonce=u'test12345678')

    def test_nonce_length(self):
        # nonce can only be 8 or 12 bytes long
        self.assertRaises(ValueError,
                          ChaCha20_Poly1305.new,
                          key=self.key_256,
                          nonce=b'0' * 7)
        self.assertRaises(ValueError,
                          ChaCha20_Poly1305.new,
                          key=self.key_256,
                          nonce=b'')

    def test_block_size(self):
        # Not based on block ciphers
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertFalse(hasattr(cipher, 'block_size'))

    def test_nonce_attribute(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)

        # By default, a 12 bytes long nonce is randomly generated
        nonce1 = ChaCha20_Poly1305.new(key=self.key_256).nonce
        nonce2 = ChaCha20_Poly1305.new(key=self.key_256).nonce
        self.assertEqual(len(nonce1), 12)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError,
                          ChaCha20_Poly1305.new,
                          key=self.key_256,
                          param=9)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            result = getattr(cipher, func)(b"")
            self.assertEqual(result, b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_data_must_be_bytes(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data_128)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Crypto.Util.strxor import strxor_c
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data_128)

        invalid_mac = strxor_c(mac, 0x01)

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct,
                          invalid_mac)

    def test_hex_mac(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_message_chunks(self):
        # Validate that both associated data and plaintext/ciphertext
        # can be broken up in chunks of arbitrary length

        auth_data = get_tag_random("authenticated data", 127)
        plaintext = get_tag_random("plaintext", 127)

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i+chunk_length] for i in range(0, len(data),
                    chunk_length)]

        # Encryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b""
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)

        # Decryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            ct2 = b""
            for chunk in break_up(plaintext, chunk_length):
                ct2 += cipher.encrypt(chunk)
            self.assertEqual(ciphertext, ct2)
            self.assertEqual(cipher.digest(), ref_mac)

    def test_bytearray(self):

        # Encrypt
        key_ba = bytearray(self.key_256)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data_128)
        data_ba = bytearray(self.data_128)

        cipher1 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        cipher1.update(self.data_128)
        ct = cipher1.encrypt(self.data_128)
        tag = cipher1.digest()

        cipher2 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        key_ba[:3] = b'\xFF\xFF\xFF'
        nonce_ba[:3] = b'\xFF\xFF\xFF'
        cipher2.update(header_ba)
        header_ba[:3] = b'\xFF\xFF\xFF'
        ct_test = cipher2.encrypt(data_ba)
        data_ba[:3] = b'\x99\x99\x99'
        tag_test = cipher2.digest()

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key_ba = bytearray(self.key_256)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data_128)
        ct_ba = bytearray(ct)
        tag_ba = bytearray(tag)
        del data_ba

        cipher3 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        key_ba[:3] = b'\xFF\xFF\xFF'
        nonce_ba[:3] = b'\xFF\xFF\xFF'
        cipher3.update(header_ba)
        header_ba[:3] = b'\xFF\xFF\xFF'
        pt_test = cipher3.decrypt(ct_ba)
        ct_ba[:3] = b'\xFF\xFF\xFF'
        cipher3.verify(tag_ba)

        self.assertEqual(pt_test, self.data_128)

    def test_memoryview(self):

        # Encrypt
        key_mv = memoryview(bytearray(self.key_256))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data_128))
        data_mv = memoryview(bytearray(self.data_128))

        cipher1 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        cipher1.update(self.data_128)
        ct = cipher1.encrypt(self.data_128)
        tag = cipher1.digest()

        cipher2 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        key_mv[:3] = b'\xFF\xFF\xFF'
        nonce_mv[:3] = b'\xFF\xFF\xFF'
        cipher2.update(header_mv)
        header_mv[:3] = b'\xFF\xFF\xFF'
        ct_test = cipher2.encrypt(data_mv)
        data_mv[:3] = b'\x99\x99\x99'
        tag_test = cipher2.digest()

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key_mv = memoryview(bytearray(self.key_256))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data_128))
        ct_mv = memoryview(bytearray(ct))
        tag_mv = memoryview(bytearray(tag))
        del data_mv

        cipher3 = ChaCha20_Poly1305.new(key=self.key_256,
                                        nonce=self.nonce_96)
        key_mv[:3] = b'\xFF\xFF\xFF'
        nonce_mv[:3] = b'\xFF\xFF\xFF'
        cipher3.update(header_mv)
        header_mv[:3] = b'\xFF\xFF\xFF'
        pt_test = cipher3.decrypt(ct_mv)
        ct_mv[:3] = b'\x99\x99\x99'
        cipher3.verify(tag_mv)

        self.assertEqual(pt_test, self.data_128)


class XChaCha20Poly1305Tests(unittest.TestCase):

    def test_encrypt(self):
        # From https://tools.ietf.org/html/draft-arciszewski-xchacha-03
        # Section A.3.1

        pt = b"""
                4c616469657320616e642047656e746c656d656e206f662074686520636c6173
                73206f66202739393a204966204920636f756c64206f6666657220796f75206f
                6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73
                637265656e20776f756c642062652069742e"""
        pt = unhexlify(pt.replace(b"\n", b"").replace(b" ", b""))

        aad = unhexlify(b"50515253c0c1c2c3c4c5c6c7")
        key = unhexlify(b"808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
        iv = unhexlify(b"404142434445464748494a4b4c4d4e4f5051525354555657")

        ct = b"""
                bd6d179d3e83d43b9576579493c0e939572a1700252bfaccbed2902c21396cbb
                731c7f1b0b4aa6440bf3a82f4eda7e39ae64c6708c54c216cb96b72e1213b452
                2f8c9ba40db5d945b11b69b982c1bb9e3f3fac2bc369488f76b2383565d3fff9
                21f9664c97637da9768812f615c68b13b52e"""
        ct = unhexlify(ct.replace(b"\n", b"").replace(b" ", b""))

        tag = unhexlify(b"c0875924c1c7987947deafd8780acf49")

        cipher = ChaCha20_Poly1305.new(key=key, nonce=iv)
        cipher.update(aad)
        ct_test, tag_test = cipher.encrypt_and_digest(pt)

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=iv)
        cipher.update(aad)
        cipher.decrypt_and_verify(ct, tag)


class ChaCha20Poly1305FSMTests(unittest.TestCase):

    key_256 = get_tag_random("key_256", 32)
    nonce_96 = get_tag_random("nonce_96", 12)
    data_128 = get_tag_random("data_128", 16)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        # No authenticated data, fixed plaintext
        # Verify path INIT->ENCRYPT->DIGEST
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()

        # Verify path INIT->DECRYPT->VERIFY
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_update_digest_verify(self):
        # No plaintext, fixed authenticated data
        # Verify path INIT->UPDATE->DIGEST
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->VERIFY
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        cipher.verify(mac)

    def test_valid_full_path(self):
        # Fixed authenticated data, fixed plaintext
        # Verify path INIT->UPDATE->ENCRYPT->DIGEST
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->DECRYPT->VERIFY
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        # Verify path INIT->DIGEST
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        # Verify path INIT->VERIFY
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        mac = cipher.digest()

        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        for method_name in "encrypt", "decrypt":
            for auth_data in (None, b"333", self.data_128,
                              self.data_128 + b"3"):
                cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                               nonce=self.nonce_96)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data_128)
                method(self.data_128)
                method(self.data_128)
                method(self.data_128)

    def test_valid_multiple_digest_or_verify(self):
        # Multiple calls to digest
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())

        # Multiple calls to verify
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        # encrypt_and_digest
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        ct, mac = cipher.encrypt_and_digest(self.data_128)

        # decrypt_and_verify
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        cipher.update(self.data_128)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data_128, pt)

    def test_invalid_mixing_encrypt_decrypt(self):
        # Once per method, with or without assoc. data
        for method1_name, method2_name in (("encrypt", "decrypt"),
                                           ("decrypt", "encrypt")):
            for assoc_data_present in (True, False):
                cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                               nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data_128)
                getattr(cipher, method1_name)(self.data_128)
                self.assertRaises(TypeError, getattr(cipher, method2_name),
                                  self.data_128)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in "encrypt", "update":
            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            cipher.encrypt(self.data_128)
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data_128)

            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data_128)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                       nonce=self.nonce_96)
        ct = cipher.encrypt(self.data_128)
        mac = cipher.digest()

        for method_name in "decrypt", "update":
            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data_128)

            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data_128)

            cipher = ChaCha20_Poly1305.new(key=self.key_256,
                                           nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data_128)


def compact(x):
    return unhexlify(x.replace(" ", "").replace(":", ""))


class TestVectorsRFC(unittest.TestCase):
    """Test cases from RFC7539"""

    # AAD, PT, CT, MAC, KEY, NONCE
    test_vectors_hex = [
        ( '50 51 52 53 c0 c1 c2 c3 c4 c5 c6 c7',
          '4c 61 64 69 65 73 20 61 6e 64 20 47 65 6e 74 6c'
          '65 6d 65 6e 20 6f 66 20 74 68 65 20 63 6c 61 73'
          '73 20 6f 66 20 27 39 39 3a 20 49 66 20 49 20 63'
          '6f 75 6c 64 20 6f 66 66 65 72 20 79 6f 75 20 6f'
          '6e 6c 79 20 6f 6e 65 20 74 69 70 20 66 6f 72 20'
          '74 68 65 20 66 75 74 75 72 65 2c 20 73 75 6e 73'
          '63 72 65 65 6e 20 77 6f 75 6c 64 20 62 65 20 69'
          '74 2e',
          'd3 1a 8d 34 64 8e 60 db 7b 86 af bc 53 ef 7e c2'
          'a4 ad ed 51 29 6e 08 fe a9 e2 b5 a7 36 ee 62 d6'
          '3d be a4 5e 8c a9 67 12 82 fa fb 69 da 92 72 8b'
          '1a 71 de 0a 9e 06 0b 29 05 d6 a5 b6 7e cd 3b 36'
          '92 dd bd 7f 2d 77 8b 8c 98 03 ae e3 28 09 1b 58'
          'fa b3 24 e4 fa d6 75 94 55 85 80 8b 48 31 d7 bc'
          '3f f4 de f0 8e 4b 7a 9d e5 76 d2 65 86 ce c6 4b'
          '61 16',
          '1a:e1:0b:59:4f:09:e2:6a:7e:90:2e:cb:d0:60:06:91',
          '80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f'
          '90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f',
          '07 00 00 00' + '40 41 42 43 44 45 46 47',
        ),
        ( 'f3 33 88 86 00 00 00 00 00 00 4e 91',
          '49 6e 74 65 72 6e 65 74 2d 44 72 61 66 74 73 20'
          '61 72 65 20 64 72 61 66 74 20 64 6f 63 75 6d 65'
          '6e 74 73 20 76 61 6c 69 64 20 66 6f 72 20 61 20'
          '6d 61 78 69 6d 75 6d 20 6f 66 20 73 69 78 20 6d'
          '6f 6e 74 68 73 20 61 6e 64 20 6d 61 79 20 62 65'
          '20 75 70 64 61 74 65 64 2c 20 72 65 70 6c 61 63'
          '65 64 2c 20 6f 72 20 6f 62 73 6f 6c 65 74 65 64'
          '20 62 79 20 6f 74 68 65 72 20 64 6f 63 75 6d 65'
          '6e 74 73 20 61 74 20 61 6e 79 20 74 69 6d 65 2e'
          '20 49 74 20 69 73 20 69 6e 61 70 70 72 6f 70 72'
          '69 61 74 65 20 74 6f 20 75 73 65 20 49 6e 74 65'
          '72 6e 65 74 2d 44 72 61 66 74 73 20 61 73 20 72'
          '65 66 65 72 65 6e 63 65 20 6d 61 74 65 72 69 61'
          '6c 20 6f 72 20 74 6f 20 63 69 74 65 20 74 68 65'
          '6d 20 6f 74 68 65 72 20 74 68 61 6e 20 61 73 20'
          '2f e2 80 9c 77 6f 72 6b 20 69 6e 20 70 72 6f 67'
          '72 65 73 73 2e 2f e2 80 9d',
          '64 a0 86 15 75 86 1a f4 60 f0 62 c7 9b e6 43 bd'
          '5e 80 5c fd 34 5c f3 89 f1 08 67 0a c7 6c 8c b2'
          '4c 6c fc 18 75 5d 43 ee a0 9e e9 4e 38 2d 26 b0'
          'bd b7 b7 3c 32 1b 01 00 d4 f0 3b 7f 35 58 94 cf'
          '33 2f 83 0e 71 0b 97 ce 98 c8 a8 4a bd 0b 94 81'
          '14 ad 17 6e 00 8d 33 bd 60 f9 82 b1 ff 37 c8 55'
          '97 97 a0 6e f4 f0 ef 61 c1 86 32 4e 2b 35 06 38'
          '36 06 90 7b 6a 7c 02 b0 f9 f6 15 7b 53 c8 67 e4'
          'b9 16 6c 76 7b 80 4d 46 a5 9b 52 16 cd e7 a4 e9'
          '90 40 c5 a4 04 33 22 5e e2 82 a1 b0 a0 6c 52 3e'
          'af 45 34 d7 f8 3f a1 15 5b 00 47 71 8c bc 54 6a'
          '0d 07 2b 04 b3 56 4e ea 1b 42 22 73 f5 48 27 1a'
          '0b b2 31 60 53 fa 76 99 19 55 eb d6 31 59 43 4e'
          'ce bb 4e 46 6d ae 5a 10 73 a6 72 76 27 09 7a 10'
          '49 e6 17 d9 1d 36 10 94 fa 68 f0 ff 77 98 71 30'
          '30 5b ea ba 2e da 04 df 99 7b 71 4d 6c 6f 2c 29'
          'a6 ad 5c b4 02 2b 02 70 9b',
          'ee ad 9d 67 89 0c bb 22 39 23 36 fe a1 85 1f 38',
          '1c 92 40 a5 eb 55 d3 8a f3 33 88 86 04 f6 b5 f0'
          '47 39 17 c1 40 2b 80 09 9d ca 5c bc 20 70 75 c0',
          '00 00 00 00 01 02 03 04 05 06 07 08',
        )
    ]

    test_vectors = [[unhexlify(x.replace(" ","").replace(":","")) for x in tv] for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:
            # Encrypt
            cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
            cipher.update(assoc_data)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)

            # Decrypt
            cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
            cipher.update(assoc_data)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = "None"

    def load_tests(self, filename):

        def filter_tag(group):
            return group['tagSize'] // 8

        def filter_algo(root):
            return root['algorithm']

        result = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                               filename,
                                               "Wycheproof ChaCha20-Poly1305",
                                               root_tag={'algo': filter_algo},
                                               group_tag={'tag_size': filter_tag})
        return result

    def setUp(self):
        self.tv = []
        self.tv.extend(self.load_tests("chacha20_poly1305_test.json"))
        self.tv.extend(self.load_tests("xchacha20_poly1305_test.json"))

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_encrypt(self, tv):
        self._id = "Wycheproof Encrypt %s Test #%s" % (tv.algo, tv.id)

        try:
            cipher = ChaCha20_Poly1305.new(key=tv.key, nonce=tv.iv)
        except ValueError as e:
            assert len(tv.iv) not in (8, 12) and "Nonce must be" in str(e)
            return

        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(ct, tv.ct)
            self.assertEqual(tag, tv.tag)
            self.warn(tv)

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt %s Test #%s" % (tv.algo, tv.id)

        try:
            cipher = ChaCha20_Poly1305.new(key=tv.key, nonce=tv.iv)
        except ValueError as e:
            assert len(tv.iv) not in (8, 12) and "Nonce must be" in str(e)
            return

        cipher.update(tv.aad)
        try:
            pt = cipher.decrypt_and_verify(tv.ct, tv.tag)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)
            self.warn(tv)

    def test_corrupt_decrypt(self, tv):
        self._id = "Wycheproof Corrupt Decrypt ChaCha20-Poly1305 Test #" + str(tv.id)
        if len(tv.iv) == 0 or len(tv.ct) < 1:
            return
        cipher = ChaCha20_Poly1305.new(key=tv.key, nonce=tv.iv)
        cipher.update(tv.aad)
        ct_corrupt = strxor(tv.ct, b"\x00" * (len(tv.ct) - 1) + b"\x01")
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct_corrupt, tv.tag)

    def runTest(self):

        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)
            self.test_corrupt_decrypt(tv)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        key = b'4' * 32
        nonce = b'5' * 12
        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)

        pt = b'5' * 16
        ct = cipher.encrypt(pt)

        output = bytearray(16)
        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        output = memoryview(bytearray(16))
        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0'*16)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0'*16)

        shorter_output = bytearray(7)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)

        cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(ChaCha20Poly1305Tests)
    tests += list_test_cases(XChaCha20Poly1305Tests)
    tests += list_test_cases(ChaCha20Poly1305FSMTests)
    tests += [TestVectorsRFC()]
    tests += [TestVectorsWycheproof(wycheproof_warnings)]
    tests += [TestOutput()]
    return tests


if __name__ == '__main__':
    def suite():
        unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
