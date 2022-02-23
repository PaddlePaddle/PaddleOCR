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
from binascii import unhexlify

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors_wycheproof
from Crypto.Util.py3compat import tobytes, bchr
from Crypto.Cipher import AES
from Crypto.Hash import SHAKE128

from Crypto.Util.strxor import strxor


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class CcmTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        # If not passed, the nonce is created randomly
        cipher = AES.new(self.key_128, AES.MODE_CCM)
        nonce1 = cipher.nonce
        cipher = AES.new(self.key_128, AES.MODE_CCM)
        nonce2 = cipher.nonce
        self.assertEqual(len(nonce1), 11)
        self.assertNotEqual(nonce1, nonce2)

        cipher = AES.new(self.key_128, AES.MODE_CCM, self.nonce_96)
        ct = cipher.encrypt(self.data)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CCM,
                          nonce=u'test12345678')

    def test_nonce_length(self):
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CCM,
                          nonce=b"")
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CCM,
                          nonce=bchr(1) * 6)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CCM,
                          nonce=bchr(1) * 14)
        for x in range(7, 13 + 1):
            AES.new(self.key_128, AES.MODE_CCM, nonce=bchr(1) * x)

    def test_block_size(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)

        # By default, a 11 bytes long nonce is randomly generated
        nonce1 = AES.new(self.key_128, AES.MODE_CCM).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_CCM).nonce
        self.assertEqual(len(nonce1), 11)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CCM,
                          self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_CCM,
                          nonce=self.nonce_96, unknown=7)

        # But some are only known by the base cipher
        # (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            result = getattr(cipher, func)(b"")
            self.assertEqual(result, b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        # Invalid MAC length
        for mac_len in range(3, 17 + 1, 2):
            self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_CCM,
                              nonce=self.nonce_96, mac_len=mac_len)

        # Valid MAC length
        for mac_len in range(4, 16 + 1, 2):
            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                             mac_len=mac_len)
            _, mac = cipher.encrypt_and_digest(self.data)
            self.assertEqual(len(mac), mac_len)

        # Default MAC length
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Crypto.Util.strxor import strxor_c
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)

        invalid_mac = strxor_c(mac, 0x01)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct,
                          invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_longer_assoc_data_than_declared(self):
        # More than zero
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=0)
        self.assertRaises(ValueError, cipher.update, b"1")

        # Too large
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=15)
        self.assertRaises(ValueError, cipher.update, self.data)

    def test_shorter_assoc_data_than_expected(self):
        DATA_LEN = len(self.data)

        # With plaintext
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=DATA_LEN + 1)
        cipher.update(self.data)
        self.assertRaises(ValueError, cipher.encrypt, self.data)

        # With empty plaintext
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=DATA_LEN + 1)
        cipher.update(self.data)
        self.assertRaises(ValueError, cipher.digest)

        # With ciphertext
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=DATA_LEN + 1)
        cipher.update(self.data)
        self.assertRaises(ValueError, cipher.decrypt, self.data)

        # With empty ciphertext
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         assoc_len=DATA_LEN + 1)
        cipher.update(self.data)
        self.assertRaises(ValueError, cipher.verify, mac)

    def test_shorter_and_longer_plaintext_than_declared(self):
        DATA_LEN = len(self.data)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         msg_len=DATA_LEN + 1)
        cipher.encrypt(self.data)
        self.assertRaises(ValueError, cipher.digest)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         msg_len=DATA_LEN - 1)
        self.assertRaises(ValueError, cipher.encrypt, self.data)

    def test_shorter_ciphertext_than_declared(self):
        DATA_LEN = len(self.data)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         msg_len=DATA_LEN + 1)
        cipher.decrypt(ct)
        self.assertRaises(ValueError, cipher.verify, mac)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                         msg_len=DATA_LEN - 1)
        self.assertRaises(ValueError, cipher.decrypt, ct)

    def test_message_chunks(self):
        # Validate that both associated data and plaintext/ciphertext
        # can be broken up in chunks of arbitrary length

        auth_data = get_tag_random("authenticated data", 127)
        plaintext = get_tag_random("plaintext", 127)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i+chunk_length] for i in range(0, len(data),
                    chunk_length)]

        # Encryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                             msg_len=127, assoc_len=127)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b""
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)

        # Decryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96,
                             msg_len=127, assoc_len=127)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            ct2 = b""
            for chunk in break_up(plaintext, chunk_length):
                ct2 += cipher.encrypt(chunk)
            self.assertEqual(ciphertext, ct2)
            self.assertEqual(cipher.digest(), ref_mac)

    def test_bytearray(self):

        # Encrypt
        key_ba = bytearray(self.key_128)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data)
        data_ba = bytearray(self.data)

        cipher1 = AES.new(self.key_128,
                          AES.MODE_CCM,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data)
        tag = cipher1.digest()

        cipher2 = AES.new(key_ba,
                          AES.MODE_CCM,
                          nonce=nonce_ba)
        key_ba[:3] = b"\xFF\xFF\xFF"
        nonce_ba[:3] = b"\xFF\xFF\xFF"
        cipher2.update(header_ba)
        header_ba[:3] = b"\xFF\xFF\xFF"
        ct_test = cipher2.encrypt(data_ba)
        data_ba[:3] = b"\xFF\xFF\xFF"
        tag_test = cipher2.digest()

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key_ba = bytearray(self.key_128)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data)
        del data_ba

        cipher4 = AES.new(key_ba,
                          AES.MODE_CCM,
                          nonce=nonce_ba)
        key_ba[:3] = b"\xFF\xFF\xFF"
        nonce_ba[:3] = b"\xFF\xFF\xFF"
        cipher4.update(header_ba)
        header_ba[:3] = b"\xFF\xFF\xFF"
        pt_test = cipher4.decrypt_and_verify(bytearray(ct_test), bytearray(tag_test))

        self.assertEqual(self.data, pt_test)

    def test_memoryview(self):

        # Encrypt
        key_mv = memoryview(bytearray(self.key_128))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data))
        data_mv = memoryview(bytearray(self.data))

        cipher1 = AES.new(self.key_128,
                          AES.MODE_CCM,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data)
        tag = cipher1.digest()

        cipher2 = AES.new(key_mv,
                          AES.MODE_CCM,
                          nonce=nonce_mv)
        key_mv[:3] = b"\xFF\xFF\xFF"
        nonce_mv[:3] = b"\xFF\xFF\xFF"
        cipher2.update(header_mv)
        header_mv[:3] = b"\xFF\xFF\xFF"
        ct_test = cipher2.encrypt(data_mv)
        data_mv[:3] = b"\xFF\xFF\xFF"
        tag_test = cipher2.digest()

        self.assertEqual(ct, ct_test)
        self.assertEqual(tag, tag_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decrypt
        key_mv = memoryview(bytearray(self.key_128))
        nonce_mv = memoryview(bytearray(self.nonce_96))
        header_mv = memoryview(bytearray(self.data))
        del data_mv

        cipher4 = AES.new(key_mv,
                          AES.MODE_CCM,
                          nonce=nonce_mv)
        key_mv[:3] = b"\xFF\xFF\xFF"
        nonce_mv[:3] = b"\xFF\xFF\xFF"
        cipher4.update(header_mv)
        header_mv[:3] = b"\xFF\xFF\xFF"
        pt_test = cipher4.decrypt_and_verify(memoryview(ct_test), memoryview(tag_test))

        self.assertEqual(self.data, pt_test)

    def test_output_param(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)
        tag = cipher.digest()

        output = bytearray(128)
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        res, tag_out = cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        self.assertEqual(tag, tag_out)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        res = cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)

        output = memoryview(bytearray(128))
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):

        pt = b'5' * 16
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0'*16)

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0'*16)

        shorter_output = bytearray(15)
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


class CcmFSMTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 16)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        # No authenticated data, fixed plaintext
        for assoc_len in (None, 0):
            for msg_len in (None, len(self.data)):
                # Verify path INIT->ENCRYPT->DIGEST
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                ct = cipher.encrypt(self.data)
                mac = cipher.digest()

                # Verify path INIT->DECRYPT->VERIFY
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                cipher.decrypt(ct)
                cipher.verify(mac)

    def test_valid_init_update_digest_verify(self):
        # No plaintext, fixed authenticated data
        for assoc_len in (None, len(self.data)):
            for msg_len in (None, 0):
                # Verify path INIT->UPDATE->DIGEST
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                cipher.update(self.data)
                mac = cipher.digest()

                # Verify path INIT->UPDATE->VERIFY
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                cipher.update(self.data)
                cipher.verify(mac)

    def test_valid_full_path(self):
        # Fixed authenticated data, fixed plaintext
        for assoc_len in (None, len(self.data)):
            for msg_len in (None, len(self.data)):
                # Verify path INIT->UPDATE->ENCRYPT->DIGEST
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                cipher.update(self.data)
                ct = cipher.encrypt(self.data)
                mac = cipher.digest()

                # Verify path INIT->UPDATE->DECRYPT->VERIFY
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 assoc_len=assoc_len,
                                 msg_len=msg_len)
                cipher.update(self.data)
                cipher.decrypt(ct)
                cipher.verify(mac)

    def test_valid_init_digest(self):
        # Verify path INIT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        # Verify path INIT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        mac = cipher.digest()

        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        # Only possible if msg_len is declared in advance
        for method_name in "encrypt", "decrypt":
            for auth_data in (None, b"333", self.data,
                              self.data + b"3"):
                if auth_data is None:
                    assoc_len = None
                else:
                    assoc_len = len(auth_data)
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 msg_len=64,
                                 assoc_len=assoc_len)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data)
                method(self.data)
                method(self.data)
                method(self.data)

    def test_valid_multiple_digest_or_verify(self):
        # Multiple calls to digest
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())

        # Multiple calls to verify
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        # encrypt_and_digest
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)

        # decrypt_and_verify
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_multiple_encrypt_decrypt_without_msg_len(self):
        # Once per method, with or without assoc. data
        for method_name in "encrypt", "decrypt":
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data)
                method = getattr(cipher, method_name)
                method(self.data)
                self.assertRaises(TypeError, method, self.data)

    def test_invalid_mixing_encrypt_decrypt(self):
        # Once per method, with or without assoc. data
        for method1_name, method2_name in (("encrypt", "decrypt"),
                                           ("decrypt", "encrypt")):
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_CCM,
                                 nonce=self.nonce_96,
                                 msg_len=32)
                if assoc_data_present:
                    cipher.update(self.data)
                getattr(cipher, method1_name)(self.data)
                self.assertRaises(TypeError, getattr(cipher, method2_name),
                                  self.data)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in "encrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            cipher.encrypt(self.data)
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()

        for method_name in "decrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_CCM, nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)


class TestVectors(unittest.TestCase):
    """Class exercising the CCM test vectors found in Appendix C
    of NIST SP 800-38C and in RFC 3610"""

    # List of test vectors, each made up of:
    # - authenticated data
    # - plaintext
    # - ciphertext
    # - MAC
    # - AES key
    # - nonce
    test_vectors_hex = [
        # NIST SP 800 38C
        ( '0001020304050607',
          '20212223',
          '7162015b',
          '4dac255d',
          '404142434445464748494a4b4c4d4e4f',
          '10111213141516'),
        ( '000102030405060708090a0b0c0d0e0f',
          '202122232425262728292a2b2c2d2e2f',
          'd2a1f0e051ea5f62081a7792073d593d',
          '1fc64fbfaccd',
          '404142434445464748494a4b4c4d4e4f',
          '1011121314151617'),
        ( '000102030405060708090a0b0c0d0e0f10111213',
          '202122232425262728292a2b2c2d2e2f3031323334353637',
          'e3b201a9f5b71a7a9b1ceaeccd97e70b6176aad9a4428aa5',
          '484392fbc1b09951',
          '404142434445464748494a4b4c4d4e4f',
          '101112131415161718191a1b'),
        ( (''.join(["%02X" % (x*16+y) for x in range(0,16) for y in range(0,16)]))*256,
          '202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f',
          '69915dad1e84c6376a68c2967e4dab615ae0fd1faec44cc484828529463ccf72',
          'b4ac6bec93e8598e7f0dadbcea5b',
          '404142434445464748494a4b4c4d4e4f',
          '101112131415161718191a1b1c'),
        # RFC3610
        ( '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e',
          '588c979a61c663d2f066d0c2c0f989806d5f6b61dac384',
          '17e8d12cfdf926e0',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000003020100a0a1a2a3a4a5'),
        (
          '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
          '72c91a36e135f8cf291ca894085c87e3cc15c439c9e43a3b',
          'a091d56e10400916',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000004030201a0a1a2a3a4a5'),
        ( '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20',
          '51b1e5f44a197d1da46b0f8e2d282ae871e838bb64da859657',
          '4adaa76fbd9fb0c5',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000005040302A0A1A2A3A4A5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e',
          'a28c6865939a9a79faaa5c4c2a9d4a91cdac8c',
          '96c861b9c9e61ef1',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000006050403a0a1a2a3a4a5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e1f',
          'dcf1fb7b5d9e23fb9d4e131253658ad86ebdca3e',
          '51e83f077d9c2d93',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000007060504a0a1a2a3a4a5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e1f20',
          '6fc1b011f006568b5171a42d953d469b2570a4bd87',
          '405a0443ac91cb94',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000008070605a0a1a2a3a4a5'),
        ( '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e',
          '0135d1b2c95f41d5d1d4fec185d166b8094e999dfed96c',
          '048c56602c97acbb7490',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '00000009080706a0a1a2a3a4a5'),
        ( '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f',
          '7b75399ac0831dd2f0bbd75879a2fd8f6cae6b6cd9b7db24',
          'c17b4433f434963f34b4',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '0000000a090807a0a1a2a3a4a5'),
        ( '0001020304050607',
          '08090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20',
          '82531a60cc24945a4b8279181ab5c84df21ce7f9b73f42e197',
          'ea9c07e56b5eb17e5f4e',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '0000000b0a0908a0a1a2a3a4a5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e',
          '07342594157785152b074098330abb141b947b',
          '566aa9406b4d999988dd',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '0000000c0b0a09a0a1a2a3a4a5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e1f',
          '676bb20380b0e301e8ab79590a396da78b834934',
          'f53aa2e9107a8b6c022c',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '0000000d0c0b0aa0a1a2a3a4a5'),
        ( '000102030405060708090a0b',
          '0c0d0e0f101112131415161718191a1b1c1d1e1f20',
          'c0ffa0d6f05bdb67f24d43a4338d2aa4bed7b20e43',
          'cd1aa31662e7ad65d6db',
          'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf',
          '0000000e0d0c0ba0a1a2a3a4a5'),
        ( '0be1a88bace018b1',
          '08e8cf97d820ea258460e96ad9cf5289054d895ceac47c',
          '4cb97f86a2a4689a877947ab8091ef5386a6ffbdd080f8',
          'e78cf7cb0cddd7b3',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00412b4ea9cdbe3c9696766cfa'),
        ( '63018f76dc8a1bcb',
          '9020ea6f91bdd85afa0039ba4baff9bfb79c7028949cd0ec',
          '4ccb1e7ca981befaa0726c55d378061298c85c92814abc33',
          'c52ee81d7d77c08a',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '0033568ef7b2633c9696766cfa'),
        ( 'aa6cfa36cae86b40',
          'b916e0eacc1c00d7dcec68ec0b3bbb1a02de8a2d1aa346132e',
          'b1d23a2220ddc0ac900d9aa03c61fcf4a559a4417767089708',
          'a776796edb723506',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00103fe41336713c9696766cfa'),
        ( 'd0d0735c531e1becf049c244',
          '12daac5630efa5396f770ce1a66b21f7b2101c',
          '14d253c3967b70609b7cbb7c49916028324526',
          '9a6f49975bcadeaf',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00764c63b8058e3c9696766cfa'),
        ( '77b60f011c03e1525899bcae',
          'e88b6a46c78d63e52eb8c546efb5de6f75e9cc0d',
          '5545ff1a085ee2efbf52b2e04bee1e2336c73e3f',
          '762c0c7744fe7e3c',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00f8b678094e3b3c9696766cfa'),
        ( 'cd9044d2b71fdb8120ea60c0',
          '6435acbafb11a82e2f071d7ca4a5ebd93a803ba87f',
          '009769ecabdf48625594c59251e6035722675e04c8',
          '47099e5ae0704551',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00d560912d3f703c9696766cfa'),
        ( 'd85bc7e69f944fb8',
          '8a19b950bcf71a018e5e6701c91787659809d67dbedd18',
          'bc218daa947427b6db386a99ac1aef23ade0b52939cb6a',
          '637cf9bec2408897c6ba',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '0042fff8f1951c3c9696766cfa'),
        ( '74a0ebc9069f5b37',
          '1761433c37c5a35fc1f39f406302eb907c6163be38c98437',
          '5810e6fd25874022e80361a478e3e9cf484ab04f447efff6',
          'f0a477cc2fc9bf548944',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '00920f40e56cdc3c9696766cfa'),
        ( '44a3aa3aae6475ca',
          'a434a8e58500c6e41530538862d686ea9e81301b5ae4226bfa',
          'f2beed7bc5098e83feb5b31608f8e29c38819a89c8e776f154',
          '4d4151a4ed3a8b87b9ce',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '0027ca0c7120bc3c9696766cfa'),
        ( 'ec46bb63b02520c33c49fd70',
          'b96b49e21d621741632875db7f6c9243d2d7c2',
          '31d750a09da3ed7fddd49a2032aabf17ec8ebf',
          '7d22c8088c666be5c197',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '005b8ccbcd9af83c9696766cfa'),
        ( '47a65ac78b3d594227e85e71',
          'e2fcfbb880442c731bf95167c8ffd7895e337076',
          'e882f1dbd38ce3eda7c23f04dd65071eb41342ac',
          'df7e00dccec7ae52987d',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '003ebe94044b9a3c9696766cfa'),
        ( '6e37a6ef546d955d34ab6059',
          'abf21c0b02feb88f856df4a37381bce3cc128517d4',
          'f32905b88a641b04b9c9ffb58cc390900f3da12ab1',
          '6dce9e82efa16da62059',
          'd7828d13b2b0bdc325a76236df93cc6b',
          '008d493b30ae8b3c9696766cfa'),
    ]

    test_vectors = [[unhexlify(x) for x in tv] for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:
            # Encrypt
            cipher = AES.new(key, AES.MODE_CCM, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)

            # Decrypt
            cipher = AES.new(key, AES.MODE_CCM, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, **extra_params):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._extra_params = extra_params
        self._id = "None"

    def setUp(self):

        def filter_tag(group):
            return group['tagSize'] // 8

        self.tv = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                               "aes_ccm_test.json",
                                               "Wycheproof AES CCM",
                                               group_tag={'tag_size': filter_tag})

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_encrypt(self, tv):
        self._id = "Wycheproof Encrypt CCM Test #" + str(tv.id)

        try:
            cipher = AES.new(tv.key, AES.MODE_CCM, tv.iv, mac_len=tv.tag_size,
                                **self._extra_params)
        except ValueError as e:
            if len(tv.iv) not in range(7, 13 + 1, 2) and "Length of parameter 'nonce'" in str(e):
                assert not tv.valid
                return
            if tv.tag_size not in range(4, 16 + 1, 2) and "Parameter 'mac_len'" in str(e):
                assert not tv.valid
                return
            raise e

        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(ct, tv.ct)
            self.assertEqual(tag, tv.tag)
            self.warn(tv)

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt CCM Test #" + str(tv.id)

        try:
            cipher = AES.new(tv.key, AES.MODE_CCM, tv.iv, mac_len=tv.tag_size,
                                **self._extra_params)
        except ValueError as e:
            if len(tv.iv) not in range(7, 13 + 1, 2) and "Length of parameter 'nonce'" in str(e):
                assert not tv.valid
                return
            if tv.tag_size not in range(4, 16 + 1, 2) and "Parameter 'mac_len'" in str(e):
                assert not tv.valid
                return
            raise e

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
        self._id = "Wycheproof Corrupt Decrypt CCM Test #" + str(tv.id)
        if len(tv.iv) not in range(7, 13 + 1, 2) or len(tv.ct) == 0:
            return
        cipher = AES.new(tv.key, AES.MODE_CCM, tv.iv, mac_len=tv.tag_size,
                            **self._extra_params)
        cipher.update(tv.aad)
        ct_corrupt = strxor(tv.ct, b"\x00" * (len(tv.ct) - 1) + b"\x01")
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct_corrupt, tv.tag)

    def runTest(self):

        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)
            self.test_corrupt_decrypt(tv)


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(CcmTests)
    tests += list_test_cases(CcmFSMTests)
    tests += [TestVectors()]
    tests += [TestVectorsWycheproof(wycheproof_warnings)]

    return tests


if __name__ == '__main__':
    def suite():
        unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
