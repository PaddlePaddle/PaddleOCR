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

import os
import re
import unittest
from binascii import hexlify, unhexlify

from Crypto.Util.py3compat import b, tobytes, bchr
from Crypto.Util.strxor import strxor_c
from Crypto.Util.number import long_to_bytes
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Cipher import AES
from Crypto.Hash import SHAKE128


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class OcbTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        pt = get_tag_random("plaintext", 16 * 100)
        ct, mac = cipher.encrypt_and_digest(pt)

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        pt2 = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        # Nonce is optional
        AES.new(self.key_128, AES.MODE_OCB)

        cipher = AES.new(self.key_128, AES.MODE_OCB, self.nonce_96)
        ct = cipher.encrypt(self.data)

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=u'test12345678')

    def test_nonce_length(self):
        # nonce cannot be empty
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=b(""))

        # nonce can be up to 15 bytes long
        for length in range(1, 16):
            AES.new(self.key_128, AES.MODE_OCB, nonce=self.data[:length])

        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=self.data)

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)

        # By default, a 15 bytes long nonce is randomly generated
        nonce1 = AES.new(self.key_128, AES.MODE_OCB).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_OCB).nonce
        self.assertEqual(len(nonce1), 15)
        self.assertNotEqual(nonce1, nonce2)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)

        # By default, a 15 bytes long nonce is randomly generated
        nonce1 = AES.new(self.key_128, AES.MODE_OCB).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_OCB).nonce
        self.assertEqual(len(nonce1), 15)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB,
                          self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=self.nonce_96, unknown=7)

        # But some are only known by the base cipher
        # (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96,
                use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            result = getattr(cipher, func)(b(""))
            self.assertEqual(result, b(""))

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.encrypt(b("xyz"))
        self.assertRaises(TypeError, cipher.decrypt, b("xyz"))

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.decrypt(b("xyz"))
        self.assertRaises(TypeError, cipher.encrypt, b("xyz"))

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        # Invalid MAC length
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=self.nonce_96, mac_len=7)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_OCB,
                          nonce=self.nonce_96, mac_len=16+1)

        # Valid MAC length
        for mac_len in range(8, 16 + 1):
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96,
                             mac_len=mac_len)
            _, mac = cipher.encrypt_and_digest(self.data)
            self.assertEqual(len(mac), mac_len)

        # Default MAC length
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Crypto.Util.strxor import strxor_c
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)

        invalid_mac = strxor_c(mac, 0x01)

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct,
                          invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_message_chunks(self):
        # Validate that both associated data and plaintext/ciphertext
        # can be broken up in chunks of arbitrary length

        auth_data = get_tag_random("authenticated data", 127)
        plaintext = get_tag_random("plaintext", 127)

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i+chunk_length] for i in range(0, len(data),
                    chunk_length)]

        # Encryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b("")
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            pt2 += cipher.decrypt()
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)

        # Decryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            ct2 = b("")
            for chunk in break_up(plaintext, chunk_length):
                ct2 += cipher.encrypt(chunk)
            ct2 += cipher.encrypt()
            self.assertEqual(ciphertext, ct2)
            self.assertEqual(cipher.digest(), ref_mac)

    def test_bytearray(self):

        # Encrypt
        key_ba = bytearray(self.key_128)
        nonce_ba = bytearray(self.nonce_96)
        header_ba = bytearray(self.data)
        data_ba = bytearray(self.data)

        cipher1 = AES.new(self.key_128,
                          AES.MODE_OCB,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data) + cipher1.encrypt()
        tag = cipher1.digest()

        cipher2 = AES.new(key_ba,
                          AES.MODE_OCB,
                          nonce=nonce_ba)
        key_ba[:3] = b"\xFF\xFF\xFF"
        nonce_ba[:3] = b"\xFF\xFF\xFF"
        cipher2.update(header_ba)
        header_ba[:3] = b"\xFF\xFF\xFF"
        ct_test = cipher2.encrypt(data_ba) + cipher2.encrypt()
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
                          AES.MODE_OCB,
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
                          AES.MODE_OCB,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data) + cipher1.encrypt()
        tag = cipher1.digest()

        cipher2 = AES.new(key_mv,
                          AES.MODE_OCB,
                          nonce=nonce_mv)
        key_mv[:3] = b"\xFF\xFF\xFF"
        nonce_mv[:3] = b"\xFF\xFF\xFF"
        cipher2.update(header_mv)
        header_mv[:3] = b"\xFF\xFF\xFF"
        ct_test = cipher2.encrypt(data_mv) + cipher2.encrypt()
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
                          AES.MODE_OCB,
                          nonce=nonce_mv)
        key_mv[:3] = b"\xFF\xFF\xFF"
        nonce_mv[:3] = b"\xFF\xFF\xFF"
        cipher4.update(header_mv)
        header_mv[:3] = b"\xFF\xFF\xFF"
        pt_test = cipher4.decrypt_and_verify(memoryview(ct_test), memoryview(tag_test))

        self.assertEqual(self.data, pt_test)


class OcbFSMTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        # No authenticated data, fixed plaintext
        # Verify path INIT->ENCRYPT->ENCRYPT(NONE)->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        ct += cipher.encrypt()
        mac = cipher.digest()

        # Verify path INIT->DECRYPT->DECRYPT(NONCE)->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.decrypt()
        cipher.verify(mac)

    def test_invalid_init_encrypt_decrypt_digest_verify(self):
        # No authenticated data, fixed plaintext
        # Verify path INIT->ENCRYPT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        self.assertRaises(TypeError, cipher.digest)

        # Verify path INIT->DECRYPT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.decrypt(ct)
        self.assertRaises(TypeError, cipher.verify)

    def test_valid_init_update_digest_verify(self):
        # No plaintext, fixed authenticated data
        # Verify path INIT->UPDATE->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.verify(mac)

    def test_valid_full_path(self):
        # Fixed authenticated data, fixed plaintext
        # Verify path INIT->UPDATE->ENCRYPT->ENCRYPT(NONE)->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        ct = cipher.encrypt(self.data)
        ct += cipher.encrypt()
        mac = cipher.digest()

        # Verify path INIT->UPDATE->DECRYPT->DECRYPT(NONE)->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.decrypt(ct)
        cipher.decrypt()
        cipher.verify(mac)

    def test_invalid_encrypt_after_final(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.encrypt(self.data)
        cipher.encrypt()
        self.assertRaises(TypeError, cipher.encrypt, self.data)

    def test_invalid_decrypt_after_final(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.decrypt(self.data)
        cipher.decrypt()
        self.assertRaises(TypeError, cipher.decrypt, self.data)

    def test_valid_init_digest(self):
        # Verify path INIT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        # Verify path INIT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        mac = cipher.digest()

        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        for method_name in "encrypt", "decrypt":
            for auth_data in (None, b("333"), self.data,
                              self.data + b("3")):
                if auth_data is None:
                    assoc_len = None
                else:
                    assoc_len = len(auth_data)
                cipher = AES.new(self.key_128, AES.MODE_OCB,
                                 nonce=self.nonce_96)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data)
                method(self.data)
                method(self.data)
                method(self.data)
                method()

    def test_valid_multiple_digest_or_verify(self):
        # Multiple calls to digest
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())

        # Multiple calls to verify
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        # encrypt_and_digest
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)

        # decrypt_and_verify
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_mixing_encrypt_decrypt(self):
        # Once per method, with or without assoc. data
        for method1_name, method2_name in (("encrypt", "decrypt"),
                                           ("decrypt", "encrypt")):
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_OCB,
                                 nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data)
                getattr(cipher, method1_name)(self.data)
                self.assertRaises(TypeError, getattr(cipher, method2_name),
                                  self.data)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in "encrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            cipher.encrypt(self.data)
            cipher.encrypt()
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        ct += cipher.encrypt()
        mac = cipher.digest()

        for method_name in "decrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.decrypt()
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_OCB, nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)


class OcbRfc7253Test(unittest.TestCase):

    # Tuple with
    # - nonce
    # - authenticated data
    # - plaintext
    # - ciphertext and 16 byte MAC tag
    tv1_key = "000102030405060708090A0B0C0D0E0F"
    tv1 = (
            (
                "BBAA99887766554433221100",
                "",
                "",
                "785407BFFFC8AD9EDCC5520AC9111EE6"
            ),
            (
                "BBAA99887766554433221101",
                "0001020304050607",
                "0001020304050607",
                "6820B3657B6F615A5725BDA0D3B4EB3A257C9AF1F8F03009"
            ),
            (
                "BBAA99887766554433221102",
                "0001020304050607",
                "",
                "81017F8203F081277152FADE694A0A00"
            ),
            (
                "BBAA99887766554433221103",
                "",
                "0001020304050607",
                "45DD69F8F5AAE72414054CD1F35D82760B2CD00D2F99BFA9"
            ),
            (
                "BBAA99887766554433221104",
                "000102030405060708090A0B0C0D0E0F",
                "000102030405060708090A0B0C0D0E0F",
                "571D535B60B277188BE5147170A9A22C3AD7A4FF3835B8C5"
                "701C1CCEC8FC3358"
            ),
            (
                "BBAA99887766554433221105",
                "000102030405060708090A0B0C0D0E0F",
                "",
                "8CF761B6902EF764462AD86498CA6B97"
            ),
            (
                "BBAA99887766554433221106",
                "",
                "000102030405060708090A0B0C0D0E0F",
                "5CE88EC2E0692706A915C00AEB8B2396F40E1C743F52436B"
                "DF06D8FA1ECA343D"
            ),
            (
                "BBAA99887766554433221107",
                "000102030405060708090A0B0C0D0E0F1011121314151617",
                "000102030405060708090A0B0C0D0E0F1011121314151617",
                "1CA2207308C87C010756104D8840CE1952F09673A448A122"
                "C92C62241051F57356D7F3C90BB0E07F"
            ),
            (
                "BBAA99887766554433221108",
                "000102030405060708090A0B0C0D0E0F1011121314151617",
                "",
                "6DC225A071FC1B9F7C69F93B0F1E10DE"
            ),
            (
                "BBAA99887766554433221109",
                "",
                "000102030405060708090A0B0C0D0E0F1011121314151617",
                "221BD0DE7FA6FE993ECCD769460A0AF2D6CDED0C395B1C3C"
                "E725F32494B9F914D85C0B1EB38357FF"
            ),
            (
                "BBAA9988776655443322110A",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F",
                "BD6F6C496201C69296C11EFD138A467ABD3C707924B964DE"
                "AFFC40319AF5A48540FBBA186C5553C68AD9F592A79A4240"
            ),
            (
                "BBAA9988776655443322110B",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F",
                "",
                "FE80690BEE8A485D11F32965BC9D2A32"
            ),
            (
                "BBAA9988776655443322110C",
                "",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F",
                "2942BFC773BDA23CABC6ACFD9BFD5835BD300F0973792EF4"
                "6040C53F1432BCDFB5E1DDE3BC18A5F840B52E653444D5DF"
            ),
            (
                "BBAA9988776655443322110D",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F2021222324252627",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F2021222324252627",
                "D5CA91748410C1751FF8A2F618255B68A0A12E093FF45460"
                "6E59F9C1D0DDC54B65E8628E568BAD7AED07BA06A4A69483"
                "A7035490C5769E60"
            ),
            (
                "BBAA9988776655443322110E",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F2021222324252627",
                "",
                "C5CD9D1850C141E358649994EE701B68"
            ),
            (
                "BBAA9988776655443322110F",
                "",
                "000102030405060708090A0B0C0D0E0F1011121314151617"
                "18191A1B1C1D1E1F2021222324252627",
                "4412923493C57D5DE0D700F753CCE0D1D2D95060122E9F15"
                "A5DDBFC5787E50B5CC55EE507BCB084E479AD363AC366B95"
                "A98CA5F3000B1479"
            )
        )

    # Tuple with
    # - key
    # - nonce
    # - authenticated data
    # - plaintext
    # - ciphertext and 12 byte MAC tag
    tv2 = (
        "0F0E0D0C0B0A09080706050403020100",
        "BBAA9988776655443322110D",
        "000102030405060708090A0B0C0D0E0F1011121314151617"
        "18191A1B1C1D1E1F2021222324252627",
        "000102030405060708090A0B0C0D0E0F1011121314151617"
        "18191A1B1C1D1E1F2021222324252627",
        "1792A4E31E0755FB03E31B22116E6C2DDF9EFD6E33D536F1"
        "A0124B0A55BAE884ED93481529C76B6AD0C515F4D1CDD4FD"
        "AC4F02AA"
        )

    # Tuple with
    # - key length
    # - MAC tag length
    # - Expected output
    tv3 = (
        (128, 128, "67E944D23256C5E0B6C61FA22FDF1EA2"),
        (192, 128, "F673F2C3E7174AAE7BAE986CA9F29E17"),
        (256, 128, "D90EB8E9C977C88B79DD793D7FFA161C"),
        (128, 96,  "77A3D8E73589158D25D01209"),
        (192, 96,  "05D56EAD2752C86BE6932C5E"),
        (256, 96,  "5458359AC23B0CBA9E6330DD"),
        (128, 64,  "192C9B7BD90BA06A"),
        (192, 64,  "0066BC6E0EF34E24"),
        (256, 64,  "7D4EA5D445501CBE"),
    )

    def test1(self):
        key = unhexlify(b(self.tv1_key))
        for tv in self.tv1:
            nonce, aad, pt, ct = [ unhexlify(b(x)) for x in tv ]
            ct, mac_tag = ct[:-16], ct[-16:]

            cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
            cipher.update(aad)
            ct2 = cipher.encrypt(pt) + cipher.encrypt()
            self.assertEqual(ct, ct2)
            self.assertEqual(mac_tag, cipher.digest())

            cipher = AES.new(key, AES.MODE_OCB, nonce=nonce)
            cipher.update(aad)
            pt2 = cipher.decrypt(ct) + cipher.decrypt()
            self.assertEqual(pt, pt2)
            cipher.verify(mac_tag)

    def test2(self):

        key, nonce, aad, pt, ct = [ unhexlify(b(x)) for x in self.tv2 ]
        ct, mac_tag = ct[:-12], ct[-12:]

        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce, mac_len=12)
        cipher.update(aad)
        ct2 = cipher.encrypt(pt) + cipher.encrypt()
        self.assertEqual(ct, ct2)
        self.assertEqual(mac_tag, cipher.digest())

        cipher = AES.new(key, AES.MODE_OCB, nonce=nonce, mac_len=12)
        cipher.update(aad)
        pt2 = cipher.decrypt(ct) + cipher.decrypt()
        self.assertEqual(pt, pt2)
        cipher.verify(mac_tag)

    def test3(self):

        for keylen, taglen, result in self.tv3:

            key = bchr(0) * (keylen // 8 - 1) + bchr(taglen)
            C = b("")

            for i in range(128):
                S = bchr(0) * i

                N = long_to_bytes(3 * i + 1, 12)
                cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
                cipher.update(S)
                C += cipher.encrypt(S) + cipher.encrypt() + cipher.digest()

                N = long_to_bytes(3 * i + 2, 12)
                cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
                C += cipher.encrypt(S) + cipher.encrypt() + cipher.digest()

                N = long_to_bytes(3 * i + 3, 12)
                cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
                cipher.update(S)
                C += cipher.encrypt() + cipher.digest()

            N = long_to_bytes(385, 12)
            cipher = AES.new(key, AES.MODE_OCB, nonce=N, mac_len=taglen // 8)
            cipher.update(C)
            result2 = cipher.encrypt() + cipher.digest()
            self.assertEqual(unhexlify(b(result)), result2)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(OcbTests)
    tests += list_test_cases(OcbFSMTests)
    tests += list_test_cases(OcbRfc7253Test)
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
