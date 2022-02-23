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

from __future__ import print_function

import unittest
from binascii import unhexlify

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof

from Crypto.Util.py3compat import tobytes, bchr
from Crypto.Cipher import AES
from Crypto.Hash import SHAKE128, SHA256

from Crypto.Util.strxor import strxor


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class GcmTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_nonce(self):
        # Nonce is optional (a random one will be created)
        AES.new(self.key_128, AES.MODE_GCM)

        cipher = AES.new(self.key_128, AES.MODE_GCM, self.nonce_96)
        ct = cipher.encrypt(self.data)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertEqual(ct, cipher.encrypt(self.data))

    def test_nonce_must_be_bytes(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_GCM,
                          nonce=u'test12345678')

    def test_nonce_length(self):
        # nonce can be of any length (but not empty)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_GCM,
                          nonce=b"")

        for x in range(1, 128):
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=bchr(1) * x)
            cipher.encrypt(bchr(1))

    def test_block_size_128(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertEqual(cipher.block_size, AES.block_size)

    def test_nonce_attribute(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertEqual(cipher.nonce, self.nonce_96)

        # By default, a 15 bytes long nonce is randomly generated
        nonce1 = AES.new(self.key_128, AES.MODE_GCM).nonce
        nonce2 = AES.new(self.key_128, AES.MODE_GCM).nonce
        self.assertEqual(len(nonce1), 16)
        self.assertNotEqual(nonce1, nonce2)

    def test_unknown_parameters(self):
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_GCM,
                          self.nonce_96, 7)
        self.assertRaises(TypeError, AES.new, self.key_128, AES.MODE_GCM,
                          nonce=self.nonce_96, unknown=7)

        # But some are only known by the base cipher
        # (e.g. use_aesni consumed by the AES module)
        AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96,
                use_aesni=False)

    def test_null_encryption_decryption(self):
        for func in "encrypt", "decrypt":
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            result = getattr(cipher, func)(b"")
            self.assertEqual(result, b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_data_must_be_bytes(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, u'test1234567890-*')

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, u'test1234567890-*')

    def test_mac_len(self):
        # Invalid MAC length
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_GCM,
                          nonce=self.nonce_96, mac_len=3)
        self.assertRaises(ValueError, AES.new, self.key_128, AES.MODE_GCM,
                          nonce=self.nonce_96, mac_len=16+1)

        # Valid MAC length
        for mac_len in range(5, 16 + 1):
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96,
                             mac_len=mac_len)
            _, mac = cipher.encrypt_and_digest(self.data)
            self.assertEqual(len(mac), mac_len)

        # Default MAC length
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        _, mac = cipher.encrypt_and_digest(self.data)
        self.assertEqual(len(mac), 16)

    def test_invalid_mac(self):
        from Crypto.Util.strxor import strxor_c
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct, mac = cipher.encrypt_and_digest(self.data)

        invalid_mac = strxor_c(mac, 0x01)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct,
                          invalid_mac)

    def test_hex_mac(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        mac_hex = cipher.hexdigest()
        self.assertEqual(cipher.digest(), unhexlify(mac_hex))

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.hexverify(mac_hex)

    def test_message_chunks(self):
        # Validate that both associated data and plaintext/ciphertext
        # can be broken up in chunks of arbitrary length

        auth_data = get_tag_random("authenticated data", 127)
        plaintext = get_tag_random("plaintext", 127)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(auth_data)
        ciphertext, ref_mac = cipher.encrypt_and_digest(plaintext)

        def break_up(data, chunk_length):
            return [data[i:i+chunk_length] for i in range(0, len(data),
                    chunk_length)]

        # Encryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)

            for chunk in break_up(auth_data, chunk_length):
                cipher.update(chunk)
            pt2 = b""
            for chunk in break_up(ciphertext, chunk_length):
                pt2 += cipher.decrypt(chunk)
            self.assertEqual(plaintext, pt2)
            cipher.verify(ref_mac)

        # Decryption
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)

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
                          AES.MODE_GCM,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data)
        tag = cipher1.digest()

        cipher2 = AES.new(key_ba,
                          AES.MODE_GCM,
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
                          AES.MODE_GCM,
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
                          AES.MODE_GCM,
                          nonce=self.nonce_96)
        cipher1.update(self.data)
        ct = cipher1.encrypt(self.data)
        tag = cipher1.digest()

        cipher2 = AES.new(key_mv,
                          AES.MODE_GCM,
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
                          AES.MODE_GCM,
                          nonce=nonce_mv)
        key_mv[:3] = b"\xFF\xFF\xFF"
        nonce_mv[:3] = b"\xFF\xFF\xFF"
        cipher4.update(header_mv)
        header_mv[:3] = b"\xFF\xFF\xFF"
        pt_test = cipher4.decrypt_and_verify(memoryview(ct_test), memoryview(tag_test))

        self.assertEqual(self.data, pt_test)

    def test_output_param(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)
        tag = cipher.digest()

        output = bytearray(128)
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        res, tag_out = cipher.encrypt_and_digest(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        self.assertEqual(tag, tag_out)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        res = cipher.decrypt_and_verify(ct, tag, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

    def test_output_param_memoryview(self):

        pt = b'5' * 128
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)

        output = memoryview(bytearray(128))
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

    def test_output_param_neg(self):
        LEN_PT = 128

        pt = b'5' * LEN_PT
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(pt)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0' * LEN_PT)

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0' * LEN_PT)

        shorter_output = bytearray(LEN_PT - 1)
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


class GcmFSMTests(unittest.TestCase):

    key_128 = get_tag_random("key_128", 16)
    nonce_96 = get_tag_random("nonce_128", 12)
    data = get_tag_random("data", 128)

    def test_valid_init_encrypt_decrypt_digest_verify(self):
        # No authenticated data, fixed plaintext
        # Verify path INIT->ENCRYPT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()

        # Verify path INIT->DECRYPT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_update_digest_verify(self):
        # No plaintext, fixed authenticated data
        # Verify path INIT->UPDATE->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.verify(mac)

    def test_valid_full_path(self):
        # Fixed authenticated data, fixed plaintext
        # Verify path INIT->UPDATE->ENCRYPT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()

        # Verify path INIT->UPDATE->DECRYPT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_GCM,
                         nonce=self.nonce_96)
        cipher.update(self.data)
        cipher.decrypt(ct)
        cipher.verify(mac)

    def test_valid_init_digest(self):
        # Verify path INIT->DIGEST
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.digest()

    def test_valid_init_verify(self):
        # Verify path INIT->VERIFY
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        mac = cipher.digest()

        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.verify(mac)

    def test_valid_multiple_encrypt_or_decrypt(self):
        for method_name in "encrypt", "decrypt":
            for auth_data in (None, b"333", self.data,
                              self.data + b"3"):
                if auth_data is None:
                    assoc_len = None
                else:
                    assoc_len = len(auth_data)
                cipher = AES.new(self.key_128, AES.MODE_GCM,
                                 nonce=self.nonce_96)
                if auth_data is not None:
                    cipher.update(auth_data)
                method = getattr(cipher, method_name)
                method(self.data)
                method(self.data)
                method(self.data)
                method(self.data)

    def test_valid_multiple_digest_or_verify(self):
        # Multiple calls to digest
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        first_mac = cipher.digest()
        for x in range(4):
            self.assertEqual(first_mac, cipher.digest())

        # Multiple calls to verify
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        for x in range(5):
            cipher.verify(first_mac)

    def test_valid_encrypt_and_digest_decrypt_and_verify(self):
        # encrypt_and_digest
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        ct, mac = cipher.encrypt_and_digest(self.data)

        # decrypt_and_verify
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        cipher.update(self.data)
        pt = cipher.decrypt_and_verify(ct, mac)
        self.assertEqual(self.data, pt)

    def test_invalid_mixing_encrypt_decrypt(self):
        # Once per method, with or without assoc. data
        for method1_name, method2_name in (("encrypt", "decrypt"),
                                           ("decrypt", "encrypt")):
            for assoc_data_present in (True, False):
                cipher = AES.new(self.key_128, AES.MODE_GCM,
                                 nonce=self.nonce_96)
                if assoc_data_present:
                    cipher.update(self.data)
                getattr(cipher, method1_name)(self.data)
                self.assertRaises(TypeError, getattr(cipher, method2_name),
                                  self.data)

    def test_invalid_encrypt_or_update_after_digest(self):
        for method_name in "encrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.encrypt(self.data)
            cipher.digest()
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.encrypt_and_digest(self.data)

    def test_invalid_decrypt_or_update_after_verify(self):
        cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
        ct = cipher.encrypt(self.data)
        mac = cipher.digest()

        for method_name in "decrypt", "update":
            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.decrypt(ct)
            cipher.verify(mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)

            cipher = AES.new(self.key_128, AES.MODE_GCM, nonce=self.nonce_96)
            cipher.decrypt_and_verify(ct, mac)
            self.assertRaises(TypeError, getattr(cipher, method_name),
                              self.data)


class TestVectors(unittest.TestCase):
    """Class exercising the GCM test vectors found in
       http://csrc.nist.gov/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf"""

    # List of test vectors, each made up of:
    # - authenticated data
    # - plaintext
    # - ciphertext
    # - MAC
    # - AES key
    # - nonce
    test_vectors_hex = [
        (
            '',
            '',
            '',
            '58e2fccefa7e3061367f1d57a4e7455a',
            '00000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (
            '',
            '00000000000000000000000000000000',
            '0388dace60b6a392f328c2b971b2fe78',
            'ab6e47d42cec13bdf53a67b21257bddf',
            '00000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (
            '',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255',
            '42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e' +
            '21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985',
            '4d5c2af327cd64a62cf35abd2ba6fab4',
            'feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e' +
            '21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091',
            '5bc94fbc3221a5db94fae95ae7121a47',
            'feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '61353b4c2806934a777ff51fa22a4755699b2a714fcdc6f83766e5f97b6c7423' +
            '73806900e49f24b22b097544d4896b424989b5e1ebac0f07c23f4598',
            '3612d2e79e3b0785561be14aaca2fccb',
            'feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbad'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '8ce24998625615b603a033aca13fb894be9112a5c3a211a8ba262a3cca7e2ca7' +
            '01e4a9a4fba43c90ccdcb281d48c7c6fd62875d2aca417034c34aee5',
            '619cc5aefffe0bfa462af43c1699d050',
            'feffe9928665731c6d6a8f9467308308',
            '9313225df88406e555909c5aff5269aa' +
            '6a7a9538534f7da1e4c303d2a318a728c3c0c95156809539fcf0e2429a6b5254' +
            '16aedbf5a0de6a57a637b39b'
        ),
        (
            '',
            '',
            '',
            'cd33b28ac773f74ba00ed1f312572435',
            '000000000000000000000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (
            '',
            '00000000000000000000000000000000',
            '98e7247c07f0fe411c267e4384b0f600',
            '2ff58d80033927ab8ef4d4587514f0fb',
            '000000000000000000000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (
            '',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255',
            '3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c' +
            '7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710acade256',
            '9924a7c8587336bfb118024db8674a14',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c' +
            '7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710',
            '2519498e80f1478f37ba55bd6d27618c',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '0f10f599ae14a154ed24b36e25324db8c566632ef2bbb34f8347280fc4507057' +
            'fddc29df9a471f75c66541d4d4dad1c9e93a19a58e8b473fa0f062f7',
            '65dcc57fcf623a24094fcca40d3533f8',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c',
            'cafebabefacedbad'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            'd27e88681ce3243c4830165a8fdcf9ff1de9a1d8e6b447ef6ef7b79828666e45' +
            '81e79012af34ddd9e2f037589b292db3e67c036745fa22e7e9b7373b',
            'dcf566ff291c25bbb8568fc3d376a6d9',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c',
            '9313225df88406e555909c5aff5269aa' +
            '6a7a9538534f7da1e4c303d2a318a728c3c0c95156809539fcf0e2429a6b5254' +
            '16aedbf5a0de6a57a637b39b'
        ),
        (
            '',
            '',
            '',
            '530f8afbc74536b9a963b4f1c4cb738b',
            '0000000000000000000000000000000000000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (
            '',
            '00000000000000000000000000000000',
            'cea7403d4d606b6e074ec5d3baf39d18',
            'd0d1c8a799996bf0265b98b5d48ab919',
            '0000000000000000000000000000000000000000000000000000000000000000',
            '000000000000000000000000'
        ),
        (   '',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255',
            '522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa' +
            '8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad',
            'b094dac5d93471bdec1a502270e3cc6c',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa' +
            '8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662',
            '76fc6ece0f4e1768cddf8853bb2d551b',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbaddecaf888'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            'c3762df1ca787d32ae47c13bf19844cbaf1ae14d0b976afac52ff7d79bba9de0' +
            'feb582d33934a4f0954cc2363bc73f7862ac430e64abe499f47c9b1f',
            '3a337dbf46a792c45e454913fe2ea8f2',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308',
            'cafebabefacedbad'
        ),
        (
            'feedfacedeadbeeffeedfacedeadbeefabaddad2',
            'd9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72' +
            '1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39',
            '5a8def2f0c9e53f1f75d7853659e2a20eeb2b22aafde6419a058ab4f6f746bf4' +
            '0fc0c3b780f244452da3ebf1c5d82cdea2418997200ef82e44ae7e3f',
            'a44a8266ee1c8eb0c8b5d4cf5ae9f19a',
            'feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308',
            '9313225df88406e555909c5aff5269aa' +
            '6a7a9538534f7da1e4c303d2a318a728c3c0c95156809539fcf0e2429a6b5254' +
            '16aedbf5a0de6a57a637b39b'
        )
    ]

    test_vectors = [[unhexlify(x) for x in tv] for tv in test_vectors_hex]

    def runTest(self):
        for assoc_data, pt, ct, mac, key, nonce in self.test_vectors:

            # Encrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            ct2, mac2 = cipher.encrypt_and_digest(pt)
            self.assertEqual(ct, ct2)
            self.assertEqual(mac, mac2)

            # Decrypt
            cipher = AES.new(key, AES.MODE_GCM, nonce, mac_len=len(mac))
            cipher.update(assoc_data)
            pt2 = cipher.decrypt_and_verify(ct, mac)
            self.assertEqual(pt, pt2)


class TestVectorsGueronKrasnov(unittest.TestCase):
    """Class exercising the GCM test vectors found in
       'The fragility of AES-GCM authentication algorithm', Gueron, Krasnov
       https://eprint.iacr.org/2013/157.pdf"""

    def test_1(self):
        key = unhexlify("3da6c536d6295579c0959a7043efb503")
        iv  = unhexlify("2b926197d34e091ef722db94")
        aad = unhexlify("00000000000000000000000000000000" +
                        "000102030405060708090a0b0c0d0e0f" +
                        "101112131415161718191a1b1c1d1e1f" +
                        "202122232425262728292a2b2c2d2e2f" +
                        "303132333435363738393a3b3c3d3e3f")
        digest = unhexlify("69dd586555ce3fcc89663801a71d957b")

        cipher = AES.new(key, AES.MODE_GCM, iv).update(aad)
        self.assertEqual(digest, cipher.digest())

    def test_2(self):
        key = unhexlify("843ffcf5d2b72694d19ed01d01249412")
        iv  = unhexlify("dbcca32ebf9b804617c3aa9e")
        aad = unhexlify("00000000000000000000000000000000" +
                        "101112131415161718191a1b1c1d1e1f")
        pt  = unhexlify("000102030405060708090a0b0c0d0e0f" +
                        "101112131415161718191a1b1c1d1e1f" +
                        "202122232425262728292a2b2c2d2e2f" +
                        "303132333435363738393a3b3c3d3e3f" +
                        "404142434445464748494a4b4c4d4e4f")
        ct  = unhexlify("6268c6fa2a80b2d137467f092f657ac0" +
                        "4d89be2beaa623d61b5a868c8f03ff95" +
                        "d3dcee23ad2f1ab3a6c80eaf4b140eb0" +
                        "5de3457f0fbc111a6b43d0763aa422a3" +
                        "013cf1dc37fe417d1fbfc449b75d4cc5")
        digest = unhexlify("3b629ccfbc1119b7319e1dce2cd6fd6d")

        cipher = AES.new(key, AES.MODE_GCM, iv).update(aad)
        ct2, digest2 = cipher.encrypt_and_digest(pt)

        self.assertEqual(ct, ct2)
        self.assertEqual(digest, digest2)


class NISTTestVectorsGCM(unittest.TestCase):

    def __init__(self, a):
        self.use_clmul = True
        unittest.TestCase.__init__(self, a)


class NISTTestVectorsGCM_no_clmul(unittest.TestCase):

    def __init__(self, a):
        self.use_clmul = False
        unittest.TestCase.__init__(self, a)


test_vectors_nist = load_test_vectors(
                        ("Cipher", "AES"),
                        "gcmDecrypt128.rsp",
                        "GCM decrypt",
                        {"count": lambda x: int(x)}) or []

test_vectors_nist += load_test_vectors(
                        ("Cipher", "AES"),
                        "gcmEncryptExtIV128.rsp",
                        "GCM encrypt",
                        {"count": lambda x: int(x)}) or []

for idx, tv in enumerate(test_vectors_nist):

    # The test vector file contains some directive lines
    if isinstance(tv, str):
        continue

    def single_test(self, tv=tv):

        self.description = tv.desc
        cipher = AES.new(tv.key, AES.MODE_GCM, nonce=tv.iv,
                             mac_len=len(tv.tag), use_clmul=self.use_clmul)
        cipher.update(tv.aad)
        if "FAIL" in tv.others:
            self.assertRaises(ValueError, cipher.decrypt_and_verify,
                                  tv.ct, tv.tag)
        else:
            pt = cipher.decrypt_and_verify(tv.ct, tv.tag)
            self.assertEqual(pt, tv.pt)

    setattr(NISTTestVectorsGCM, "test_%d" % idx, single_test)
    setattr(NISTTestVectorsGCM_no_clmul, "test_%d" % idx, single_test)


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
                                               "aes_gcm_test.json",
                                               "Wycheproof GCM",
                                               group_tag={'tag_size': filter_tag})

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_encrypt(self, tv):
        self._id = "Wycheproof Encrypt GCM Test #" + str(tv.id)

        try:
            cipher = AES.new(tv.key, AES.MODE_GCM, tv.iv, mac_len=tv.tag_size,
                        **self._extra_params)
        except ValueError as e:
            if len(tv.iv) == 0 and "Nonce cannot be empty" in str(e):
                return
            raise e

        cipher.update(tv.aad)
        ct, tag = cipher.encrypt_and_digest(tv.msg)
        if tv.valid:
            self.assertEqual(ct, tv.ct)
            self.assertEqual(tag, tv.tag)
            self.warn(tv)

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt GCM Test #" + str(tv.id)

        try:
            cipher = AES.new(tv.key, AES.MODE_GCM, tv.iv, mac_len=tv.tag_size,
                        **self._extra_params)
        except ValueError as e:
            if len(tv.iv) == 0 and "Nonce cannot be empty" in str(e):
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
        self._id = "Wycheproof Corrupt Decrypt GCM Test #" + str(tv.id)
        if len(tv.iv) == 0 or len(tv.ct) < 1:
            return
        cipher = AES.new(tv.key, AES.MODE_GCM, tv.iv, mac_len=tv.tag_size,
                    **self._extra_params)
        cipher.update(tv.aad)
        ct_corrupt = strxor(tv.ct, b"\x00" * (len(tv.ct) - 1) + b"\x01")
        self.assertRaises(ValueError, cipher.decrypt_and_verify, ct_corrupt, tv.tag)

    def runTest(self):

        for tv in self.tv:
            self.test_encrypt(tv)
            self.test_decrypt(tv)
            self.test_corrupt_decrypt(tv)


class TestVariableLength(unittest.TestCase):

    def __init__(self, **extra_params):
        unittest.TestCase.__init__(self)
        self._extra_params = extra_params

    def runTest(self):
        key = b'0' * 16
        h = SHA256.new()

        for length in range(160):
            nonce = '{0:04d}'.format(length).encode('utf-8')
            data = bchr(length) * length
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, **self._extra_params)
            ct, tag = cipher.encrypt_and_digest(data)
            h.update(ct)
            h.update(tag)

        self.assertEqual(h.hexdigest(), "7b7eb1ffbe67a2e53a912067c0ec8e62ebc7ce4d83490ea7426941349811bdf4")


def get_tests(config={}):
    from Crypto.Util import _cpu_features

    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(GcmTests)
    tests += list_test_cases(GcmFSMTests)
    tests += [TestVectors()]
    tests += [TestVectorsWycheproof(wycheproof_warnings)]
    tests += list_test_cases(TestVectorsGueronKrasnov)
    tests += [TestVariableLength()]
    if config.get('slow_tests'):
        tests += list_test_cases(NISTTestVectorsGCM)

    if _cpu_features.have_clmul():
        tests += [TestVectorsWycheproof(wycheproof_warnings, use_clmul=False)]
        tests += [TestVariableLength(use_clmul=False)]
        if config.get('slow_tests'):
            tests += list_test_cases(NISTTestVectorsGCM_no_clmul)
    else:
        print("Skipping test of PCLMULDQD in AES GCM")

    return tests


if __name__ == '__main__':
    def suite():
        unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
