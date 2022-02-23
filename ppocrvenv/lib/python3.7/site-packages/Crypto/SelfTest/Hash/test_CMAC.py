#
#  SelfTest/Hash/CMAC.py: Self-test for the CMAC module
#
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

"""Self-test suite for Crypto.Hash.CMAC"""

import json
import unittest
from binascii import unhexlify

from Crypto.Util.py3compat import tobytes

from Crypto.Hash import CMAC
from Crypto.Cipher import AES, DES3
from Crypto.Hash import SHAKE128

from Crypto.Util.strxor import strxor

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors_wycheproof

# This is a list of (key, data, result, description, module) tuples.
test_data = [

    ## Test vectors from RFC 4493 ##
    ## The are also in NIST SP 800 38B D.2 ##
    (   '2b7e151628aed2a6abf7158809cf4f3c',
        '',
        'bb1d6929e95937287fa37d129b756746',
        'RFC 4493 #1',
        AES
    ),

    (   '2b7e151628aed2a6abf7158809cf4f3c',
        '6bc1bee22e409f96e93d7e117393172a',
        '070a16b46b4d4144f79bdd9dd04a287c',
        'RFC 4493 #2',
        AES
    ),

    (   '2b7e151628aed2a6abf7158809cf4f3c',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411',
        'dfa66747de9ae63030ca32611497c827',
        'RFC 4493 #3',
        AES
    ),

    (   '2b7e151628aed2a6abf7158809cf4f3c',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411e5fbc1191a0a52ef'+
        'f69f2445df4f9b17ad2b417be66c3710',
        '51f0bebf7e3b9d92fc49741779363cfe',
        'RFC 4493 #4',
        AES
    ),

    ## The rest of Appendix D of NIST SP 800 38B
    ## was not totally correct.
    ## Values in Examples 14, 15, 18, and 19 were wrong.
    ## The updated test values are published in:
    ## http://csrc.nist.gov/publications/nistpubs/800-38B/Updated_CMAC_Examples.pdf

    (   '8e73b0f7da0e6452c810f32b809079e5'+
        '62f8ead2522c6b7b',
        '',
        'd17ddf46adaacde531cac483de7a9367',
        'NIST SP 800 38B D.2 Example 5',
        AES
    ),

    (   '8e73b0f7da0e6452c810f32b809079e5'+
        '62f8ead2522c6b7b',
        '6bc1bee22e409f96e93d7e117393172a',
        '9e99a7bf31e710900662f65e617c5184',
        'NIST SP 800 38B D.2 Example 6',
        AES
    ),

    (   '8e73b0f7da0e6452c810f32b809079e5'+
        '62f8ead2522c6b7b',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411',
        '8a1de5be2eb31aad089a82e6ee908b0e',
        'NIST SP 800 38B D.2 Example 7',
        AES
    ),

    (   '8e73b0f7da0e6452c810f32b809079e5'+
        '62f8ead2522c6b7b',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411e5fbc1191a0a52ef'+
        'f69f2445df4f9b17ad2b417be66c3710',
        'a1d5df0eed790f794d77589659f39a11',
        'NIST SP 800 38B D.2 Example 8',
        AES
    ),

    (   '603deb1015ca71be2b73aef0857d7781'+
        '1f352c073b6108d72d9810a30914dff4',
        '',
        '028962f61b7bf89efc6b551f4667d983',
        'NIST SP 800 38B D.3 Example 9',
        AES
    ),

    (   '603deb1015ca71be2b73aef0857d7781'+
        '1f352c073b6108d72d9810a30914dff4',
        '6bc1bee22e409f96e93d7e117393172a',
        '28a7023f452e8f82bd4bf28d8c37c35c',
        'NIST SP 800 38B D.3 Example 10',
        AES
    ),

    (   '603deb1015ca71be2b73aef0857d7781'+
        '1f352c073b6108d72d9810a30914dff4',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411',
        'aaf3d8f1de5640c232f5b169b9c911e6',
        'NIST SP 800 38B D.3 Example 11',
        AES
    ),

    (   '603deb1015ca71be2b73aef0857d7781'+
        '1f352c073b6108d72d9810a30914dff4',
        '6bc1bee22e409f96e93d7e117393172a'+
        'ae2d8a571e03ac9c9eb76fac45af8e51'+
        '30c81c46a35ce411e5fbc1191a0a52ef'+
        'f69f2445df4f9b17ad2b417be66c3710',
        'e1992190549f6ed5696a2c056c315410',
        'NIST SP 800 38B D.3 Example 12',
        AES
    ),

    (   '8aa83bf8cbda1062'+
        '0bc1bf19fbb6cd58'+
        'bc313d4a371ca8b5',
        '',
        'b7a688e122ffaf95',
        'NIST SP 800 38B D.4 Example 13',
        DES3
    ),

    (   '8aa83bf8cbda1062'+
        '0bc1bf19fbb6cd58'+
        'bc313d4a371ca8b5',
        '6bc1bee22e409f96',
        '8e8f293136283797',
        'NIST SP 800 38B D.4 Example 14',
        DES3
    ),

    (   '8aa83bf8cbda1062'+
        '0bc1bf19fbb6cd58'+
        'bc313d4a371ca8b5',
        '6bc1bee22e409f96'+
        'e93d7e117393172a'+
        'ae2d8a57',
        '743ddbe0ce2dc2ed',
        'NIST SP 800 38B D.4 Example 15',
        DES3
    ),

    (   '8aa83bf8cbda1062'+
        '0bc1bf19fbb6cd58'+
        'bc313d4a371ca8b5',
        '6bc1bee22e409f96'+
        'e93d7e117393172a'+
        'ae2d8a571e03ac9c'+
        '9eb76fac45af8e51',
        '33e6b1092400eae5',
        'NIST SP 800 38B D.4 Example 16',
        DES3
    ),

    (   '4cf15134a2850dd5'+
        '8a3d10ba80570d38',
        '',
        'bd2ebf9a3ba00361',
        'NIST SP 800 38B D.7 Example 17',
        DES3
    ),

    (   '4cf15134a2850dd5'+
        '8a3d10ba80570d38',
        '6bc1bee22e409f96',
        '4ff2ab813c53ce83',
        'NIST SP 800 38B D.7 Example 18',
        DES3
    ),

    (   '4cf15134a2850dd5'+
        '8a3d10ba80570d38',
        '6bc1bee22e409f96'+
        'e93d7e117393172a'+
        'ae2d8a57',
        '62dd1b471902bd4e',
        'NIST SP 800 38B D.7 Example 19',
        DES3
    ),

    (   '4cf15134a2850dd5'+
        '8a3d10ba80570d38',
        '6bc1bee22e409f96'+
        'e93d7e117393172a'+
        'ae2d8a571e03ac9c'+
        '9eb76fac45af8e51',
        '31b1e431dabc4eb8',
        'NIST SP 800 38B D.7 Example 20',
        DES3
    ),

]


def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)


class TestCMAC(unittest.TestCase):

    def test_internal_caching(self):
        """Verify that internal caching is implemented correctly"""

        data_to_mac = get_tag_random("data_to_mac", 128)
        key = get_tag_random("key", 16)
        ref_mac = CMAC.new(key, msg=data_to_mac, ciphermod=AES).digest()

        # Break up in chunks of different length
        # The result must always be the same
        for chunk_length in 1, 2, 3, 7, 10, 13, 16, 40, 80, 128:

            chunks = [data_to_mac[i:i+chunk_length] for i in
                      range(0, len(data_to_mac), chunk_length)]

            mac = CMAC.new(key, ciphermod=AES)
            for chunk in chunks:
                mac.update(chunk)
            self.assertEqual(ref_mac, mac.digest())

    def test_update_after_digest(self):
        msg = b"rrrrttt"
        key = b"4" * 16

        # Normally, update() cannot be done after digest()
        h = CMAC.new(key, msg[:4], ciphermod=AES)
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = CMAC.new(key, msg, ciphermod=AES).digest()

        # With the proper flag, it is allowed
        h2 = CMAC.new(key, msg[:4], ciphermod=AES, update_after_digest=True)
        self.assertEqual(h2.digest(), dig1)
        # ... and the subsequent digest applies to the entire message
        # up to that point
        h2.update(msg[4:])
        self.assertEqual(h2.digest(), dig2)


class ByteArrayTests(unittest.TestCase):

    def runTest(self):

        key = b"0" * 16
        data = b"\x00\x01\x02"

        # Data and key can be a bytearray (during initialization)
        key_ba = bytearray(key)
        data_ba = bytearray(data)

        h1 = CMAC.new(key, data, ciphermod=AES)
        h2 = CMAC.new(key_ba, data_ba, ciphermod=AES)
        key_ba[:1] = b'\xFF'
        data_ba[:1] = b'\xFF'
        self.assertEqual(h1.digest(), h2.digest())

        # Data can be a bytearray (during operation)
        key_ba = bytearray(key)
        data_ba = bytearray(data)

        h1 = CMAC.new(key, ciphermod=AES)
        h2 = CMAC.new(key, ciphermod=AES)
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xFF'
        self.assertEqual(h1.digest(), h2.digest())


class MemoryViewTests(unittest.TestCase):

    def runTest(self):

        key = b"0" * 16
        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in (get_mv_ro, get_mv_rw):

            # Data and key can be a memoryview (during initialization)
            key_mv = get_mv(key)
            data_mv = get_mv(data)

            h1 = CMAC.new(key, data, ciphermod=AES)
            h2 = CMAC.new(key_mv, data_mv, ciphermod=AES)
            if not data_mv.readonly:
                key_mv[:1] = b'\xFF'
                data_mv[:1] = b'\xFF'
            self.assertEqual(h1.digest(), h2.digest())

            # Data can be a memoryview (during operation)
            data_mv = get_mv(data)

            h1 = CMAC.new(key, ciphermod=AES)
            h2 = CMAC.new(key, ciphermod=AES)
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'
            self.assertEqual(h1.digest(), h2.digest())


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = "None"

    def setUp(self):

        def filter_tag(group):
            return group['tagSize'] // 8

        self.tv = load_test_vectors_wycheproof(("Hash", "wycheproof"),
                                               "aes_cmac_test.json",
                                               "Wycheproof CMAC",
                                               group_tag={'tag_size': filter_tag})

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_create_mac(self, tv):
        self._id = "Wycheproof MAC creation Test #" + str(tv.id)

        try:
            tag = CMAC.new(tv.key, tv.msg, ciphermod=AES, mac_len=tv.tag_size).digest()
        except ValueError as e:
            if len(tv.key) not in (16, 24, 32) and "key length" in str(e):
                return
            raise e
        if tv.valid:
            self.assertEqual(tag, tv.tag)
            self.warn(tv)

    def test_verify_mac(self, tv):
        self._id = "Wycheproof MAC verification Test #" + str(tv.id)

        try:
            mac = CMAC.new(tv.key, tv.msg, ciphermod=AES, mac_len=tv.tag_size)
        except ValueError as e:
            if len(tv.key) not in (16, 24, 32) and "key length" in str(e):
                return
            raise e
        try:
            mac.verify(tv.tag)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):

        for tv in self.tv:
            self.test_create_mac(tv)
            self.test_verify_mac(tv)


def get_tests(config={}):
    global test_data
    import types
    from .common import make_mac_tests

    wycheproof_warnings = config.get('wycheproof_warnings')

    # Add new() parameters to the back of each test vector
    params_test_data = []
    for row in test_data:
        t = list(row)
        t[4] = dict(ciphermod=t[4])
        params_test_data.append(t)

    tests = make_mac_tests(CMAC, "CMAC", params_test_data)
    tests.append(ByteArrayTests())
    tests.append(list_test_cases(TestCMAC))
    tests.append(MemoryViewTests())
    tests += [ TestVectorsWycheproof(wycheproof_warnings) ]
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
