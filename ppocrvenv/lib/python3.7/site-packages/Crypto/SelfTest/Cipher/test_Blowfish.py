# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/test_Blowfish.py: Self-test for the Blowfish cipher
#
# Written in 2008 by Dwayne C. Litzenberger <dlitz@dlitz.net>
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

"""Self-test suite for Crypto.Cipher.Blowfish"""

import unittest

from Crypto.Util.py3compat import bchr

from Crypto.Cipher import Blowfish

# This is a list of (plaintext, ciphertext, key) tuples.
test_data = [
    # Test vectors from http://www.schneier.com/code/vectors.txt
    ('0000000000000000', '4ef997456198dd78', '0000000000000000'),
    ('ffffffffffffffff', '51866fd5b85ecb8a', 'ffffffffffffffff'),
    ('1000000000000001', '7d856f9a613063f2', '3000000000000000'),
    ('1111111111111111', '2466dd878b963c9d', '1111111111111111'),
    ('1111111111111111', '61f9c3802281b096', '0123456789abcdef'),
    ('0123456789abcdef', '7d0cc630afda1ec7', '1111111111111111'),
    ('0000000000000000', '4ef997456198dd78', '0000000000000000'),
    ('0123456789abcdef', '0aceab0fc6a0a28d', 'fedcba9876543210'),
    ('01a1d6d039776742', '59c68245eb05282b', '7ca110454a1a6e57'),
    ('5cd54ca83def57da', 'b1b8cc0b250f09a0', '0131d9619dc1376e'),
    ('0248d43806f67172', '1730e5778bea1da4', '07a1133e4a0b2686'),
    ('51454b582ddf440a', 'a25e7856cf2651eb', '3849674c2602319e'),
    ('42fd443059577fa2', '353882b109ce8f1a', '04b915ba43feb5b6'),
    ('059b5e0851cf143a', '48f4d0884c379918', '0113b970fd34f2ce'),
    ('0756d8e0774761d2', '432193b78951fc98', '0170f175468fb5e6'),
    ('762514b829bf486a', '13f04154d69d1ae5', '43297fad38e373fe'),
    ('3bdd119049372802', '2eedda93ffd39c79', '07a7137045da2a16'),
    ('26955f6835af609a', 'd887e0393c2da6e3', '04689104c2fd3b2f'),
    ('164d5e404f275232', '5f99d04f5b163969', '37d06bb516cb7546'),
    ('6b056e18759f5cca', '4a057a3b24d3977b', '1f08260d1ac2465e'),
    ('004bd6ef09176062', '452031c1e4fada8e', '584023641aba6176'),
    ('480d39006ee762f2', '7555ae39f59b87bd', '025816164629b007'),
    ('437540c8698f3cfa', '53c55f9cb49fc019', '49793ebc79b3258f'),
    ('072d43a077075292', '7a8e7bfa937e89a3', '4fb05e1515ab73a7'),
    ('02fe55778117f12a', 'cf9c5d7a4986adb5', '49e95d6d4ca229bf'),
    ('1d9d5c5018f728c2', 'd1abb290658bc778', '018310dc409b26d6'),
    ('305532286d6f295a', '55cb3774d13ef201', '1c587f1c13924fef'),
    ('0123456789abcdef', 'fa34ec4847b268b2', '0101010101010101'),
    ('0123456789abcdef', 'a790795108ea3cae', '1f1f1f1f0e0e0e0e'),
    ('0123456789abcdef', 'c39e072d9fac631d', 'e0fee0fef1fef1fe'),
    ('ffffffffffffffff', '014933e0cdaff6e4', '0000000000000000'),
    ('0000000000000000', 'f21e9a77b71c49bc', 'ffffffffffffffff'),
    ('0000000000000000', '245946885754369a', '0123456789abcdef'),
    ('ffffffffffffffff', '6b5c5a9c5d9e0a5a', 'fedcba9876543210'),
    #('fedcba9876543210', 'f9ad597c49db005e', 'f0'),
    #('fedcba9876543210', 'e91d21c1d961a6d6', 'f0e1'),
    #('fedcba9876543210', 'e9c2b70a1bc65cf3', 'f0e1d2'),
    ('fedcba9876543210', 'be1e639408640f05', 'f0e1d2c3'),
    ('fedcba9876543210', 'b39e44481bdb1e6e', 'f0e1d2c3b4'),
    ('fedcba9876543210', '9457aa83b1928c0d', 'f0e1d2c3b4a5'),
    ('fedcba9876543210', '8bb77032f960629d', 'f0e1d2c3b4a596'),
    ('fedcba9876543210', 'e87a244e2cc85e82', 'f0e1d2c3b4a59687'),
    ('fedcba9876543210', '15750e7a4f4ec577', 'f0e1d2c3b4a5968778'),
    ('fedcba9876543210', '122ba70b3ab64ae0', 'f0e1d2c3b4a596877869'),
    ('fedcba9876543210', '3a833c9affc537f6', 'f0e1d2c3b4a5968778695a'),
    ('fedcba9876543210', '9409da87a90f6bf2', 'f0e1d2c3b4a5968778695a4b'),
    ('fedcba9876543210', '884f80625060b8b4', 'f0e1d2c3b4a5968778695a4b3c'),
    ('fedcba9876543210', '1f85031c19e11968', 'f0e1d2c3b4a5968778695a4b3c2d'),
    ('fedcba9876543210', '79d9373a714ca34f', 'f0e1d2c3b4a5968778695a4b3c2d1e'),
    ('fedcba9876543210', '93142887ee3be15c',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f'),
    ('fedcba9876543210', '03429e838ce2d14b',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f00'),
    ('fedcba9876543210', 'a4299e27469ff67b',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f0011'),
    ('fedcba9876543210', 'afd5aed1c1bc96a8',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f001122'),
    ('fedcba9876543210', '10851c0e3858da9f',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f00112233'),
    ('fedcba9876543210', 'e6f51ed79b9db21f',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f0011223344'),
    ('fedcba9876543210', '64a6e14afd36b46f',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f001122334455'),
    ('fedcba9876543210', '80c7d7d45a5479ad',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f00112233445566'),
    ('fedcba9876543210', '05044b62fa52d080',
        'f0e1d2c3b4a5968778695a4b3c2d1e0f0011223344556677'),
]


class KeyLength(unittest.TestCase):

    def runTest(self):
        self.assertRaises(ValueError, Blowfish.new, bchr(0) * 3,
                          Blowfish.MODE_ECB)
        self.assertRaises(ValueError, Blowfish.new, bchr(0) * 57,
                          Blowfish.MODE_ECB)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        cipher = Blowfish.new(b'4'*16, Blowfish.MODE_ECB)

        pt = b'5' * 16
        ct = cipher.encrypt(pt)

        output = bytearray(16)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)
        
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        output = memoryview(bytearray(16))
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0'*16)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0'*16)

        shorter_output = bytearray(7)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


def get_tests(config={}):
    from .common import make_block_tests
    tests = make_block_tests(Blowfish, "Blowfish", test_data)
    tests.append(KeyLength())
    tests += [TestOutput()]
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
