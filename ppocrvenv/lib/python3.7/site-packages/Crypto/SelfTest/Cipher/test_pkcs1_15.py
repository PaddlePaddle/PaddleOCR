# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/test_pkcs1_15.py: Self-test for PKCS#1 v1.5 encryption
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

from __future__ import print_function

import unittest

from Crypto.PublicKey import RSA
from Crypto.SelfTest.st_common import list_test_cases, a2b_hex
from Crypto import Random
from Crypto.Cipher import PKCS1_v1_5 as PKCS
from Crypto.Util.py3compat import b
from Crypto.Util.number import bytes_to_long, long_to_bytes
from Crypto.SelfTest.loader import load_test_vectors_wycheproof


def rws(t):
    """Remove white spaces, tabs, and new lines from a string"""
    for c in ['\n', '\t', ' ']:
        t = t.replace(c, '')
    return t


def t2b(t):
    """Convert a text string with bytes in hex form to a byte string"""
    clean = b(rws(t))
    if len(clean) % 2 == 1:
        raise ValueError("Even number of characters expected")
    return a2b_hex(clean)


class PKCS1_15_Tests(unittest.TestCase):

        def setUp(self):
                self.rng = Random.new().read
                self.key1024 = RSA.generate(1024, self.rng)

        # List of tuples with test data for PKCS#1 v1.5.
        # Each tuple is made up by:
        #       Item #0: dictionary with RSA key component, or key to import
        #       Item #1: plaintext
        #       Item #2: ciphertext
        #       Item #3: random data

        _testData = (

                #
                # Generated with openssl 0.9.8o
                #
                (
                # Private key
                '''-----BEGIN RSA PRIVATE KEY-----
MIICXAIBAAKBgQDAiAnvIAOvqVwJTaYzsKnefZftgtXGE2hPJppGsWl78yz9jeXY
W/FxX/gTPURArNhdnhP6n3p2ZaDIBrO2zizbgIXs0IsljTTcr4vnI8fMXzyNUOjA
zP3nzMqZDZK6757XQAobOssMkBFqRWwilT/3DsBhRpl3iMUhF+wvpTSHewIDAQAB
AoGAC4HV/inOrpgTvSab8Wj0riyZgQOZ3U3ZpSlsfR8ra9Ib9Uee3jCYnKscu6Gk
y6zI/cdt8EPJ4PuwAWSNJzbpbVaDvUq25OD+CX8/uRT08yBS4J8TzBitZJTD4lS7
atdTnKT0Wmwk+u8tDbhvMKwnUHdJLcuIsycts9rwJVapUtkCQQDvDpx2JMun0YKG
uUttjmL8oJ3U0m3ZvMdVwBecA0eebZb1l2J5PvI3EJD97eKe91Nsw8T3lwpoN40k
IocSVDklAkEAzi1HLHE6EzVPOe5+Y0kGvrIYRRhncOb72vCvBZvD6wLZpQgqo6c4
d3XHFBBQWA6xcvQb5w+VVEJZzw64y25sHwJBAMYReRl6SzL0qA0wIYrYWrOt8JeQ
8mthulcWHXmqTgC6FEXP9Es5GD7/fuKl4wqLKZgIbH4nqvvGay7xXLCXD/ECQH9a
1JYNMtRen5unSAbIOxRcKkWz92F0LKpm9ZW/S9vFHO+mBcClMGoKJHiuQxLBsLbT
NtEZfSJZAeS2sUtn3/0CQDb2M2zNBTF8LlM0nxmh0k9VGm5TVIyBEMcipmvOgqIs
HKukWBcq9f/UOmS0oEhai/6g+Uf7VHJdWaeO5LzuvwU=
-----END RSA PRIVATE KEY-----''',
                # Plaintext
                '''THIS IS PLAINTEXT\x0A''',
                # Ciphertext
                '''3f dc fd 3c cd 5c 9b 12  af 65 32 e3 f7 d0 da 36
                8f 8f d9 e3 13 1c 7f c8  b3 f9 c1 08 e4 eb 79 9c
                91 89 1f 96 3b 94 77 61  99 a4 b1 ee 5d e6 17 c9
                5d 0a b5 63 52 0a eb 00  45 38 2a fb b0 71 3d 11
                f7 a1 9e a7 69 b3 af 61  c0 bb 04 5b 5d 4b 27 44
                1f 5b 97 89 ba 6a 08 95  ee 4f a2 eb 56 64 e5 0f
                da 7c f9 9a 61 61 06 62  ed a0 bc 5f aa 6c 31 78
                70 28 1a bb 98 3c e3 6a  60 3c d1 0b 0f 5a f4 75''',
                # Random data
                '''eb d7 7d 86 a4 35 23 a3 54 7e 02 0b 42 1d
                61 6c af 67 b8 4e 17 56 80 66 36 04 64 34 26 8a
                47 dd 44 b3 1a b2 17 60 f4 91 2e e2 b5 95 64 cc
                f9 da c8 70 94 54 86 4c ef 5b 08 7d 18 c4 ab 8d
                04 06 33 8f ca 15 5f 52 60 8a a1 0c f5 08 b5 4c
                bb 99 b8 94 25 04 9c e6 01 75 e6 f9 63 7a 65 61
                13 8a a7 47 77 81 ae 0d b8 2c 4d 50 a5'''
                ),
        )

        def testEncrypt1(self):
                for test in self._testData:
                        # Build the key
                        key = RSA.importKey(test[0])
                        # RNG that takes its random numbers from a pool given
                        # at initialization
                        class randGen:
                            def __init__(self, data):
                                self.data = data
                                self.idx = 0
                            def __call__(self, N):
                                r = self.data[self.idx:self.idx+N]
                                self.idx += N
                                return r
                        # The real test
                        cipher = PKCS.new(key, randfunc=randGen(t2b(test[3])))
                        ct = cipher.encrypt(b(test[1]))
                        self.assertEqual(ct, t2b(test[2]))

        def testEncrypt2(self):
                # Verify that encryption fail if plaintext is too long
                pt = '\x00'*(128-11+1)
                cipher = PKCS.new(self.key1024)
                self.assertRaises(ValueError, cipher.encrypt, pt)

        def testVerify1(self):
            for test in self._testData:
                key = RSA.importKey(test[0])
                expected_pt = b(test[1])
                ct = t2b(test[2])
                cipher = PKCS.new(key)

                # The real test
                pt = cipher.decrypt(ct, None)
                self.assertEqual(pt, expected_pt)

                pt = cipher.decrypt(ct, b'\xFF' * len(expected_pt))
                self.assertEqual(pt, expected_pt)

        def testVerify2(self):
            # Verify that decryption fails if ciphertext is not as long as
            # RSA modulus
            cipher = PKCS.new(self.key1024)
            self.assertRaises(ValueError, cipher.decrypt, '\x00'*127, "---")
            self.assertRaises(ValueError, cipher.decrypt, '\x00'*129, "---")

            # Verify that decryption fails if there are less then 8 non-zero padding
            # bytes
            pt = b('\x00\x02' + '\xFF'*7 + '\x00' + '\x45'*118)
            pt_int = bytes_to_long(pt)
            ct_int = self.key1024._encrypt(pt_int)
            ct = long_to_bytes(ct_int, 128)
            self.assertEqual(b"---", cipher.decrypt(ct, b"---"))

        def testEncryptVerify1(self):
            # Encrypt/Verify messages of length [0..RSAlen-11]
            # and therefore padding [8..117]
            for pt_len in range(0, 128 - 11 + 1):
                pt = self.rng(pt_len)
                cipher = PKCS.new(self.key1024)
                ct = cipher.encrypt(pt)
                pt2 = cipher.decrypt(ct, b'\xAA' * pt_len)
                self.assertEqual(pt, pt2)

        def test_encrypt_verify_exp_pt_len(self):

            cipher = PKCS.new(self.key1024)
            pt = b'5' * 16
            ct = cipher.encrypt(pt)
            sentinel = b'\xAA' * 16

            pt_A = cipher.decrypt(ct, sentinel, 16)
            self.assertEqual(pt, pt_A)

            pt_B = cipher.decrypt(ct, sentinel, 15)
            self.assertEqual(sentinel, pt_B)

            pt_C = cipher.decrypt(ct, sentinel, 17)
            self.assertEqual(sentinel, pt_C)

        def testByteArray(self):
            pt = b"XER"
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(bytearray(pt))
            pt2 = cipher.decrypt(bytearray(ct), '\xFF' * len(pt))
            self.assertEqual(pt, pt2)

        def testMemoryview(self):
            pt = b"XER"
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(memoryview(bytearray(pt)))
            pt2 = cipher.decrypt(memoryview(bytearray(ct)), b'\xFF' * len(pt))
            self.assertEqual(pt, pt2)

        def test_return_type(self):
            pt = b"XYZ"
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(pt)
            self.assertTrue(isinstance(ct, bytes))
            pt2 = cipher.decrypt(ct, b'\xAA' * 3)
            self.assertTrue(isinstance(pt2, bytes))


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, skip_slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._skip_slow_tests = skip_slow_tests
        self._id = "None"

    def load_tests(self, filename):

        def filter_rsa(group):
            return RSA.import_key(group['privateKeyPem'])

        result = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                              filename,
                                              "Wycheproof PKCS#1v1.5 (%s)" % filename,
                                              group_tag={'rsa_key': filter_rsa}
                                              )
        return result

    def setUp(self):
        self.tv = []
        self.tv.extend(self.load_tests("rsa_pkcs1_2048_test.json"))
        if not self._skip_slow_tests:
            self.tv.extend(self.load_tests("rsa_pkcs1_3072_test.json"))
            self.tv.extend(self.load_tests("rsa_pkcs1_4096_test.json"))

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt PKCS#1v1.5 Test #%s" % tv.id
        sentinel = b'\xAA' * max(3, len(tv.msg))
        cipher = PKCS.new(tv.rsa_key)
        try:
            pt = cipher.decrypt(tv.ct, sentinel=sentinel)
        except ValueError:
            assert not tv.valid
        else:
            if pt == sentinel:
                assert not tv.valid
            else:
                assert tv.valid
                self.assertEqual(pt, tv.msg)
                self.warn(tv)

    def runTest(self):

        for tv in self.tv:
            self.test_decrypt(tv)


def get_tests(config={}):
    skip_slow_tests = not config.get('slow_tests')
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(PKCS1_15_Tests)
    tests += [TestVectorsWycheproof(wycheproof_warnings, skip_slow_tests)]
    return tests


if __name__ == '__main__':
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
