# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/DES3.py: Self-test for the Triple-DES cipher
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

"""Self-test suite for Crypto.Cipher.DES3"""

import unittest
from binascii import hexlify, unhexlify

from Crypto.Cipher import DES3

from Crypto.Util.strxor import strxor_c
from Crypto.Util.py3compat import bchr, tostr
from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases

# This is a list of (plaintext, ciphertext, key, description) tuples.
test_data = [
    # Test vector from Appendix B of NIST SP 800-67
    # "Recommendation for the Triple Data Encryption Algorithm (TDEA) Block
    # Cipher"
    # http://csrc.nist.gov/publications/nistpubs/800-67/SP800-67.pdf
    ('54686520717566636b2062726f776e20666f78206a756d70',
        'a826fd8ce53b855fcce21c8112256fe668d5c05dd9b6b900',
        '0123456789abcdef23456789abcdef01456789abcdef0123',
        'NIST SP800-67 B.1'),

    # This test is designed to test the DES3 API, not the correctness of the
    # output.
    ('21e81b7ade88a259', '5c577d4d9b20c0f8',
        '9b397ebf81b1181e282f4bb8adbadc6b', 'Two-key 3DES'),
]

# NIST CAVP test vectors

nist_tdes_mmt_files = ("TECBMMT2.rsp", "TECBMMT3.rsp")

for tdes_file in nist_tdes_mmt_files:

    test_vectors = load_test_vectors(
                        ("Cipher", "TDES"),
                        tdes_file,
                        "TDES ECB (%s)" % tdes_file,
                        {"count": lambda x: int(x)}) or []

    for index, tv in enumerate(test_vectors):

        # The test vector file contains some directive lines
        if isinstance(tv, str):
            continue

        key = tv.key1 + tv.key2 + tv.key3
        test_data_item = (tostr(hexlify(tv.plaintext)),
                          tostr(hexlify(tv.ciphertext)),
                          tostr(hexlify(key)),
                          "%s (%s)" % (tdes_file, index))
        test_data.append(test_data_item)


class CheckParity(unittest.TestCase):

    def test_parity_option2(self):
        before_2k = unhexlify("CABF326FA56734324FFCCABCDEFACABF")
        after_2k = DES3.adjust_key_parity(before_2k)
        self.assertEqual(after_2k,
                         unhexlify("CBBF326EA46734324FFDCBBCDFFBCBBF"))

    def test_parity_option3(self):
        before_3k = unhexlify("AAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBCCCCCCCCCCCCCCCC")
        after_3k = DES3.adjust_key_parity(before_3k)
        self.assertEqual(after_3k,
                         unhexlify("ABABABABABABABABBABABABABABABABACDCDCDCDCDCDCDCD"))

    def test_degradation(self):
        sub_key1 = bchr(1) * 8
        sub_key2 = bchr(255) * 8

        # K1 == K2
        self.assertRaises(ValueError, DES3.adjust_key_parity,
                          sub_key1 * 2 + sub_key2)

        # K2 == K3
        self.assertRaises(ValueError, DES3.adjust_key_parity,
                          sub_key1 + sub_key2 * 2)

        # K1 == K2 == K3
        self.assertRaises(ValueError, DES3.adjust_key_parity,
                          sub_key1 * 3)

        # K1 == K2 (with different parity)
        self.assertRaises(ValueError, DES3.adjust_key_parity,
                          sub_key1 + strxor_c(sub_key1, 1) + sub_key2)


class DegenerateToDESTest(unittest.TestCase):

    def runTest(self):
        sub_key1 = bchr(1) * 8
        sub_key2 = bchr(255) * 8

        # K1 == K2
        self.assertRaises(ValueError, DES3.new,
                          sub_key1 * 2 + sub_key2,
                          DES3.MODE_ECB)

        # K2 == K3
        self.assertRaises(ValueError, DES3.new,
                          sub_key1 + sub_key2 * 2,
                          DES3.MODE_ECB)

        # K1 == K2 == K3
        self.assertRaises(ValueError, DES3.new,
                          sub_key1 * 3,
                          DES3.MODE_ECB)

        # K2 == K3 (parity is ignored)
        self.assertRaises(ValueError, DES3.new,
                          sub_key1 + sub_key2 + strxor_c(sub_key2, 0x1),
                          DES3.MODE_ECB)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        cipher = DES3.new(b'4'*8 + b'G'*8 + b'T'*8, DES3.MODE_ECB)

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

    tests = []
    tests = make_block_tests(DES3, "DES3", test_data)
    tests.append(DegenerateToDESTest())
    tests += list_test_cases(CheckParity)
    tests += [TestOutput()]
    return tests


if __name__ == '__main__':
    import unittest

    def suite():
        unittest.TestSuite(get_tests())

    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
