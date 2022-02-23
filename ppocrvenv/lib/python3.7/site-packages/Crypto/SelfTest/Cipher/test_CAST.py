# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/CAST.py: Self-test for the CAST-128 (CAST5) cipher
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

"""Self-test suite for Crypto.Cipher.CAST"""

import unittest

from Crypto.Util.py3compat import bchr

from Crypto.Cipher import CAST

# This is a list of (plaintext, ciphertext, key) tuples.
test_data = [
    # Test vectors from RFC 2144, B.1
    ('0123456789abcdef', '238b4fe5847e44b2',
        '0123456712345678234567893456789a',
        '128-bit key'),

    ('0123456789abcdef', 'eb6a711a2c02271b',
        '01234567123456782345',
        '80-bit key'),

    ('0123456789abcdef', '7ac816d16e9b302e',
        '0123456712',
        '40-bit key'),
]


class KeyLength(unittest.TestCase):

    def runTest(self):
        self.assertRaises(ValueError, CAST.new, bchr(0) * 4, CAST.MODE_ECB)
        self.assertRaises(ValueError, CAST.new, bchr(0) * 17, CAST.MODE_ECB)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        cipher = CAST.new(b'4'*16, CAST.MODE_ECB)

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

    tests = make_block_tests(CAST, "CAST", test_data)
    tests.append(KeyLength())
    tests.append(TestOutput())
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
