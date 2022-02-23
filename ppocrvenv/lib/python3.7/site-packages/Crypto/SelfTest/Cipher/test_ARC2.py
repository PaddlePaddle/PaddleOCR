# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/ARC2.py: Self-test for the Alleged-RC2 cipher
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

"""Self-test suite for Crypto.Cipher.ARC2"""

import unittest

from Crypto.Util.py3compat import b, bchr

from Crypto.Cipher import ARC2

# This is a list of (plaintext, ciphertext, key[, description[, extra_params]]) tuples.
test_data = [
    # Test vectors from RFC 2268

    # 63-bit effective key length
    ('0000000000000000', 'ebb773f993278eff', '0000000000000000',
        'RFC2268-1', dict(effective_keylen=63)),

    # 64-bit effective key length
    ('ffffffffffffffff', '278b27e42e2f0d49', 'ffffffffffffffff',
        'RFC2268-2', dict(effective_keylen=64)),
    ('1000000000000001', '30649edf9be7d2c2', '3000000000000000',
        'RFC2268-3', dict(effective_keylen=64)),
    #('0000000000000000', '61a8a244adacccf0', '88',
    #    'RFC2268-4', dict(effective_keylen=64)),
    ('0000000000000000', '6ccf4308974c267f', '88bca90e90875a',
        'RFC2268-5', dict(effective_keylen=64)),
    ('0000000000000000', '1a807d272bbe5db1', '88bca90e90875a7f0f79c384627bafb2',
        'RFC2268-6', dict(effective_keylen=64)),

    # 128-bit effective key length
    ('0000000000000000', '2269552ab0f85ca6', '88bca90e90875a7f0f79c384627bafb2',
        "RFC2268-7", dict(effective_keylen=128)),
    ('0000000000000000', '5b78d3a43dfff1f1',
        '88bca90e90875a7f0f79c384627bafb216f80a6f85920584c42fceb0be255daf1e',
        "RFC2268-8", dict(effective_keylen=129)),

    # Test vectors from PyCrypto 2.0.1's testdata.py
    # 1024-bit effective key length
    ('0000000000000000', '624fb3e887419e48', '5068696c6970476c617373',
        'PCTv201-0'),
    ('ffffffffffffffff', '79cadef44c4a5a85', '5068696c6970476c617373',
        'PCTv201-1'),
    ('0001020304050607', '90411525b34e4c2c', '5068696c6970476c617373',
        'PCTv201-2'),
    ('0011223344556677', '078656aaba61cbfb', '5068696c6970476c617373',
        'PCTv201-3'),
    ('0000000000000000', 'd7bcc5dbb4d6e56a', 'ffffffffffffffff',
        'PCTv201-4'),
    ('ffffffffffffffff', '7259018ec557b357', 'ffffffffffffffff',
        'PCTv201-5'),
    ('0001020304050607', '93d20a497f2ccb62', 'ffffffffffffffff',
        'PCTv201-6'),
    ('0011223344556677', 'cb15a7f819c0014d', 'ffffffffffffffff',
        'PCTv201-7'),
    ('0000000000000000', '63ac98cdf3843a7a', 'ffffffffffffffff5065746572477265656e6177617953e5ffe553',
        'PCTv201-8'),
    ('ffffffffffffffff', '3fb49e2fa12371dd', 'ffffffffffffffff5065746572477265656e6177617953e5ffe553',
        'PCTv201-9'),
    ('0001020304050607', '46414781ab387d5f', 'ffffffffffffffff5065746572477265656e6177617953e5ffe553',
        'PCTv201-10'),
    ('0011223344556677', 'be09dc81feaca271', 'ffffffffffffffff5065746572477265656e6177617953e5ffe553',
        'PCTv201-11'),
    ('0000000000000000', 'e64221e608be30ab', '53e5ffe553',
        'PCTv201-12'),
    ('ffffffffffffffff', '862bc60fdcd4d9a9', '53e5ffe553',
        'PCTv201-13'),
    ('0001020304050607', '6a34da50fa5e47de', '53e5ffe553',
        'PCTv201-14'),
    ('0011223344556677', '584644c34503122c', '53e5ffe553',
        'PCTv201-15'),
]

class BufferOverflowTest(unittest.TestCase):
    # Test a buffer overflow found in older versions of PyCrypto

    def runTest(self):
        """ARC2 with keylength > 128"""
        key = b("x") * 16384
        self.assertRaises(ValueError, ARC2.new, key, ARC2.MODE_ECB)

class KeyLength(unittest.TestCase):

    def runTest(self):
        ARC2.new(b'\x00' * 16, ARC2.MODE_ECB, effective_keylen=40)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 4, ARC2.MODE_ECB)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 129, ARC2.MODE_ECB)

        self.assertRaises(ValueError, ARC2.new, bchr(0) * 16, ARC2.MODE_ECB,
                          effective_keylen=39)
        self.assertRaises(ValueError, ARC2.new, bchr(0) * 16, ARC2.MODE_ECB,
                          effective_keylen=1025)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        cipher = ARC2.new(b'4'*16, ARC2.MODE_ECB)

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
    from Crypto.Cipher import ARC2
    from .common import make_block_tests

    tests = make_block_tests(ARC2, "ARC2", test_data)
    tests.append(BufferOverflowTest())
    tests.append(KeyLength())
    tests += [TestOutput()]

    return tests

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
