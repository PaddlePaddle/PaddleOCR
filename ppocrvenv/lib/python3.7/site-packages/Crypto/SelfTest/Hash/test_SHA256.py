# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/test_SHA256.py: Self-test for the SHA-256 hash function
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

"""Self-test suite for Crypto.Hash.SHA256"""

import unittest
from Crypto.Util.py3compat import *

class LargeSHA256Test(unittest.TestCase):
    def runTest(self):
        """SHA256: 512/520 MiB test"""
        from Crypto.Hash import SHA256
        zeros = bchr(0x00) * (1024*1024)

        h = SHA256.new(zeros)
        for i in range(511):
            h.update(zeros)

        # This test vector is from PyCrypto's old testdata.py file.
        self.assertEqual('9acca8e8c22201155389f65abbf6bc9723edc7384ead80503839f49dcc56d767', h.hexdigest()) # 512 MiB

        for i in range(8):
            h.update(zeros)

        # This test vector is from PyCrypto's old testdata.py file.
        self.assertEqual('abf51ad954b246009dfe5a50ecd582fd5b8f1b8b27f30393853c3ef721e7fa6e', h.hexdigest()) # 520 MiB

def get_tests(config={}):
    # Test vectors from FIPS PUB 180-2
    # This is a list of (expected_result, input[, description]) tuples.
    test_data = [
        # FIPS PUB 180-2, B.1 - "One-Block Message"
        ('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
            'abc'),

        # FIPS PUB 180-2, B.2 - "Multi-Block Message"
        ('248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
            'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'),

        # FIPS PUB 180-2, B.3 - "Long Message"
        ('cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0',
            'a' * 10**6,
             '"a" * 10**6'),

        # Test for an old PyCrypto bug.
        ('f7fd017a3c721ce7ff03f3552c0813adcc48b7f33f07e5e2ba71e23ea393d103',
            'This message is precisely 55 bytes long, to test a bug.',
            'Length = 55 (mod 64)'),

        # Example from http://de.wikipedia.org/wiki/Secure_Hash_Algorithm
        ('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', ''),

        ('d32b568cd1b96d459e7291ebf4b25d007f275c9f13149beeb782fac0716613f8',
         'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'),
    ]

    from Crypto.Hash import SHA256
    from .common import make_hash_tests
    tests = make_hash_tests(SHA256, "SHA256", test_data,
        digest_size=32,
        oid="2.16.840.1.101.3.4.2.1")

    if config.get('slow_tests'):
        tests += [LargeSHA256Test()]

    return tests

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
