# -*- coding: utf-8 -*-
#
# SelfTest/Hash/test_SHA3_384.py: Self-test for the SHA-3/384 hash function
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

"""Self-test suite for Crypto.Hash.SHA3_384"""

import unittest
from binascii import hexlify

from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases
from Crypto.Hash import SHA3_384 as SHA3
from Crypto.Util.py3compat import b


class APITest(unittest.TestCase):

    def test_update_after_digest(self):
        msg=b("rrrrttt")

        # Normally, update() cannot be done after digest()
        h = SHA3.new(data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = SHA3.new(data=msg).digest()

        # With the proper flag, it is allowed
        h = SHA3.new(data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        # ... and the subsequent digest applies to the entire message
        # up to that point
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)


def get_tests(config={}):
    from .common import make_hash_tests

    tests = []

    test_vectors = load_test_vectors(("Hash", "SHA3"),
                                "ShortMsgKAT_SHA3-384.txt",
                                "KAT SHA-3 384",
                                { "len" : lambda x: int(x) } ) or []

    test_data = []
    for tv in test_vectors:
        if tv.len == 0:
            tv.msg = b("")
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests += make_hash_tests(SHA3, "SHA3_384", test_data,
                             digest_size=SHA3.digest_size,
                             oid="2.16.840.1.101.3.4.2.9")
    tests += list_test_cases(APITest)
    return tests

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
