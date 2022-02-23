# ===================================================================
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

"""Self-test suite for Crypto.Hash.cSHAKE128 and cSHAKE256"""

import unittest

from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import cSHAKE128, cSHAKE256, SHAKE128, SHAKE256
from Crypto.Util.py3compat import b, bchr, tobytes


class cSHAKETest(unittest.TestCase):

    def test_left_encode(self):
        from Crypto.Hash.cSHAKE128 import _left_encode
        self.assertEqual(_left_encode(0), b'\x01\x00')
        self.assertEqual(_left_encode(1), b'\x01\x01')
        self.assertEqual(_left_encode(256), b'\x02\x01\x00')

    def test_bytepad(self):
        from Crypto.Hash.cSHAKE128 import _bytepad
        self.assertEqual(_bytepad(b'', 4), b'\x01\x04\x00\x00')
        self.assertEqual(_bytepad(b'A', 4), b'\x01\x04A\x00')
        self.assertEqual(_bytepad(b'AA', 4), b'\x01\x04AA')
        self.assertEqual(_bytepad(b'AAA', 4), b'\x01\x04AAA\x00\x00\x00')
        self.assertEqual(_bytepad(b'AAAA', 4), b'\x01\x04AAAA\x00\x00')
        self.assertEqual(_bytepad(b'AAAAA', 4), b'\x01\x04AAAAA\x00')
        self.assertEqual(_bytepad(b'AAAAAA', 4), b'\x01\x04AAAAAA')
        self.assertEqual(_bytepad(b'AAAAAAA', 4), b'\x01\x04AAAAAAA\x00\x00\x00')

    def test_new_positive(self):

        xof1 = self.cshake.new()
        xof2 = self.cshake.new(data=b("90"))
        xof3 = self.cshake.new().update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

        xof1 = self.cshake.new()
        ref = xof1.read(10)
        xof2 = self.cshake.new(custom=b(""))
        xof3 = self.cshake.new(custom=b("foo"))

        self.assertEqual(ref, xof2.read(10))
        self.assertNotEqual(ref, xof3.read(10))

        xof1 = self.cshake.new(custom=b("foo"))
        xof2 = self.cshake.new(custom=b("foo"), data=b("90"))
        xof3 = self.cshake.new(custom=b("foo")).update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = self.cshake.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = self.cshake.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = self.cshake.new()
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = self.cshake.new()
        digest = h.read(90)

        # read returns a byte string of the right length
        self.assertTrue(isinstance(digest, type(b("digest"))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = self.cshake.new()
        mac.update(b("rrrr"))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b("ttt"))

    def test_shake(self):
        # When no customization string is passed, results must match SHAKE
        for digest_len in range(64):
            xof1 = self.cshake.new(b'TEST')
            xof2 = self.shake.new(b'TEST')
            self.assertEqual(xof1.read(digest_len), xof2.read(digest_len))


class cSHAKE128Test(cSHAKETest):
    cshake = cSHAKE128
    shake = SHAKE128


class cSHAKE256Test(cSHAKETest):
    cshake = cSHAKE256
    shake = SHAKE256


class cSHAKEVectors(unittest.TestCase):
    pass


vector_files = [("ShortMsgSamples_cSHAKE128.txt", "Short Message Samples cSHAKE128", "128_cshake", cSHAKE128),
                ("ShortMsgSamples_cSHAKE256.txt", "Short Message Samples cSHAKE256", "256_cshake", cSHAKE256),
                ("CustomMsgSamples_cSHAKE128.txt", "Custom Message Samples cSHAKE128", "custom_128_cshake", cSHAKE128),
                ("CustomMsgSamples_cSHAKE256.txt", "Custom Message Samples cSHAKE256", "custom_256_cshake", cSHAKE256),
                ]

for file, descr, tag, test_class in vector_files:

    test_vectors = load_test_vectors(("Hash", "SHA3"), file, descr,
                                     {"len": lambda x: int(x),
                                      "nlen": lambda x: int(x),
                                      "slen": lambda x: int(x)}) or []

    for idx, tv in enumerate(test_vectors):
        if getattr(tv, "len", 0) == 0:
            data = b("")
        else:
            data = tobytes(tv.msg)
            assert(tv.len == len(tv.msg)*8)
        if getattr(tv, "nlen", 0) != 0:
            raise ValueError("Unsupported cSHAKE test vector")
        if getattr(tv, "slen", 0) == 0:
            custom = b("")
        else:
            custom = tobytes(tv.s)
            assert(tv.slen == len(tv.s)*8)

        def new_test(self, data=data, result=tv.md, custom=custom, test_class=test_class):
            hobj = test_class.new(data=data, custom=custom)
            digest = hobj.read(len(result))
            self.assertEqual(digest, result)

        setattr(cSHAKEVectors, "test_%s_%d" % (tag, idx), new_test)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(cSHAKE128Test)
    tests += list_test_cases(cSHAKE256Test)
    tests += list_test_cases(cSHAKEVectors)
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
