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

"""Self-test suite for Crypto.Hash.SHAKE128 and SHAKE256"""

import unittest
from binascii import hexlify, unhexlify

from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import SHAKE128, SHAKE256
from Crypto.Util.py3compat import b, bchr, bord, tobytes

class SHAKETest(unittest.TestCase):

    def test_new_positive(self):

        xof1 = self.shake.new()
        xof2 = self.shake.new(data=b("90"))
        xof3 = self.shake.new().update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = self.shake.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = self.shake.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = self.shake.new()
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = self.shake.new()
        digest = h.read(90)

        # read returns a byte string of the right length
        self.assertTrue(isinstance(digest, type(b("digest"))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = self.shake.new()
        mac.update(b("rrrr"))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b("ttt"))


class SHAKE128Test(SHAKETest):
        shake = SHAKE128


class SHAKE256Test(SHAKETest):
        shake = SHAKE256


class SHAKEVectors(unittest.TestCase):
    pass


test_vectors_128 = load_test_vectors(("Hash", "SHA3"),
                               "ShortMsgKAT_SHAKE128.txt",
                               "Short Messages KAT SHAKE128",
                               { "len" : lambda x: int(x) } ) or []

for idx, tv in enumerate(test_vectors_128):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = SHAKE128.new(data=data)
        digest = hobj.read(len(result))
        self.assertEqual(digest, result)

    setattr(SHAKEVectors, "test_128_%d" % idx, new_test)


test_vectors_256 = load_test_vectors(("Hash", "SHA3"),
                               "ShortMsgKAT_SHAKE256.txt",
                               "Short Messages KAT SHAKE256",
                               { "len" : lambda x: int(x) } ) or []

for idx, tv in enumerate(test_vectors_256):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = SHAKE256.new(data=data)
        digest = hobj.read(len(result))
        self.assertEqual(digest, result)

    setattr(SHAKEVectors, "test_256_%d" % idx, new_test)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(SHAKE128Test)
    tests += list_test_cases(SHAKE256Test)
    tests += list_test_cases(SHAKEVectors)
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
