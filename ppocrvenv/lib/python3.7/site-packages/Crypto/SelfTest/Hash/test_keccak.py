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

"""Self-test suite for Crypto.Hash.keccak"""

import unittest
from binascii import hexlify, unhexlify

from Crypto.SelfTest.loader import load_test_vectors
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import keccak
from Crypto.Util.py3compat import b, tobytes, bchr

class KeccakTest(unittest.TestCase):

    def test_new_positive(self):

        for digest_bits in (224, 256, 384, 512):
            hobj = keccak.new(digest_bits=digest_bits)
            self.assertEqual(hobj.digest_size, digest_bits // 8)

            hobj2 = hobj.new()
            self.assertEqual(hobj2.digest_size, digest_bits // 8)

        for digest_bytes in (28, 32, 48, 64):
            hobj = keccak.new(digest_bytes=digest_bytes)
            self.assertEqual(hobj.digest_size, digest_bytes)

            hobj2 = hobj.new()
            self.assertEqual(hobj2.digest_size, digest_bytes)

    def test_new_positive2(self):

        digest1 = keccak.new(data=b("\x90"), digest_bytes=64).digest()
        digest2 = keccak.new(digest_bytes=64).update(b("\x90")).digest()
        self.assertEqual(digest1, digest2)

    def test_new_negative(self):

        # keccak.new needs digest size
        self.assertRaises(TypeError, keccak.new)

        h = keccak.new(digest_bits=512)

        # Either bits or bytes can be specified
        self.assertRaises(TypeError, keccak.new,
                              digest_bytes=64,
                              digest_bits=512)

        # Range
        self.assertRaises(ValueError, keccak.new, digest_bytes=0)
        self.assertRaises(ValueError, keccak.new, digest_bytes=1)
        self.assertRaises(ValueError, keccak.new, digest_bytes=65)
        self.assertRaises(ValueError, keccak.new, digest_bits=0)
        self.assertRaises(ValueError, keccak.new, digest_bits=1)
        self.assertRaises(ValueError, keccak.new, digest_bits=513)

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = keccak.new(digest_bytes=64)
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = keccak.new(digest_bytes=64)
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = keccak.new(digest_bytes=64)
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = keccak.new(digest_bytes=64)
        digest = h.digest()

        # hexdigest does not change the state
        self.assertEqual(h.digest(), digest)
        # digest returns a byte string
        self.assertTrue(isinstance(digest, type(b("digest"))))

    def test_hex_digest(self):
        mac = keccak.new(digest_bits=512)
        digest = mac.digest()
        hexdigest = mac.hexdigest()

        # hexdigest is equivalent to digest
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        # hexdigest does not change the state
        self.assertEqual(mac.hexdigest(), hexdigest)
        # hexdigest returns a string
        self.assertTrue(isinstance(hexdigest, type("digest")))

    def test_update_after_digest(self):
        msg=b("rrrrttt")

        # Normally, update() cannot be done after digest()
        h = keccak.new(digest_bits=512, data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = keccak.new(digest_bits=512, data=msg).digest()

        # With the proper flag, it is allowed
        h = keccak.new(digest_bits=512, data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        # ... and the subsequent digest applies to the entire message
        # up to that point
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)


class KeccakVectors(unittest.TestCase):
    pass

    # TODO: add ExtremelyLong tests


test_vectors_224 =  load_test_vectors(("Hash", "keccak"),
                                "ShortMsgKAT_224.txt",
                                "Short Messages KAT 224",
                                {"len": lambda x: int(x)}) or []

test_vectors_224 += load_test_vectors(("Hash", "keccak"),
                                "LongMsgKAT_224.txt",
                                "Long Messages KAT 224",
                                {"len": lambda x: int(x)}) or []

for idx, tv in enumerate(test_vectors_224):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = keccak.new(digest_bits=224, data=data)
        self.assertEqual(hobj.digest(), result)

    setattr(KeccakVectors, "test_224_%d" % idx, new_test)

# ---

test_vectors_256 =  load_test_vectors(("Hash", "keccak"),
                                "ShortMsgKAT_256.txt",
                                "Short Messages KAT 256",
                                { "len" : lambda x: int(x) } ) or []

test_vectors_256 += load_test_vectors(("Hash", "keccak"),
                                "LongMsgKAT_256.txt",
                                "Long Messages KAT 256",
                                { "len" : lambda x: int(x) } ) or []

for idx, tv in enumerate(test_vectors_256):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = keccak.new(digest_bits=256, data=data)
        self.assertEqual(hobj.digest(), result)

    setattr(KeccakVectors, "test_256_%d" % idx, new_test)


# ---

test_vectors_384 =  load_test_vectors(("Hash", "keccak"),
                                "ShortMsgKAT_384.txt",
                                "Short Messages KAT 384",
                                {"len": lambda x: int(x)}) or []

test_vectors_384 += load_test_vectors(("Hash", "keccak"),
                                "LongMsgKAT_384.txt",
                                "Long Messages KAT 384",
                                {"len": lambda x: int(x)}) or []

for idx, tv in enumerate(test_vectors_384):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = keccak.new(digest_bits=384, data=data)
        self.assertEqual(hobj.digest(), result)

    setattr(KeccakVectors, "test_384_%d" % idx, new_test)

# ---

test_vectors_512 =  load_test_vectors(("Hash", "keccak"),
                                "ShortMsgKAT_512.txt",
                                "Short Messages KAT 512",
                                {"len": lambda x: int(x)}) or []

test_vectors_512 += load_test_vectors(("Hash", "keccak"),
                                "LongMsgKAT_512.txt",
                                "Long Messages KAT 512",
                                {"len": lambda x: int(x)}) or []

for idx, tv in enumerate(test_vectors_512):
    if tv.len == 0:
        data = b("")
    else:
        data = tobytes(tv.msg)

    def new_test(self, data=data, result=tv.md):
        hobj = keccak.new(digest_bits=512, data=data)
        self.assertEqual(hobj.digest(), result)

    setattr(KeccakVectors, "test_512_%d" % idx, new_test)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(KeccakTest)
    tests += list_test_cases(KeccakVectors)
    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
