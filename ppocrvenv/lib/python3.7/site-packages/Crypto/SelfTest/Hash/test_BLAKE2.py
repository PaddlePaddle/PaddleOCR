# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
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

import os
import re
import unittest
import warnings
from binascii import unhexlify, hexlify

from Crypto.Util.py3compat import tobytes
from Crypto.Util.strxor import strxor_c
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import BLAKE2b, BLAKE2s


class Blake2Test(unittest.TestCase):

    def test_new_positive(self):

        h = self.BLAKE2.new(digest_bits=self.max_bits)
        for new_func in self.BLAKE2.new, h.new:

            for dbits in range(8, self.max_bits + 1, 8):
                hobj = new_func(digest_bits=dbits)
                self.assertEqual(hobj.digest_size, dbits // 8)

            for dbytes in range(1, self.max_bytes + 1):
                hobj = new_func(digest_bytes=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)

            digest1 = new_func(data=b"\x90", digest_bytes=self.max_bytes).digest()
            digest2 = new_func(digest_bytes=self.max_bytes).update(b"\x90").digest()
            self.assertEqual(digest1, digest2)

            new_func(data=b"A", key=b"5", digest_bytes=self.max_bytes)

        hobj = h.new()
        self.assertEqual(hobj.digest_size, self.max_bytes)

    def test_new_negative(self):

        h = self.BLAKE2.new(digest_bits=self.max_bits)
        for new_func in self.BLAKE2.new, h.new:
            self.assertRaises(TypeError, new_func,
                              digest_bytes=self.max_bytes,
                              digest_bits=self.max_bits)
            self.assertRaises(ValueError, new_func, digest_bytes=0)
            self.assertRaises(ValueError, new_func,
                              digest_bytes=self.max_bytes + 1)
            self.assertRaises(ValueError, new_func, digest_bits=7)
            self.assertRaises(ValueError, new_func, digest_bits=15)
            self.assertRaises(ValueError, new_func,
                              digest_bits=self.max_bits + 1)
            self.assertRaises(TypeError, new_func,
                              digest_bytes=self.max_bytes,
                              key=u"string")
            self.assertRaises(TypeError, new_func,
                              digest_bytes=self.max_bytes,
                              data=u"string")

    def test_default_digest_size(self):
        digest = self.BLAKE2.new(data=b'abc').digest()
        self.assertEqual(len(digest), self.max_bytes)

    def test_update(self):
        pieces = [b"\x0A" * 200, b"\x14" * 300]
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes)
        digest = h.digest()

        # hexdigest does not change the state
        self.assertEqual(h.digest(), digest)
        # digest returns a byte string
        self.assertTrue(isinstance(digest, type(b"digest")))

    def test_update_after_digest(self):
        msg = b"rrrrttt"

        # Normally, update() cannot be done after digest()
        h = self.BLAKE2.new(digest_bits=256, data=msg[:4])
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])
        dig2 = self.BLAKE2.new(digest_bits=256, data=msg).digest()

        # With the proper flag, it is allowed
        h = self.BLAKE2.new(digest_bits=256, data=msg[:4], update_after_digest=True)
        self.assertEqual(h.digest(), dig1)
        # ... and the subsequent digest applies to the entire message
        # up to that point
        h.update(msg[4:])
        self.assertEqual(h.digest(), dig2)

    def test_hex_digest(self):
        mac = self.BLAKE2.new(digest_bits=self.max_bits)
        digest = mac.digest()
        hexdigest = mac.hexdigest()

        # hexdigest is equivalent to digest
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        # hexdigest does not change the state
        self.assertEqual(mac.hexdigest(), hexdigest)
        # hexdigest returns a string
        self.assertTrue(isinstance(hexdigest, type("digest")))

    def test_verify(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes, key=b"4")
        mac = h.digest()
        h.verify(mac)
        wrong_mac = strxor_c(mac, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)

    def test_hexverify(self):
        h = self.BLAKE2.new(digest_bytes=self.max_bytes, key=b"4")
        mac = h.hexdigest()
        h.hexverify(mac)
        self.assertRaises(ValueError, h.hexverify, "4556")

    def test_oid(self):

        prefix = "1.3.6.1.4.1.1722.12.2." + self.oid_variant + "."

        for digest_bits in self.digest_bits_oid:
            h = self.BLAKE2.new(digest_bits=digest_bits)
            self.assertEqual(h.oid, prefix + str(digest_bits // 8))

            h = self.BLAKE2.new(digest_bits=digest_bits, key=b"secret")
            self.assertRaises(AttributeError, lambda: h.oid)

        for digest_bits in (8, self.max_bits):
            if digest_bits in self.digest_bits_oid:
                continue
            self.assertRaises(AttributeError, lambda: h.oid)

    def test_bytearray(self):

        key = b'0' * 16
        data = b"\x00\x01\x02"

        # Data and key can be a bytearray (during initialization)
        key_ba = bytearray(key)
        data_ba = bytearray(data)

        h1 = self.BLAKE2.new(data=data, key=key)
        h2 = self.BLAKE2.new(data=data_ba, key=key_ba)
        key_ba[:1] = b'\xFF'
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

        # Data can be a bytearray (during operation)
        data_ba = bytearray(data)

        h1 = self.BLAKE2.new()
        h2 = self.BLAKE2.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):

        key = b'0' * 16
        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in (get_mv_ro, get_mv_rw):

            # Data and key can be a memoryview (during initialization)
            key_mv = get_mv(key)
            data_mv = get_mv(data)

            h1 = self.BLAKE2.new(data=data, key=key)
            h2 = self.BLAKE2.new(data=data_mv, key=key_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'
                key_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())

            # Data can be a memoryview (during operation)
            data_mv = get_mv(data)

            h1 = self.BLAKE2.new()
            h2 = self.BLAKE2.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())


class Blake2bTest(Blake2Test):
    #: Module
    BLAKE2 = BLAKE2b
    #: Max output size (in bits)
    max_bits = 512
    #: Max output size (in bytes)
    max_bytes = 64
    #: Bit size of the digests for which an ASN OID exists
    digest_bits_oid = (160, 256, 384, 512)
    # http://tools.ietf.org/html/draft-saarinen-blake2-02
    oid_variant = "1"


class Blake2sTest(Blake2Test):
    #: Module
    BLAKE2 = BLAKE2s
    #: Max output size (in bits)
    max_bits = 256
    #: Max output size (in bytes)
    max_bytes = 32
    #: Bit size of the digests for which an ASN OID exists
    digest_bits_oid = (128, 160, 224, 256)
    # http://tools.ietf.org/html/draft-saarinen-blake2-02
    oid_variant = "2"


class Blake2OfficialTestVector(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        expected = "in"
        test_vectors = []
        with open(test_vector_file, "rt") as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):

                if line.strip() == "" or line.startswith("#"):
                    continue

                res = re.match("%s:\t([0-9A-Fa-f]*)" % expected, line)
                if not res:
                    raise ValueError("Incorrect test vector format (line %d)"
                                     % line_number)

                if res.group(1):
                    bin_value = unhexlify(tobytes(res.group(1)))
                else:
                    bin_value = b""
                if expected == "in":
                    input_data = bin_value
                    expected = "key"
                elif expected == "key":
                    key = bin_value
                    expected = "hash"
                else:
                    result = bin_value
                    expected = "in"
                    test_vectors.append((input_data, key, result))
        return test_vectors

    def setUp(self):

        dir_comps = ("Hash", self.name)
        file_name = self.name.lower() + "-test.txt"
        self.description = "%s tests" % self.name

        try:
            import pycryptodome_test_vectors  # type: ignore
        except ImportError:
            warnings.warn("Warning: skipping extended tests for %s" % self.name,
                           UserWarning)
            self.test_vectors = []
            return

        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        self.test_vectors = self._load_tests(full_file_name)

    def runTest(self):
        for (input_data, key, result) in self.test_vectors:
            mac = self.BLAKE2.new(key=key, digest_bytes=self.max_bytes)
            mac.update(input_data)
            self.assertEqual(mac.digest(), result)


class Blake2bOfficialTestVector(Blake2OfficialTestVector):
    #: Module
    BLAKE2 = BLAKE2b
    #: Hash name
    name = "BLAKE2b"
    #: Max digest size
    max_bytes = 64


class Blake2sOfficialTestVector(Blake2OfficialTestVector):
    #: Module
    BLAKE2 = BLAKE2s
    #: Hash name
    name = "BLAKE2s"
    #: Max digest size
    max_bytes = 32


class Blake2TestVector1(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        test_vectors = []
        with open(test_vector_file, "rt") as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):
                if line.strip() == "" or line.startswith("#"):
                    continue
                res = re.match("digest: ([0-9A-Fa-f]*)", line)
                if not res:
                    raise ValueError("Incorrect test vector format (line %d)"
                                     % line_number)

                test_vectors.append(unhexlify(tobytes(res.group(1))))
        return test_vectors

    def setUp(self):
        dir_comps = ("Hash", self.name)
        file_name = "tv1.txt"
        self.description = "%s tests" % self.name

        try:
            import pycryptodome_test_vectors
        except ImportError:
            warnings.warn("Warning: skipping extended tests for %s" % self.name,
                           UserWarning)
            self.test_vectors = []
            return

        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        self.test_vectors = self._load_tests(full_file_name)

    def runTest(self):

        for tv in self.test_vectors:
            digest_bytes = len(tv)
            next_data = b""
            for _ in range(100):
                h = self.BLAKE2.new(digest_bytes=digest_bytes)
                h.update(next_data)
                next_data = h.digest() + next_data
            self.assertEqual(h.digest(), tv)


class Blake2bTestVector1(Blake2TestVector1):
    #: Module
    BLAKE2 = BLAKE2b
    #: Hash name
    name = "BLAKE2b"


class Blake2sTestVector1(Blake2TestVector1):
    #: Module
    BLAKE2 = BLAKE2s
    #: Hash name
    name = "BLAKE2s"


class Blake2TestVector2(unittest.TestCase):

    def _load_tests(self, test_vector_file):
        test_vectors = []
        with open(test_vector_file, "rt") as test_vector_fd:
            for line_number, line in enumerate(test_vector_fd):
                if line.strip() == "" or line.startswith("#"):
                    continue
                res = re.match(r"digest\(([0-9]+)\): ([0-9A-Fa-f]*)", line)
                if not res:
                    raise ValueError("Incorrect test vector format (line %d)"
                                     % line_number)
                key_size = int(res.group(1))
                result = unhexlify(tobytes(res.group(2)))
                test_vectors.append((key_size, result))
        return test_vectors

    def setUp(self):
        dir_comps = ("Hash", self.name)
        file_name = "tv2.txt"
        self.description = "%s tests" % self.name

        try:
            import pycryptodome_test_vectors  # type: ignore
        except ImportError:
            warnings.warn("Warning: skipping extended tests for %s" % self.name,
                           UserWarning)
            self.test_vectors = []
            return

        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        self.test_vectors = self._load_tests(full_file_name)

    def runTest(self):

        for key_size, result in self.test_vectors:
            next_data = b""
            for _ in range(100):
                h = self.BLAKE2.new(digest_bytes=self.max_bytes,
                                    key=b"A" * key_size)
                h.update(next_data)
                next_data = h.digest() + next_data
            self.assertEqual(h.digest(), result)


class Blake2bTestVector2(Blake2TestVector1):
    #: Module
    BLAKE2 = BLAKE2b
    #: Hash name
    name = "BLAKE2b"
    #: Max digest size in bytes
    max_bytes = 64


class Blake2sTestVector2(Blake2TestVector1):
    #: Module
    BLAKE2 = BLAKE2s
    #: Hash name
    name = "BLAKE2s"
    #: Max digest size in bytes
    max_bytes = 32


def get_tests(config={}):
    tests = []

    tests += list_test_cases(Blake2bTest)
    tests.append(Blake2bOfficialTestVector())
    tests.append(Blake2bTestVector1())
    tests.append(Blake2bTestVector2())

    tests += list_test_cases(Blake2sTest)
    tests.append(Blake2sOfficialTestVector())
    tests.append(Blake2sTestVector1())
    tests.append(Blake2sTestVector2())

    return tests


if __name__ == '__main__':
    import unittest
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
