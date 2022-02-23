# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/common.py: Common code for Crypto.SelfTest.Hash
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

"""Self-testing for PyCrypto hash modules"""

import re
import sys
import unittest
import binascii
import Crypto.Hash
from binascii import hexlify, unhexlify
from Crypto.Util.py3compat import b, tobytes
from Crypto.Util.strxor import strxor_c

def t2b(hex_string):
    shorter = re.sub(br'\s+', b'', tobytes(hex_string))
    return unhexlify(shorter)


class HashDigestSizeSelfTest(unittest.TestCase):

    def __init__(self, hashmod, description, expected, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.expected = expected
        self.description = description
        self.extra_params = extra_params

    def shortDescription(self):
        return self.description

    def runTest(self):
        if "truncate" not in self.extra_params:
            self.assertTrue(hasattr(self.hashmod, "digest_size"))
            self.assertEqual(self.hashmod.digest_size, self.expected)
        h = self.hashmod.new(**self.extra_params)
        self.assertTrue(hasattr(h, "digest_size"))
        self.assertEqual(h.digest_size, self.expected)


class HashSelfTest(unittest.TestCase):

    def __init__(self, hashmod, description, expected, input, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.expected = expected.lower()
        self.input = input
        self.description = description
        self.extra_params = extra_params

    def shortDescription(self):
        return self.description

    def runTest(self):
        h = self.hashmod.new(**self.extra_params)
        h.update(self.input)

        out1 = binascii.b2a_hex(h.digest())
        out2 = h.hexdigest()

        h = self.hashmod.new(self.input, **self.extra_params)

        out3 = h.hexdigest()
        out4 = binascii.b2a_hex(h.digest())

        # PY3K: hexdigest() should return str(), and digest() bytes
        self.assertEqual(self.expected, out1)   # h = .new(); h.update(data); h.digest()
        if sys.version_info[0] == 2:
            self.assertEqual(self.expected, out2)   # h = .new(); h.update(data); h.hexdigest()
            self.assertEqual(self.expected, out3)   # h = .new(data); h.hexdigest()
        else:
            self.assertEqual(self.expected.decode(), out2)   # h = .new(); h.update(data); h.hexdigest()
            self.assertEqual(self.expected.decode(), out3)   # h = .new(data); h.hexdigest()
        self.assertEqual(self.expected, out4)   # h = .new(data); h.digest()

        # Verify that the .new() method produces a fresh hash object, except
        # for MD5 and SHA1, which are hashlib objects.  (But test any .new()
        # method that does exist.)
        if self.hashmod.__name__ not in ('Crypto.Hash.MD5', 'Crypto.Hash.SHA1') or hasattr(h, 'new'):
            h2 = h.new()
            h2.update(self.input)
            out5 = binascii.b2a_hex(h2.digest())
            self.assertEqual(self.expected, out5)


class HashTestOID(unittest.TestCase):
    def __init__(self, hashmod, oid, extra_params):
        unittest.TestCase.__init__(self)
        self.hashmod = hashmod
        self.oid = oid
        self.extra_params = extra_params

    def runTest(self):
        h = self.hashmod.new(**self.extra_params)
        self.assertEqual(h.oid, self.oid)


class ByteArrayTest(unittest.TestCase):

    def __init__(self, module, extra_params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.extra_params = extra_params

    def runTest(self):
        data = b("\x00\x01\x02")

        # Data can be a bytearray (during initialization)
        ba = bytearray(data)

        h1 = self.module.new(data, **self.extra_params)
        h2 = self.module.new(ba, **self.extra_params)
        ba[:1] = b'\xFF'
        self.assertEqual(h1.digest(), h2.digest())

        # Data can be a bytearray (during operation)
        ba = bytearray(data)

        h1 = self.module.new(**self.extra_params)
        h2 = self.module.new(**self.extra_params)

        h1.update(data)
        h2.update(ba)

        ba[:1] = b'\xFF'
        self.assertEqual(h1.digest(), h2.digest())


class MemoryViewTest(unittest.TestCase):

    def __init__(self, module, extra_params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.extra_params = extra_params

    def runTest(self):

        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in get_mv_ro, get_mv_rw:

            # Data can be a memoryview (during initialization)
            mv = get_mv(data)

            h1 = self.module.new(data, **self.extra_params)
            h2 = self.module.new(mv, **self.extra_params)
            if not mv.readonly:
                mv[:1] = b'\xFF'
            self.assertEqual(h1.digest(), h2.digest())

            # Data can be a memoryview (during operation)
            mv = get_mv(data)

            h1 = self.module.new(**self.extra_params)
            h2 = self.module.new(**self.extra_params)
            h1.update(data)
            h2.update(mv)
            if not mv.readonly:
                mv[:1] = b'\xFF'
            self.assertEqual(h1.digest(), h2.digest())


class MACSelfTest(unittest.TestCase):

    def __init__(self, module, description, result, data, key, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.result = t2b(result)
        self.data = t2b(data)
        self.key = t2b(key)
        self.params = params
        self.description = description

    def shortDescription(self):
        return self.description

    def runTest(self):

        result_hex = hexlify(self.result)

        # Verify result
        h = self.module.new(self.key, **self.params)
        h.update(self.data)
        self.assertEqual(self.result, h.digest())
        self.assertEqual(hexlify(self.result).decode('ascii'), h.hexdigest())

        # Verify that correct MAC does not raise any exception
        h.verify(self.result)
        h.hexverify(result_hex)

        # Verify that incorrect MAC does raise ValueError exception
        wrong_mac = strxor_c(self.result, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)
        self.assertRaises(ValueError, h.hexverify, "4556")

        # Verify again, with data passed to new()
        h = self.module.new(self.key, self.data, **self.params)
        self.assertEqual(self.result, h.digest())
        self.assertEqual(hexlify(self.result).decode('ascii'), h.hexdigest())

        # Test .copy()
        try:
            h = self.module.new(self.key, self.data, **self.params)
            h2 = h.copy()
            h3 = h.copy()

            # Verify that changing the copy does not change the original
            h2.update(b"bla")
            self.assertEqual(h3.digest(), self.result)

            # Verify that both can reach the same state
            h.update(b"bla")
            self.assertEqual(h.digest(), h2.digest())
        except NotImplementedError:
            pass

        # PY3K: Check that hexdigest() returns str and digest() returns bytes
        self.assertTrue(isinstance(h.digest(), type(b"")))
        self.assertTrue(isinstance(h.hexdigest(), type("")))

        # PY3K: Check that .hexverify() accepts bytes or str
        h.hexverify(h.hexdigest())
        h.hexverify(h.hexdigest().encode('ascii'))


def make_hash_tests(module, module_name, test_data, digest_size, oid=None,
                    extra_params={}):
    tests = []
    for i in range(len(test_data)):
        row = test_data[i]
        (expected, input) = map(tobytes,row[0:2])
        if len(row) < 3:
            description = repr(input)
        else:
            description = row[2]
        name = "%s #%d: %s" % (module_name, i+1, description)
        tests.append(HashSelfTest(module, name, expected, input, extra_params))

    name = "%s #%d: digest_size" % (module_name, len(test_data) + 1)
    tests.append(HashDigestSizeSelfTest(module, name, digest_size, extra_params))

    if oid is not None:
        tests.append(HashTestOID(module, oid, extra_params))

    tests.append(ByteArrayTest(module, extra_params))

    tests.append(MemoryViewTest(module, extra_params))

    return tests


def make_mac_tests(module, module_name, test_data):
    tests = []
    for i, row in enumerate(test_data):
        if len(row) == 4:
            (key, data, results, description, params) = list(row) + [ {} ]
        else:
            (key, data, results, description, params) = row
        name = "%s #%d: %s" % (module_name, i+1, description)
        tests.append(MACSelfTest(module, name, results, data, key, params))
    return tests

# vim:set ts=4 sw=4 sts=4 expandtab:
