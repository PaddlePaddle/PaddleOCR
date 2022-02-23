import unittest
from binascii import unhexlify, hexlify

from Crypto.Util.py3compat import tobytes
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import TupleHash128, TupleHash256


class TupleHashTest(unittest.TestCase):

    def new(self, *args, **kwargs):
        return self.TupleHash.new(*args, **kwargs)

    def test_new_positive(self):

        h = self.new()
        for new_func in self.TupleHash.new, h.new:

            for dbits in range(64, 1024 + 1, 8):
                hobj = new_func(digest_bits=dbits)
                self.assertEqual(hobj.digest_size * 8, dbits)

            for dbytes in range(8, 128 + 1):
                hobj = new_func(digest_bytes=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)

        hobj = h.new()
        self.assertEqual(hobj.digest_size, self.default_bytes)

    def test_new_negative(self):

        h = self.new()
        for new_func in self.TupleHash.new, h.new:
            self.assertRaises(TypeError, new_func,
                              digest_bytes=self.minimum_bytes,
                              digest_bits=self.minimum_bits)
            self.assertRaises(ValueError, new_func, digest_bytes=0)
            self.assertRaises(ValueError, new_func,
                              digest_bits=self.minimum_bits + 7)
            self.assertRaises(ValueError, new_func,
                              digest_bits=self.minimum_bits - 8)
            self.assertRaises(ValueError, new_func,
                              digest_bits=self.minimum_bytes - 1)

    def test_default_digest_size(self):
        digest = self.new().digest()
        self.assertEqual(len(digest), self.default_bytes)

    def test_update(self):
        h = self.new()
        h.update(b'')
        h.digest()

        h = self.new()
        h.update(b'')
        h.update(b'STRING1')
        h.update(b'STRING2')
        mac1 = h.digest()

        h = self.new()
        h.update(b'STRING1')
        h.update(b'STRING2')
        mac2 = h.digest()

        self.assertNotEqual(mac1, mac2)

    def test_update_negative(self):
        h = self.new()
        self.assertRaises(TypeError, h.update, u"string")
        self.assertRaises(TypeError, h.update, None)

    def test_digest(self):
        h = self.new()
        digest = h.digest()

        # hexdigest does not change the state
        self.assertEqual(h.digest(), digest)
        # digest returns a byte string
        self.assertTrue(isinstance(digest, type(b"digest")))

    def test_update_after_digest(self):
        msg = b"rrrrttt"

        # Normally, update() cannot be done after digest()
        h = self.new()
        h.update(msg)
        dig1 = h.digest()
        self.assertRaises(TypeError, h.update, dig1)

    def test_hex_digest(self):
        mac = self.new()
        digest = mac.digest()
        hexdigest = mac.hexdigest()

        # hexdigest is equivalent to digest
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        # hexdigest does not change the state
        self.assertEqual(mac.hexdigest(), hexdigest)
        # hexdigest returns a string
        self.assertTrue(isinstance(hexdigest, type("digest")))

    def test_bytearray(self):

        data = b"\x00\x01\x02"

        # Data can be a bytearray (during operation)
        data_ba = bytearray(data)

        h1 = self.new()
        h2 = self.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):

        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in (get_mv_ro, get_mv_rw):

            # Data can be a memoryview (during operation)
            data_mv = get_mv(data)

            h1 = self.new()
            h2 = self.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())


class TupleHash128Test(TupleHashTest):

    TupleHash = TupleHash128

    minimum_bytes = 8
    default_bytes = 64

    minimum_bits = 64
    default_bits = 512


class TupleHash256Test(TupleHashTest):

    TupleHash = TupleHash256

    minimum_bytes = 8
    default_bytes = 64

    minimum_bits = 64
    default_bits = 512


class NISTExampleTestVectors(unittest.TestCase):

    # http://csrc.nist.gov/groups/ST/toolkit/documents/Examples/TupleHash_samples.pdf
    test_data = [
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
            ),
            "",
            "C5 D8 78 6C 1A FB 9B 82 11 1A B3 4B 65 B2 C0 04"
            "8F A6 4E 6D 48 E2 63 26 4C E1 70 7D 3F FC 8E D1",
            "KMAC128 Sample #1 NIST",
            TupleHash128
        ),
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
            ),
            "My Tuple App",
            "75 CD B2 0F F4 DB 11 54 E8 41 D7 58 E2 41 60 C5"
            "4B AE 86 EB 8C 13 E7 F5 F4 0E B3 55 88 E9 6D FB",
            "KMAC128 Sample #2 NIST",
            TupleHash128
        ),
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
                "20 21 22 23 24 25 26 27 28",
            ),
            "My Tuple App",
            "E6 0F 20 2C 89 A2 63 1E DA 8D 4C 58 8C A5 FD 07"
            "F3 9E 51 51 99 8D EC CF 97 3A DB 38 04 BB 6E 84",
            "KMAC128 Sample #3 NIST",
            TupleHash128
        ),
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
            ),
            "",
            "CF B7 05 8C AC A5 E6 68 F8 1A 12 A2 0A 21 95 CE"
            "97 A9 25 F1 DB A3 E7 44 9A 56 F8 22 01 EC 60 73"
            "11 AC 26 96 B1 AB 5E A2 35 2D F1 42 3B DE 7B D4"
            "BB 78 C9 AE D1 A8 53 C7 86 72 F9 EB 23 BB E1 94",
            "KMAC256 Sample #4 NIST",
            TupleHash256
        ),
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
            ),
            "My Tuple App",
            "14 7C 21 91 D5 ED 7E FD 98 DB D9 6D 7A B5 A1 16"
            "92 57 6F 5F E2 A5 06 5F 3E 33 DE 6B BA 9F 3A A1"
            "C4 E9 A0 68 A2 89 C6 1C 95 AA B3 0A EE 1E 41 0B"
            "0B 60 7D E3 62 0E 24 A4 E3 BF 98 52 A1 D4 36 7E",
            "KMAC256 Sample #5 NIST",
            TupleHash256
        ),
        (
            (
                "00 01 02",
                "10 11 12 13 14 15",
                "20 21 22 23 24 25 26 27 28",
            ),
            "My Tuple App",
            "45 00 0B E6 3F 9B 6B FD 89 F5 47 17 67 0F 69 A9"
            "BC 76 35 91 A4 F0 5C 50 D6 88 91 A7 44 BC C6 E7"
            "D6 D5 B5 E8 2C 01 8D A9 99 ED 35 B0 BB 49 C9 67"
            "8E 52 6A BD 8E 85 C1 3E D2 54 02 1D B9 E7 90 CE",
            "KMAC256 Sample #6 NIST",
            TupleHash256
        ),



    ]

    def setUp(self):
        td = []
        for tv_in in self.test_data:
            tv_out = [None] * len(tv_in)

            tv_out[0] = []
            for string in tv_in[0]:
                tv_out[0].append(unhexlify(string.replace(" ", "")))

            tv_out[1] = tobytes(tv_in[1])    # Custom
            tv_out[2] = unhexlify(tv_in[2].replace(" ", ""))
            tv_out[3] = tv_in[3]
            tv_out[4] = tv_in[4]
            td.append(tv_out)
        self.test_data = td

    def runTest(self):

        for data, custom, digest, text, module in self.test_data:
            hd = module.new(custom=custom, digest_bytes=len(digest))
            for string in data:
                hd.update(string)
            self.assertEqual(hd.digest(), digest, msg=text)


def get_tests(config={}):
    tests = []

    tests += list_test_cases(TupleHash128Test)
    tests += list_test_cases(TupleHash256Test)
    tests.append(NISTExampleTestVectors())

    return tests


if __name__ == '__main__':
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
