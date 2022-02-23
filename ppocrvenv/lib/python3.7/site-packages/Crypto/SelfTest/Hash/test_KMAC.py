import unittest
from binascii import unhexlify, hexlify

from Crypto.Util.py3compat import tobytes
from Crypto.Util.strxor import strxor_c
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import KMAC128, KMAC256


class KMACTest(unittest.TestCase):

    def new(self, *args, **kwargs):
        return self.KMAC.new(key=b'X' * (self.minimum_key_bits // 8), *args, **kwargs)

    def test_new_positive(self):

        key = b'X' * 32

        h = self.new()
        for new_func in self.KMAC.new, h.new:

            for dbytes in range(self.minimum_bytes, 128 + 1):
                hobj = new_func(key=key, mac_len=dbytes)
                self.assertEqual(hobj.digest_size, dbytes)

            digest1 = new_func(key=key, data=b"\x90").digest()
            digest2 = new_func(key=key).update(b"\x90").digest()
            self.assertEqual(digest1, digest2)

            new_func(data=b"A", key=key, custom=b"g")

        hobj = h.new(key=key)
        self.assertEqual(hobj.digest_size, self.default_bytes)

    def test_new_negative(self):

        h = self.new()
        for new_func in self.KMAC.new, h.new:
            self.assertRaises(ValueError, new_func, key=b'X'*32,
                              mac_len=0)
            self.assertRaises(ValueError, new_func, key=b'X'*32,
                              mac_len=self.minimum_bytes - 1)
            self.assertRaises(TypeError, new_func,
                              key=u"string")
            self.assertRaises(TypeError, new_func,
                              data=u"string")

    def test_default_digest_size(self):
        digest = self.new(data=b'abc').digest()
        self.assertEqual(len(digest), self.default_bytes)

    def test_update(self):
        pieces = [b"\x0A" * 200, b"\x14" * 300]
        h = self.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.digest()
        h = self.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.digest(), digest)

    def test_update_negative(self):
        h = self.new()
        self.assertRaises(TypeError, h.update, u"string")

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
        h = self.new(mac_len=32, data=msg[:4])
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

    def test_verify(self):
        h = self.new()
        mac = h.digest()
        h.verify(mac)
        wrong_mac = strxor_c(mac, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)

    def test_hexverify(self):
        h = self.new()
        mac = h.hexdigest()
        h.hexverify(mac)
        self.assertRaises(ValueError, h.hexverify, "4556")

    def test_oid(self):

        oid = "2.16.840.1.101.3.4.2." + self.oid_variant
        h = self.new()
        self.assertEqual(h.oid, oid)

    def test_bytearray(self):

        key = b'0' * 32
        data = b"\x00\x01\x02"

        # Data and key can be a bytearray (during initialization)
        key_ba = bytearray(key)
        data_ba = bytearray(data)

        h1 = self.KMAC.new(data=data, key=key)
        h2 = self.KMAC.new(data=data_ba, key=key_ba)
        key_ba[:1] = b'\xFF'
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

        # Data can be a bytearray (during operation)
        data_ba = bytearray(data)

        h1 = self.new()
        h2 = self.new()
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):

        key = b'0' * 32
        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in (get_mv_ro, get_mv_rw):

            # Data and key can be a memoryview (during initialization)
            key_mv = get_mv(key)
            data_mv = get_mv(data)

            h1 = self.KMAC.new(data=data, key=key)
            h2 = self.KMAC.new(data=data_mv, key=key_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'
                key_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())

            # Data can be a memoryview (during operation)
            data_mv = get_mv(data)

            h1 = self.new()
            h2 = self.new()
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())


class KMAC128Test(KMACTest):

    KMAC = KMAC128

    minimum_key_bits = 128

    minimum_bytes = 8
    default_bytes = 64

    oid_variant = "19"


class KMAC256Test(KMACTest):

    KMAC = KMAC256

    minimum_key_bits = 256

    minimum_bytes = 8
    default_bytes = 64

    oid_variant = "20"


class NISTExampleTestVectors(unittest.TestCase):

    # https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/KMAC_samples.pdf
    test_data = [
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03",
            "",
            "E5 78 0B 0D 3E A6 F7 D3 A4 29 C5 70 6A A4 3A 00"
            "FA DB D7 D4 96 28 83 9E 31 87 24 3F 45 6E E1 4E",
            "Sample #1 NIST",
            KMAC128
        ),
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03",
            "My Tagged Application",
            "3B 1F BA 96 3C D8 B0 B5 9E 8C 1A 6D 71 88 8B 71"
            "43 65 1A F8 BA 0A 70 70 C0 97 9E 28 11 32 4A A5",
            "Sample #2 NIST",
            KMAC128
        ),
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F"
            "10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F"
            "20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F"
            "30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F"
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F"
            "60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F"
            "70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F"
            "80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F"
            "90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F"
            "A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF"
            "B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF"
            "C0 C1 C2 C3 C4 C5 C6 C7",
            "My Tagged Application",
            "1F 5B 4E 6C CA 02 20 9E 0D CB 5C A6 35 B8 9A 15"
            "E2 71 EC C7 60 07 1D FD 80 5F AA 38 F9 72 92 30",
            "Sample #3 NIST",
            KMAC128
        ),
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03",
            "My Tagged Application",
            "20 C5 70 C3 13 46 F7 03 C9 AC 36 C6 1C 03 CB 64"
            "C3 97 0D 0C FC 78 7E 9B 79 59 9D 27 3A 68 D2 F7"
            "F6 9D 4C C3 DE 9D 10 4A 35 16 89 F2 7C F6 F5 95"
            "1F 01 03 F3 3F 4F 24 87 10 24 D9 C2 77 73 A8 DD",
            "Sample #4 NIST",
            KMAC256
        ),
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F"
            "10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F"
            "20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F"
            "30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F"
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F"
            "60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F"
            "70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F"
            "80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F"
            "90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F"
            "A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF"
            "B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF"
            "C0 C1 C2 C3 C4 C5 C6 C7",
            "",
            "75 35 8C F3 9E 41 49 4E 94 97 07 92 7C EE 0A F2"
            "0A 3F F5 53 90 4C 86 B0 8F 21 CC 41 4B CF D6 91"
            "58 9D 27 CF 5E 15 36 9C BB FF 8B 9A 4C 2E B1 78"
            "00 85 5D 02 35 FF 63 5D A8 25 33 EC 6B 75 9B 69",
            "Sample #5 NIST",
            KMAC256
        ),
        (
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F",
            "00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F"
            "10 11 12 13 14 15 16 17 18 19 1A 1B 1C 1D 1E 1F"
            "20 21 22 23 24 25 26 27 28 29 2A 2B 2C 2D 2E 2F"
            "30 31 32 33 34 35 36 37 38 39 3A 3B 3C 3D 3E 3F"
            "40 41 42 43 44 45 46 47 48 49 4A 4B 4C 4D 4E 4F"
            "50 51 52 53 54 55 56 57 58 59 5A 5B 5C 5D 5E 5F"
            "60 61 62 63 64 65 66 67 68 69 6A 6B 6C 6D 6E 6F"
            "70 71 72 73 74 75 76 77 78 79 7A 7B 7C 7D 7E 7F"
            "80 81 82 83 84 85 86 87 88 89 8A 8B 8C 8D 8E 8F"
            "90 91 92 93 94 95 96 97 98 99 9A 9B 9C 9D 9E 9F"
            "A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 AA AB AC AD AE AF"
            "B0 B1 B2 B3 B4 B5 B6 B7 B8 B9 BA BB BC BD BE BF"
            "C0 C1 C2 C3 C4 C5 C6 C7",
            "My Tagged Application",
            "B5 86 18 F7 1F 92 E1 D5 6C 1B 8C 55 DD D7 CD 18"
            "8B 97 B4 CA 4D 99 83 1E B2 69 9A 83 7D A2 E4 D9"
            "70 FB AC FD E5 00 33 AE A5 85 F1 A2 70 85 10 C3"
            "2D 07 88 08 01 BD 18 28 98 FE 47 68 76 FC 89 65",
            "Sample #6 NIST",
            KMAC256
        ),
    ]

    def setUp(self):
        td = []
        for key, data, custom, mac, text, module in self.test_data:
            ni = (
                unhexlify(key.replace(" ", "")),
                unhexlify(data.replace(" ", "")),
                custom.encode(),
                unhexlify(mac.replace(" ", "")),
                text,
                module
            )
            td.append(ni)
        self.test_data = td

    def runTest(self):

        for key, data, custom, mac, text, module in self.test_data:
            h = module.new(data=data, key=key, custom=custom, mac_len=len(mac))
            mac_tag = h.digest()
            self.assertEqual(mac_tag, mac, msg=text)


def get_tests(config={}):
    tests = []

    tests += list_test_cases(KMAC128Test)
    tests += list_test_cases(KMAC256Test)
    tests.append(NISTExampleTestVectors())

    return tests


if __name__ == '__main__':
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
