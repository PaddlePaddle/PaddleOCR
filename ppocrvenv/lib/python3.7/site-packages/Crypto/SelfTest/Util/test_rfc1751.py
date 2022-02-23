import unittest

import binascii
from Crypto.Util.RFC1751 import key_to_english, english_to_key


class RFC1751_Tests(unittest.TestCase):

    def test1(self):
        data = [
                ('EB33F77EE73D4053', 'TIDE ITCH SLOW REIN RULE MOT'),
                ('CCAC2AED591056BE4F90FD441C534766', 'RASH BUSH MILK LOOK BAD BRIM AVID GAFF BAIT ROT POD LOVE'),
                ('EFF81F9BFBC65350920CDD7416DE8009', 'TROD MUTE TAIL WARM CHAR KONG HAAG CITY BORE O TEAL AWL')
                ]

        for key_hex, words in data:
            key_bin = binascii.a2b_hex(key_hex)

            w2 = key_to_english(key_bin)
            self.assertEqual(w2, words)

            k2 = english_to_key(words)
            self.assertEqual(k2, key_bin)

    def test_error_key_to_english(self):

        self.assertRaises(ValueError, key_to_english, b'0' * 7)


def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases
    tests = list_test_cases(RFC1751_Tests)
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
