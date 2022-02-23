#
# Test script for Crypto.Util.RFC1751.
#
# Part of the Python Cryptography Toolkit
#
# Written by Andrew Kuchling and others
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

__revision__ = "$Id$"

import binascii
import unittest
from Crypto.Util import RFC1751
from Crypto.Util.py3compat import *

test_data = [('EB33F77EE73D4053', 'TIDE ITCH SLOW REIN RULE MOT'),
             ('CCAC2AED591056BE4F90FD441C534766',
              'RASH BUSH MILK LOOK BAD BRIM AVID GAFF BAIT ROT POD LOVE'),
             ('EFF81F9BFBC65350920CDD7416DE8009',
              'TROD MUTE TAIL WARM CHAR KONG HAAG CITY BORE O TEAL AWL')
             ]

class RFC1751Test_k2e (unittest.TestCase):

    def runTest (self):
        "Check converting keys to English"
        for key, words in test_data:
            key=binascii.a2b_hex(b(key))
            self.assertEqual(RFC1751.key_to_english(key), words)

class RFC1751Test_e2k (unittest.TestCase):

    def runTest (self):
        "Check converting English strings to keys"
        for key, words in test_data:
            key=binascii.a2b_hex(b(key))
            self.assertEqual(RFC1751.english_to_key(words), key)

# class RFC1751Test

def get_tests(config={}):
    return [RFC1751Test_k2e(), RFC1751Test_e2k()]

if __name__ == "__main__":
    unittest.main()
