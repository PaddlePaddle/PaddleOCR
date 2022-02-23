# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/test_RIPEMD160.py: Self-test for the RIPEMD-160 hash function
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

#"""Self-test suite for Crypto.Hash.RIPEMD160"""

from Crypto.Util.py3compat import *

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors downloaded 2008-09-12 from
    #   http://homes.esat.kuleuven.be/~bosselae/ripemd160.html
    ('9c1185a5c5e9fc54612808977ee8f548b2258d31', '', "'' (empty string)"),
    ('0bdc9d2d256b3ee9daae347be6f4dc835a467ffe', 'a'),
    ('8eb208f7e05d987a9b044a8e98c6b087f15a0bfc', 'abc'),
    ('5d0689ef49d2fae572b881b123a85ffa21595f36', 'message digest'),

    ('f71c27109c692c1b56bbdceb5b9d2865b3708dbc',
        'abcdefghijklmnopqrstuvwxyz',
        'a-z'),

    ('12a053384a9c0c88e405a06c27dcf49ada62eb2b',
        'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
        'abcdbcd...pnopq'),

    ('b0e20b6e3116640286ed3a87a5713079b21f5189',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        'A-Z, a-z, 0-9'),

    ('9b752e45573d4b39f4dbd3323cab82bf63326bfb',
        '1234567890' * 8,
        "'1234567890' * 8"),

    ('52783243c1697bdbe16d37f97f68f08325dc1528',
        'a' * 10**6,
        '"a" * 10**6'),
]

def get_tests(config={}):
    from Crypto.Hash import RIPEMD160
    from .common import make_hash_tests
    return make_hash_tests(RIPEMD160, "RIPEMD160", test_data,
        digest_size=20,
        oid="1.3.36.3.2.1")

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
