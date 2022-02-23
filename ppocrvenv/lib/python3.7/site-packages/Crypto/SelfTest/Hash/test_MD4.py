# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/MD4.py: Self-test for the MD4 hash function
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

"""Self-test suite for Crypto.Hash.MD4"""

__revision__ = "$Id$"

from Crypto.Util.py3compat import *

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors from RFC 1320
    ('31d6cfe0d16ae931b73c59d7e0c089c0', '', "'' (empty string)"),
    ('bde52cb31de33e46245e05fbdbd6fb24', 'a'),
    ('a448017aaf21d8525fc10ae87aa6729d', 'abc'),
    ('d9130a8164549fe818874806e1c7014b', 'message digest'),

    ('d79e1c308aa5bbcdeea8ed63df412da9', 'abcdefghijklmnopqrstuvwxyz',
        'a-z'),

    ('043f8582f241db351ce627e153e7f0e4',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        'A-Z, a-z, 0-9'),

    ('e33b4ddc9c38f2199c3e7b164fcc0536',
        '1234567890123456789012345678901234567890123456'
        + '7890123456789012345678901234567890',
        "'1234567890' * 8"),
]

def get_tests(config={}):
    from Crypto.Hash import MD4
    from .common import make_hash_tests
    return make_hash_tests(MD4, "MD4", test_data,
        digest_size=16,
        oid="1.2.840.113549.2.4")

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
