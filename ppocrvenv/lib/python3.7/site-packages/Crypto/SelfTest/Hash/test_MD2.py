# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/MD2.py: Self-test for the MD2 hash function
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

"""Self-test suite for Crypto.Hash.MD2"""

from Crypto.Util.py3compat import *

# This is a list of (expected_result, input[, description]) tuples.
test_data = [
    # Test vectors from RFC 1319
    ('8350e5a3e24c153df2275c9f80692773', '', "'' (empty string)"),
    ('32ec01ec4a6dac72c0ab96fb34c0b5d1', 'a'),
    ('da853b0d3f88d99b30283a69e6ded6bb', 'abc'),
    ('ab4f496bfb2a530b219ff33031fe06b0', 'message digest'),

    ('4e8ddff3650292ab5a4108c3aa47940b', 'abcdefghijklmnopqrstuvwxyz',
        'a-z'),

    ('da33def2a42df13975352846c30338cd',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
        'A-Z, a-z, 0-9'),

    ('d5976f79d83d3a0dc9806c3c66f3efd8',
        '1234567890123456789012345678901234567890123456'
        + '7890123456789012345678901234567890',
        "'1234567890' * 8"),
]

def get_tests(config={}):
    from Crypto.Hash import MD2
    from .common import make_hash_tests
    return make_hash_tests(MD2, "MD2", test_data,
        digest_size=16,
        oid="1.2.840.113549.2.2")

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
