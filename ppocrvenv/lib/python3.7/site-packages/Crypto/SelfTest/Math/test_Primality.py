#
#  SelfTest/Math/test_Primality.py: Self-test for Primality module
#
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

"""Self-test for Math.Numbers"""

import unittest

from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Util.py3compat import *

from Crypto.Math.Numbers import Integer
from Crypto.Math.Primality import (
        PROBABLY_PRIME, COMPOSITE,
        miller_rabin_test, lucas_test,
        test_probable_prime,
        generate_probable_prime,
        generate_probable_safe_prime,
        )


class TestPrimality(unittest.TestCase):

    primes = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 2**127-1, 175637383534939453397801320455508570374088202376942372758907369518414308188137781042871856139027160010343454418881888953150175357127346872102307696660678617989191485418582475696230580407111841072614783095326672517315988762029036079794994990250662362650625650262324085116467511357592728695033227611029693067539)
    composites = (0, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 7*23, (2**19-1)*(2**67-1), 9746347772161,)

    def test_miller_rabin(self):
        for prime in self.primes:
            self.assertEqual(miller_rabin_test(prime, 3), PROBABLY_PRIME)
        for composite in self.composites:
            self.assertEqual(miller_rabin_test(composite, 3), COMPOSITE)
        self.assertRaises(ValueError, miller_rabin_test, -1, 3)

    def test_lucas(self):
        for prime in self.primes:
            res = lucas_test(prime)
            self.assertEqual(res, PROBABLY_PRIME)
        for composite in self.composites:
            res = lucas_test(composite)
            self.assertEqual(res, COMPOSITE)
        self.assertRaises(ValueError, lucas_test, -1)

    def test_is_prime(self):
        primes = (170141183460469231731687303715884105727,
                  19175002942688032928599,
                  1363005552434666078217421284621279933627102780881053358473,
                  2 ** 521 - 1)
        for p in primes:
            self.assertEqual(test_probable_prime(p), PROBABLY_PRIME)

        not_primes = (
                    4754868377601046732119933839981363081972014948522510826417784001,
                    1334733877147062382486934807105197899496002201113849920496510541601,
                    260849323075371835669784094383812120359260783810157225730623388382401,
                    )
        for np in not_primes:
            self.assertEqual(test_probable_prime(np), COMPOSITE)

        from Crypto.Util.number import sieve_base
        for p in sieve_base[:100]:
            res = test_probable_prime(p)
            self.assertEqual(res, PROBABLY_PRIME)

    def test_generate_prime_bit_size(self):
        p = generate_probable_prime(exact_bits=512)
        self.assertEqual(p.size_in_bits(), 512)

    def test_generate_prime_filter(self):
        def ending_with_one(number):
            return number % 10 == 1

        for x in range(20):
            q = generate_probable_prime(exact_bits=160,
                    prime_filter=ending_with_one)
            self.assertEqual(q % 10, 1)

    def test_generate_safe_prime(self):
        p = generate_probable_safe_prime(exact_bits=161)
        self.assertEqual(p.size_in_bits(), 161)

def get_tests(config={}):
    tests = []
    tests += list_test_cases(TestPrimality)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
