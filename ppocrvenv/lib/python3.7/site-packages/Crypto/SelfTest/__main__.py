#! /usr/bin/env python
#
#  __main__.py : Stand-along loader for PyCryptodome test suite
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

from __future__ import print_function

import sys

from Crypto import SelfTest

slow_tests = not "--skip-slow-tests" in sys.argv
if not slow_tests:
    print("Skipping slow tests")

wycheproof_warnings = "--wycheproof-warnings" in sys.argv
if wycheproof_warnings:
    print("Printing Wycheproof warnings")

config = {'slow_tests' : slow_tests, 'wycheproof_warnings' : wycheproof_warnings }
SelfTest.run(stream=sys.stdout, verbosity=1, config=config)
