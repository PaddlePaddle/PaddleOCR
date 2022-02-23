# ===================================================================
#
# Copyright (c) 2016, Legrandin <helderijs@gmail.com>
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

import os
import re
import json
import errno
import binascii
import warnings
from binascii import unhexlify
from Crypto.Util.py3compat import FileNotFoundError


try:
    import pycryptodome_test_vectors  # type: ignore
    test_vectors_available = True
except ImportError:
    test_vectors_available = False


def _load_tests(dir_comps, file_in, description, conversions):
    """Load and parse a test vector file

    Return a list of objects, one per group of adjacent
    KV lines or for a single line in the form "[.*]".

    For a group of lines, the object has one attribute per line.
    """

    line_number = 0
    results = []

    class TestVector(object):
        def __init__(self, description, count):
            self.desc = description
            self.count = count
            self.others = []

    test_vector = None
    count = 0
    new_group = True

    while True:
        line_number += 1
        line = file_in.readline()
        if not line:
            if test_vector is not None:
                results.append(test_vector)
            break
        line = line.strip()

        # Skip comments and empty lines
        if line.startswith('#') or not line:
            new_group = True
            continue

        if line.startswith("["):
            if test_vector is not None:
                results.append(test_vector)
            test_vector = None
            results.append(line)
            continue

        if new_group:
            count += 1
            new_group = False
            if test_vector is not None:
                results.append(test_vector)
            test_vector = TestVector("%s (#%d)" % (description, count), count)

        res = re.match("([A-Za-z0-9]+) = ?(.*)", line)
        if not res:
            test_vector.others += [line]
        else:
            token = res.group(1).lower()
            data = res.group(2).lower()

            conversion = conversions.get(token, None)
            if conversion is None:
                if len(data) % 2 != 0:
                    data = "0" + data
                setattr(test_vector, token, binascii.unhexlify(data))
            else:
                setattr(test_vector, token, conversion(data))

        # This line is ignored
    return results


def load_test_vectors(dir_comps, file_name, description, conversions):
    """Load and parse a test vector file

    This function returns a list of objects, one per group of adjacent
    KV lines or for a single line in the form "[.*]".

    For a group of lines, the object has one attribute per line.
    """

    results = None

    try:
        if not test_vectors_available:
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    file_name)

        description = "%s test (%s)" % (description, file_name)

        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        with open(full_file_name) as file_in:
            results = _load_tests(dir_comps, file_in, description, conversions)

    except FileNotFoundError:
        warnings.warn("Warning: skipping extended tests for " + description,
                      UserWarning,
                      stacklevel=2)

    return results


def load_test_vectors_wycheproof(dir_comps, file_name, description,
                                 root_tag={}, group_tag={}, unit_tag={}):

    result = []
    try:
        if not test_vectors_available:
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    file_name)

        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        with open(full_file_name) as file_in:
            tv_tree = json.load(file_in)

    except FileNotFoundError:
        warnings.warn("Warning: skipping extended tests for " + description,
                      UserWarning,
                      stacklevel=2)
        return result

    class TestVector(object):
        pass

    common_root = {}
    for k, v in root_tag.items():
        common_root[k] = v(tv_tree)

    for group in tv_tree['testGroups']:

        common_group = {}
        for k, v in group_tag.items():
            common_group[k] = v(group)

        for test in group['tests']:
            tv = TestVector()

            for k, v in common_root.items():
                setattr(tv, k, v)
            for k, v in common_group.items():
                setattr(tv, k, v)

            tv.id = test['tcId']
            tv.comment = test['comment']
            for attr in 'key', 'iv', 'aad', 'msg', 'ct', 'tag', 'label', 'ikm', 'salt', 'info', 'okm', 'sig':
                if attr in test:
                    setattr(tv, attr, unhexlify(test[attr]))
            tv.filename = file_name

            for k, v in unit_tag.items():
                setattr(tv, k, v(test))

            tv.valid = test['result'] != "invalid"
            tv.warning = test['result'] == "acceptable"
            result.append(tv)

    return result

