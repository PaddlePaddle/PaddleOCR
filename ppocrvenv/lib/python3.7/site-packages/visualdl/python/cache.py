# Copyright (c) 2017 VisualDL Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================

import time


class MemCache(object):
    class Record:
        def __init__(self, value):
            self.time = time.time()
            self.value = value

        def clear(self):
            self.value = None

        def expired(self, timeout):
            return timeout > 0 and time.time() - self.time >= timeout

    '''
    A global dict to help cache some temporary data.
    '''

    def __init__(self, timeout=-1):
        self._timeout = timeout
        self._data = {}

    def set(self, key, value):
        self._data[key] = MemCache.Record(value)

    def get(self, key):
        rcd = self._data.get(key, None)
        if not rcd:
            return None
        # do not delete the key to accelerate speed
        if rcd.expired(self._timeout):
            rcd.clear()
            return None
        return rcd.value


if __name__ == '__main__':
    import unittest

    class TestMemCacheTest(unittest.TestCase):
        def setUp(self):
            self.cache = MemCache(timeout=1)

        def expire(self):
            self.cache.set("message", "hello")
            self.assertFalse(self.cache.expired(1))
            time.sleep(4)
            self.assertTrue(self.cache.expired(1))

        def test_have_key(self):
            self.cache.set('message', 'hello')
            self.assertTrue(self.cache.get('message'))
            time.sleep(1.1)
            self.assertFalse(self.cache.get('message'))
            self.assertTrue(self.cache.get("message") is None)

    unittest.main()
