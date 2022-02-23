# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

"""
This module defines some constants for BOS
"""

MAX_PUT_OBJECT_LENGTH = 5 * 1024 * 1024 * 1024
MAX_APPEND_OBJECT_LENGTH = 5 * 1024 * 1024 * 1024
MAX_USER_METADATA_SIZE = 2 * 1024
MIN_PART_NUMBER = 1
MAX_PART_NUMBER = 10000
URL_PREFIX = b"/"
