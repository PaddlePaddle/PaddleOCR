# Copyright 2014 Baidu, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License") you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.

"""
This module defines string constants for HTTP headers
"""

# Standard HTTP Headers

AUTHORIZATION = b"Authorization"

CACHE_CONTROL = b"Cache-Control"

CONTENT_DISPOSITION = b"Content-Disposition"

CONTENT_ENCODING = b"Content-Encoding"

CONTENT_LENGTH = b"Content-Length"

CONTENT_MD5 = b"Content-MD5"

CONTENT_RANGE = b"Content-Range"

CONTENT_TYPE = b"Content-Type"

DATE = b"Date"

ETAG = b"ETag"

EXPIRES = b"Expires"

HOST = b"Host"

LAST_MODIFIED = b"Last-Modified"

RANGE = b"Range"

SERVER = b"Server"

USER_AGENT = b"User-Agent"

# BCE Common HTTP Headers

BCE_PREFIX = b"x-bce-"

BCE_ACL = b"x-bce-acl"

BCE_CONTENT_SHA256 = b"x-bce-content-sha256"

BCE_COPY_METADATA_DIRECTIVE = b"x-bce-metadata-directive"

BCE_COPY_SOURCE = b"x-bce-copy-source"

BCE_COPY_SOURCE_IF_MATCH = b"x-bce-copy-source-if-match"

BCE_COPY_SOURCE_IF_MODIFIED_SINCE = b"x-bce-copy-source-if-modified-since"

BCE_COPY_SOURCE_IF_NONE_MATCH = b"x-bce-copy-source-if-none-match"

BCE_COPY_SOURCE_IF_UNMODIFIED_SINCE = b"x-bce-copy-source-if-unmodified-since"

BCE_COPY_SOURCE_RANGE = b"x-bce-copy-source-range"

BCE_DATE = b"x-bce-date"

BCE_USER_METADATA_PREFIX = b"x-bce-meta-"

BCE_REQUEST_ID = b"x-bce-request-id"

# BOS HTTP Headers

BOS_DEBUG_ID = b"x-bce-bos-debug-id"

BOS_STORAGE_CLASS = b"x-bce-storage-class"

BOS_GRANT_READ = b'x-bce-grant-read'

BOS_GRANT_FULL_CONTROL = b'x-bce-grant-full-control'

BOS_FETCH_SOURCE = b"x-bce-fetch-source"

BOS_FETCH_MODE = b"x-bce-fetch-mode"

BOS_SERVER_SIDE_ENCRYPTION = b"x-bce-server-side-encryption"

BOS_SERVER_SIDE_ENCRYPTION_CUSTOMER_KEY = b"x-bce-server-side-encryption-customer-key"

BOS_SERVER_SIDE_ENCRYPTION_CUSTOMER_KEY_MD5 = b"x-bce-server-side-encryption-customer-key-md5"

BOS_RESTORE_TIER = b"x-bce-restore-tier"

BOS_RESTORE_DAYS = b"x-bce-restore-days"

BOS_SYMLINK_TARGET = b"x-bce-symlink-target"

BOS_FORBID_OVERWRITE = b"x-bce-forbid-overwrite"

# STS HTTP Headers

STS_SECURITY_TOKEN = b"x-bce-security-token"
