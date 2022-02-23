# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from cryptography.hazmat.backends.interfaces import Backend


def default_backend() -> Backend:
    from cryptography.hazmat.backends.openssl.backend import backend

    return backend
